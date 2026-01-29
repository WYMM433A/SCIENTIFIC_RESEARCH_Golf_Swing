"""
SwingAI Coach - Unified Pipeline
================================
Complete end-to-end golf swing analysis pipeline.

Input: Raw video file
Output: 8 key frames (one per swing phase)

Pipeline Steps:
1. Clean video (motion detection crop)
2. Extract poses (MediaPipe keypoints)
3. Calculate metrics (biomechanical analysis)
4. Detect 8 phases (rule-based detector)
5. Extract key frames (one per phase)

Usage:
    python pipeline.py <video_path>
    python pipeline.py data/raw_videos/golf_swing_001.mp4
    python pipeline.py C:/path/to/my_swing.mp4
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime

# Import from src package
from src.pose import SwingAnalyzer
from src.phase import create_predictor
from src.video.cleaner import detect_swing_bounds, crop_video
from src import config


class GolfSwingPipeline:
    """
    Unified pipeline for golf swing analysis.
    
    Takes a raw video and outputs:
    - Cleaned video (cropped to swing bounds)
    - Pose CSV (33 landmarks per frame)
    - Metrics CSV (biomechanical measurements)
    - 8 key frames (one per phase)
    - Phase information CSV
    """
    
    def __init__(self, output_base_dir='data', phase_method='rule-based', model_path=None):
        """
        Initialize pipeline.
        
        Args:
            output_base_dir: Base directory for all outputs
            phase_method: 'rule-based' or 'neural-network'
            model_path: Path to trained model (required for neural-network)
        """
        self.output_base_dir = Path(output_base_dir)
        self.phase_method = phase_method
        self.model_path = model_path
        
        # Create output directories
        self.cleaned_video_dir = self.output_base_dir / 'cleaned_videos'
        self.poses_dir = self.output_base_dir / 'extracted_poses'
        self.metrics_dir = self.output_base_dir / 'metrics'
        self.keyframes_dir = self.output_base_dir / 'keyframes'
        
        for directory in [self.cleaned_video_dir, self.poses_dir, 
                          self.metrics_dir, self.keyframes_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.analyzer = SwingAnalyzer()
        
        # Pipeline state
        self.video_name = None
        self.cleaned_video_path = None
        self.poses_csv_path = None
        self.metrics_csv_path = None
        self.keyframes = {}
    
    def run(self, video_path, show_preview=False):
        """
        Run the complete pipeline.
        
        Args:
            video_path: Path to raw video file
            show_preview: Whether to show live preview during processing
            
        Returns:
            dict: Results containing paths to all outputs
        """
        video_path = Path(video_path)
        
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        self.video_name = video_path.stem
        
        print("\n" + "=" * 70)
        print("🏌️ SWINGAI COACH - UNIFIED PIPELINE")
        print("=" * 70)
        print(f"📹 Input: {video_path}")
        print(f"📁 Output: {self.output_base_dir}")
        print("=" * 70 + "\n")
        
        # Step 1: Clean video
        print("📋 STEP 1/4: Cleaning Video...")
        self._clean_video(video_path)
        
        # Step 2: Extract poses
        print("\n📋 STEP 2/4: Extracting Poses...")
        self._extract_poses(show_preview)
        
        # Step 3: Detect phases
        print("\n📋 STEP 3/4: Detecting 8 Swing Phases...")
        self._detect_phases()
        
        # Step 4: Extract key frames
        print("\n📋 STEP 4/4: Extracting 8 Key Frames...")
        self._extract_keyframes()
        
        # Summary
        self._print_summary()
        
        return self._get_results()
    
    def _clean_video(self, video_path, motion_threshold=0.3):
        """
        Step 1: Clean video by cropping to swing bounds.
        
        Uses motion detection to find swing start/end and removes
        pre/post-swing idle footage.
        """
        output_path = self.cleaned_video_dir / f"{self.video_name}_cleaned.mp4"
        
        print(f"   Analyzing motion...", end='', flush=True)
        
        # Use shared cleaner functions
        swing_start, swing_end = detect_swing_bounds(
            str(video_path), 
            motion_threshold=motion_threshold,
            buffer_frames=0
        )
        
        # Get original video dimensions for cropping
        cap = cv2.VideoCapture(str(video_path))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        
        # Crop video using shared function
        frames_written = crop_video(
            str(video_path), 
            str(output_path), 
            swing_start, 
            swing_end,
            target_width=width,
            target_height=height
        )
        
        self.cleaned_video_path = output_path
        
        print(f"   ✓ Cleaned video saved: {output_path.name}")
        print(f"   ✓ Frames: {swing_start}-{swing_end} → {frames_written} frames kept")
    
    def _extract_poses(self, show_preview=False):
        """
        Step 2: Extract poses from cleaned video.
        
        Uses MediaPipe PoseLandmarker to extract 33 body keypoints
        per frame, plus biomechanical metrics.
        """
        csv_name = f"{self.video_name}_cleaned_poses.csv"
        metrics_name = f"{self.video_name}_cleaned_metrics.csv"
        
        self.poses_csv_path = self.poses_dir / csv_name
        self.metrics_csv_path = self.metrics_dir / metrics_name
        
        # Process video
        pose_df = self.analyzer.processVideo(
            video_path=str(self.cleaned_video_path),
            output_csv=str(self.poses_csv_path),
            show_preview=show_preview
        )
        
        # Save metrics
        if pose_df is not None:
            metrics_df = self.analyzer.getMetricsDataFrame()
            metrics_df.to_csv(str(self.metrics_csv_path), index=False)
            
            print(f"   ✓ Pose CSV saved: {self.poses_csv_path.name}")
            print(f"   ✓ Metrics CSV saved: {self.metrics_csv_path.name}")
            print(f"   ✓ Frames processed: {len(pose_df)}, Features: {len(pose_df.columns) - 1}")
        else:
            raise RuntimeError("Pose extraction failed. Check video quality.")
    
    def _detect_phases(self):
        """
        Step 3: Detect 8 golf swing phases.
        
        Uses either rule-based (wrist trajectory) or neural network (Bi-LSTM)
        to identify: Address, Takeaway, Mid-backswing, Top,
        Mid-downswing, Impact, Follow-through, Finish.
        """
        # Create video-specific keyframes directory with method suffix
        # _rb for rule-based, _nn for neural-network
        method_suffix = '_rb' if self.phase_method == 'rule-based' else '_nn'
        self.output_folder_name = f"{self.video_name}{method_suffix}"
        output_dir = self.keyframes_dir / self.output_folder_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create phase predictor using adapter
        self.predictor = create_predictor(self.phase_method, self.model_path)
        
        # Process video
        self.phase_results = self.predictor.process(
            csv_path=str(self.poses_csv_path),
            video_path=str(self.cleaned_video_path),
            output_dir=str(output_dir)
        )
        
        self.phase_ranges = self.phase_results['phase_ranges']
        self.keyframes = self.phase_results['keyframes']
        self.phases_csv_path = self.phase_results['phases_csv']
        
        print(f"   ✓ 8 phases detected (method: {self.phase_method}):")
        for phase_name in self.predictor.PHASE_NAMES:
            if phase_name in self.phase_ranges:
                start, end = self.phase_ranges[phase_name]
                duration = end - start + 1
                print(f"      • {phase_name:<18} frames {start:>4d}-{end:>4d} ({duration:>3d} frames)")
    
    def _extract_keyframes(self):
        """
        Step 4: Extract 8 key frames (one per phase).
        
        Keyframes are already extracted by the adapter in _detect_phases.
        This step just prints the summary.
        """
        print(f"\n   ✓ 8 key frames extracted to: {self.keyframes_dir / self.output_folder_name}")
        print(f"   ✓ Phase info saved to: {self.phases_csv_path}")
    
    def _print_summary(self):
        """Print pipeline completion summary."""
        print("\n" + "=" * 70)
        print("✅ PIPELINE COMPLETE!")
        print("=" * 70)
        
        print("\n📂 OUTPUT FILES:")
        print(f"   📹 Cleaned Video: {self.cleaned_video_path}")
        print(f"   📊 Pose Data:     {self.poses_csv_path}")
        print(f"   📈 Metrics:       {self.metrics_csv_path}")
        print(f"   📋 Phase Info:    {self.phases_csv_path}")
        
        print(f"\n🖼️  KEY FRAMES ({self.keyframes_dir / self.output_folder_name}):")
        for phase_name in self.predictor.PHASE_NAMES:
            if phase_name in self.keyframes:
                frame_info = self.keyframes[phase_name]
                print(f"   • {phase_name:<18} → Frame {frame_info['key_frame']}")
        
        print("\n" + "=" * 70 + "\n")
    
    def _get_results(self):
        """Return dictionary of all output paths."""
        return {
            'video_name': self.video_name,
            'cleaned_video': str(self.cleaned_video_path),
            'poses_csv': str(self.poses_csv_path),
            'metrics_csv': str(self.metrics_csv_path),
            'phases_csv': str(self.phases_csv_path),
            'keyframes_dir': str(self.keyframes_dir / self.output_folder_name),
            'keyframes': self.keyframes,
            'phase_ranges': self.phase_ranges
        }


def main():
    """
    Main entry point for the unified pipeline.
    
    Usage:
        python pipeline.py <video_path> [--method rule-based|neural-network] [--model path]
        python pipeline.py data/raw_videos/golf_swing_001.mp4
        python pipeline.py video.mp4 --method neural-network --model models/pose_swingnet_trained.pth
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='SwingAI Coach - Golf Swing Analysis Pipeline')
    parser.add_argument('video', nargs='?', help='Path to video file')
    parser.add_argument('--method', '-m', choices=['rule-based', 'neural-network'], 
                        default='rule-based', help='Phase detection method (default: rule-based)')
    parser.add_argument('--model', type=str, default='models/pose_swingnet_trained.pth',
                        help='Path to trained model (for neural-network method)')
    parser.add_argument('--preview', '-p', action='store_true', help='Show live preview')
    
    args = parser.parse_args()
    
    if args.video is None:
        print("=" * 70)
        print("SwingAI Coach - Unified Pipeline")
        print("=" * 70)
        print("\nUsage: python pipeline.py <video_path> [options]")
        print("\nOptions:")
        print("  --method, -m    Phase detection: 'rule-based' or 'neural-network'")
        print("  --model         Path to trained model (for neural-network)")
        print("  --preview, -p   Show live preview during processing")
        print("\nExamples:")
        print("  python pipeline.py video.mp4")
        print("  python pipeline.py video.mp4 --method neural-network")
        print("  python pipeline.py video.mp4 -m neural-network --model models/pose_swingnet_best.pth")
        print("\nPipeline:")
        print("  1. Clean video (remove pre/post swing)")
        print("  2. Extract poses (MediaPipe keypoints)")
        print("  3. Detect 8 phases (rule-based or neural-network)")
        print("  4. Extract 8 key frames")
        print("=" * 70)
        return
    
    # Set model path only for neural-network
    model_path = args.model if args.method == 'neural-network' else None
    
    # Run pipeline
    pipeline = GolfSwingPipeline(
        output_base_dir='data',
        phase_method=args.method,
        model_path=model_path
    )
    
    try:
        results = pipeline.run(args.video, show_preview=args.preview)
        print(f"🎉 Success! 8 key frames saved to: {results['keyframes_dir']}")
    except FileNotFoundError as e:
        print(f"\n❌ ERROR: {e}")
        print("Please check the video path and try again.")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
