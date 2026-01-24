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
from src.phase import EightPhaseDetector
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
    
    def __init__(self, output_base_dir='data'):
        """
        Initialize pipeline.
        
        Args:
            output_base_dir: Base directory for all outputs
        """
        self.output_base_dir = Path(output_base_dir)
        
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
        
        # Detect swing bounds
        swing_start, swing_end = self._detect_swing_bounds(
            str(video_path), 
            motion_threshold=motion_threshold
        )
        
        # Crop video
        frames_written = self._crop_video(
            str(video_path), 
            str(output_path), 
            swing_start, 
            swing_end
        )
        
        self.cleaned_video_path = output_path
        
        print(f"   ✓ Cleaned video saved: {output_path.name}")
        print(f"   ✓ Frames: {swing_start}-{swing_end} → {frames_written} frames kept")
    
    def _detect_swing_bounds(self, video_path, motion_threshold=0.3, buffer_frames=0):
        """Detect swing start/end using motion detection."""
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        ret, prev_frame = cap.read()
        if not ret:
            return 0, total_frames
        
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        motion_scores = []
        
        print(f"   Analyzing motion in {total_frames} frames...", end='', flush=True)
        
        for frame_idx in range(1, total_frames):
            ret, frame = cap.read()
            if not ret:
                break
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            diff = cv2.absdiff(prev_gray, gray)
            motion_score = np.mean(diff)
            motion_scores.append(motion_score)
            prev_gray = gray
        
        cap.release()
        
        motion_scores = np.array(motion_scores)
        motion_threshold_value = motion_threshold * np.max(motion_scores) if len(motion_scores) > 0 else 0
        motion_frames = np.where(motion_scores > motion_threshold_value)[0]
        
        if len(motion_frames) == 0:
            print(f" No motion detected, keeping full video")
            return 0, total_frames
        
        swing_start = max(0, motion_frames[0] - buffer_frames)
        swing_end = min(total_frames - 1, motion_frames[-1] + buffer_frames + 1)
        
        print(f" Done!")
        return swing_start, swing_end
    
    def _crop_video(self, input_path, output_path, swing_start, swing_end):
        """Crop and save video to swing bounds."""
        cap = cv2.VideoCapture(input_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        total_frames = swing_end - swing_start
        frames_written = 0
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, swing_start)
        
        for i in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
            frames_written += 1
        
        cap.release()
        out.release()
        
        return frames_written
    
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
        
        Uses rule-based detector analyzing wrist Y trajectory
        to identify: Address, Takeaway, Mid-backswing, Top,
        Mid-downswing, Impact, Follow-through, Finish.
        """
        self.detector = EightPhaseDetector(
            csv_path=str(self.poses_csv_path),
            video_path=str(self.cleaned_video_path),
            output_dir=str(self.keyframes_dir / self.video_name)
        )
        
        # Create video-specific keyframes directory
        (self.keyframes_dir / self.video_name).mkdir(parents=True, exist_ok=True)
        
        # Detect phases
        self.phase_ranges = self.detector.detect_phases()
        
        print(f"   ✓ 8 phases detected:")
        for phase_name in self.detector.PHASE_NAMES:
            if phase_name in self.phase_ranges:
                start, end = self.phase_ranges[phase_name]
                duration = end - start + 1
                print(f"      • {phase_name:<18} frames {start:>4d}-{end:>4d} ({duration:>3d} frames)")
    
    def _extract_keyframes(self):
        """
        Step 4: Extract 8 key frames (one per phase).
        
        Selects the middle frame from each phase range
        and saves as JPEG images.
        """
        # Extract frames
        self.keyframes = self.detector.extract_8_frames()
        
        # Save phase CSV
        phases_csv_path = self.detector.save_phase_csv()
        self.phases_csv_path = phases_csv_path
        
        print(f"\n   ✓ 8 key frames extracted to: {self.keyframes_dir / self.video_name}")
        print(f"   ✓ Phase info saved to: {phases_csv_path}")
    
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
        
        print(f"\n🖼️  KEY FRAMES ({self.keyframes_dir / self.video_name}):")
        for phase_name in self.detector.PHASE_NAMES:
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
            'keyframes_dir': str(self.keyframes_dir / self.video_name),
            'keyframes': self.keyframes,
            'phase_ranges': self.phase_ranges
        }


def main():
    """
    Main entry point for the unified pipeline.
    
    Usage:
        python pipeline.py <video_path>
        python pipeline.py data/raw_videos/golf_swing_001.mp4
        python pipeline.py C:/path/to/my_swing.mp4
    """
    # Parse command line arguments
    if len(sys.argv) < 2:
        print("=" * 70)
        print("SwingAI Coach - Unified Pipeline")
        print("=" * 70)
        print("\nUsage: python pipeline.py <video_path>")
        print("\nExamples:")
        print("  python pipeline.py data/raw_videos/golf_swing_001.mp4")
        print("  python pipeline.py C:/Users/Videos/my_swing.mp4")
        print("\nPipeline:")
        print("  1. Clean video (remove pre/post swing)")
        print("  2. Extract poses (MediaPipe keypoints)")
        print("  3. Detect 8 phases (rule-based)")
        print("  4. Extract 8 key frames")
        print("=" * 70)
        return
    
    video_path = sys.argv[1]
    
    # Optional: show preview flag
    show_preview = '--preview' in sys.argv or '-p' in sys.argv
    
    # Run pipeline
    pipeline = GolfSwingPipeline(output_base_dir='data')
    
    try:
        results = pipeline.run(video_path, show_preview=show_preview)
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
