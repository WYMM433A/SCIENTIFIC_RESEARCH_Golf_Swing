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
from src.biomechanics import PhaseScorer, GolfBiomechanics
from train_phase_scorer import predict_scores, MODEL_OUT_PATH
from src import config
import pandas as pd


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
    
    def __init__(self, output_base_dir='data', phase_method='rule-based', model_path=None,
                 scoring_backend='xgboost'):
        """
        Initialize pipeline.
        
        Args:
            output_base_dir: Base directory for all outputs
            phase_method: 'rule-based' or 'neural-network'
            model_path: Path to trained model (required for neural-network)
            scoring_backend: 'xgboost' or 'biomechanics'
        """
        self.output_base_dir = Path(output_base_dir)
        self.phase_method = phase_method
        self.model_path = model_path
        self.scoring_backend = scoring_backend
        
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
        
        # Step 3b: Score phases (NEW)
        print("\n📋 STEP 3b/4: Scoring Phases...")
        self._score_swing()
        
        # Step 4: Extract key frames
        print("\n📋 STEP 4/4: Extracting 8 Key Frames...")
        self._extract_keyframes()
        
        # Summary
        self._print_summary()
        
        return self._get_results()

    def _calibrate_overall_score(self, overall_score, phase_scores, phase_metrics):
        """
        Apply targeted calibration so catastrophic finish faults are penalized hard
        without globally crushing pro-level scores.

        Args:
            overall_score: Weighted score from phase aggregation
            phase_scores: Dict of phase -> phase score
            phase_metrics: Dict of normalized phase -> metrics dict

        Returns:
            (calibrated_score, calibration_details)
        """
        calibrated = float(overall_score)
        calibration_details = {
            'base_score': float(overall_score),
            'finish_rotation': None,
            'finish_rotation_penalty': 0.0,
        }

        finish_metrics = phase_metrics.get('finish', {})
        finish_rotation = finish_metrics.get('hip_rotation')
        if finish_rotation is not None:
            finish_rotation = float(finish_rotation)
            calibration_details['finish_rotation'] = finish_rotation

            # Catastrophic finish-rotation penalty:
            # front-view clips with near-zero finish rotation indicate severe quality issues.
            if finish_rotation < 120.0:
                rotation_gap = 120.0 - finish_rotation
                rotation_penalty = min(35.0, rotation_gap * 0.30)
                calibrated -= rotation_penalty
                calibration_details['finish_rotation_penalty'] = float(rotation_penalty)

        calibrated = float(np.clip(calibrated, 0.0, 100.0))
        calibration_details['calibrated_score'] = calibrated
        return calibrated, calibration_details
    
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
    
    def _score_swing(self):
        """Step 3b dispatcher for scoring backend."""
        if self.scoring_backend == 'xgboost':
            self._score_swing_xgboost()
        else:
            self._score_swing_biomechanics()

    def _score_swing_xgboost(self):
        """
        Step 3b: Score each phase with the trained XGBoost phase scorer.

        Uses phase outputs from Step 3 and metrics from Step 2.
        Writes summary scores and detailed feedback CSVs.
        """
        try:
            swing_id = self.output_folder_name  # e.g. video_nn or video_rb

            print("   Scoring with XGBoost model...")
            result = predict_scores(
                swing_id=swing_id,
                model_path=MODEL_OUT_PATH,
                annotate=True,
            )

            if result is None:
                raise RuntimeError(
                    "predict_scores() returned None. "
                    "Ensure model exists and metrics/phase CSV files are available."
                )

            scores = result.get("scores", {})
            feedback = result.get("feedback", {})
            total = result.get("total", 0.0) or 0.0
            images = result.get("images", {})

            scoring_details = []
            detailed_feedback_rows = []

            print("   Phase scores:")
            for phase_name in self.predictor.PHASE_NAMES:
                score = float(scores.get(phase_name, 0.0))
                msgs = feedback.get(phase_name, []) or []

                print(f"      • {phase_name:<18} Score: {score:>6.1f}/100")

                key_frame = self.keyframes.get(phase_name, {}).get('key_frame', -1)
                scoring_details.append({
                    'phase': phase_name,
                    'score': score,
                    'confidence': 1.0,
                    'key_frame': int(key_frame) if key_frame is not None else -1,
                    'feedback': " | ".join(msgs) if msgs else "No issues detected",
                })

                for msg in msgs:
                    detailed_feedback_rows.append({
                        'phase': phase_name,
                        'feedback': msg,
                        'phase_score': score,
                        'key_frame': int(key_frame) if key_frame is not None else -1,
                    })

            overall_score = float(total)
            print(f"\n   ✓ Overall Score: {overall_score:.1f}/100")

            # Save scoring results
            self.scores_csv_path = self.metrics_dir / f"{self.video_name}_scores.csv"
            scoring_details.append({
                'phase': 'Overall',
                'score': overall_score,
                'confidence': 1.0,
                'key_frame': -1,
                'feedback': f"OVERALL: {overall_score:.1f}/100",
            })
            pd.DataFrame(scoring_details).to_csv(self.scores_csv_path, index=False)
            print(f"   ✓ Scores saved to: {self.scores_csv_path}")

            # Save detailed feedback diagnostics
            self.feedback_details_csv_path = self.metrics_dir / f"{self.video_name}_feedback_detailed.csv"
            pd.DataFrame(detailed_feedback_rows).to_csv(self.feedback_details_csv_path, index=False)
            print(f"   ✓ Detailed feedback saved to: {self.feedback_details_csv_path}")

            if images:
                print("\n   ✓ Annotated keyframes saved:")
                for phase_name, img_path in images.items():
                    print(f"      • {phase_name:<18} → {img_path}")

            # Store for results
            self.phase_scores = {k: float(v) for k, v in scores.items()}
            self.overall_score = overall_score

        except Exception as e:
            print(f"   ✗ Error scoring swing: {e}")
            import traceback
            traceback.print_exc()
            self.phase_scores = {}
            self.overall_score = 0

    def _score_swing_biomechanics(self):
        """
        Step 3b: Score each phase biomechanically (legacy backend).
        
        Evaluates:
        - Individual phase scores (0-100)
        - Overall swing score
        - Kinematic sequence (mid-downswing)
        - Feedback and recommendations
        
        Saves results to CSV for analysis.
        """
        try:
            # Load pose data
            pose_df = pd.read_csv(self.poses_csv_path)

            def _normalize_phase_name(name: str) -> str:
                return str(name).strip().lower().replace('-', '_').replace(' ', '_')

            def _get_phase_range(phase_name: str):
                target = _normalize_phase_name(phase_name)
                for key, value in self.phase_ranges.items():
                    if _normalize_phase_name(key) == target:
                        return value
                return None

            def _get_phase_keyframe(phase_name: str):
                target = _normalize_phase_name(phase_name)
                for key, value in self.keyframes.items():
                    if _normalize_phase_name(key) == target:
                        return value.get('key_frame')
                return None
            
            # Initialize scorer
            scorer = PhaseScorer()
            
            # Initialize biomechanics for reference setting
            biomechanics = GolfBiomechanics()
            biomechanics.df = pose_df
            
            # Get address frame for reference
            address_phase = _get_phase_range('address') or (0, 10)
            address_key_frame = _get_phase_keyframe('address')
            if address_key_frame is None:
                address_key_frame = address_phase[0]
            
            # Set reference position at address
            biomechanics.set_reference_position(frame=int(address_key_frame))
            print(f"   ✓ Reference position set at address (frame {int(address_key_frame)})")
            
            # Score each phase
            phase_scores = {}
            phase_metrics = {}
            scoring_details = []
            detailed_feedback_rows = []
            
            print(f"   Scoring phases:")
            for phase_name in self.predictor.PHASE_NAMES:
                phase_range = _get_phase_range(phase_name)
                if phase_range is None:
                    continue
                
                start_frame, end_frame = phase_range
                key_frame = _get_phase_keyframe(phase_name)
                if key_frame is None:
                    key_frame = start_frame
                
                # Get metrics for this frame
                frame_df = pose_df[pose_df['frame'] == int(key_frame)]
                
                if len(frame_df) == 0:
                    phase_scores[phase_name] = 0
                    continue

                phase_normalized = phase_name.replace('-', '_').lower()
                
                # Calculate metrics
                metrics = biomechanics.calculate_all_metrics(frame=int(key_frame))
                phase_metrics[phase_normalized] = metrics
                
                # Convert metrics dict into a pandas Series for scorer input
                metrics_series = pd.Series(metrics)
                
                # Get kinematic sequence for mid-downswing
                kinematic_data = None
                if phase_name.lower().replace('-', '_') == 'mid_downswing':
                    kin_start = int(start_frame)
                    kin_end = int(end_frame)

                    # Sequence is more stable across a broader transition window.
                    top_phase = _get_phase_range('top')
                    impact_phase = _get_phase_range('impact')
                    if top_phase and impact_phase and int(impact_phase[1]) > int(top_phase[0]):
                        kin_start = int(top_phase[0])
                        kin_end = int(impact_phase[1])

                    kinematic_data = biomechanics.compute_angular_velocity_sequence(
                        pose_df, kin_start, kin_end
                    )
                
                # Score the phase
                score, details = scorer.score_phase_with_metrics(
                    phase_normalized, 
                    metrics_series, 
                    kinematic_data
                )
                
                phase_scores[phase_name] = score
                
                # Generate feedback
                feedback = scorer.generate_feedback(
                    phase_name, 
                    score, 
                    metrics,
                    details.get('components', {})
                )

                # Build detailed actionable feedback rows for export
                feedback_details = scorer.generate_feedback_details(
                    phase_name,
                    metrics,
                    details.get('components', {})
                )
                for item in feedback_details:
                    item_row = dict(item)
                    item_row['key_frame'] = int(key_frame)
                    item_row['phase_score'] = float(score)
                    detailed_feedback_rows.append(item_row)
                
                print(f"      • {phase_name:<18} Score: {score:>6.1f}/100")
                
                # Store details
                scoring_details.append({
                    'phase': phase_name,
                    'score': score,
                    'confidence': details.get('confidence', 0.5),
                    'key_frame': int(key_frame),
                    'feedback': feedback
                })
            
            # Calculate overall score
            overall_score_raw, overall_details = scorer.score_full_swing(phase_scores)
            overall_score, calibration_details = self._calibrate_overall_score(
                overall_score_raw,
                phase_scores,
                phase_metrics,
            )
            
            print(f"\n   ✓ Overall Score: {overall_score:.1f}/100")
            if calibration_details.get('finish_rotation_penalty', 0.0) > 0:
                print(
                    "   ✓ Calibration applied: "
                    f"finish rotation {calibration_details.get('finish_rotation', 0.0):.2f}°, "
                    f"penalty {calibration_details['finish_rotation_penalty']:.1f}"
                )
            
            # Save scoring results
            self.scores_csv_path = self.metrics_dir / f"{self.video_name}_scores.csv"
            scoring_details.append({
                'phase': 'Overall',
                'score': float(overall_score),
                'confidence': 1.0,
                'key_frame': -1,
                'feedback': (
                    f"OVERALL: {overall_score:.1f}/100 "
                    f"(raw={overall_score_raw:.1f}, "
                    f"finish_rotation_penalty={calibration_details.get('finish_rotation_penalty', 0.0):.1f})"
                ),
            })
            scoring_df = pd.DataFrame(scoring_details)
            scoring_df.to_csv(self.scores_csv_path, index=False)
            print(f"   ✓ Scores saved to: {self.scores_csv_path}")

            # Save detailed actionable feedback diagnostics
            self.feedback_details_csv_path = self.metrics_dir / f"{self.video_name}_feedback_detailed.csv"
            detailed_feedback_df = pd.DataFrame(detailed_feedback_rows)
            detailed_feedback_df.to_csv(self.feedback_details_csv_path, index=False)
            print(f"   ✓ Detailed feedback saved to: {self.feedback_details_csv_path}")
            
            # Store for results
            self.phase_scores = phase_scores
            self.overall_score = overall_score
            
        except Exception as e:
            print(f"   ✗ Error scoring swing: {e}")
            import traceback
            traceback.print_exc()
            self.phase_scores = {}
            self.overall_score = 0
    
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
        print(f"   🎯 Scores:        {getattr(self, 'scores_csv_path', 'N/A')}")
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
            'scores_csv': str(getattr(self, 'scores_csv_path', '')),
            'feedback_details_csv': str(getattr(self, 'feedback_details_csv_path', '')),
            'phases_csv': str(self.phases_csv_path),
            'keyframes_dir': str(self.keyframes_dir / self.output_folder_name),
            'keyframes': self.keyframes,
            'phase_ranges': self.phase_ranges,
            'overall_score': getattr(self, 'overall_score', 0),
            'phase_scores': getattr(self, 'phase_scores', {})
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
    parser.add_argument('--scoring-backend', choices=['xgboost', 'biomechanics'],
                        default='xgboost', help='Scoring backend (default: xgboost)')
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
        print("  --scoring-backend  Scoring: 'xgboost' or 'biomechanics'")
        print("  --preview, -p   Show live preview during processing")
        print("\nExamples:")
        print("  python pipeline.py video.mp4")
        print("  python pipeline.py video.mp4 --method neural-network")
        print("  python pipeline.py video.mp4 --method neural-network --scoring-backend xgboost")
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
        model_path=model_path,
        scoring_backend=args.scoring_backend,
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
