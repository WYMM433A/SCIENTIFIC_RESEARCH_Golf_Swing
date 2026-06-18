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
    python pipeline.py data/raw_videos/videos_160/1.mp4
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
from src.biomechanics import GolfBiomechanics, SwingBiomechanicsEvaluator
from src.video.cleaner import detect_swing_bounds, crop_video
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
    
    def __init__(self, output_base_dir='data', phase_method='rule-based', model_path=None, player_level='amateur'):
        """
        Initialize pipeline.
        
        Args:
            output_base_dir: Base directory for all outputs
            phase_method: 'rule-based' or 'neural-network'
            model_path: Path to trained model (required for neural-network)
            player_level: 'beginner', 'amateur', or 'pro'
        """
        self.output_base_dir = Path(output_base_dir)
        self.phase_method = phase_method
        self.model_path = model_path
        self.player_level = player_level
        
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
        self.evaluator = SwingBiomechanicsEvaluator(player_level=player_level)
        
        # Pipeline state
        self.video_name = None
        self.cleaned_video_path = None
        self.poses_csv_path = None
        self.metrics_csv_path = None
        self.keyframes = {}
        self.evaluation_report = None
    
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
        print("[GOLF] SWINGAI COACH - UNIFIED PIPELINE")
        print("=" * 70)
        print(f"[VIDEO] Input: {video_path}")
        print(f"[OUTPUT] Output: {self.output_base_dir}")
        print("=" * 70 + "\n")
        
        # Step 1: Clean video
        print("[STEP 1/5] Cleaning Video...")
        self._clean_video(video_path)
        
        # Step 2: Extract poses
        print("\n[TASK] STEP 2/5: Extracting Poses...")
        self._extract_poses(show_preview)
        
        # Step 3: Detect phases
        print("\n[TASK] STEP 3/5: Detecting 8 Swing Phases...")
        self._detect_phases()
        
        # Step 4: Score phases
        print("\n[TASK] STEP 4/5: Scoring Phases...")
        self._score_swing()
        
        # Step 4b: Evaluate Biomechanics
        print("\n[TASK] STEP 4b/5: Performing Biomechanics Evaluation...")
        self._evaluate_biomechanics()
        
        # Step 5: Extract key frames
        print("\n[TASK] STEP 5/5: Extracting 8 Key Frames...")
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
        
        # Motion analysis for swing boundaries
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
        
        print(f"   [OK] Cleaned video saved: {output_path.name}")
        print(f"   [OK] Frames: {swing_start}-{swing_end} → {frames_written} frames kept")
    
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
            
            print(f"   [OK] Pose CSV saved: {self.poses_csv_path.name}")
            print(f"   [OK] Metrics CSV saved: {self.metrics_csv_path.name}")
            print(f"   [OK] Frames processed: {len(pose_df)}, Features: {len(pose_df.columns) - 1}")
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
        
        print(f"   [OK] 8 phases detected (method: {self.phase_method}):")
        for phase_name in self.predictor.PHASE_NAMES:
            if phase_name in self.phase_ranges:
                start, end = self.phase_ranges[phase_name]
                duration = end - start + 1
                print(f"      • {phase_name:<18} frames {start:>4d}-{end:>4d} ({duration:>3d} frames)")
    
    def _score_swing(self):
        print(f"   [DEBUG] phases_csv   = {self.phases_csv_path}")
        print(f"   [DEBUG] metrics_csv  = {self.metrics_csv_path}")
        print(f"   [DEBUG] swing_id     = {self.output_folder_name}")

        try:
            from train_phase_scorer import predict_scores, MODEL_OUT_PATH

            # predict_scores needs:
            # - data/metrics/{video_name}_cleaned_metrics.csv  ← already saved by _extract_poses()
            # - data/keyframes/{swing_id}/{video_name}_cleaned_8phases.csv  ← need to verify this exists

            swing_id = self.output_folder_name  # e.g. "Minh_Random_nn"

            print(f"   Scoring with XGBoost model...")
            result = predict_scores(
                swing_id   = swing_id,
                model_path = MODEL_OUT_PATH,
                annotate   = False,
            )

            if result is None:
                raise RuntimeError("predict_scores() returned None — check file paths")

            scores   = result["scores"]    # {phase_name: float}
            feedback = result["feedback"]  # {phase_name: [str, ...]}
            total    = result["total"]     # float

            # Build scoring_details in the same format the rest of pipeline expects
            scoring_details = []
            phase_scores    = {}

            for phase_name in self.predictor.PHASE_NAMES:
                score = scores.get(phase_name, 0.0)
                msgs  = feedback.get(phase_name, [])
                phase_scores[phase_name] = score

                print(f"      • {phase_name:<18} Score: {score:>6.1f}/100"
                    + (f"  → {msgs[0]}" if msgs else ""))

                scoring_details.append({
                    "phase":      phase_name,
                    "score":      score,
                    "confidence": 1.0,
                    "key_frame":  self.keyframes.get(phase_name, {}).get("key_frame", -1),
                    "feedback":   " | ".join(msgs) if msgs else "No issues detected",
                })

            overall_score = total or 0.0
            print(f"\n   [OK] Overall Score: {overall_score:.1f}/100")

            # Add overall row
            scoring_details.append({
                "phase":      "Overall",
                "score":      overall_score,
                "confidence": 1.0,
                "key_frame":  -1,
                "feedback":   f"OVERALL: {overall_score:.1f}/100",
            })

            # Save scores CSV
            self.scores_csv_path = self.metrics_dir / f"{self.video_name}_scores.csv"
            pd.DataFrame(scoring_details).to_csv(self.scores_csv_path, index=False)
            print(f"   [OK] Scores saved to: {self.scores_csv_path}")

            # Save detailed feedback CSV
            detailed_rows = []
            for phase_name in self.predictor.PHASE_NAMES:
                msgs = feedback.get(phase_name, [])
                for msg in msgs:
                    detailed_rows.append({
                        "phase":       phase_name,
                        "feedback":    msg,
                        "phase_score": scores.get(phase_name, 0.0),
                        "key_frame":   self.keyframes.get(phase_name, {}).get("key_frame", -1),
                    })

            self.feedback_details_csv_path = self.metrics_dir / f"{self.video_name}_feedback_detailed.csv"
            pd.DataFrame(detailed_rows).to_csv(self.feedback_details_csv_path, index=False)
            print(f"   [OK] Detailed feedback saved to: {self.feedback_details_csv_path}")

            self.phase_scores  = phase_scores
            self.overall_score = overall_score

        except Exception as e:
            print(f"   [FAIL] Error scoring swing: {e}")
            import traceback
            traceback.print_exc()
            self.phase_scores  = {}
            self.overall_score = 0
        print(f"\n   [OK] Generating visual feedback...")
        try:
            from train_phase_scorer import (
                draw_annotated_keyframe,
                _top_deviated_metric_names,
                PHASE_LABEL_MAP
            )

            with open(MODEL_OUT_PATH, "rb") as f:
                import pickle
                bundle = pickle.load(f)

            feature_cols = bundle["feature_cols"]
            benchmarks   = bundle.get("benchmarks", {})
            models       = bundle["models"]
            imputer      = bundle["imputer"]

            # Get features for this swing
            from train_phase_scorer import extract_features_for_swing
            feats = extract_features_for_swing(self.output_folder_name)

            if feats is not None:
                import numpy as np
                row   = {col: feats.get(col, np.nan) for col in feature_cols}
                X_raw = np.array([[row[c] for c in feature_cols]])
                X     = imputer.transform(X_raw)

                for score_col, phase_name in PHASE_LABEL_MAP.items():
                    if score_col not in models:
                        continue
                    score = self.phase_scores.get(phase_name, 0.0)
                    if score >= 80:
                        continue  # no annotation needed for good phases

                    deviated = _top_deviated_metric_names(
                        phase_name   = phase_name,
                        score_col    = score_col,
                        features     = feats,
                        benchmark    = benchmarks.get(score_col, {}),
                        model        = models[score_col],
                        feature_cols = feature_cols,
                        score        = score,
                    )
                    if deviated:
                        img_path = draw_annotated_keyframe(
                            swing_id         = self.output_folder_name,
                            phase_name       = phase_name,
                            deviated_metrics = deviated,
                            score            = score,
                        )
                        if img_path:
                            print(f"      • {phase_name:<18} → {img_path}")
            else:
                print(f"   [WARN] Could not extract features for visual annotation")

        except Exception as e:
            print(f"   [WARN] Visual annotation failed: {e}")
            import traceback
            traceback.print_exc()

    def _evaluate_biomechanics(self):
        """
        Step 4b: Perform scientific biomechanics evaluation using detected keyframes.
        """
        import json

        # Load extracted pose data
        poses_df = pd.read_csv(str(self.poses_csv_path))

        # Convert keyframes to simple format {phase_name: frame_idx} for the evaluator
        eval_keyframes = {name: info['key_frame'] for name, info in self.keyframes.items()}

        # Run scientific evaluation
        self.evaluation_report = self.evaluator.evaluate(poses_df, eval_keyframes)

        # Save detailed report as JSON
        report_path = self.metrics_dir / f"{self.video_name}_evaluation.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(self.evaluation_report, f, indent=4, ensure_ascii=False)

        print(f"   [OK] Evaluation complete. Overall Score: {self.evaluation_report['summary']['overall_score']}/100")
        print(f"   [OK] Report saved: {report_path.name}")

    def _extract_keyframes(self):
        """
        Step 5: Extract 8 key frames (one per phase).
        
        Keyframes are already extracted by the adapter in _detect_phases.
        This step just prints the summary.
        """
        print(f"\n   [OK] 8 key frames extracted to: {self.keyframes_dir / self.output_folder_name}")
        print(f"   [OK] Phase info saved to: {self.phases_csv_path}")
    
    def _print_summary(self):
        """Print pipeline completion summary."""
        print("\n" + "=" * 70)
        print("[DONE] PIPELINE COMPLETE!")
        print("=" * 70)
        
        print("\n📂 OUTPUT FILES:")
        print(f"   [VIDEO] Cleaned Video: {self.cleaned_video_path}")
        print(f"   [CHART] Pose Data:     {self.poses_csv_path}")
        print(f"   📈 Metrics:       {self.metrics_csv_path}")
        print(f"   🎯 Scores:        {getattr(self, 'scores_csv_path', 'N/A')}")
        print(f"   [TASK] Phase Info:    {self.phases_csv_path}")
        
        if self.evaluation_report:
            print(f"\nBIOMECHANICS SUMMARY (Score: {self.evaluation_report['summary']['overall_score']}/100):")
            if self.evaluation_report.get('priority_fixes'):
                print("   TOP IMPROVEMENTS:")
                for fix in self.evaluation_report['priority_fixes'][:3]:
                    print(f"      • {fix['issue']}: {fix['advice']}")
            else:
                print("   Excellent form! No major issues detected.")
        
        print(f"\n🖼[GOLF]  KEY FRAMES ({self.keyframes_dir / self.output_folder_name}):")
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
            'phase_scores': getattr(self, 'phase_scores', {}),
            'evaluation': self.evaluation_report
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
                        default='neural-network', help='Phase detection method (default: neural-network)')
    parser.add_argument('--model', type=str, default='models/pose_swingnet_trained.pth',
                        help='Path to trained model (for neural-network method)')
    parser.add_argument('--level', choices=['beginner', 'amateur', 'pro'], default='amateur',
                        help='Player skill level for benchmarking')
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
        model_path=model_path,
        player_level=args.level
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
