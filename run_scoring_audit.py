#!/usr/bin/env python3
"""
Scoring Audit Runner — Execute all diagnostic checks automatically.

Matches EXACT pipeline scoring methodology:
- Uses neural network phase detection (same as pipeline)
- Passes kinematic_data for mid_downswing (same as pipeline)
- Sets reference position at address (same as pipeline)

Usage:
    python run_scoring_audit.py
    
Output:
    - Console report with phase-by-phase comparison
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import json
import tempfile
import shutil

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.biomechanics.phase_scorer import PhaseScorer
from src.biomechanics.angles import GolfBiomechanics
from src.phase import create_predictor


class ScoringAudit:
    """Run all diagnostic checks on current scorer using EXACT pipeline methodology."""
    
    def __init__(self, poses_dir='data/extracted_poses'):
        self.poses_dir = poses_dir
        self.scorer = PhaseScorer()
        self.results = {}
        
    def _score_swing_from_poses(self, poses_df, poses_csv_path, video_path=None, name=""):
        """Score a full swing using EXACT pipeline approach with neural network phase detection."""
        
        # Create temporary output directory for phase detection
        temp_output_dir = tempfile.mkdtemp(prefix="audit_phases_")
        
        try:
            # Use neural network phase detection (same as pipeline)
            model_path = 'models/pose_swingnet_trained.pth'
            predictor = create_predictor('neural-network', model_path)
            phase_results = predictor.process(
                csv_path=str(poses_csv_path),
                video_path=str(video_path) if video_path else str(poses_csv_path),
                output_dir=str(temp_output_dir)
            )
            
            phase_ranges = phase_results.get('phase_ranges', {})
            keyframes = phase_results.get('keyframes', {})
            
        except Exception as e:
            print(f"   ❌ Phase detection failed: {e}")
            import traceback
            traceback.print_exc()
            return {
                'phase_scores': {phase: 0.0 for phase in 
                                ['address', 'takeaway', 'mid_backswing', 'top', 'mid_downswing', 
                                 'impact', 'follow_through', 'finish']},
                'overall_score': 0.0,
                'metrics': {}
            }
        
        # Step 2: Initialize biomechanics
        biomechanics = GolfBiomechanics()
        biomechanics.df = poses_df
        
        # Step 3: Set reference position at address
        # Handle both lowercase and capitalized keys
        address_phase = None
        for key in phase_ranges:
            if key.lower() == 'address':
                address_phase = phase_ranges[key]
                break
        
        if address_phase is None:
            address_phase = (0, 10)
        
        address_keyframe = address_phase[0]
        biomechanics.set_reference_position(frame=int(address_keyframe))
        
        # Step 4: Score each phase (MATCHING PIPELINE EXACTLY)
        phase_scores = {}
        phase_metrics = {}  # Store actual metrics for each phase
        phase_names = ['address', 'takeaway', 'mid_backswing', 'top', 'mid_downswing', 
                       'impact', 'follow_through', 'finish']
        
        for phase_name in phase_names:
            try:
                # Find matching phase key (handle case mismatch)
                phase_key = None
                for key in phase_ranges:
                    if key.lower().replace('-', '_') == phase_name:
                        phase_key = key
                        break
                
                if phase_key is None:
                    phase_scores[phase_name] = 0.0
                    continue
                
                phase_range = phase_ranges[phase_key]
                start_frame, end_frame = phase_range
                
                # Compute keyframe (midpoint) since keyframes dict is empty
                key_frame = (start_frame + end_frame) // 2
                
                # Calculate metrics at keyframe
                metrics = biomechanics.calculate_all_metrics(frame=int(key_frame))
                
                if not metrics:
                    phase_scores[phase_name] = 0.0
                    continue
                
                # Store metrics for this phase
                phase_metrics[phase_name] = metrics
                
                # Convert to pandas Series (same as pipeline)
                metrics_series = pd.Series(metrics)
                
                # Get kinematic sequence for mid_downswing (PIPELINE DOES THIS)
                kinematic_data = None
                if phase_name == 'mid_downswing':
                    kin_start = int(start_frame)
                    kin_end = int(end_frame)
                    
                    # Use broader window across top and impact
                    top_key = None
                    for key in phase_ranges:
                        if key.lower() == 'top':
                            top_key = key
                            break
                    
                    impact_key = None
                    for key in phase_ranges:
                        if key.lower() == 'impact':
                            impact_key = key
                            break
                    
                    if top_key and impact_key:
                        top_phase = phase_ranges[top_key]
                        impact_phase = phase_ranges[impact_key]
                        if int(impact_phase[1]) > int(top_phase[0]):
                            kin_start = int(top_phase[0])
                            kin_end = int(impact_phase[1])
                    
                    kinematic_data = biomechanics.compute_angular_velocity_sequence(
                        poses_df, kin_start, kin_end
                    )
                
                # Score phase (with kinematic_data for mid_downswing)
                score, details = self.scorer.score_phase_with_metrics(
                    phase_name,
                    metrics_series,
                    kinematic_data
                )
                
                phase_scores[phase_name] = float(score)
                
            except Exception as e:
                phase_scores[phase_name] = 0.0
        
        # Calculate overall score
        overall_score, overall_details = self.scorer.score_full_swing(phase_scores)
        
        # Cleanup
        shutil.rmtree(temp_output_dir, ignore_errors=True)
        
        return {
            'phase_scores': phase_scores,
            'overall_score': float(overall_score),
            'metrics': phase_metrics  # Return all metrics for each phase
        }
    
    def check_1_1_score_known_swings(self):
        """CHECK 1.1: Score known pro and beginner swings."""
        print("\n" + "="*60)
        print("CHECK 1.1: Score Known Pro & Beginner Swings")
        print("="*60)
        
        # Use specific files as requested
        pro_file = "B90_cleaned_poses.csv"           # Pro 9-10
        beginner_file = "B12_cleaned_poses.csv"      # Beginner 1-2
        
        pro_path = os.path.join(self.poses_dir, pro_file)
        beginner_path = os.path.join(self.poses_dir, beginner_file)
        
        if not os.path.exists(pro_path) or not os.path.exists(beginner_path):
            print("❌ Files not found:")
            print(f"   Pro:      {pro_path} {'✓' if os.path.exists(pro_path) else '✗'}")
            print(f"   Beginner: {beginner_path} {'✓' if os.path.exists(beginner_path) else '✗'}")
            return False
        
        print(f"Pro file:      {pro_file}")
        print(f"Beginner file: {beginner_file}\n")
        
        try:
            pro_poses = pd.read_csv(pro_path)
            beginner_poses = pd.read_csv(beginner_path)
            
            pro_result = self._score_swing_from_poses(pro_poses, pro_path)
            beginner_result = self._score_swing_from_poses(beginner_poses, beginner_path)
            
            pro_score = pro_result['overall_score']
            beginner_score = beginner_result['overall_score']
            difference = abs(pro_score - beginner_score)
            
            print("PRO SWING SCORES (min2):")
            for phase, score in pro_result['phase_scores'].items():
                print(f"  {phase:20s}: {score:6.1f}")
            print(f"  {'OVERALL':20s}: {pro_score:6.1f}")
            
            print("\nBEGINNER SWING SCORES (me):")
            for phase, score in beginner_result['phase_scores'].items():
                print(f"  {phase:20s}: {score:6.1f}")
            print(f"  {'OVERALL':20s}: {beginner_score:6.1f}")
            
            print(f"\n📊 Difference: {difference:.1f} pts")
            
            # Diagnosis
            print("\n🔍 DIAGNOSIS:")
            if difference > 15:
                print("   ✓ Good separation! Scoring working well.")
            elif difference < 5:
                print("   ⚠️  PROBLEM: Weak separation. Continue to Check 1.2 for metric breakdown")
            else:
                print(f"   ⚠️  Acceptable ({difference:.1f} pts). Check 1.2 for details.")
            
            self.results['check_1_1'] = {
                'pro_score': pro_score,
                'beginner_score': beginner_score,
                'difference': difference,
                'pro_result': pro_result,
                'beginner_result': beginner_result,
            }
            
            return True
            
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def check_1_2_metric_breakdown(self):
        """CHECK 1.2: Compare metrics between pro and beginner."""
        print("\n" + "="*60)
        print("CHECK 1.2: Metric Comparison Pro vs Beginner")
        print("="*60)
        
        try:
            pro_result = self.results['check_1_1']['pro_result']
            beginner_result = self.results['check_1_1']['beginner_result']
            
            # Extract phase scores for comparison
            print("\nPHASE-BY-PHASE BREAKDOWN:\n")
            
            for phase in pro_result['phase_scores'].keys():
                pro_score = pro_result['phase_scores'][phase]
                beginner_score = beginner_result['phase_scores'][phase]
                diff = pro_score - beginner_score
                
                # Visual indicator
                if diff > 10:
                    indicator = "🟢 (Good)"
                elif diff > 5:
                    indicator = "🟡 (OK)"
                elif diff > 0:
                    indicator = "🟠 (Weak)"
                else:
                    indicator = "🔴 (BROKEN)"
                
                print(f"{phase:20s}: Pro={pro_score:6.1f}  Beginner={beginner_score:6.1f}  Diff={diff:+6.1f}  {indicator}")
            
            print("\n" + "-"*60)
            print("INTERPRETATION:")
            print("-"*60)
            
            # Count how many phases have poor separation
            poor_phases = 0
            for phase in pro_result['phase_scores'].keys():
                pro_score = pro_result['phase_scores'][phase]
                beginner_score = beginner_result['phase_scores'][phase]
                diff = pro_score - beginner_score
                if diff < 5:
                    poor_phases += 1
            
            total_phases = len(pro_result['phase_scores'])
            
            if poor_phases > 0:
                print(f"\n⚠️  {poor_phases}/{total_phases} phases show weak separation (<5 pts)")
                print("\nPhases with poor discrimination likely use metrics that are:")
                print("  • View-dependent (shoulder_rotation, hip_rotation from side view)")
                print("  • Derived from poor metrics (x_factor from rotations)")
                print("  • Proxy estimates (lag_angle, wrist_hinge)")
                print("\n💡 Recommendation:")
                print("   1. Check which metrics these phases use")
                print("   2. Remove or reduce weight of problematic metrics")
                print("   3. Increase weight of reliable metrics (spine_angle, arm angles)")
                
                # Show detailed metrics for broken phases
                print("\n" + "="*60)
                print("DETAILED METRICS FOR BROKEN PHASES:")
                print("="*60)
                
                for phase in pro_result['phase_scores'].keys():
                    pro_score = pro_result['phase_scores'][phase]
                    beginner_score = beginner_result['phase_scores'][phase]
                    diff = pro_score - beginner_score
                    
                    if diff < 5:  # Only show broken phases
                        print(f"\n{phase.upper()} (Pro={pro_score:.1f} vs Beginner={beginner_score:.1f}, Diff={diff:+.1f})")
                        print("-" * 60)
                        
                        # Get actual metrics
                        pro_metrics = pro_result['metrics'].get(phase, {})
                        beginner_metrics = beginner_result['metrics'].get(phase, {})
                        
                        # Show key metrics for this phase
                        key_metrics = ['spine_angle', 'shoulder_rotation', 'hip_rotation', 'x_factor', 
                                      'wrist_angle', 'arm_extension', 'lag_angle', 'head_displacement']
                        
                        for metric in key_metrics:
                            if metric in pro_metrics or metric in beginner_metrics:
                                pro_val = pro_metrics.get(metric, 'N/A')
                                beginner_val = beginner_metrics.get(metric, 'N/A')
                                
                                if isinstance(pro_val, (int, float)) and isinstance(beginner_val, (int, float)):
                                    metric_diff = pro_val - beginner_val
                                    print(f"  {metric:20s}: Pro={pro_val:8.2f}  Beginner={beginner_val:8.2f}  (Diff={metric_diff:+.2f})")
                                else:
                                    print(f"  {metric:20s}: Pro={pro_val}  Beginner={beginner_val}")
            else:
                print(f"\n✓ All {total_phases} phases show good separation (>5 pts)")
            
            self.results['check_1_2'] = {
                'poor_phases': poor_phases,
                'total_phases': total_phases
            }
            
            return True
            
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def check_1_3_test_multiple_pairs(self):
        """CHECK 1.3: Test multiple swing pairs to validate scoring robustness."""
        print("\n" + "="*60)
        print("CHECK 1.3: Multi-Pair Validation")
        print("="*60)
        
        # Define multiple known pro vs beginner pairs and comparisons
        test_pairs = [
            ("min2_cleaned_poses.csv", "me_cleaned_poses.csv", "min2 (pro) vs me (beginner)"),
            ("min_indoor_cleaned_poses.csv", "me_cleaned_poses.csv", "min_indoor (pro) vs me (beginner)"),
            ("min3_cleaned_poses.csv", "me_cleaned_poses.csv", "min3 (pro) vs me (beginner)"),
            ("206_cleaned_poses.csv", "me_cleaned_poses.csv", "206 (WORLD PRO) vs me (beginner)"),
            ("72_cleaned_poses.csv", "me_cleaned_poses.csv", "72 (WORLD PRO) vs me (beginner)"),
            ("golf_swing_001_cleaned_poses.csv", "golf_swing_003_cleaned_poses.csv", "golf_swing_001 (pro) vs 003 (beginner)"),
            ("golf_swing_004_cleaned_poses.csv", "golf_swing_003_cleaned_poses.csv", "golf_swing_004 (pro) vs 003 (beginner)"),
            ("min2_cleaned_poses.csv", "min3_cleaned_poses.csv", "min2 vs min3 (pro vs pro)"),
        ]
        
        print("\nTesting swing pairs for consistency...\n")
        pair_results = []
        
        for pro_file, beginner_file, label in test_pairs:
            pro_path = os.path.join(self.poses_dir, pro_file)
            beginner_path = os.path.join(self.poses_dir, beginner_file)
            
            if not os.path.exists(pro_path) or not os.path.exists(beginner_path):
                print(f"⏭️  SKIP: {label} - files not found")
                continue
            
            try:
                pro_poses = pd.read_csv(pro_path)
                beginner_poses = pd.read_csv(beginner_path)
                
                pro_result = self._score_swing_from_poses(pro_poses, pro_path)
                beginner_result = self._score_swing_from_poses(beginner_poses, beginner_path)
                
                pro_score = pro_result['overall_score']
                beginner_score = beginner_result['overall_score']
                diff = pro_score - beginner_score
                
                # Check for anomalies in metrics
                anomalies = self._detect_metric_anomalies(pro_result['metrics'], beginner_result['metrics'])
                
                status = "✅" if diff > 5 and not anomalies else "⚠️" if not anomalies else "❌"
                print(f"{status} {label:30s}: Pro={pro_score:6.1f}  Beginner={beginner_score:6.1f}  Diff={diff:+6.1f}")
                
                if anomalies:
                    print(f"   🚨 ANOMALIES DETECTED:")
                    for anomaly in anomalies[:3]:  # Show first 3 anomalies
                        print(f"      - {anomaly}")
                
                pair_results.append({
                    'label': label,
                    'pro_score': pro_score,
                    'beginner_score': beginner_score,
                    'diff': diff,
                    'anomalies': anomalies,
                    'status': 'OK' if diff > 5 and not anomalies else 'POOR'
                })
                
            except Exception as e:
                print(f"❌ {label:30s}: ERROR - {str(e)[:50]}")
        
        self.results['check_1_3'] = pair_results
        return True
    
    def _detect_metric_anomalies(self, pro_metrics, beginner_metrics):
        """Detect physically impossible or suspicious metric values."""
        anomalies = []
        
        # Define expected ranges for key metrics
        ranges = {
            'spine_angle': (-40, 40),
            'shoulder_rotation': (150, 185),
            'hip_rotation': (150, 185),
            'arm_extension': (90, 180),
            'wrist_angle': (140, 180),
            'x_factor': (-5, 20),
            'lag_angle': (140, 180),
        }
        
        for phase_name in ['finish', 'impact', 'mid_downswing']:
            pro_phase = pro_metrics.get(phase_name, {})
            beginner_phase = beginner_metrics.get(phase_name, {})
            
            for metric, (min_val, max_val) in ranges.items():
                pro_val = pro_phase.get(metric)
                beginner_val = beginner_phase.get(metric)
                
                if isinstance(pro_val, (int, float)) and (pro_val < min_val or pro_val > max_val):
                    anomalies.append(f"{phase_name}.{metric}(pro)={pro_val:.1f}")
                
                if isinstance(beginner_val, (int, float)) and (beginner_val < min_val or beginner_val > max_val):
                    anomalies.append(f"{phase_name}.{metric}(beginner)={beginner_val:.1f}")
        
        return anomalies
    
    def run_all(self):
        """Execute all checks."""
        print("\n" + "█"*60)
        print("█  SCORING AUDIT — CHECK 1.1 & 1.2 & 1.3")
        print("█"*60)
        
        checks = [
            self.check_1_1_score_known_swings,
            self.check_1_2_metric_breakdown,
            self.check_1_3_test_multiple_pairs,
        ]
        
        for check_func in checks:
            try:
                check_func()
            except Exception as e:
                print(f"❌ Check failed: {e}")
                import traceback
                traceback.print_exc()
        
        # Summary
        self._print_summary()
    
    def _print_summary(self):
        """Print summary and recommendations."""
        print("\n" + "="*60)
        print("SUMMARY & RECOMMENDATIONS")
        print("="*60)
        
        check_1_1 = self.results.get('check_1_1', {})
        check_1_2 = self.results.get('check_1_2', {})
        
        print("\n1️⃣  Pro vs Beginner Separation (Overall):")
        diff = check_1_1.get('difference', 0)
        if diff > 15:
            print(f"   ✓ Excellent: {diff:.1f} pts difference")
        elif diff > 5:
            print(f"   ⚠️  Acceptable: {diff:.1f} pts difference")
        else:
            print(f"   ✗ BROKEN: {diff:.1f} pts difference (should be >15)")
        
        print("\n2️⃣  Phase-by-Phase Discrimination:")
        poor_phases = check_1_2.get('poor_phases', 0)
        total_phases = check_1_2.get('total_phases', 8)
        if poor_phases > 0:
            print(f"   ✗ {poor_phases}/{total_phases} phases show weak discrimination")
            print("   → These phases use unreliable metrics")
        else:
            print(f"   ✓ All {total_phases} phases discriminate well")
        
        print("\n" + "-"*60)
        print("NEXT STEPS:")
        print("-"*60)
        if diff < 5:
            print("1. Poor overall separation detected")
            print("   → Metrics are not differentiating pros from beginners")
            print("\n2. Check which PHASES have weak discrimination (see above)")
            print("   → Remove metrics that don't discriminate")
            print("   → Increase weight of reliable metrics")
            print("\n3. After changes: Re-run audit to verify improvement")
        else:
            print("1. Overall separation is acceptable")
            print("2. Focus on improving weak phases identified above")
            print("3. Consider removing low-discriminating metrics")


if __name__ == '__main__':
    # Create outputs directory if needed
    os.makedirs('outputs', exist_ok=True)
    
    try:
        audit = ScoringAudit()
        audit.run_all()
        
        print("\n" + "█"*60)
        print("✓ Audit complete!")
        print("█"*60 + "\n")
    except Exception as e:
        print(f"\n❌ Audit failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
