"""
Swing Comparator - Compare user swing to benchmarks

Analyzes biomechanical metrics and identifies areas for improvement.
"""

import pandas as pd
import numpy as np
from typing import Dict, List
from .angles import extract_metrics_from_row
from .benchmarks import BENCHMARKS, get_status

class SwingBiomechanicsEvaluator:
    """
    Phân tích toàn bộ swing dựa trên các keyframes đã detect.
    """
    def __init__(self, player_level: str = "amateur"):
        self.player_level = player_level

    def evaluate(self, poses_df: pd.DataFrame, phase_keyframes: Dict[str, int]) -> Dict:
        """
        Analyzes the entire swing based on detected keyframes.
        phase_keyframes: e.g., {'Top': 124, 'Impact': 150, 'Address': 20}
        """
        report = {
            "summary": {},
            "detailed_metrics": [],
            "priority_fixes": []
        }
        
        # 1. Determine Baseline Target Line from Address (if available) for relative rotation
        baseline_h_vec = None
        if 'Address' in phase_keyframes:
            addr_idx = phase_keyframes['Address']
            if addr_idx < len(poses_df):
                row = poses_df.iloc[addr_idx]
                # Create hip vector on X-Z plane as baseline
                # Ensure columns exist before access
                if not all(col in row for col in ['left_hip_x', 'left_hip_z', 'right_hip_x', 'right_hip_z']):
                    print(f"WARNING: Missing hip data at Address frame ({addr_idx}). Cannot determine baseline.")
                else:
                    h_l = np.array([row['left_hip_x'], row['left_hip_z']])
                    h_r = np.array([row['right_hip_x'], row['right_hip_z']])
                    baseline_h_vec = h_r - h_l
            else:
                print(f"WARNING: Address frame ({addr_idx}) is out of bounds for poses_df.")

        # 2. Calculate metrics for critical frames
        key_metrics = {}
        for phase, frame_idx in phase_keyframes.items():
            if frame_idx < len(poses_df):
                row = poses_df.iloc[frame_idx].to_dict()
                metrics_for_frame = extract_metrics_from_row(row, baseline_h_vec=baseline_h_vec)
                if metrics_for_frame is not None:
                    key_metrics[phase] = metrics_for_frame
                else:
                    print(f"WARNING: Could not extract metrics for phase '{phase}' at frame {frame_idx} due to missing landmarks.")
            else:
                print(f"WARNING: Phase frame '{phase}' ({frame_idx}) is out of bounds for poses_df.")

        # 3. Temporal Analysis (Head Stability Enhancement)
        # Collect all head (nose) positions from available frames
        all_head_positions = []
        for _, row_data in poses_df.iterrows():
            if 'nose_x' in row_data and 'nose_y' in row_data and 'nose_z' in row_data:
                # MediaPipe output via SwingAnalyzer uses 'nose_visibility'
                vis = row_data.get('nose_visibility', 0.0)
                if vis >= 0.5:
                    all_head_positions.append(np.array([row_data['nose_x'], row_data['nose_y'], row_data['nose_z']]))
        
        head_stability_val = 0.0
        if len(all_head_positions) > 1: # Minimum 2 points required for standard deviation
            head_pts = np.array(all_head_positions)
            # Standard deviation across X, Y, Z axes, normalized and scaled
            head_stability_val = np.linalg.norm(np.std(head_pts, axis=0)) * 100 
        
        # Append global metrics
        key_metrics["Global"] = {"head_stability": head_stability_val}

        # 4. Benchmarking and Weighted Scoring
        scores = []
        total_weight = 0
        for metric_id, cfg in BENCHMARKS.items():
            phase = cfg["phase"]
            
            # Verify phase and metric exist in extracted data
            if phase in key_metrics and metric_id in key_metrics[phase]:
                val = key_metrics[phase][metric_id]
                
                # Skip if value is None
                if val is None:
                    continue
                
                status, color = get_status(val, cfg, self.player_level)
                weight = cfg.get("priority_weight", 1.0)
                
                metric_result = {
                    "name": cfg["label"],
                    "value": round(val, 1),
                    "unit": cfg["unit"],
                    "status": status,
                    "color": color,
                    "ideal_range": f"{cfg[self.player_level]['min']}-{cfg[self.player_level]['max']}",
                    "feedback": cfg["hint"] if status != "Good" else "Excellent!"
                }
                report["detailed_metrics"].append(metric_result)
                
                score = 100 if status == "Good" else (60 if status == "Fair" else 20)
                scores.append(score * weight)
                total_weight += weight
                
                # 5. Priority Fix Logic based on weights and deviation
                if status != "Good":
                    deviation = abs(val - cfg[self.player_level]["ideal"])
                    report["priority_fixes"].append({
                        "score_impact": deviation * weight, # Calculate error impact
                        "issue": cfg["label"],
                        "advice": cfg["hint"]
                    })

        # Sort fixes by score_impact descending
        report["priority_fixes"].sort(key=lambda x: x["score_impact"], reverse=True)
        
        # Final summary
        report["summary"] = {
            "overall_score": int(np.sum(scores) / total_weight) if total_weight > 0 else 0,
            "player_level": self.player_level,
            "total_metrics_analyzed": len(scores)
        }
        
        return report
