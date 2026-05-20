"""Debug why all impact scores are 85"""
import pandas as pd
from src.biomechanics.phase_scorer import PhaseScorer
from src.biomechanics.scoring_config import SCORING_THRESHOLDS, METRIC_WEIGHTS

# Load metrics
metrics_df = pd.read_csv('data/metrics/B90_cleaned_metrics.csv')

# Impact is roughly frames 50-65% 
total = len(metrics_df)
impact_start = int(total * 0.5)
impact_end = int(total * 0.65)

print("="*80)
print("IMPACT SCORING DEBUG - Multiple Frames")
print("="*80)
print()

scorer = PhaseScorer()
phase = "impact"

for frame_idx in [impact_start, impact_start+5, impact_start+10]:
    row = metrics_df.iloc[frame_idx]
    metrics_dict = row.to_dict()
    
    print(f"Frame {frame_idx}:")
    print(f"  x_factor: {metrics_dict.get('x_factor', 'N/A')}")
    
    # Manually calculate like score_impact does
    components = {}
    
    # x_factor_unwind
    if "x_factor" in metrics_dict and metrics_dict["x_factor"] is not None:
        x_factor_val = metrics_dict["x_factor"]
        ideal = SCORING_THRESHOLDS["x_factor"]["impact_ideal"]
        
        x_score = scorer._evaluate_metric(x_factor_val, ideal, metric_name="x_factor")
        components["x_factor_unwind"] = x_score * METRIC_WEIGHTS[phase]["x_factor_unwind"]
        
        print(f"    x_factor raw score: {x_score}")
        print(f"    x_factor weighted: {components['x_factor_unwind']}")
    
    # lag_release
    lag_val = metrics_dict.get('lag_angle') or metrics_dict.get('wrist_angle') or metrics_dict.get('wrist_hinge')
    if lag_val is not None:
        ideal = SCORING_THRESHOLDS["lag_angle"]["impact_ideal"]
        lag_score = scorer._evaluate_metric(lag_val, ideal, metric_name="lag_angle")
        components["lag_release"] = lag_score * METRIC_WEIGHTS[phase]["lag_release"]
        print(f"    lag_release raw score: {lag_score}")
        print(f"    lag_release weighted: {components['lag_release']}")
    
    # wrist_angle
    wrist_val = metrics_dict.get('wrist_angle') or metrics_dict.get('wrist_hinge')
    if wrist_val is not None:
        ideal = SCORING_THRESHOLDS["wrist_angle"]["impact_ideal"]
        wrist_score = scorer._evaluate_metric(wrist_val, ideal, metric_name="wrist_angle")
        components["wrist_angle"] = wrist_score * METRIC_WEIGHTS[phase]["wrist_angle"]
        print(f"    wrist_angle raw score: {wrist_score}")
        print(f"    wrist_angle weighted: {components['wrist_angle']}")
    
    # arm_extension
    arm_val = metrics_dict.get('arm_extension') or metrics_dict.get('lead_arm_angle')
    if arm_val is not None:
        ideal = SCORING_THRESHOLDS["lead_arm_angle"]["impact_ideal"]
        arm_score = scorer._evaluate_metric(arm_val, ideal, metric_name="lead_arm_angle")
        components["arm_extension"] = arm_score * METRIC_WEIGHTS[phase]["arm_extension"]
        print(f"    arm_extension raw score: {arm_score}")
        print(f"    arm_extension weighted: {components['arm_extension']}")
    
    # Calculate normalized score
    total_weight = sum(
        METRIC_WEIGHTS[phase].get(comp, 0.0)
        for comp in components
    )
    if total_weight > 0:
        final_score = sum(components.values()) / total_weight
    else:
        final_score = 0.0
    
    print(f"  Total weight: {total_weight:.4f}")
    print(f"  Final score: {final_score:.1f}")
    print()
