"""Trace exact scoring path for B90 impact"""
import pandas as pd
from src.biomechanics.phase_scorer import PhaseScorer
from src.biomechanics.scoring_config import SCORING_THRESHOLDS

# Load metrics
metrics_df = pd.read_csv('data/metrics/B90_cleaned_metrics.csv')

# Get impact frame range (50-65% of total)
total = len(metrics_df)
impact_start = int(total * 0.5)
impact_end = int(total * 0.65)

# Get sample metrics for impact
impact_row = metrics_df.iloc[impact_start + 5]  # Mid-impact frame

# Create metrics dict like score_impact expects
metrics = {col: impact_row[col] for col in metrics_df.columns if col != 'frame'}

print("="*100)
print("B90 IMPACT SCORING - TRACE")
print("="*100)
print()

print(f"Sample impact metrics:")
print(f"  x_factor: {metrics.get('x_factor', 'N/A')}")
print(f"  lead_arm_angle: {metrics.get('lead_arm_angle', 'N/A')}")
print(f"  wrist_hinge: {metrics.get('wrist_hinge', 'N/A')}")
print()

# Create scorer and check context
scorer = PhaseScorer()
context = getattr(scorer, '_window_context', {})

print(f"PhaseScorer context:")
print(f"  use_window: {context.get('use_window', False)}")
print(f"  has biomechanics: {'biomechanics' in context}")
print()

# Manually evaluate x_factor
print(f"Direct x_factor_unwind evaluation:")
x_factor_value = metrics.get('x_factor')
ideal_range = SCORING_THRESHOLDS["x_factor"]["impact_ideal"]

print(f"  Value: {x_factor_value}")
print(f"  Ideal range: {ideal_range}")

if x_factor_value is not None:
    score = scorer._evaluate_metric(
        x_factor_value,
        ideal_range,
        metric_name="x_factor"
    )
    print(f"  Evaluated score: {score}")
else:
    print(f"  ERROR: x_factor is None!")

print()
print("This explains why raw_score should be 100 (not 0)")
