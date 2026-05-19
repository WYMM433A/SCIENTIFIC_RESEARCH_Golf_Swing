import pandas as pd
from src.biomechanics.angles import GolfBiomechanics
from src.biomechanics.scoring_config import SCORING_THRESHOLDS, METRIC_WEIGHTS

# Load pose data
pose_csv = 'data/extracted_poses/B80O_cleaned_poses.csv'
df = pd.read_csv(pose_csv)

# Use correct top frame from pipeline
phases_csv = 'data/keyframes/B80O_nn/B80O_cleaned_8phases.csv'
phases_df = pd.read_csv(phases_csv)
top_frame = phases_df[phases_df['Phase'] == 'Top']['Key_Frame'].values[0]

biomech = GolfBiomechanics(df)
metrics = biomech.calculate_all_metrics(frame=top_frame)

print(f"TOP PHASE - Frame {top_frame}")
print("="*70)

# Debug what score_top() is doing
print("\nDEBUG score_top() logic:")
print(f"  shoulder_rotation: {metrics.get('shoulder_rotation', 'NOT FOUND')}")
print(f"  x_factor:          {metrics.get('x_factor', 'NOT FOUND')}")
print(f"  spine_angle:       {metrics.get('spine_angle', 'NOT FOUND')}")
print(f"  wrist_angle:       {metrics.get('wrist_angle', 'NOT FOUND')}")
print(f"  head_displacement: {metrics.get('head_displacement', 'NOT FOUND')}")

# score_top() uses shoulder_rotation for coil, not x_factor
shoulder_value = metrics.get('shoulder_rotation')
print(f"\nCoil evaluation (using shoulder_rotation):")
print(f"  Value: {shoulder_value}°")
print(f"  Threshold: {SCORING_THRESHOLDS['shoulder_rotation']['backswing_ideal']}")

# The problem: score_top() is using shoulder_rotation, but that's always ~176° (front-view limitation)
# It should use x_factor instead!
