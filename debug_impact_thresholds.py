import pandas as pd
from src.biomechanics.angles import GolfBiomechanics
from src.biomechanics.scoring_config import SCORING_THRESHOLDS

# Load pose data
pose_csv = 'data/extracted_poses/B80O_cleaned_poses.csv'
df = pd.read_csv(pose_csv)

# Use correct frames from pipeline
phases_csv = 'data/keyframes/B80O_nn/B80O_cleaned_8phases.csv'
phases_df = pd.read_csv(phases_csv)
phases_dict = dict(zip(phases_df['Phase'], phases_df['Key_Frame']))

biomech = GolfBiomechanics(df)

impact_frame = phases_dict['Impact']
metrics = biomech.calculate_all_metrics(frame=impact_frame)

print("IMPACT PHASE COMPONENT ANALYSIS")
print("="*70)

# 1. LAG RELEASE
lag_angle = metrics.get('lag_angle', 0)
lag_thresholds = SCORING_THRESHOLDS['lag_angle']['impact_ideal']
print(f"\nLAG_RELEASE:")
print(f"  Value: {lag_angle:.2f}°")
print(f"  Ideal range: {lag_thresholds}")
print(f"  Status: {'✓ IN RANGE' if lag_thresholds[0] <= lag_angle <= lag_thresholds[1] else '✗ OUT OF RANGE'}")

# 2. WRIST ANGLE
wrist_angle = metrics.get('wrist_angle', 0)
wrist_thresholds = SCORING_THRESHOLDS['wrist_angle']['impact_ideal']
print(f"\nWRIST_ANGLE:")
print(f"  Value: {wrist_angle:.2f}°")
print(f"  Ideal range: {wrist_thresholds}")
print(f"  Status: {'✓ IN RANGE' if wrist_thresholds[0] <= wrist_angle <= wrist_thresholds[1] else '✗ OUT OF RANGE'}")

# 3. HEAD DISPLACEMENT
head_disp = metrics.get('head_displacement', 0)
head_thresholds = SCORING_THRESHOLDS['head_displacement']['impact_ideal']
print(f"\nHEAD_DISPLACEMENT:")
print(f"  Value: {head_disp:.2f}px")
print(f"  Ideal range: {head_thresholds}")
print(f"  Status: {'✓ IN RANGE' if head_thresholds[0] <= head_disp <= head_thresholds[1] else '✗ OUT OF RANGE'}")

# 4. X_FACTOR
x_factor = metrics.get('x_factor', 0)
x_thresholds = SCORING_THRESHOLDS['x_factor']['impact_ideal']
print(f"\nX_FACTOR (unwind):")
print(f"  Value: {x_factor:.2f}°")
print(f"  Ideal range: {x_thresholds}")
print(f"  Status: {'✓ IN RANGE' if x_thresholds[0] <= x_factor <= x_thresholds[1] else '✗ OUT OF RANGE'}")
