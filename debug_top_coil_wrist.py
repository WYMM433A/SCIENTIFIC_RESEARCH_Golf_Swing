import pandas as pd
from src.biomechanics.angles import GolfBiomechanics
from src.biomechanics.phase_scorer import PhaseScorer
from src.biomechanics.scoring_config import SCORING_THRESHOLDS

# Load pose data
pose_csv = 'data/extracted_poses/B80O_cleaned_poses.csv'
df = pd.read_csv(pose_csv)

# Use correct top frame from pipeline
phases_csv = 'data/keyframes/B80O_nn/B80O_cleaned_8phases.csv'
phases_df = pd.read_csv(phases_csv)
top_frame = phases_df[phases_df['Phase'] == 'Top']['Key_Frame'].values[0]

biomech = GolfBiomechanics(df)
scorer = PhaseScorer()

# Get metrics at TOP frame
metrics = biomech.calculate_all_metrics(frame=top_frame)

print(f"TOP PHASE - Frame {top_frame}")
print("="*70)

# 1. Check X_FACTOR (coil)
x_factor = metrics.get('x_factor', 0)
x_thresholds = SCORING_THRESHOLDS['x_factor']['top_ideal']
print(f"\nX_FACTOR (coil):")
print(f"  Value: {x_factor:.2f}°")
print(f"  Ideal range: {x_thresholds}")
print(f"  Status: {'✓ IN RANGE' if x_thresholds[0] <= x_factor <= x_thresholds[1] else '✗ OUT OF RANGE'}")
coil_score = scorer._evaluate_metric(x_factor, x_thresholds, metric_name='x_factor')
print(f"  Raw score: {coil_score:.2f}/100")

# 2. Check WRIST_ANGLE
wrist_angle = metrics.get('wrist_angle', 0)
wrist_thresholds = SCORING_THRESHOLDS['wrist_angle']['top_ideal']
print(f"\nWRIST_ANGLE:")
print(f"  Value: {wrist_angle:.2f}°")
print(f"  Ideal range: {wrist_thresholds}")
print(f"  Status: {'✓ IN RANGE' if wrist_thresholds[0] <= wrist_angle <= wrist_thresholds[1] else '✗ OUT OF RANGE'}")
wrist_score = scorer._evaluate_metric(wrist_angle, wrist_thresholds, metric_name='wrist_angle')
print(f"  Raw score: {wrist_score:.2f}/100")

# 3. Score full top phase
score, details = scorer.score_top(metrics)
print(f"\nTOP PHASE COMPONENTS:")
for k, v in details.get('components', {}).items():
    print(f"  {k:15}: {v:.2f}")
print(f"\n🎯 FINAL TOP SCORE: {score:.1f}/100")
