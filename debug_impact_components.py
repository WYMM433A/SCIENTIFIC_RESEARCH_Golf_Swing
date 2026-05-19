import pandas as pd
from src.biomechanics.angles import GolfBiomechanics
from src.biomechanics.phase_scorer import PhaseScorer

# Load pose data
pose_csv = 'data/extracted_poses/B80O_cleaned_poses.csv'
df = pd.read_csv(pose_csv)

# Use correct frames from pipeline
phases_csv = 'data/keyframes/B80O_nn/B80O_cleaned_8phases.csv'
phases_df = pd.read_csv(phases_csv)
phases_dict = dict(zip(phases_df['Phase'], phases_df['Key_Frame']))

biomech = GolfBiomechanics(df)
scorer = PhaseScorer()

impact_frame = phases_dict['Impact']
top_frame = phases_dict['Top']

# Get metrics at impact frame
metrics = biomech.calculate_all_metrics(frame=impact_frame)

print(f"IMPACT PHASE - Frame {impact_frame}")
print("="*70)
print(f"\nMetrics at impact:")
print(f"  lead_arm_angle:      {metrics.get('lead_arm_angle', 0):.2f}°")
print(f"  lag_angle:           {metrics.get('lag_angle', 0):.2f}°")
print(f"  wrist_angle:         {metrics.get('wrist_angle', 0):.2f}°")
print(f"  x_factor:            {metrics.get('x_factor', 0):.2f}°")
print(f"  head_displacement:   {metrics.get('head_displacement', 0):.2f}px")

# Set window context for Fix 1
scorer._window_context = {
    'biomechanics': biomech,
    'start_frame': phases_dict['Impact'] - 3,  # approximate
    'end_frame': impact_frame,
    'top_frame': top_frame,
    'use_window': True
}

score, details = scorer.score_impact(metrics)

print(f"\nComponent breakdown:")
for k, v in details.get('components', {}).items():
    print(f"  {k:15}: {v:.2f}")

print(f"\n🎯 IMPACT PHASE SCORE: {score:.1f}/100")
