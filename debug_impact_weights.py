import pandas as pd
from src.biomechanics.angles import GolfBiomechanics
from src.biomechanics.phase_scorer import PhaseScorer
from src.biomechanics.scoring_config import SCORING_THRESHOLDS, METRIC_WEIGHTS

# Load pose data
pose_csv = 'data/extracted_poses/me_cleaned_poses.csv'
df = pd.read_csv(pose_csv)

# Use correct frames from pipeline
phases_csv = 'data/keyframes/me_nn/me_cleaned_8phases.csv'
try:
    phases_df = pd.read_csv(phases_csv)
    phases_dict = dict(zip(phases_df['Phase'], phases_df['Key_Frame']))
except:
    print("CSV not found, skipping")
    exit()

biomech = GolfBiomechanics(df)
scorer = PhaseScorer()

impact_frame = phases_dict['Impact']
top_frame = phases_dict['Top']
metrics = biomech.calculate_all_metrics(frame=impact_frame)

print(f"IMPACT PHASE COMPONENT SCORING - Frame {impact_frame}")
print("="*70)

# Check each component
components = {
    'lag_release': ('lag_angle', SCORING_THRESHOLDS['lag_angle']['impact_ideal']),
    'x_factor_unwind': ('x_factor', SCORING_THRESHOLDS['x_factor']['impact_ideal']),
    'wrist_angle': ('wrist_angle', SCORING_THRESHOLDS['wrist_angle']['impact_ideal']),
    'stability': ('head_displacement', SCORING_THRESHOLDS['head_displacement']['impact_ideal']),
}

for comp_name, (metric_name, threshold) in components.items():
    value = metrics.get(metric_name, 0)
    raw_score = scorer._evaluate_metric(value, threshold, metric_name=metric_name)
    weight = METRIC_WEIGHTS['impact'][comp_name]
    weighted = raw_score * weight
    
    print(f"\n{comp_name}:")
    print(f"  Metric value: {value:.2f}")
    print(f"  Ideal range:  {threshold}")
    print(f"  Raw score:    {raw_score:.2f}/100")
    print(f"  Weight:       {weight}")
    print(f"  Weighted:     {weighted:.2f}")

# Calculate total
score_components = {}
for comp_name, (metric_name, threshold) in components.items():
    value = metrics.get(metric_name, 0)
    raw_score = scorer._evaluate_metric(value, threshold, metric_name=metric_name)
    weight = METRIC_WEIGHTS['impact'][comp_name]
    score_components[comp_name] = raw_score * weight

total_weight = sum(METRIC_WEIGHTS['impact'].values())
normalized = sum(score_components.values()) / total_weight if total_weight > 0 else 0

print(f"\n{'='*70}")
print(f"Total weight: {total_weight}")
print(f"Normalized score: {normalized:.1f}/100")
