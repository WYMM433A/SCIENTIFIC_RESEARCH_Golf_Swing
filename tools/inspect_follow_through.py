import pandas as pd
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.biomechanics.phase_scorer import PhaseScorer
from src.biomechanics.scoring_config import SCORING_THRESHOLDS, METRIC_WEIGHTS

if len(sys.argv) >= 3:
    scores_path = sys.argv[1]
    metrics_path = sys.argv[2]
else:
    scores_path = 'data/metrics/min2_scores.csv'
    metrics_path = 'data/metrics/min2_cleaned_metrics.csv'

scores = pd.read_csv(scores_path)
metrics_df = pd.read_csv(metrics_path)

row = scores[scores['phase'] == 'Follow-through'].iloc[0]
key = int(row['key_frame'])
metrics = metrics_df[metrics_df['frame'] == key].iloc[0].to_dict()

scorer = PhaseScorer()
score, details = scorer.score_phase_with_metrics('follow-through', metrics)

print(f"source_scores={scores_path}")
print(f"source_metrics={metrics_path}")
print(f"Follow-through key_frame={key} score={score:.6f} confidence={details.get('confidence', 0):.2f}")
print('--- Raw metric values ---')
print(f"x_factor={metrics.get('x_factor'):.6f}")
print(f"spine_angle={metrics.get('spine_angle'):.6f}")
print(f"lead_arm_angle={metrics.get('lead_arm_angle'):.6f}")
print(f"hip_rotation={metrics.get('hip_rotation'):.6f}")

print('--- Thresholds ---')
print('x_factor.follow_through_ideal =', SCORING_THRESHOLDS['x_factor']['follow_through_ideal'])
print('spine_angle.follow_through_ideal =', SCORING_THRESHOLDS['spine_angle']['follow_through_ideal'])
print('lead_arm_angle.follow_through_ideal =', SCORING_THRESHOLDS['lead_arm_angle']['follow_through_ideal'])
print('hip_rotation.follow_through_ideal =', SCORING_THRESHOLDS['hip_rotation']['follow_through_ideal'])

print('--- Weighted components ---')
for component, weighted_value in details.get('components', {}).items():
    print(f"{component}={weighted_value:.6f}")

print('--- Reconstructed raw sub-scores ---')
weights = METRIC_WEIGHTS['follow_through']
for component, weighted_value in details.get('components', {}).items():
    weight = weights[component]
    raw_score = weighted_value / weight if weight else weighted_value
    print(f"{component}: raw_score={raw_score:.3f} weight={weight}")
