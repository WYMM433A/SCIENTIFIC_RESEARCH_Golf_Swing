"""Debug why Impact scoring is low for pro swing"""
import pandas as pd
from src.biomechanics.phase_scorer import PhaseScorer
from src.biomechanics.scoring_config import SCORING_THRESHOLDS, METRIC_WEIGHTS

# Load feedback
feedback = pd.read_csv('data/metrics/B90_feedback_detailed.csv')
impact_rows = feedback[feedback['phase'].str.lower() == 'impact']

print("="*100)
print("IMPACT PHASE DEBUG - Why is it scoring 35.7?")
print("="*100)
print()

# Show each impact component
scorer = PhaseScorer()

for idx, row in impact_rows.iterrows():
    component = row['component']
    measured = row['measured_value']
    target_min = row['target_min']
    target_max = row['target_max']
    raw_score = row['raw_score']
    weight = METRIC_WEIGHTS.get('impact', {}).get(component, 0)
    
    weighted_score = raw_score * weight if pd.notna(raw_score) else 0
    
    print(f"Component: {component}")
    print(f"  Measured Value: {measured}")
    print(f"  Target Range:  {target_min} to {target_max}")
    print(f"  Raw Score:     {raw_score}")
    print(f"  Weight:        {weight}")
    print(f"  Weighted Score: {weighted_score}")
    print()

# Calculate what the total should be
total_weighted = 0
for idx, row in impact_rows.iterrows():
    component = row['component']
    raw_score = row['raw_score']
    weight = METRIC_WEIGHTS.get('impact', {}).get(component, 0)
    if pd.notna(raw_score):
        total_weighted += raw_score * weight

print(f"Total Impact Score (sum of weighted): {total_weighted:.1f}")
