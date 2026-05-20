"""Test x_factor_unwind scoring directly"""
from src.biomechanics.phase_scorer import PhaseScorer
from src.biomechanics.scoring_config import SCORING_THRESHOLDS, METRIC_WEIGHTS

scorer = PhaseScorer()

# Test x_factor_unwind with value 5.16
test_value = 5.16
ideal_range = SCORING_THRESHOLDS["x_factor"]["impact_ideal"]

print(f"Testing x_factor_unwind scoring:")
print(f"  Value: {test_value}")
print(f"  Ideal range from config: {ideal_range}")
print()

# Call _evaluate_metric directly
score = scorer._evaluate_metric(
    test_value,
    ideal_range,
    metric_name="x_factor"
)

print(f"Direct _evaluate_metric result: {score}")
print()

# Also test what happens if we pass the full threshold dict
print(f"Full x_factor thresholds:")
print(SCORING_THRESHOLDS.get("x_factor", {}))
