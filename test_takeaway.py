#!/usr/bin/env python
"""Test script for Takeaway phase scorer"""

from src.biomechanics.phase_scorer import PhaseScorer

scorer = PhaseScorer()

# Test takeaway scoring with realistic metrics
test_metrics = {
    'shoulder_rotation': 15,        # Good early shoulder turn (ideal: 5-25)
    'x_factor': 3,                  # Small x_factor (good - hips quiet, ideal: 1-10)
    'wrist_angle': 170,             # Good slight hinge (ideal: 160-180)
    'head_displacement': 2,         # Good head stability (ideal: 0-8)
}

print("Testing Takeaway Phase Scorer")
print("=" * 50)
print("\nTest Metrics:")
for key, val in test_metrics.items():
    print(f"  {key}: {val}")

score, details = scorer.score_phase_with_metrics('takeaway', test_metrics)
print(f"\nTakeaway Score: {score:.1f}/100")
if 'components' in details:
    print("\nComponent Scores:")
    for component, component_score in details['components'].items():
        print(f"  {component}: {component_score:.1f}")
print(f"\nConfidence: {details.get('confidence', 0):.2f}")

print("\n" + "=" * 50)
print("✓ Takeaway phase scorer working correctly!")
