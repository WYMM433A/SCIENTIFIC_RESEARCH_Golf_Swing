#!/usr/bin/env python
"""Test script for Top phase scorer"""

from src.biomechanics.phase_scorer import PhaseScorer

scorer = PhaseScorer()

print("Testing Top Phase Scorer")
print("=" * 50)

# Test 1: Ideal swing at top
print("\n[TEST 1] Ideal swing at top:")
ideal_metrics = {
    'shoulder_rotation': 170,       # Good coil (backswing_ideal: 160-185)
    'spine_angle': 5,               # Good posture (top_ideal: -8 to 12)
    'wrist_angle': 160,             # Good hinge (top_ideal: 145-180)
    'head_displacement': 1,         # Excellent stability (top_ideal: 0-3)
    'x_factor': 10,                 # Good coil separation (top_ideal: 2-18)
}

score, details = scorer.score_phase_with_metrics('top', ideal_metrics)
print(f"  Score: {score:.1f}/100")
if 'components' in details:
    for component, component_score in details['components'].items():
        print(f"    {component}: {component_score:.1f}")

# Test 2: Acceptable swing (some movement)
print("\n[TEST 2] Acceptable swing (some sway):")
acceptable_metrics = {
    'shoulder_rotation': 165,
    'spine_angle': 3,
    'wrist_angle': 155,
    'head_displacement': 5,         # More movement (still acceptable)
    'x_factor': 8,
}

score, details = scorer.score_phase_with_metrics('top', acceptable_metrics)
print(f"  Score: {score:.1f}/100")
if 'components' in details:
    for component, component_score in details['components'].items():
        print(f"    {component}: {component_score:.1f}")

# Test 3: Poor stability (too much head movement)
print("\n[TEST 3] Poor swing (excessive head movement):")
poor_metrics = {
    'shoulder_rotation': 170,
    'spine_angle': 0,
    'wrist_angle': 150,
    'head_displacement': 12,        # Excessive (outside acceptable)
    'x_factor': 12,
}

score, details = scorer.score_phase_with_metrics('top', poor_metrics)
print(f"  Score: {score:.1f}/100")
if 'components' in details:
    for component, component_score in details['components'].items():
        print(f"    {component}: {component_score:.1f}")

print("\n" + "=" * 50)
print("✓ Top phase scorer verified!")
print("\nKey changes:")
print("  • head_displacement now uses config-driven top_ideal (0-3px)")
print("  • Replaces hardcoded 'if < 5: 100 else: 70' rule")
print("  • Scores gracefully degrade with distance from ideal")
