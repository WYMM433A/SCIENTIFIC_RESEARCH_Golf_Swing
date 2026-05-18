"""
Simple diagnostic: Show which metrics have LOOSE thresholds that allow 
intermediate swings to score as high as pros.

Uses the actual metric values from the audit runs to identify problematic thresholds.
"""

from src.biomechanics.scoring_config import SCORING_THRESHOLDS

# From the audit runs, here are metric ranges we're seeing:
OBSERVED_VALUES = {
    "impact": {
        "arm_extension": {"beginner_1-2": 160.6, "intermediate_6-8": 160.6, "pro_9-10": 154.3},
        "wrist_angle": {"beginner_1-2": 172.7, "intermediate_6-8": 172.7, "pro_9-10": 169.15},
        "spine_angle": {"beginner_1-2": -22.9, "intermediate_6-8": -22.9, "pro_9-10": -16.2},
    },
    "follow_through": {
        "arm_extension": {"beginner_1-2": 174.5, "intermediate_6-8": 174.5, "pro_9-10": 168.96},
        "x_factor": {"beginner_1-2": 10.85, "intermediate_6-8": 10.85, "pro_9-10": 0.29},
        "spine_angle": {"beginner_1-2": -23.4, "intermediate_6-8": -23.4, "pro_9-10": -14.41},
    },
    "address": {
        "arm_extension": {"beginner_1-2": 168.8, "intermediate_6-8": 168.8, "pro_9-10": 164.2},
        "wrist_angle": {"beginner_1-2": 178.6, "intermediate_6-8": 178.6, "pro_9-10": 165.7},
    },
}

def check_threshold_tightness(phase, metric_name, values_dict):
    """Check if threshold is too loose by seeing if it allows intermediate ≥ pro."""
    if phase not in SCORING_THRESHOLDS:
        return
    
    thresholds = SCORING_THRESHOLDS[phase]
    if metric_name not in thresholds:
        return
    
    threshold = thresholds[metric_name]
    ideal_range = threshold.get("ideal_range", None)
    acceptable_range = threshold.get("acceptable_range", ideal_range)
    
    beg = values_dict.get("beginner_1-2", 0)
    inter = values_dict.get("intermediate_6-8", 0)
    pro = values_dict.get("pro_9-10", 0)
    
    print(f"\n{phase.upper()} — {metric_name}:")
    print(f"  Ideal range:      {ideal_range}")
    print(f"  Acceptable range: {acceptable_range}")
    print(f"  Beginner:    {beg:>7.2f}")
    print(f"  Intermediate: {inter:>7.2f}  {'⚠️ SAME AS BEGINNER' if abs(inter - beg) < 0.1 else ''}")
    print(f"  Pro:         {pro:>7.2f}  {'❌ LOWER THAN INTERMEDIATE!' if pro < inter else ''}")
    
    # Check if they're all in ideal range
    if ideal_range:
        in_ideal = []
        for level, val in [("Beginner", beg), ("Intermediate", inter), ("Pro", pro)]:
            in_ideal.append(ideal_range[0] <= val <= ideal_range[1])
        
        if all(in_ideal):
            print(f"  🔴 PROBLEM: All three skill levels IN IDEAL RANGE = No discrimination!")
        elif in_ideal[1] and in_ideal[2]:  # intermediate and pro both in ideal
            print(f"  🔴 PROBLEM: Intermediate and Pro BOTH in ideal range = No discrimination!")

print("\n" + "█"*80)
print("█  CEILING EFFECT ANALYSIS - Which metrics allow intermediate to score like pro?")
print("█"*80)

for phase, metrics in OBSERVED_VALUES.items():
    print(f"\n\n{'='*80}")
    print(f"PHASE: {phase.upper()}")
    print(f"{'='*80}")
    for metric_name, values in metrics.items():
        check_threshold_tightness(phase, metric_name, values)

print("\n" + "█"*80)
print("█  RECOMMENDATION: Tighten thresholds where intermediate ≥ pro")
print("█"*80 + "\n")
