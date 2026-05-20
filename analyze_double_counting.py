"""
DOUBLE-COUNTING ANALYSIS
========================

Showing which metrics get scored MULTIPLE TIMES in each phase because of aliases.

"""

from src.biomechanics.scoring_config import METRIC_WEIGHTS

# Component → Actual Metric Mapping
COMPONENT_METRIC_MAP = {
    "address": {
        "posture": "spine_angle",
        "grip": "wrist_angle",
        "weight_distribution": "stance_width_ratio",
    },
    "takeaway": {
        "shoulder_rotation": "shoulder_rotation",
        "hip_lag": "x_factor",
        "wrist_position": "wrist_angle",
        "club_path": "head_displacement",
    },
    "mid_backswing": {
        "coil": "x_factor",
        "shoulder_rotation": "shoulder_rotation",
        "wrist_hinge": "wrist_angle",
        "shaft_plane": "lead_arm_angle",
    },
    "top": {
        "coil": "shoulder_rotation",
        "posture": "spine_angle",
        "stability": "head_displacement",
        "wrist_angle": "wrist_angle",
    },
    "mid_downswing": {
        "kinematic_sequence": "kinematic_sequence",
        "lag": "lag_angle",
        "hip_rotation": "hip_rotation",
        "upper_body_lag": "x_factor",
    },
    "impact": {
        "lag_release": "lag_angle",
        "x_factor_unwind": "x_factor",
        "arm_extension": "lead_arm_angle",
        "wrist_angle": "wrist_angle",
        "stability": "head_displacement",
    },
    "follow_through": {
        "deceleration": "x_factor",
        "posture": "spine_angle",
        "arm_swing": "lead_arm_angle",
        "rotation": "hip_rotation",
    },
    "finish": {
        "balance": "head_displacement",
        "posture": "spine_angle",
        "rotation": "hip_rotation",
        "symmetry": "x_factor",
    },
}

# Aliases in metrics dict
ALIASES = {
    "hip_angle": "hip_rotation",
    "arm_extension": "lead_arm_angle",
    "wrist_angle": "wrist_hinge",
    # lag_angle is kept separate from wrist_hinge to avoid double-counting in IMPACT phase
}

print("=" * 100)
print("DOUBLE-COUNTING ANALYSIS: Which metrics get scored multiple times?")
print("=" * 100)

for phase, components in COMPONENT_METRIC_MAP.items():
    print(f"\n{'─' * 100}")
    print(f"PHASE: {phase.upper()}")
    print(f"{'─' * 100}")
    
    # Build a reverse map: metric → components that use it
    metric_usage = {}
    for component_name, metric_key in components.items():
        # Resolve aliases
        actual_metric = ALIASES.get(metric_key, metric_key)
        
        if actual_metric not in metric_usage:
            metric_usage[actual_metric] = []
        metric_usage[actual_metric].append(component_name)
    
    # Find doubles
    doubles_found = False
    for metric, component_list in sorted(metric_usage.items()):
        weight_info = METRIC_WEIGHTS.get(phase, {})
        total_weight = sum(weight_info.get(comp, 0) for comp in component_list)
        
        if len(component_list) > 1:
            doubles_found = True
            print(f"\n🔴 DOUBLE-COUNTED: {metric}")
            print(f"   Used by {len(component_list)} components:")
            for comp in component_list:
                weight = weight_info.get(comp, "unknown")
                print(f"     • {comp:25s} (weight: {weight})")
            print(f"   Total weight for this metric: {total_weight}")
        else:
            print(f"\n✓  Single use: {metric:25s} (via {component_list[0]:20s}, weight: {weight_info.get(component_list[0], 'unknown')})")
    
    if not doubles_found:
        print("\n✓ No double-counting in this phase")

print("\n" + "=" * 100)
print("SUMMARY OF DOUBLE-COUNTS ACROSS ALL PHASES")
print("=" * 100)

all_doubles = {}
for phase, components in COMPONENT_METRIC_MAP.items():
    metric_usage = {}
    for component_name, metric_key in components.items():
        actual_metric = ALIASES.get(metric_key, metric_key)
        if actual_metric not in metric_usage:
            metric_usage[actual_metric] = []
        metric_usage[actual_metric].append((phase, component_name))
    
    for metric, usage_list in metric_usage.items():
        if len(usage_list) > 1:
            if metric not in all_doubles:
                all_doubles[metric] = []
            all_doubles[metric].extend(usage_list)

if all_doubles:
    print("\nMetrics that get DOUBLE (or more) COUNTED across phases:\n")
    for metric in sorted(all_doubles.keys()):
        usage = all_doubles[metric]
        print(f"\n{metric}:")
        for phase, component in usage:
            print(f"  • {phase:20s} → {component}")
else:
    print("\nNo double-counting found!")

print("\n" + "=" * 100)
print("RECOMMENDATION")
print("=" * 100)
print("""
To fix double-counting, you have 3 options:

OPTION A: Remove the aliases
  - Delete 'hip_angle', 'arm_extension', 'wrist_angle', 'lag_angle'
  - Use only the base metrics
  - Simplifies scoring logic

OPTION B: Keep aliases but update thresholds
  - Make each alias measure something DIFFERENT
  - E.g., 'lag_angle' = real 3D lag (calculated differently)
  - 'arm_extension' = arm extension from top to impact (delta metric)

OPTION C: Update METRIC_WEIGHTS to avoid double-counting
  - Don't use the same metric under multiple component names
  - Create unique components that aren't duplicates
  - More complex but more flexible

For your feedback system, I recommend OPTION A (simplify) or OPTION B (make them real metrics).
""")
