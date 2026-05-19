"""
Scientific Benchmarks for Golf Swing Biometrics.
Based on Trackman, GolfTEC, and Literature (Meister et al. 2011).
"""

BENCHMARKS = {
    "shoulder_turn": {
        "phase": "Top",
        "pro": {"min": 85, "max": 105, "ideal": 95},
        "amateur": {"min": 70, "max": 90, "ideal": 80},
        "label": "Shoulder Turn",
        "priority_weight": 1.5,
        "unit": "°",
        "hint": "Increase shoulder rotation to generate better coil/power."
    },
    "x_factor": {
        "phase": "Top",
        "pro": {"min": 45, "max": 65, "ideal": 55},
        "amateur": {"min": 30, "max": 45, "ideal": 38},
        "label": "X-Factor Separation",
        "priority_weight": 2.0,
        "unit": "°",
        "hint": "Increase separation between shoulders and hips to optimize power."
    },
    "spine_angle": {
        "phase": "Impact",
        "pro": {"min": 35, "max": 45, "ideal": 40},
        "amateur": {"min": 30, "max": 42, "ideal": 36},
        "label": "Spine Angle",
        "priority_weight": 1.2,
        "unit": "°",
        "hint": "Maintain spine angle stability through impact."
    },
    "trail_knee_flex": {
        "phase": "Top",
        "pro": {"min": 20, "max": 30, "ideal": 25},
        "amateur": {"min": 15, "max": 35, "ideal": 25},
        "label": "Trail Knee Stability",
        "priority_weight": 0.8,
        "unit": "°",
        "hint": "Maintain trail knee flex for lower body stability."
    },
    "head_stability": {
        "phase": "Global",
        "pro": {"min": 0, "max": 5, "ideal": 2},
        "amateur": {"min": 0, "max": 10, "ideal": 5},
        "label": "Head Stability (Fluctuation)",
        "priority_weight": 1.0,
        "unit": "cm",
        "hint": "Maintain head stability throughout the swing."
    }
}

def get_status(value, benchmark_cfg, level="amateur"):
    cfg = benchmark_cfg[level]
    if cfg["min"] <= value <= cfg["max"]:
        return "Good", "Success"
    
    diff = value - cfg["ideal"]
    if abs(diff) < 15:
        return "Fair", "Warning"
    else:
        return "Poor", "Danger"
