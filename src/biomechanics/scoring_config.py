"""
Scoring configuration for phase-specific biomechanical analysis.
Defines weights, thresholds, and ideal values for golf swing evaluation.
"""

# ============================================================================
# PHASE WEIGHTS - Contribution of each phase to overall swing score
# ============================================================================
# Higher weight = more critical to overall performance
PHASE_WEIGHTS = {
    "address": 0.10,              # Setup foundation
    "takeaway": 0.08,             # Sequence initiation
    "mid_backswing": 0.13,        # Coil building (reliable discrimination)
    "top": 0.08,                  # Max coil/stability
    "mid_downswing": 0.25,        # Sequencing (CRITICAL: power generation)
    "impact": 0.30,               # Contact efficiency (MOST CRITICAL: shot outcome)
    "follow_through": 0.05,       # Deceleration (reduced: front-view unreliable)
    "finish": 0.01,               # Balance/consistency (reduced: front-view late-swing issues)
}

# ============================================================================
# METRIC WEIGHTS - Importance of each metric within phases
# ============================================================================
# Higher weight = metric contributes more to phase score
METRIC_WEIGHTS = {
    "address": {
        "posture": 0.35,           # Spine angle/posture setup
        "grip": 0.10,              # Grip pressure indicator (wrist angle)
        "weight_distribution": 0.20,
    },
    "takeaway": {
        "shoulder_rotation": 0.15, # Shoulder turn
        "hip_lag": 0.50,           # X-factor
        "wrist_position": 0.15,    # Wrist
        "club_path": 0.20,         # Head displacement
    },
    "mid_backswing": {
        "coil": 0.30,             # X-factor development
        "shoulder_rotation": 0.25, # Continued turn
        "wrist_hinge": 0.25,      # Wrist angle progression
        "shaft_plane": 0.20,      # Club on plane
    },
    "top": {
        "coil": 0.35,             # Max X-factor (critical)
        "posture": 0.20,          # Spine angle maintained
        "stability": 0.25,        # Balance/no sway
        "wrist_angle": 0.20,      # Wrist hinge maintained
    },
    "mid_downswing": {
        # Down-weight estimated/proxy signals and prioritize directly
        # measurable pose-based body rotation metrics.
        "kinematic_sequence": 0.15,
        "lag": 0.05,
        "hip_rotation": 0.50,
        "upper_body_lag": 0.30,
    },
    "impact": {
        # Down-weight estimated lag proxy; emphasize directly observed pose.
        "lag_release": 0.02,
        "x_factor_unwind": 0.45,   # Increased from 0.35
        "arm_extension": 0.30,     # Increased from 0.25
        "wrist_angle": 0.10,       # Reduced from 0.20
        "stability": 0.13,         # Reduced from 0.15
    },
    "follow_through": {
        # Deceleration is proxy-based, so keep it lower.
        "deceleration": 0.10,
        "posture": 0.35,
        "arm_swing": 0.30,
        "rotation": 0.25,
    },
    "finish": {
        "balance": 0.35,          # Weight on left side
        "posture": 0.25,          # Spine angle maintained
        "rotation": 0.20,         # Full rotation achieved
        "symmetry": 0.20,         # Balanced finish position
    },
}

# ============================================================================
# SCORING THRESHOLDS - Acceptable deviation ranges and ideal values
# ============================================================================
# Format: (min_acceptable, ideal_low, ideal_high, max_acceptable)
# Deviations within ideal range = 100 points
# Deviations outside ideal but within acceptable = 50-99 points
# Deviations beyond acceptable = 0-49 points

SCORING_THRESHOLDS = {
    # SPINE ANGLE (degrees) - Should be relatively upright but not rigid
    "spine_angle": {
        "acceptable": (-20, 35),   # Min, Max acceptable
        "ideal": (-5, 15),         # Min, Max ideal
        "direction": "absolute",   # Target value near 0
        "address_ideal": (-5, 10),
        "top_ideal": (-8, 12),
        "follow_through_ideal": (-30, 20),  # Relaxed for natural post-release posture
        "finish_ideal": (-10, 12),
        "impact_ideal": (-3, 8),
    },
    
    # X-FACTOR (degrees) - Hip-shoulder differential
    "x_factor": {
        "acceptable": (0, 20),
        "ideal": (2, 15),
        "direction": "higher_better",  # More is better for power, but this metric is small in 2D projection
        "takeaway_ideal": (1, 10),
        "mid_backswing_ideal": (2, 14),
        "top_ideal": (2, 18),
        "mid_downswing_ideal": (0, 5),
        "follow_through_ideal": (10, 20),
        "finish_ideal": (0, 6),
        "impact_ideal": (0, 18),
    },
    
    # X-FACTOR STRETCH - Hip-shoulder differential speed ratio
    "x_factor_stretch": {
        "acceptable": (0, 40),
        "ideal": (5, 30),
        "direction": "higher_better",
    },
    
    # WRIST ANGLE (degrees) - Hinge angle
    "wrist_angle": {
        "acceptable": (110, 180),
        "ideal": (150, 180),
        "direction": "context_dependent",
        "address_ideal": (160, 180),
        "takeaway_ideal": (160, 180),
        "mid_backswing_ideal": (150, 180),
        "top_ideal": (130, 180),  # Relaxed from (145, 180) to accommodate front-view measurement variance
        "impact_ideal": (150, 180),
        "follow_through_ideal": (150, 180),
    },
    
    # HIP ROTATION (degrees) - Hip line rotation from horizontal
    "hip_rotation": {
        "acceptable": (150, 185),
        "ideal": (165, 185),
        "direction": "higher_better",
        "backswing_ideal": (170, 185),
        "downswing_ideal": (170, 185),
        "follow_through_ideal": (165, 180),
        "finish_ideal": (160, 175),
    },
    
    # SHOULDER ROTATION (degrees) - Shoulder line rotation from horizontal
    "shoulder_rotation": {
        "acceptable": (150, 185),
        "ideal": (165, 185),
        "direction": "higher_better",
        "takeaway_ideal": (165, 180),
        "backswing_ideal": (170, 185),
        "downswing_ideal": (160, 180),
    },
    
    # LAG ANGLE (degrees) - Approximate wrist lag using wrist hinge
    "lag_angle": {
        "acceptable": (140, 180),
        "ideal": (150, 180),
        "direction": "higher_better",  # More lag = more power potential
        "mid_downswing_ideal": (160, 180),
        "impact_ideal": (160, 180),
    },

    # LEAD ARM ANGLE (degrees) - Approximate shaft/arm plane quality
    "lead_arm_angle": {
        "acceptable": (90, 180),
        "ideal": (150, 180),
        "direction": "higher_better",
        "mid_backswing_ideal": (155, 180),
        "impact_ideal": (155, 175),
        "follow_through_ideal": (160, 175),
    },
    
    # HEAD MOVEMENT (pixels/cm) - Movement from address
    "head_displacement": {
        "acceptable": (0, 15),
        "ideal": (0, 5),
        "direction": "lower_better",   # Minimize head movement
        "takeaway_ideal": (0, 8),       # Allow slight movement during takeaway
        "top_ideal": (0, 3),            # Very still at top of swing
        "finish_ideal": (0, 4),         # Balanced finish position
        "impact_ideal": (0, 3),         # Minimal movement at impact
    },
    
    # KNEE FLEXION CHANGE (degrees) - Knee angle change
    "knee_flexion": {
        "acceptable": (5, 40),
        "ideal": (15, 35),
        "direction": "context_dependent",
    },
    
    # STANCE WIDTH RATIO - Shoulder width to stance width ratio
    "stance_width_ratio": {
        "acceptable": (0.8, 2.5),    # Wider acceptable range for golfer variability
        "ideal": (0.9, 1.2),
        "direction": "context_dependent",
        "address_ideal": (0.9, 1.2),
    },
}

# ============================================================================
# SCORE INTERPRETATION - Points and feedback thresholds
# ============================================================================
SCORE_RANGES = {
    "excellent": (90, 100),      # 90-100: Excellent
    "good": (75, 89),             # 75-89: Good
    "acceptable": (60, 74),       # 60-74: Acceptable
    "needs_work": (40, 59),       # 40-59: Needs work
    "poor": (0, 39),              # 0-39: Poor
}

# ============================================================================
# FEEDBACK MESSAGES - Based on score ranges and deviations
# ============================================================================
FEEDBACK_TEMPLATES = {
    "excellent": "Excellent {metric} in {phase}. This is a strength.",
    "good": "Good {metric} in {phase}. Room for minor improvement.",
    "acceptable": "Acceptable {metric} in {phase}. Focus on improvement.",
    "needs_work": "Your {metric} in {phase} needs work. Target: {target}",
    "poor": "Poor {metric} in {phase}. Significant improvement needed. Target: {target}",
    "deviation": "Your {metric} is {deviation:.1f}° off ideal. Target: {target}",
}

# ============================================================================
# ACTIONABLE FEEDBACK PLAYBOOK
# ============================================================================
# Keys use "<phase>.<component>"
FEEDBACK_PLAYBOOK = {
    "address.posture": {
        "cue": "Set your chest over the ball with a neutral spine and soft knees.",
        "drill": "Mirror setup holds: 3 x 20s maintaining neutral posture.",
    },
    "address.grip": {
        "cue": "Keep wrists quiet and avoid excessive cupping at setup.",
        "drill": "Setup-and-freeze repetitions: 10 reps, check wrist angle each rep.",
    },
    "address.weight_distribution": {
        "cue": "Center pressure between your feet with shoulder-width stance.",
        "drill": "Feet-line setup drill: align heels to a marked shoulder-width line.",
    },
    "takeaway.shoulder_rotation": {
        "cue": "Start takeaway with one-piece turn from chest and shoulders.",
        "drill": "Cross-arm takeaway drill: 2 x 10 slow reps.",
    },
    "takeaway.hip_lag": {
        "cue": "Let shoulders start first; keep hips quieter early.",
        "drill": "Pause-at-P2 drill with hips quiet: 8 reps.",
    },
    "takeaway.wrist_position": {
        "cue": "Allow a gentle hinge, not an abrupt set.",
        "drill": "Slow takeaway to shaft-parallel with 2s pause: 10 reps.",
    },
    "takeaway.club_path": {
        "cue": "Keep head steady and move club on-plane early.",
        "drill": "Alignment-stick takeaway drill: 2 x 8 reps.",
    },
    "mid_backswing.coil": {
        "cue": "Increase torso coil while keeping lower body stable.",
        "drill": "Resistance-band turn drill: 2 x 10 reps.",
    },
    "mid_backswing.shoulder_rotation": {
        "cue": "Complete shoulder turn to the top without lifting.",
        "drill": "Wall-turn drill to feel depth: 2 x 8 reps.",
    },
    "mid_backswing.wrist_hinge": {
        "cue": "Set wrists progressively through mid-backswing.",
        "drill": "Half-swing hinge checkpoints: 10 reps.",
    },
    "mid_backswing.shaft_plane": {
        "cue": "Keep lead arm structure and shaft on a repeatable plane.",
        "drill": "Pump-to-top drill with mirror feedback: 8 reps.",
    },
    "top.coil": {
        "cue": "Reach a full but balanced top turn.",
        "drill": "Top-position hold drill: 6 reps x 3s hold.",
    },
    "top.posture": {
        "cue": "Maintain spine posture at the top without standing up.",
        "drill": "Top freeze with posture check: 8 reps.",
    },
    "top.stability": {
        "cue": "Minimize sway and keep head stable at transition.",
        "drill": "Head-stability wall reference drill: 2 x 8 reps.",
    },
    "top.wrist_angle": {
        "cue": "Preserve wrist set at the top into transition.",
        "drill": "Top-to-transition pump drill: 10 reps.",
    },
    "mid_downswing.kinematic_sequence": {
        "cue": "Start down with lower body, then torso, then arms.",
        "drill": "Step-through sequencing drill: 2 x 8 reps.",
    },
    "mid_downswing.lag": {
        "cue": "Retain lag longer before release.",
        "drill": "Pump-downswing lag drill: 3 x 6 reps.",
    },
    "mid_downswing.hip_rotation": {
        "cue": "Rotate hips through the downswing instead of sliding.",
        "drill": "Chair-hip rotation drill: 2 x 10 reps.",
    },
    "mid_downswing.upper_body_lag": {
        "cue": "Let torso follow hips; avoid throwing arms early.",
        "drill": "Split-grip transition drill: 2 x 8 reps.",
    },
    "impact.lag_release": {
        "cue": "Release lag at impact, not too early.",
        "drill": "Impact bag with delayed release feel: 12 reps.",
    },
    "impact.x_factor_unwind": {
        "cue": "Unwind through impact with chest and hips synced.",
        "drill": "Slow-motion impact checkpoints: 10 reps.",
    },
    "impact.arm_extension": {
        "cue": "Extend lead arm through strike for compression.",
        "drill": "One-arm extension drill: 2 x 8 reps.",
    },
    "impact.wrist_angle": {
        "cue": "Maintain wrist structure through contact.",
        "drill": "Punch-shot impact control drill: 2 x 10 reps.",
    },
    "impact.stability": {
        "cue": "Stabilize head and trunk through impact.",
        "drill": "Impact hold drill: freeze 1s after contact, 10 reps.",
    },
    "follow_through.deceleration": {
        "cue": "Decelerate smoothly after release.",
        "drill": "Three-quarter finish hold drill: 8 reps.",
    },
    "follow_through.posture": {
        "cue": "Keep posture as chest rotates to target.",
        "drill": "Post-impact posture checkpoints: 10 reps.",
    },
    "follow_through.arm_swing": {
        "cue": "Allow natural arm fold around the body.",
        "drill": "Finish-wrap rehearsal drill: 2 x 8 reps.",
    },
    "follow_through.rotation": {
        "cue": "Continue rotating to a complete exit.",
        "drill": "Belt-buckle-to-target drill: 2 x 10 reps.",
    },
    "finish.balance": {
        "cue": "Finish tall and balanced over lead side.",
        "drill": "Finish pose holds: 6 reps x 3s.",
    },
    "finish.posture": {
        "cue": "Keep stable spine through the finish pose.",
        "drill": "Finish mirror holds: 2 x 8 reps.",
    },
    "finish.rotation": {
        "cue": "Complete body rotation fully toward target.",
        "drill": "Step-and-turn finish drill: 2 x 8 reps.",
    },
    "finish.symmetry": {
        "cue": "Hold a controlled, symmetric finish position.",
        "drill": "Count-to-two finish freeze drill: 10 reps.",
    },
}

# ============================================================================
# KINEMATIC SEQUENCE THRESHOLDS
# ============================================================================
# For mid-downswing sequencing (hips → torso → arms → club)
KINEMATIC_SEQUENCE = {
    "hip_lead": {
        "ideal_ms": (0, 50),           # Hips lead by 0-50ms
        "acceptable_ms": (-50, 100),
    },
    "torso_follow": {
        "ideal_ms_after_hips": (50, 150),  # Torso 50-150ms after hips
        "acceptable_ms_after_hips": (0, 200),
    },
    "arm_follow": {
        "ideal_ms_after_torso": (50, 150), # Arms 50-150ms after torso
        "acceptable_ms_after_torso": (0, 200),
    },
    "proper_sequence_bonus": 5,        # Bonus points for proper sequencing
}

# ============================================================================
# OVERALL SCORING CALCULATION
# ============================================================================
SCORING_STRATEGY = {
    "method": "weighted_average",      # "weighted_average" or "hierarchical"
    "phase_weight_factor": 0.7,        # 70% from phase scores
    "consistency_factor": 0.2,         # 20% from consistency across phases
    "improvement_factor": 0.1,         # 10% from historical improvement
    "minimum_confidence": 0.70,        # Min confidence (0-1) to report score
}

# ============================================================================
# BENCHMARK COMPARISONS - Pro benchmarks from research
# ============================================================================
PROFESSIONAL_BENCHMARKS = {
    "x_factor_at_top": {
        "tour_average": 38,
        "elite": (40, 50),
        "amateur_typical": 28,
    },
    "lag_angle_mid_downswing": {
        "tour_average": 58,
        "elite": (60, 70),
        "amateur_typical": 40,
    },
    "hip_shoulder_separation": {
        "tour_average": 35,
        "elite": (40, 50),
        "amateur_typical": 20,
    },
}
