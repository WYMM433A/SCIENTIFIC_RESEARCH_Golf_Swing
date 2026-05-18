# Scoring Audit Results - Min2 (Pro) vs Akh (Beginner)

**Run Date:** May 15, 2026

---

## Executive Summary

✅ **Good News**: Overall separation exists (9.8 pts)  
⚠️ **Problem**: 3 phases show weak/inverted discrimination

---

## Overall Score Comparison

| Metric | Pro (min2) | Beginner (akh) | Difference |
|--------|-----------|---------------|-----------|
| **Overall Score** | **61.0** | **51.2** | **+9.8 pts** ✓ |

---

## Phase-by-Phase Breakdown

### ✅ Working Well (6/8 phases)

| Phase | Pro | Beginner | Diff | Status |
|-------|-----|----------|------|--------|
| **address** | 77.1 | 49.9 | +27.2 | 🟢 Excellent |
| **takeaway** | 70.0 | 50.0 | +20.0 | 🟢 Excellent |
| **mid_backswing** | 55.0 | 40.0 | +15.0 | 🟢 Good |
| **top** | 63.8 | 43.1 | +20.7 | 🟢 Excellent |
| **impact** | 75.0 | 62.5 | +12.5 | 🟢 Good |

### ⚠️ Problem Phases (3/8 phases)

| Phase | Pro | Beginner | Diff | Status | Issue |
|-------|-----|----------|------|--------|-------|
| **mid_downswing** | 38.7 | 35.3 | +3.4 | 🟠 Weak | Metrics don't discriminate |
| **finish** | 80.0 | 79.7 | +0.3 | 🟠 Weak | Virtually identical scores |
| **follow_through** | 45.0 | 60.0 | -15.0 | 🔴 **INVERTED** | **Beginner scores HIGHER!** |

---

## Root Cause Analysis

### Problem 1: **follow_through is inverted** ❌
- Pro scores 45.0
- Beginner scores 60.0 (higher!)
- This means the metrics reward beginner behavior

**Likely cause**: The metrics in follow_through (probably based on x_factor/rotation unwind) favor a certain pattern that beginners happen to match better at this frame.

### Problem 2: **mid_downswing weak discrimination** ⚠️
- Only 3.4 pts difference
- Root: This phase uses view-dependent metrics (hip_rotation from side view)

### Problem 3: **finish weak discrimination** ⚠️
- Only 0.3 pts difference
- Both score ~80 (ceiling effect)

---

## Metrics Comparison (at key frames)

### Pro Swing (min2) - Frame 70
```
arm_extension:        47.02°
lead_arm_angle:       47.02°
wrist_hinge:          168.34°
trail_elbow_angle:    115.73°
spine_angle:          -9.48°
shoulder_rotation:    0.20°
hip_rotation:         2.78°
x_factor:             2.58°
```

### Beginner Swing (akh) - Frame 78
```
arm_extension:        90.36°   (MORE extended)
lead_arm_angle:       90.36°   (MORE extended)
wrist_hinge:          121.18°  (LESS hinge)
trail_elbow_angle:    130.07°  (MORE bent)
spine_angle:          -10.28°  (slightly more flexed)
shoulder_rotation:    1.35°
hip_rotation:         5.18°
x_factor:             3.83°
```

---

## Action Plan

### Priority 1: Fix follow_through (INVERTED LOGIC)
**File**: `src/biomechanics/scoring_config.py`

Check the `METRIC_WEIGHTS["follow_through"]` - likely issues:
- `x_factor` is being rewarded when it's small (wrong direction)
- The "deceleration" metric isn't actually measuring deceleration properly

**Action**: Review the follow_through scoring thresholds and invert if needed.

### Priority 2: Fix mid_downswing (WEAK DISCRIMINATION)
**File**: `src/biomechanics/scoring_config.py`

The `mid_downswing` weights:
```python
"mid_downswing": {
    "kinematic_sequence": 0.15,  # Low weight
    "lag": 0.05,                 # Very low
    "hip_rotation": 0.50,        # PROBLEM: view-dependent!
    "upper_body_lag": 0.30,
}
```

**Action**: 
- Reduce `hip_rotation` weight (it's view-dependent from side view)
- Increase `kinematic_sequence` weight (it's the real discriminator)

### Priority 3: Fix finish (CEILING EFFECT)
**File**: `src/biomechanics/scoring_config.py`

Both score ~80, which means thresholds are too loose.

**Action**: Tighten the "ideal" range thresholds in `SCORING_THRESHOLDS["finish"]` to allow differentiation.

---

## Next Steps

1. **Identify which metrics are causing follow_through inversion**
   - Check if pro's "45.0" is because of low x_factor (good deceleration)
   - Check if beginner's "60.0" is because of high x_factor (bad at this frame)
   - If so, invert the threshold direction

2. **Remove hip_rotation from mid_downswing weighting**
   - Replace with kinematic_sequence or other 3D-derived metric

3. **Lower follow_through and finish weights** (they're not discriminating well)

4. **Re-run this audit** after each change to measure improvement

---

## Files to Edit

1. `src/biomechanics/scoring_config.py` - Adjust METRIC_WEIGHTS and SCORING_THRESHOLDS
2. Test with: `python run_scoring_audit.py`

**Target**: 15+ pts overall difference with all 8 phases showing >5pt separation
