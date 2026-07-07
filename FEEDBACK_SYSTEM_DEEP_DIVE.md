# Feedback System Deep Dive
> Current state as of 2026-07-07 — based on `train_phase_scorer.py`  
> This document describes the system **before** any redesign changes.

---

## 1. What the Feedback System Does

After the XGBoost model predicts a score for each of the 8 phases, the feedback system answers the question: **"What specifically is wrong in the weak phases, and what should the golfer fix?"**

It produces two outputs per swing:
- `data/metrics/{video}_scores.csv` — one feedback text string per phase
- `data/metrics/{video}_feedback_detailed.csv` — one row per individual feedback message
- (optional) `outputs/visual_feedback/{swing_id}/{phase}_annotated.jpg` — skeleton image with problem joints highlighted red

---

## 2. Control Constants

```python
FEEDBACK_SCORE_THRESHOLD = 85.0   # Only phases scoring BELOW this get feedback
FEEDBACK_MAX_ITEMS       = 2      # Max feedback messages per phase (flat cap)
MIN_LOCAL_CONTRIB        = 0.02   # XGBoost per-feature contribution must exceed this
MIN_NORM_DEVIATION       = 0.50   # Deviation from benchmark (in IQR units) must exceed this
```

These four values act as a combined gate. **All three conditions** (threshold, local contrib, norm deviation) must be true simultaneously for a metric to produce a feedback message. If any one fails, the metric is dropped entirely.

---

## 3. The Three Inputs to Feedback Generation

For each phase, feedback uses three sources of information:

| Input | Where it comes from | What it represents |
|---|---|---|
| `features` | `data/metrics/{video}_cleaned_metrics.csv` at the phase keyframe | Actual measured biomechanical values for this swing |
| `benchmark` | Built during training, stored in `models/phase_scorer.pkl` | Median + IQR of the same metrics in good/pro swings |
| `local_contribs` | XGBoost SHAP-style `pred_contribs` | How much each feature influenced the score prediction |

---

## 4. How Benchmarks Are Built

During training (`train_phase_scorer.py --train`), `_build_phase_benchmarks()` computes per-phase, per-feature statistics from the training data.

**Priority order for which swings to use as the benchmark:**

1. **Pro-only** — if `skill_level == 'pro'` and there are at least 5 pro samples
2. **Score ≥ 75** — fallback if not enough pros
3. **Top 50% of scores** — last resort if even score ≥ 75 has fewer than 3 samples

For each phase-feature combination, it stores:
```python
{
  "median": float,   # central benchmark value
  "q1":     float,   # 25th percentile
  "q3":     float,   # 75th percentile
  "iqr":    float,   # q3 - q1 (spread; used for normalization)
}
```

At inference, deviation from benchmark is computed as:
```
norm_dev = abs(actual_value - median) / IQR
```

So `norm_dev = 1.0` means the value is exactly one IQR away from the benchmark median. The gate requires `norm_dev >= 0.50` — i.e. at least half an IQR off.

---

## 5. The Feedback Generation Pipeline (Step by Step)

This is the core of `generate_phase_feedback()`.

### Step A — Gate check
```python
if score >= FEEDBACK_SCORE_THRESHOLD:
    return []   # Phase is good enough — no feedback needed
```

### Step B — Loop over all features for this phase

Only features whose name starts with the phase prefix are considered. For example, for Top phase, only features like `top_shoulder_rotation`, `top_hip_rotation`, etc. are examined.

For each feature, the function computes:
```python
deviation   = actual_value - benchmark_median
norm_dev    = abs(deviation) / IQR
local       = SHAP contribution of this feature to the score
weighted    = local * norm_dev        # combined priority score
direction   = "high" if deviation > 0 else "low"
```

Note: there is currently a duplicate `weighted = local * norm_dev` line (line ~510 and ~515 in the file). The second assignment is dead code but harmless.

### Step C — Three separate candidate lists

Every feature with `weighted > 0` is added to `relaxed_candidates` (no gates applied).

If it also passes both strict gates (`local >= MIN_LOCAL_CONTRIB` AND `norm_dev >= MIN_NORM_DEVIATION`):
- If `(prefix, metric, direction)` key exists in `FEATURE_FEEDBACK_MAP` → added to `deviations` (mapped text)
- If no map key exists → added to `fallback_candidates` (needs generic text)

### Step D — Fallback chain (priority order)

The function tries each level in order and stops as soon as it has at least one message:

```
Level 1: FEATURE_FEEDBACK_MAP mapped messages (strict gates + exact key match)
   ↓ if empty
Level 2: fallback_candidates — strict gates passed but no map key (generic metric text)
   ↓ if empty
Level 3: relaxed_candidates — no strict gates, any non-zero weighted deviation (relaxed metric text)
   ↓ if empty
Level 4: absolute safety net — "phase score is below target; review posture, rotation, and balance"
```

### Step E — Deduplication and cap
```python
seen, feedback = set(), []
for _, msg in deviations:
    if msg not in seen and len(feedback) < max_items:
        seen.add(msg)
        feedback.append(msg)
```

The flat cap of `FEEDBACK_MAX_ITEMS = 2` applies to the final output regardless of score severity.

---

## 6. FEATURE_FEEDBACK_MAP — Coverage and Structure

The map is a Python dict with `(phase_prefix, metric_name, direction)` as key and a coaching string as value.

```python
("impact", "hip_rotation", "low"): "Hip rotation low at impact position"
```

### Current coverage

| Phase | Metrics mapped | Coverage |
|---|---|---|
| Address | spine_angle (×2), spine_lateral_tilt, stance_width_ratio (×2), wrist_angle (×2), lead_knee_flex, trail_knee_flex | Good |
| Takeaway | x_factor, shoulder_rotation, wrist_angle, wrist_hinge, lead_arm_angle, head_displacement | Partial |
| Mid-backswing | x_factor (×2), shoulder_rotation, lead_arm_angle, wrist_angle, wrist_hinge, head_displacement, trail_knee_flex | Partial |
| Top | shoulder_rotation (×2), head_displacement, wrist_angle, wrist_hinge, spine_angle, x_factor, lead_arm_angle | Partial |
| Mid-downswing | lag_angle, hip_rotation (×2), x_factor, shoulder_rotation, head_displacement | Partial |
| Impact | hip_rotation (×2), lead_arm_angle, x_factor, lag_angle, head_displacement | Partial |
| Follow-through | shoulder_rotation, lead_arm_angle, hip_rotation (×2), spine_angle | Partial |
| Finish | shoulder_rotation, lead_knee_flex, hip_rotation (×2), head_displacement, spine_angle | Partial |

### Completely unmapped metrics (no entry for any phase)

- `trail_elbow_angle` — both directions, all phases
- `arm_extension` — both directions, all phases
- `shoulder_width` — both directions, all phases
- `stance_width` — both directions, all phases
- `hip_angle` — both directions, all phases
- `head_lateral` — both directions, all phases
- `head_vertical` — both directions, all phases

These 7 metrics (out of 23 total) have zero coverage. When they are the top deviation, the system falls through to fallback messages.

### Partially unmapped combinations (present in some phases but missing in others)

- `lag_angle low` — missing at most phases (only `lag_angle high` is mapped at mid-downswing and impact)
- `x_factor high` — only mapped at impact
- `trail_knee_flex high` — missing at all phases
- `lead_knee_flex high` — only mapped at finish
- `hip_rotation high` — missing at all phases
- `shoulder_rotation high` — only mapped at mid-downswing
- `x_factor_3d` — only mapped at mid-backswing

Total mapped entries: ~52 out of a possible ~368 (23 metrics × 8 phases × 2 directions) = **~14% coverage**.

---

## 7. Visual Annotation — `_top_deviated_metric_names()`

This runs separately from text feedback, but uses the same deviation logic, to decide which skeleton joints to highlight red in the keyframe image.

It returns up to 3 metric names. These are converted to joint segments via `METRIC_TO_JOINTS`, which maps each metric to the body segments it represents:

```python
"shoulder_rotation": [("left_shoulder", "right_shoulder")]
"hip_rotation":      [("left_hip", "right_hip")]
"lead_arm_angle":    [("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist")]
```

Joints are drawn in two passes:
1. Gray skeleton first (all bones except the ones being highlighted)
2. Red thick lines on top (the deviated joints)

A banner is printed at the top of the image: `{phase} | {score}/100 | [NEEDS WORK / OK / GOOD]`

Annotation is only triggered for phases with `score < FEEDBACK_SCORE_THRESHOLD`.

---

## 8. How Feedback Integrates with Pipeline

In `pipeline.py`, the full call chain is:

```
_score_swing_xgboost()
    → predict_scores(swing_id, model_path, annotate=True)
        → generate_phase_feedback() per phase
        → _top_deviated_metric_names() per weak phase (for annotation)
        → draw_annotated_keyframe() per weak phase with deviations
    ← returns: scores, feedback, total, images
→ writes scores CSV
→ writes feedback_detailed CSV
→ prints annotated keyframe paths
```

For each phase, the pipeline also maps the feedback key using normalized phase names (lowercase alphanumeric) to handle any phase-name mismatch between model output and pipeline's phase name list.

---

## 9. Output File Formats

### `_scores.csv`

```
phase, score, confidence, key_frame, feedback
Address, 82.2, 1.0, 175, Spine angle too upright at address — add more forward tilt
Takeaway, 59.1, 1.0, 47, Shoulder rotation low at takeaway checkpoint
...
Overall, 71.4, 1.0, -1, OVERALL: 71.4/100
```

- One row per phase plus Overall
- `feedback` column contains all messages for that phase joined by ` | `
- For phases scoring ≥ 85: `"No issues detected"`
- For phases scoring < 85 with empty feedback after all fallbacks: `"Needs attention (no mapped coaching text; see annotated keyframe)"`

### `_feedback_detailed.csv`

```
phase, feedback, phase_score, key_frame
Takeaway, Shoulder rotation low at takeaway checkpoint, 59.1, 47
Mid-backswing, Lead arm bent at mid-backswing checkpoint, 70.7, 52
```

- One row per individual feedback message (not per phase)
- Only populated for phases that produced at least one message
- Empty (whitespace only) if no phases produced messages

---

## 10. Known Issues and Gaps

| Issue | Root Cause | Current Effect |
|---|---|---|
| ~86% of metric-direction combinations have no mapped text | FEATURE_FEEDBACK_MAP is only ~14% complete | Most feedback falls to generic fallback text |
| Flat FEEDBACK_MAX_ITEMS=2 cap | Constant not scaled by severity | A score of 30 gets same max messages as a score of 82 |
| Double weighted computation | Duplicate line in generate_phase_feedback | Dead code, no functional impact |
| No cross-phase summary | Feedback is generated independently per phase | Repeated issues (e.g. hip rotation low in 5 phases) appear as 5 separate messages |
| Pro free-text annotations unused | Only numeric scores are read from pro_annotation_sample.csv | Valuable coaching language in annotations column is ignored |
| Confidence column used as input feature only | Not used as sample weight during training | Low-confidence annotations influence training equally as high-confidence ones |
| lag_angle direction semantics | lag_angle high at mid-downswing is mapped as "wrist lag low" | Correct behaviour because high lag_angle number corresponds to early release — but confusing to read in code |
| Benchmark uses score≥75 fallback for non-pro data | Fallback may include mediocre swings | Benchmark may not represent truly good technique |
