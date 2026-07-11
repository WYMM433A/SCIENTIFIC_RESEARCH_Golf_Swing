# Feedback System Reference

Last updated: 2026-07-07 (branch `feat/feedback-system-improvements`)

---

## Metrics: What Gets Measured, Which Joints, How

All metrics are computed per-frame in `src/biomechanics/angles.py → calculate_all_metrics()`
and written to `data/metrics/{video}_cleaned_metrics.csv` by `pipeline.py → _extract_poses()`.
At scoring time, `train_phase_scorer.py → extract_features_for_swing()` reads the value at the
key frame for each phase.

All coordinates come from MediaPipe PoseLandmarker — normalized (0–1) screen coordinates.
Angles are in **degrees** unless stated otherwise.

### Posture

| Metric | Joints used | Formula | What it means |
|---|---|---|---|
| `spine_angle` | hip_mid → shoulder_mid | `atan2(dx, dy)` from vertical | Forward tilt of spine. ~40–50° at address is normal. Low = too upright; high = too bent over |
| `spine_lateral_tilt` | left_shoulder, right_shoulder | `atan2(right_y − left_y, abs(dx))` | Side bend. Positive = right shoulder lower. Should be near 0 at address, increases in backswing |

### Rotation (2D — camera-plane projection)

| Metric | Joints used | Formula | What it means |
|---|---|---|---|
| `shoulder_rotation` | left_shoulder, right_shoulder | `abs(atan2(dy, dx))` of shoulder line | Absolute angle of shoulder line from horizontal. 0 = square, increases as shoulders turn. Camera-view only |
| `hip_rotation` | left_hip, right_hip | same as above | Absolute angle of hip line from horizontal |
| `hip_angle` | left_hip, right_hip | **identical to `hip_rotation`** — same code, different name | Duplicate column; carries zero extra information |
| `x_factor` | computed from above | `min(│shoulder_rot − hip_rot│ % 360, 360 − ...)` | Hip-shoulder separation. Larger = more coil. Key power indicator |

### Rotation (3D — depth-aware)

| Metric | Status | Notes |
|---|---|---|
| `shoulder_rotation_3d` | **Always NaN** | Never implemented. Planned to use MediaPipe z-coordinates for depth-aware rotation. Imputed to a constant at training/inference time |
| `hip_rotation_3d` | **Always NaN** | Same — not implemented |
| `x_factor_3d` | **Always NaN** | Same — not implemented |

These three columns exist in `METRIC_COLS` and the model was trained with them as features, but they carry no real information. The XGBoost imputer fills them with a constant (imputed mean of all-NaN = 0 or similar).

### Arms

| Metric | Joints used | Formula | What it means | Alias of |
|---|---|---|---|---|
| `lead_arm_angle` | left_shoulder, left_elbow, left_wrist | `angle_3points(shoulder, elbow, wrist)` | Angle at lead (left) elbow. 180° = fully straight. Below 160° = noticeably bent | — |
| `arm_extension` | left_shoulder, left_elbow, left_wrist | **identical to `lead_arm_angle`** | Duplicate — same three-point angle computation | `lead_arm_angle` |
| `trail_elbow_angle` | right_shoulder, right_elbow, right_wrist | `angle_3points(shoulder, elbow, wrist)` | Angle at trail (right) elbow. ~170° at address, ~90° at top, ~150° at impact | — |
| `wrist_hinge` | left_elbow, left_wrist, left_index | `angle_3points(elbow, wrist, index)` | Angle at lead wrist (elbow-wrist-fingertip). Lower = more hinged/cocked. ~90° at top = good lag | — |
| `wrist_angle` | left_elbow, left_wrist, left_index | **identical to `wrist_hinge`** | Duplicate — same computation | `wrist_hinge` |
| `lag_angle` | left_elbow, left_wrist, left_index | **identical to `wrist_hinge`** | Duplicate — same computation. **Visual annotation draws trail-arm joints (right side) but metric is computed from lead arm (left side)** | `wrist_hinge` |

**Important:** `arm_extension`, `wrist_angle`, and `lag_angle` carry zero unique information.
Any feedback triggered on these will say the same thing as `lead_arm_angle` or `wrist_hinge` respectively.
The `lag_angle` visual annotation draws the wrong joints (right wrist/elbow) while the value comes from the left wrist.

### Lower Body

| Metric | Joints used | Formula | What it means |
|---|---|---|---|
| `lead_knee_flex` | left_hip, left_knee, left_ankle | `angle_3points(hip, knee, ankle)` | Bend angle at lead (left) knee. 180° = straight. ~155° at address, straightens toward 170–180° at impact |
| `trail_knee_flex` | right_hip, right_knee, right_ankle | same | Bend angle at trail (right) knee. ~155° at address, should stay flexed through top |
| `stance_width` | left_ankle, right_ankle | `euclidean_distance(left_ankle, right_ankle)` | Raw ankle-to-ankle distance in normalized units (0–1 screen width). Camera-scale dependent |
| `shoulder_width` | left_shoulder, right_shoulder | `euclidean_distance(...)` | Raw shoulder-to-shoulder distance in normalized units. Used only as denominator for ratio |
| `stance_width_ratio` | both ankles + both shoulders | `stance_width / shoulder_width` | Stance as multiple of shoulder width. 1.0 = same as shoulder width, 1.2 = 20% wider (typical for driver). Most reliable stance metric |

### Head / Stability

| Metric | Joints used | Formula | What it means |
|---|---|---|---|
| `head_lateral` | nose (current vs address) | `nose_x_current − nose_x_address` | Horizontal head drift. Computed relative to nose position at **address frame**. Near zero is ideal |
| `head_vertical` | nose (current vs address) | `nose_y_current − nose_y_address` | Vertical head movement. Positive = moved down in image |
| `head_displacement` | nose (current vs address) | `sqrt(lateral² + vertical²)` | Combined Euclidean head movement from address. Most reliable of the three for gating |

**Note:** `head_lateral` and `head_vertical` require the address frame to have been seen first to set the reference position. If this fails, both return 0.0.

---

### Per-Phase: What Gets Checked

The XGBoost model uses **all 23 metrics × 8 phases = 184 features** (plus 8 duration features = 192 total).
The feedback system evaluates metrics at the **key frame** for each phase, comparing to the benchmark.
Below is what each phase typically reveals and which metrics tend to carry signal.

**Address**
- Static setup check — no movement relative to previous phases
- Key metrics: `spine_angle`, `stance_width_ratio`, `lead_knee_flex`, `trail_knee_flex`, `shoulder_rotation` (should be near 0), `spine_lateral_tilt`
- `head_displacement` is always ~0 here (this is the reference frame)

**Takeaway**
- Club and arms begin moving; hips should stay relatively stable
- Key metrics: `shoulder_rotation` (should start loading), `x_factor` (separation begins), `lead_arm_angle` (should stay extended), `wrist_hinge` (small amount expected), `head_displacement` (should be minimal)

**Mid-backswing**
- Roughly when lead arm is parallel to ground (~9 o'clock position)
- Key metrics: `shoulder_rotation`, `x_factor`, `wrist_hinge` (loading), `lead_arm_angle`, `trail_knee_flex` (should maintain flex), `head_displacement`

**Top**
- Peak of backswing; maximum rotation checkpoint
- Key metrics: `shoulder_rotation` (~90° ideal), `x_factor` (max coil), `lead_arm_angle` (should be near 180°), `wrist_hinge` (~90° = good lag), `spine_angle` (should be maintained), `trail_knee_flex` (maintain for stability)
- `shoulder_rotation_3d` is theoretically the most important here but is always NaN

**Impact**
- Ball contact frame — most critical for strike quality
- Key metrics: `hip_rotation` (should be clear), `lead_arm_angle` (should approach 180° again), `lag_angle`/`wrist_hinge` (should be releasing), `x_factor` (hips ahead of shoulders), `head_displacement` (should be near 0 — head behind ball)

**Mid-downswing**
- Transition and lag retention check; roughly when lead arm is parallel to ground on the way down
- Key metrics: `lag_angle`/`wrist_hinge` (lag should still be retained here), `hip_rotation` (hips leading), `shoulder_rotation`, `x_factor` stretch

**Follow-through**
- Post-impact extension; arms extending through the ball
- Key metrics: `shoulder_rotation` (clearing), `hip_rotation`, `lead_arm_angle`, `spine_angle`

**Finish**
- Full rotation completion and balance check
- Key metrics: `shoulder_rotation`, `hip_rotation`, `stance_width_ratio`, `lead_knee_flex` (weight transfer), `wrist_hinge`/`wrist_angle`
- **Special case:** Finish XGBoost model has near-zero feature attribution for all features (bias-only model). The feedback system bypasses the contrib gate and uses pure benchmark deviation ranking. The underlying deviations are real; the model just can't attribute them to specific features.

---

## Benchmark Values (from `models/phase_scorer.pkl`)

Built at training time from **pro swings only** (or score ≥ 75 fallback per phase if < 5 pro labels).
All values are in degrees unless the metric is a distance or ratio (see notes below).
`norm_dev = |value − median| / IQR` — this is what `MIN_NORM_DEVIATION = 0.35` is gating against.

### Known Anomalies Before Reading the Table

| Issue | Cause | Impact on feedback |
|---|---|---|
| **head_lateral / head_vertical / head_displacement always 0** | `set_reference_position()` is never called in the current pipeline — the reference nose position is never stored, so all three metrics return 0.0 for every frame and every swing | These three metrics can never generate feedback. head_displacement _does_ exist in FEATURE_FEEDBACK_MAP but will never be triggered |
| **stance_width_ratio spikes at late phases (Top, Finish)** | When the golfer rotates sideways, both shoulders overlap in the 2D camera view — `shoulder_width` drops to near 0 pixels, making the ratio explode to 5×–32×. These outlier frames pollute the benchmark IQR | Feedback for stance_width_ratio at late phases may trigger on legitimate camera-geometry artefacts, not actual stance changes |
| **shoulder_rotation / hip_rotation wide IQR at Follow-through and Finish** | Some videos were recorded face-on, others from the side. The metric is camera-view dependent. At late phases the golfer's body orientation causes drastically different angles depending on camera angle | Low reliability for rotation feedback at Follow-through and Finish |
| **3D metrics (shoulder_rotation_3d, hip_rotation_3d, x_factor_3d) always NaN** | Never implemented — no code computes z-coordinate-based rotation | These features carry zero information; imputed to a constant at training/inference. The XGBoost model effectively ignores them |
| **arm_extension = lead_arm_angle exactly; wrist_angle = wrist_hinge = lag_angle exactly** | Aliases in `calculate_all_metrics()` — same function call, different dict key | The model has 3 duplicate wrist-hinge features and 2 duplicate lead-arm features. Feedback messages triggered on the alias names describe the same physical motion |
| **wrist_hinge benchmark at Top = 161°** | A well-hinged wrist at the top should be ~90°. This high median suggests either the computation measures something different from what we expect, or the training data contains many swings with poor lag at top | Feedback for "wrist hinge low at top" may be unusually easy to trigger |

---

### Benchmark Table: All 8 Phases × 23 Metrics

Format: `median (Q1 – Q3)  IQR`
Metrics with IQR = 0.00 are always zero and can never trigger feedback.

```
METRIC                  ADDRESS              TAKEAWAY          MID-BACKSWING
spine_angle             27.0° (7.3–34.5)  25.4   (4.4–33.8)   14.2  (1.9–30.1)  27.2
spine_lateral_tilt       4.9° (2.1–37.8)  -19.5 (-27.8–-15.3)  -22.4 (-35.5–-8.1)  35.8
shoulder_rotation       171.6 (89.1–177.1) 160.5 (152.2–164.7)  157.6 (144.5–171.8)
hip_rotation            173.6 (148.7–177.8) 172.0 (162.5–176.3) 169.5 (158.9–176.2)
x_factor                  4.0 (1.4–54.8)   11.1  (9.1–13.7)    10.1  (4.6–14.1)
lead_arm_angle          165.5 (157.8–172.9) 156.9 (148.3–168.9) 133.9 (117.3–148.6)
trail_elbow_angle       165.8 (162.6–168.6) 165.7 (149.7–175.3) 110.4  (57.4–147.0)
wrist_hinge             173.7 (168.2–177.1) 172.6 (168.2–177.2) 162.5 (154.2–169.7)
lead_knee_flex          174.4 (171.1–176.7) 170.6 (164.6–174.9) 165.3 (155.4–175.1)
trail_knee_flex         175.2 (167.0–177.6) 174.7 (166.2–177.2) 174.8 (171.8–176.3)
stance_width             89.7 (53.2–130.3)   70.0  (49.7–129.4)   84.7  (54.6–130.4)  [pixels]
shoulder_width           49.9 (11.6–99.7)    90.1  (75.9–105.7)   91.8  (70.8–121.8)  [pixels]
stance_width_ratio        1.46 (1.32–5.03)    1.26  (0.54–1.41)    0.76  (0.47–1.88)
head_lateral              0.0 (0–0)  IQR=0   0.0  (0–0)  IQR=0    0.0  (0–0)  IQR=0  [ALWAYS ZERO]
head_vertical             0.0 (0–0)  IQR=0   0.0  (0–0)  IQR=0    0.0  (0–0)  IQR=0  [ALWAYS ZERO]
head_displacement         0.0 (0–0)  IQR=0   0.0  (0–0)  IQR=0    0.0  (0–0)  IQR=0  [ALWAYS ZERO]

METRIC                  TOP                  IMPACT            MID-DOWNSWING
spine_angle             14.1° (-0.3–28.9)     6.7° (-6.7–22.4)   12.0°  (0.2–27.4)
spine_lateral_tilt      -23.9 (-31.0–-6.2)   24.3  (8.8–47.2)   -12.4 (-29.4–-5.3)
shoulder_rotation       155.4 (148.1–172.8)  152.5  (53.4–170.3) 166.5 (149.8–173.9)
hip_rotation            163.8 (158.6–175.9)  168.5 (102.0–173.7) 172.6 (159.2–176.9)
x_factor                  7.9 (3.7–11.3)      13.2  (5.4–20.4)    6.0  (3.1–9.5)
lead_arm_angle          147.2 (130.2–154.9)  146.1 (133.2–162.9) 120.1 (104.2–131.6)
trail_elbow_angle       108.7 (71.8–141.0)   152.8 (138.1–160.2)  64.8  (36.3–121.6)
wrist_hinge             161.4 (150.3–165.9)  168.4 (163.0–174.4) 164.8 (154.7–173.4)
lead_knee_flex          160.6 (146.7–172.7)  175.2 (169.8–178.2) 166.1 (156.6–172.1)
trail_knee_flex         174.2 (165.7–177.6)  166.1 (160.4–170.5) 163.8 (155.5–174.0)
stance_width             90.1 (53.8–130.1)    66.1  (34.9–140.5)   88.5  (42.1–131.3)  [pixels]
shoulder_width           84.0 (58.8–105.7)    74.9  (29.1–94.5)    85.5  (73.7–99.7)  [pixels]
stance_width_ratio        0.82 (0.52–2.18)     1.45  (0.89–1.64)    1.40  (0.55–1.59)
head metrics:   all 0.0  IQR=0                all 0.0  IQR=0        all 0.0  IQR=0    [ALWAYS ZERO]

METRIC                  FOLLOW-THROUGH       FINISH
spine_angle              8.2° (-10.1–20.2)    5.9° (-0.7–13.6)
spine_lateral_tilt      33.2  (16.9–42.2)     4.0  (-4.4–20.8)
shoulder_rotation        58.4  (40.9–161.8)   41.4  (19.7–175.2)   IQR=120.9 / 155.5 — HIGH NOISE ⚠
hip_rotation             60.2  (27.5–171.1)  135.0  (11.8–176.1)   IQR=143.6 / 164.3 — HIGH NOISE ⚠
x_factor                 13.8  (6.6–21.3)     8.5   (4.0–13.7)
lead_arm_angle          112.2  (67.9–155.0)   73.2  (48.0–107.9)
trail_elbow_angle       125.9  (38.8–149.4)   71.9  (58.1–98.0)
wrist_hinge             167.9 (154.7–175.4)  158.4 (151.7–166.0)
lead_knee_flex          175.1 (171.1–177.3)  172.4 (170.3–177.6)
trail_knee_flex         166.0 (159.2–170.7)  168.5 (153.1–174.8)
stance_width             52.8  (29.0–123.0)   62.2  (25.1–101.3)  [pixels]
shoulder_width           81.6  (63.1–119.4)   64.0  (40.7–93.9)   [pixels]
stance_width_ratio        1.34  (0.31–1.74)    0.97  (0.51–1.52)
head metrics:    all 0.0  IQR=0               all 0.0  IQR=0       [ALWAYS ZERO]
```

Note: `arm_extension`, `hip_angle`, `wrist_angle`, `lag_angle`, and all 3D variants are exact aliases of other metrics shown above —
they will always have the same value and benchmark.

The full raw CSV is at `outputs/benchmark_values.csv` (generated on 2026-07-07).

---



```
pipeline.py  →  predict_scores()  →  generate_phase_feedback()  →  text feedback
                                  →  _top_deviated_metric_names()  →  draw_annotated_keyframe()  →  red joints image
```

Every time `pipeline.py` processes a video with `--scoring-backend xgboost`, it calls
`_score_swing_xgboost()` which calls `predict_scores()` from `train_phase_scorer.py`.
That one function handles both text feedback and the visual annotations.

---

## File Responsibilities

| File | Responsible for |
|---|---|
| `train_phase_scorer.py` | Everything: constants, feedback map, all feedback logic, visual drawing |
| `pipeline.py` `_score_swing_xgboost()` ~line 288 | Calls `predict_scores()`, formats output, writes CSVs |
| `models/phase_scorer.pkl` | Trained XGBoost models + imputer + feature_cols + **benchmark stats** |

---

## Constants (`train_phase_scorer.py` lines ~44–47)

```python
FEEDBACK_SCORE_THRESHOLD = 85.0   # phases scoring >= this get NO feedback
FEEDBACK_MAX_ITEMS       = 2      # max text messages returned per phase
MIN_LOCAL_CONTRIB        = 0.02   # XGBoost per-feature attribution minimum
MIN_NORM_DEVIATION       = 0.35   # how far from benchmark (in IQR units) to qualify
```

**What they gate:**
- `FEEDBACK_SCORE_THRESHOLD` — the entire feedback system is off for phases scoring ≥ 85. Checked first, returns `[]` immediately.
- `MIN_NORM_DEVIATION` — filters out small real deviations. Value is in IQR units (e.g. 0.35 = 35% of one interquartile range).
- `MIN_LOCAL_CONTRIB` — filters out features that XGBoost says barely influenced the score prediction. **Bypassed when all contribs are zero** (see Finish phase fix below).

---

## FEATURE_FEEDBACK_MAP (`train_phase_scorer.py` lines ~79–245)

A Python dict:
```python
(phase_prefix, metric_name, "high"|"low") → "human readable coaching message"
```

**phase_prefix** is the phase name lowercased with spaces/dashes replaced by `_`:
- `"address"`, `"takeaway"`, `"mid_backswing"`, `"top"`, `"impact"`, `"mid_downswing"`, `"follow_through"`, `"finish"`

**metric_name** is the raw column name without the phase prefix:
- e.g. for column `"top_shoulder_rotation"` → metric is `"shoulder_rotation"`

**196 entries** as of 2026-07-07 (was 52 before this session).

**Completely unmapped metrics** (no entry for any phase/direction): `hip_angle`, `head_lateral`, `head_vertical`, `spine_lateral_tilt` (only address covered). Adding an entry here is the lowest-risk way to add new coaching text.

**To add a new message**, just append a new entry:
```python
("impact", "hip_angle", "low"): "Hip angle too upright at impact — maintain flex",
```

---

## Benchmark Stats (inside `models/phase_scorer.pkl`)

The benchmark is built at **training time** from pro swings (or score ≥ 75 swings as fallback).
For each `(phase × metric)` it stores `{median, q1, q3, iqr}`.

At inference, each feature value is compared to its benchmark:
```
deviation = value - median
norm_dev  = |deviation| / IQR
direction = "high" if deviation > 0 else "low"
```

**The benchmark is frozen in the .pkl file.** To update it you must retrain with `train_phase_scorer.py`.

---

## Text Feedback Pipeline: `generate_phase_feedback()` (~line 614)

Called once per phase per swing. Steps in order:

### Step 0 — Early exit
```python
if score >= FEEDBACK_SCORE_THRESHOLD:
    return []
```
No feedback is generated for good phases.

### Step 1 — Detect bias-only model (all-local-zero check)
```python
phase_locals = [local_contribs[col] for col in feature_cols if col starts with prefix]
all_local_zero = all values are < 1e-6
```
When `all_local_zero = True`, the XGBoost model scored this phase entirely via its bias term (no feature had meaningful attribution). This happens on the **Finish phase** because training labels for Finish had low variance. When this flag is set, the contrib gate is bypassed.

### Step 2 — Loop through all features for this phase
For each feature column:
1. Compute `deviation`, `norm_dev`, `local` (XGBoost attribution), `weighted`
2. `weighted = norm_dev` if `all_local_zero` else `local × norm_dev`
3. **Gate 1:** skip if `norm_dev < MIN_NORM_DEVIATION` (0.35)
4. **Gate 2:** skip if `local < MIN_LOCAL_CONTRIB` **AND** NOT `all_local_zero`
5. Determine `direction` ("high" or "low")
6. Look up `(prefix, metric, direction)` in `FEATURE_FEEDBACK_MAP`
   - Found → add to `deviations` list (primary)
   - Not found → add to `fallback_candidates` list (generic)

### Step 3 — Select top messages
```python
deviations.sort(by weighted, descending)
return top FEEDBACK_MAX_ITEMS (2) unique messages
```

### Step 4 — Fallback chain (if deviations list is empty)
1. **`fallback_candidates`** — strict gates passed but no map key: `"Phase: metric deviates from benchmark; prioritize correcting this checkpoint"`
2. **`relaxed_candidates`** — weighted > 0 but below strict gates: `"Phase: metric is the largest measurable deviation in this phase"`
3. **Absolute safety net** — everything filtered: `"Phase: phase score is below target; review posture, rotation, and balance checkpoints"`

**Note:** The pipeline.py wrapper adds one more layer — if `feedback` list is still empty for a weak phase (shouldn't happen after the safety net, but guards against None): `"Needs attention (no mapped coaching text; see annotated keyframe)"`

---

## Visual Annotation Pipeline

### `_top_deviated_metric_names()` (~line 730)

Runs **separately** from `generate_phase_feedback()` — it picks which metrics to highlight in red. This is **NOT** the same selection as the text feedback.

**Current behaviour (known issue):**
- Uses the old strict gates: `local >= MIN_LOCAL_CONTRIB AND norm_dev >= MIN_NORM_DEVIATION AND weighted > 0`
- Falls back to `relaxed_deviations` (sorted by `local × norm_dev`) if strict gates return nothing
- **Does NOT use the `all_local_zero` bypass** — so for Finish phase, it falls back to the relaxed list which may pick metrics based on very small non-zero `local × norm_dev` values, causing seemingly unrelated joints to be highlighted

**Why you see red lines on unrelated body parts:** The relaxed fallback sorts by `local × norm_dev`. If all local contribs are near zero (Finish phase), every metric has a near-zero weighted score and the ranking is essentially noise — whichever metric has even the tiniest attribution wins, which may not correspond to the actual deviation.

### `draw_annotated_keyframe()` (~line 777)

Takes a list of metric names, draws the skeleton, and highlights joints red:

1. Load the key frame JPG from `data/keyframes/{swing_id}/{phase_name}.jpg`
2. Load pose landmarks from `data/extracted_poses/{video}_cleaned_poses.csv`
3. Get the key frame index from the phases CSV
4. Build pixel coords from normalized (0–1) landmark coordinates
5. Look up each deviated metric in `METRIC_TO_JOINTS` dict (~line 318) to get joint segment pairs
6. **Pass 1** — draw all SKELETON_CONNECTIONS in gray, **skipping** segments that will be red
7. **Pass 2** — draw red segments thick (4px) and red joints as filled circles (7px radius)
8. Add black banner at top with phase name, score, and tag (GOOD / OK / NEEDS WORK)
9. Save to `outputs/visual_feedback/{swing_id}/{phase_name}_annotated.jpg`

### `METRIC_TO_JOINTS` (~line 318)

Maps metric name → list of `(landmark_a, landmark_b)` segment pairs to highlight.
All landmark names are MediaPipe names (e.g. `"left_shoulder"`, `"right_elbow"`).

Example:
```python
"shoulder_rotation": [("left_shoulder", "right_shoulder")]
"x_factor":          [("left_shoulder", "right_shoulder"), ("left_hip", "right_hip")]
"lag_angle":         [("right_elbow", "right_wrist"), ("right_wrist", "right_index")]
```

**Missing from METRIC_TO_JOINTS** (these metrics would draw nothing if selected): currently all metrics have entries, but `right_index` landmark may not always be detected by MediaPipe, silently skipping the segment.

---

## Output Files

| File | Contents |
|---|---|
| `data/metrics/{video}_scores.csv` | One row per phase + Overall, columns: phase, score, confidence, key_frame, feedback (single string) |
| `data/metrics/{video}_feedback_detailed.csv` | One row per feedback message, columns: phase, feedback, phase_score, key_frame |
| `outputs/visual_feedback/{swing_id}/{phase}_annotated.jpg` | Annotated keyframe image, only generated for phases scoring < 85 |

---

## Known Issues / Future Fixes

### 1. ~~Visual annotation uses different logic from text feedback~~ — FIXED
`_top_deviated_metric_names()` now uses the same `all_local_zero` bypass as `generate_phase_feedback()`.
For akh Finish (score 53.2), both now agree: `stance_width_ratio`, `wrist_hinge` highlighted red.
Fixed in commit `85c0f41`.

### 2. Finish phase XGBoost model relies on bias
The Finish model has near-zero local contribs for all features because Finish training labels had low variance. The `all_local_zero` bypass is a workaround. The real fix is retraining with more diverse Finish phase labels.

### 3. FEEDBACK_MAX_ITEMS = 2
Only 2 feedback messages max per phase. Players with multiple issues will only see the top 2 by weighted score.

### 4. Benchmark is fixed at training time
The pro benchmark is locked in `phase_scorer.pkl`. As more annotated data is collected and the model is retrained, the benchmark updates automatically. No manual intervention needed.
