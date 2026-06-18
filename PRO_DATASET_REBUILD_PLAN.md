# Pro Dataset Rebuild Plan

## Purpose
This plan defines exactly what to update in the current dataset, how to change the annotation process, and what to ask the pro reviewer to annotate for each video so phase scores and feedback become more accurate.

## Current Baseline
- Total labeled swings: 88
- Pro swings: 35
- Intermediate swings: very limited
- Phase score model performance: strong separation in training, but feedback quality is less stable
- Main gap: feedback benchmark and annotation consistency are weaker than scoring quality

## Target Outcome
1. Improve score generalization (not only in-sample quality).
2. Make feedback match true swing faults more reliably.
3. Build phase-specific benchmark profiles that can be selected intentionally.

---

## Part 1: Dataset Changes You Should Make Now

### 1.1 Clean and normalize `pro_annotation_sample.csv`
Update these rules before adding new rows:
- Use one header format only.
- Keep `skill_level` strictly lowercase: `pro`, `intermediate`, `beginner`.
- Keep `camera_angle` strictly lowercase: `front`, `side`.
- Keep `confidence` in integer range 1-5.
- Ensure all score columns are numeric in 0-100.
- Remove fully empty rows.
- Quote feedback text consistently (always quote if text has commas or semicolons).

### 1.2 Keep this column order fixed
- `swing_id`
- `camera_angle`
- `skill_level`
- `confidence`
- `address_score`
- `take_away_score`
- `mid_backswing_score`
- `top_score`
- `mid_downswing_score`
- `impact_score`
- `follow_through_score`
- `finish_score`
- `feedback`

### 1.3 Add metadata columns (recommended)
Add these optional columns to support better filtering and benchmark design:
- `annotator_id`
- `session_id`
- `club_type` (driver, iron, etc.)
- `view_quality` (good, medium, poor)
- `pose_quality` (good, medium, poor)
- `phase_quality` (good, medium, poor)
- `exclude_from_benchmark` (0/1)

If you do not want to change the main CSV immediately, keep these in a sidecar CSV keyed by `swing_id`.

---

## Part 2: Data Collection Mix for Next Batch

Do not only increase pro count. Increase diversity.

### 2.1 Recommended next milestone
- Pro: +80 to +120 swings
- Intermediate: +40 to +60 swings
- Beginner: +20 to +40 swings

### 2.2 Diversity constraints
For each skill group, target:
- Front/side balance close to 50/50
- Multiple golfers, not one golfer repeated
- Varied body types and tempos
- Varied shot intent (stock, controlled, aggressive)

### 2.3 Duplicate control
- Maximum 3 highly similar swings per player-session in training set
- Keep extra similar swings in a holdout folder for later stress tests

---

## Part 3: What to Ask the Pro to Annotate for Each Video

Ask the pro to provide BOTH numeric scores and structured fault tags.

### 3.1 Required per-video annotation package
For each `swing_id`, collect:
1. Eight phase scores (0-100)
2. Confidence (1-5)
3. Top 1-2 faults per phase (from a controlled tag list)
4. Severity per fault (`minor`, `moderate`, `major`)
5. Optional short free-text coaching note

### 3.2 Controlled fault tags by phase
Use these as the standard vocabulary.

#### Address
- `address_spine_too_upright`
- `address_spine_too_bent`
- `address_stance_too_narrow`
- `address_stance_too_wide`
- `address_knee_flex_low`
- `address_hands_too_close`
- `address_hands_too_far`

#### Takeaway
- `takeaway_low_shoulder_turn`
- `takeaway_low_xfactor`
- `takeaway_early_wrist_break`
- `takeaway_low_wrist_set`
- `takeaway_head_drift`
- `takeaway_lead_arm_bent`

#### Mid-backswing
- `mbs_low_xfactor`
- `mbs_low_xfactor_3d`
- `mbs_low_shoulder_turn`
- `mbs_low_wrist_hinge`
- `mbs_lead_arm_bent`
- `mbs_trail_knee_straightening`
- `mbs_head_drift`

#### Top
- `top_low_shoulder_turn`
- `top_low_shoulder_turn_3d`
- `top_low_wrist_hinge`
- `top_low_xfactor`
- `top_lead_arm_bent`
- `top_spine_loss`
- `top_head_drift`

#### Mid-downswing
- `mds_low_hip_rotation`
- `mds_low_hip_rotation_3d`
- `mds_low_xfactor`
- `mds_early_shoulder_fire`
- `mds_early_release`
- `mds_head_drift`

#### Impact
- `impact_low_hip_rotation`
- `impact_low_hip_rotation_3d`
- `impact_lead_arm_bent`
- `impact_low_lag`
- `impact_body_arm_desync`
- `impact_head_drift`

#### Follow-through
- `ft_low_shoulder_rotation`
- `ft_low_hip_rotation`
- `ft_low_hip_rotation_3d`
- `ft_lead_arm_bent`
- `ft_spine_loss`

#### Finish
- `finish_low_shoulder_rotation`
- `finish_low_hip_rotation`
- `finish_low_hip_rotation_3d`
- `finish_balance_loss`
- `finish_incomplete_release`

### 3.3 Pro annotation instruction text (copy/paste)
Use this instruction when requesting annotations:

- Score each of 8 phases from 0-100 based on model target criteria.
- For each phase, select up to 2 fault tags from the approved tag list.
- Add severity for each selected tag: minor, moderate, major.
- Use confidence 1-5 for your overall reliability on this video.
- If view or pose quality is poor, mark quality and still score only if reliable.
- Keep free text short and optional; rely on tags first.

---

## Part 4: Phase-Specific Benchmark Design (Critical for Feedback Accuracy)

Do not use one generic benchmark pool.

### 4.1 Build benchmark pools per phase
For each phase, include rows that satisfy:
- `skill_level == pro`
- `confidence >= 4`
- phase score threshold for that phase
- `pose_quality != poor`
- `phase_quality != poor`

### 4.2 Suggested starting thresholds
- Address: >= 90
- Takeaway: >= 90
- Mid-backswing: >= 88
- Top: >= 90
- Mid-downswing: >= 88
- Impact: >= 92
- Follow-through: >= 88
- Finish: >= 85

### 4.3 Use robust statistics, not only mean
For each phase-feature benchmark, store:
- median
- Q1
- Q3
- optional p10/p90

Feedback trigger should require meaningful out-of-band deviation, not tiny distance from mean.

---

## Part 5: Model and Feedback Pipeline Changes

### 5.1 Keep score model as current base
- Continue per-phase XGBoost regressors.
- Keep GroupKFold by player group.

### 5.2 Upgrade feedback ranking
Current approach uses global feature importance. Improve to local explanation:
- Use per-sample SHAP (or equivalent local attribution) per phase model.
- Rank candidate faults by local contribution strength and benchmark deviation.

### 5.3 Add feedback gates
Only show feedback if all are true:
- phase score below threshold (example < 85)
- local contribution above threshold
- deviation outside benchmark band by minimum amount

### 5.4 Keep output concise
- max 2 feedback items per phase
- avoid conflicting messages

---

## Part 6: Validation Plan (Score + Feedback)

### 6.1 Split strategy
- Keep GroupKFold for model development.
- Maintain a fixed holdout set from unseen golfers for final checks.

### 6.2 Metrics to track every retrain
Score metrics:
- MAE per phase
- weighted MAE (weights by confidence)
- holdout pro/beginner and pro/intermediate gap

Feedback metrics:
- Top-1 tag match rate
- Top-2 tag match rate
- precision/recall on major faults
- no-fault correctness for high-scoring phases

### 6.3 Acceptance targets (practical)
- MAE <= 10 on most phases
- Finish MAE <= 14
- Top-2 feedback tag match >= 70% on holdout
- no-feedback correctness >= 85% for phases >= 90 score

---

## Part 7: Operational Workflow

### 7.1 Weekly loop
1. Ingest new videos.
2. Run pose + phase extraction.
3. Annotate with pro template.
4. Run dataset validator.
5. Rebuild training features.
6. Retrain model.
7. Run evaluation report.
8. Compare against previous checkpoint.

### 7.2 Retrain rules
- Retrain only when at least 20 new high-quality rows added, or one major data-cleaning pass completed.
- Keep versioned artifacts:
  - model pkl
  - benchmark stats
  - evaluation report

---

## Part 8: Immediate Next Actions (Do These First)

1. Normalize current CSV formatting and labels.
2. Add canonical fault tags to each existing row (start with pro rows).
3. Implement phase-specific benchmark pools with thresholds above.
4. Switch feedback ranking to local attribution + deviation gates.
5. Re-evaluate on a golfer-holdout set and review top-2 feedback matches.

---

## Quick Summary
- More pro data helps, but benchmark design and annotation structure matter just as much.
- Use phase-specific benchmark pools and robust bands.
- Standardize pro feedback into canonical tags.
- Evaluate feedback quality explicitly, not only score MAE.
