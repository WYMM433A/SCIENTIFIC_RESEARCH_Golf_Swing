# Feedback Dataset Collection Guide

This guide explains how to collect and label data to improve feedback quality while keeping the current scoring + benchmark approach.

## 1. Goal

Improve feedback precision and coverage by adding better data for:
- issue coverage in each phase
- stable pro benchmark statistics
- view-balanced examples (front and side)

This guide does not require replacing the current model architecture.

## 2. What Data To Collect Next

Priority order:

1. More clean pro swings
- Target: 150 to 300 total pro swings
- Why: stabilizes benchmark median and IQR for each phase metric

2. Intermediate swings
- Target: 80 to 150
- Why: improves feedback quality for "OK" scores (not only pro vs beginner extremes)

3. Fault-rich swings for late phases
- Focus: Impact, Follow-through, Finish
- Why: these phases currently have weaker feedback coverage

4. Balanced camera views
- Keep front and side in similar proportions for each skill level
- Why: metric distributions differ by view

5. Small set of low-quality clips (labeled)
- Include occlusion, low light, cropped body
- Why: helps confidence rules and suppression behavior

## 3. Recommended Target Counts

Use this as a minimum planning matrix:

| Group | Front | Side | Total |
|---|---:|---:|---:|
| Pro | 75 | 75 | 150 |
| Intermediate | 40 | 40 | 80 |
| Beginner | 50 | 50 | 100 |

Phase-fault targets (minimum):
- each common issue in Impact, Follow-through, Finish: 40+ positives
- each common issue in Address, Takeaway, Mid-backswing, Top, Mid-downswing: 25+ positives

## 4. Label Schema (Recommended)

Create a separate file for feedback labels, for example:
- data/feedback_labels.csv

Columns:

| Column | Type | Description |
|---|---|---|
| swing_id | string | Example: B12_F1_I_nn |
| phase | string | One of 8 phases |
| camera_view | string | front or side |
| quality_ok | int | 1 if labelable, 0 if not reliable |
| primary_issue_tag | string | one main issue tag |
| secondary_issue_tags | string | pipe-separated, max 2 |
| severity | int | 0 to 3 (none, mild, moderate, severe) |
| confidence | int | 1 to 5 |
| cue_id | string | coaching cue ID |
| drill_id | string | drill ID |
| pro_note | string | optional free text |

## 5. Issue Tag Taxonomy Design

Use phase-aware, direction-aware tags:

Examples:
- address.wrist_angle_high
- takeaway.lead_arm_angle_low
- mid_backswing.x_factor_low
- top.wrist_hinge_low
- impact.hip_rotation_low
- follow_through.shoulder_rotation_low
- finish.lead_knee_flex_high

Rules:
- one clear meaning per tag
- avoid overlapping tags that mean the same thing
- keep naming consistent with metrics in your codebase

## 6. Instructions For Pro Labelers

Ask pros to label each phase using this sequence:

1. Is this phase labelable from this view?
- yes or no

2. Primary issue
- choose exactly one from taxonomy

3. Secondary issues
- choose up to two

4. Severity
- 0 none, 1 mild, 2 moderate, 3 severe

5. Confidence
- 1 low to 5 high

6. Cue and drill
- select from predefined libraries

7. Optional note
- one short sentence

Labeling policy:
- use what is visible in the clip only
- do not infer hidden motion that the view does not support
- if uncertain, lower confidence instead of guessing

## 7. Sample Label Examples

### Example A: Pro swing with minor issue

| swing_id | phase | camera_view | quality_ok | primary_issue_tag | secondary_issue_tags | severity | confidence | cue_id | drill_id | pro_note |
|---|---|---|---:|---|---|---:|---:|---|---|---|
| pro_021_nn | Mid-backswing | front | 1 | mid_backswing.x_factor_low | mid_backswing.shoulder_rotation_low | 1 | 4 | CUE_COIL_01 | DRILL_BAND_TURN | Slightly more torso turn needed. |

### Example B: Beginner with strong takeaway fault

| swing_id | phase | camera_view | quality_ok | primary_issue_tag | secondary_issue_tags | severity | confidence | cue_id | drill_id | pro_note |
|---|---|---|---:|---|---|---:|---:|---|---|---|
| beg_143_nn | Takeaway | side | 1 | takeaway.lead_arm_angle_low | takeaway.x_factor_low | 3 | 5 | CUE_ONE_PIECE_01 | DRILL_CROSS_ARM_01 | Arm collapses early; keep lead arm longer. |

### Example C: Low-quality frame, suppress feedback

| swing_id | phase | camera_view | quality_ok | primary_issue_tag | secondary_issue_tags | severity | confidence | cue_id | drill_id | pro_note |
|---|---|---|---:|---|---|---:|---:|---|---|---|
| clip_088_nn | Impact | front | 0 | unknown |  | 0 | 1 | NONE | NONE | Club and lead wrist occluded at impact. |

### CSV snippet example

```
swing_id,phase,camera_view,quality_ok,primary_issue_tag,secondary_issue_tags,severity,confidence,cue_id,drill_id,pro_note
pro_021_nn,Mid-backswing,front,1,mid_backswing.x_factor_low,mid_backswing.shoulder_rotation_low,1,4,CUE_COIL_01,DRILL_BAND_TURN,Slightly more torso turn needed.
beg_143_nn,Takeaway,side,1,takeaway.lead_arm_angle_low,takeaway.x_factor_low,3,5,CUE_ONE_PIECE_01,DRILL_CROSS_ARM_01,Arm collapses early; keep lead arm longer.
clip_088_nn,Impact,front,0,unknown,,0,1,NONE,NONE,Club and lead wrist occluded at impact.
```

## 8. Quality Control Checklist

Run this every labeling batch:

1. Agreement check
- 10 to 15 percent of samples labeled by two pros
- track disagreement rate by phase

2. Tag frequency check
- no critical tag should have fewer than 20 examples

3. View balance check
- front and side should be close for each skill level

4. Confidence check
- review all labels with confidence <= 2

5. Missingness check
- no empty primary_issue_tag when quality_ok = 1

## 9. Integration With Current Pipeline

Current system remains unchanged for scoring.

Use this new dataset to improve feedback mapping and thresholds by:
- identifying missing metric-direction mappings
- tuning phase-specific gates
- creating stronger fallback feedback templates

## 10. Practical Rollout Plan

Week 1:
- finalize taxonomy and label form
- label 30 swings pilot

Week 2:
- review inter-rater disagreement
- refine tags and instructions

Week 3-4:
- label full batch to target counts
- run feedback audit and update feedback map

---

If you want, next step is to add a starter template file at data/feedback_labels_template.csv and a small validator script to check label quality before training.
