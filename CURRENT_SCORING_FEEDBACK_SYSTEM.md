# Current Scoring and Feedback System (XGBoost)

This document explains the current production scoring and feedback logic in plain language and technical detail.

## 1. One-minute non-technical explanation

The system evaluates a golf swing in 8 phases:

1. Address
2. Takeaway
3. Mid-backswing
4. Top
5. Mid-downswing
6. Impact
7. Follow-through
8. Finish

For each phase, the system measures body mechanics (rotation, posture, arm and wrist positions, stability) and predicts a score out of 100 using a trained XGBoost model.

Then it compares the swing against benchmark values learned from good swings. If a phase is weak, it outputs short coaching feedback for the most important issues.

Final score = average of the 8 phase scores.

## 2. Current runtime flow in the pipeline

The pipeline currently runs these steps:

1. Clean video
2. Extract poses and frame metrics
3. Detect 8 phases (rule-based or neural-network)
4. Score phases (XGBoost backend by default)
5. Export key frames and reports

Important:

- The separate biomechanics evaluation step that used to print `STEP 4b/5` and `BIOMECHANICS SUMMARY` has been removed from the normal flow.
- Scoring and feedback are now centered on the XGBoost scorer path.

## 3. Inputs used by the scorer

For each swing id (`video_nn` or `video_rb`), the scorer reads:

- Metrics CSV: `data/metrics/{video}_cleaned_metrics.csv`
- Phase CSV: `data/keyframes/{swing_id}/{video}_cleaned_8phases.csv`
- Trained model bundle: `models/phase_scorer.pkl`

From these, it builds phase features:

- Key-frame biomechanics values per phase
- Phase duration per phase

## 4. Model design

- One XGBoost regressor per phase (8 models total)
- Output: one 0-100 score per phase
- Overall score: arithmetic average of all predicted phase scores

### Scoring vs Feedback (important distinction)

| Part | Purpose | Uses what | Output |
|------|---------|-----------|--------|
| Scoring | Predict numeric quality per phase | Wider phase feature set (keyframe biomechanics + durations + engineered features) | 8 phase scores + overall score |
| Feedback | Explain what to fix in low phases | Mapped checkpoint subset + benchmark deviation + local model contribution | 1-2 coaching messages per weak phase |

Key point:

- Scoring and feedback share the same model bundle, but feedback is a filtered explanation layer on top of scoring.

## 5. Feedback decision rules (current constants)

Feedback and visual annotation use these gates:

- `FEEDBACK_SCORE_THRESHOLD = 85.0`
- `FEEDBACK_MAX_ITEMS = 2`
- `MIN_LOCAL_CONTRIB = 0.02`
- `MIN_NORM_DEVIATION = 0.50`

Interpretation:

1. If phase score is >= 85, no corrective feedback is produced.
2. If phase score is < 85, candidate issues are ranked by:
   - normalized deviation from benchmark
   - local model contribution for that feature
3. Only meaningful issues pass the minimum gates.
4. Top 1-2 mapped coaching messages are returned.

## 6. How benchmarks are built

Benchmarks are learned per phase-feature and stored as robust statistics:

- median
- q1 (25th percentile)
- q3 (75th percentile)
- iqr = q3 - q1

Selection priority during training:

1. Use pro-only data if enough pro samples exist
2. Else use swings with phase score >= 75
3. Else fallback to top half of available samples

So benchmark is not one fixed number. It is a learned "good range" per checkpoint.

## 7. What exactly is output after scoring

Primary output files:

- `data/metrics/{video}_scores.csv`
  - per-phase score + overall row
- `data/metrics/{video}_feedback_detailed.csv`
  - one row per generated feedback message
- `outputs/visual_feedback/{swing_id}/{phase}_annotated.jpg`
  - generated when weak phases have mappable deviations

## 8. Phase checkpoint list (what we check for feedback)

Below is the current checkpoint map used for phase-level feedback messages.

### Address

- spine angle
- spine lateral tilt 
- stance width ratio (too narrow / too wide)
- wrist angle (hands too far / too close)
- lead knee flex (too straight)
- trail knee flex (too straight)

### Takeaway

- x-factor (hip-shoulder separation too low)
- shoulder rotation (too low)
- wrist angle (too flat)
- wrist hinge (too low)
- lead arm angle (too bent)
- head displacement (too high)

### Mid-backswing

- x-factor (too low)
- x-factor 3d (too low)
- shoulder rotation (too low)
- lead arm angle (too bent)
- wrist angle (limited hinge)
- wrist hinge (too low)
- head displacement (too high)
- trail knee flex (too straight)

### Top

- shoulder rotation (too low)
- shoulder rotation 3d (too low)
- x-factor (too low)
- lead arm angle (too bent)
- wrist angle (limited hinge)
- wrist hinge (too low)
- spine angle (too high)
- head displacement (too high)

### Mid-downswing

- lag angle (lag too low / early release)
- hip rotation (too low)
- hip rotation 3d (too low)
- x-factor (too low)
- shoulder rotation (too high relative to hips)
- head displacement (too high)

### Impact

- hip rotation (too low)
- hip rotation 3d (too low)
- lead arm angle (too bent)
- x-factor (too high relative to desired impact timing)
- lag angle (too high means lag already lost at impact)
- head displacement (too high)

### Follow-through

- shoulder rotation (too low)
- lead arm angle (too bent)
- hip rotation (too low)
- hip rotation 3d (too low)
- spine angle (too high)

### Finish

- shoulder rotation (too low)
- hip rotation (too low)
- hip rotation 3d (too low)
- lead knee flex (too high, weak weight transfer)
- spine angle (too high)
- head displacement (too high)

## 9. Full numeric benchmark tables (current snapshot)

For exact median/q1/q3/iqr values currently exported in this repo, see:

- `outputs/current_benchmarks_feedback_rules.csv`
  - checkpoints that currently map to feedback messages
- `outputs/current_benchmarks_all_features.csv`
  - all phase-feature benchmark stats in the model space

## 10. How visual feedback highlighting works

For low-scoring phases, top deviated metrics are converted to body segments using a metric-to-joint map, then the keyframe skeleton is drawn:

- gray = base skeleton
- red = segments tied to deviated metrics

This gives image-level "where to fix" guidance.

## 11. Known behavior notes

1. Text feedback and image highlights are both gated by score and deviation thresholds.
2. A low phase can still show limited text if few deviations pass gates or mapping.
3. Benchmark quality depends on training data quality, camera angle consistency, and pose extraction quality.

## 12. Commands (current)

Run pipeline with XGBoost scoring:

```bash
python pipeline.py data/raw_videos/your_video.mp4 --method neural-network --scoring-backend xgboost
```

Train scorer bundle:

```bash
python train_phase_scorer.py --prepare --train
```

Single swing prediction + optional visual annotation:

```bash
python train_phase_scorer.py --predict your_video_nn --annotate
```

## 13. Teacher-friendly summary sentence

The system grades each swing phase separately, compares each phase against good-swing benchmark ranges, and gives short, prioritized feedback based on the most important biomechanical deviations.
