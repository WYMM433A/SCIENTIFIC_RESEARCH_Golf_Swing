# Scoring Audit Checklist — No Code Changes Required

**Goal:** Diagnose whether your current rule-based scorer fails due to **metric choice**, **weighting**, or **threshold calibration**.

**Time to complete:** 2–3 hours  
**What you'll need:** Python terminal, pandas, matplotlib  
**Output:** Clear answer on what's broken and where to fix it

---

## Part 1: Visual Inspection — Is the Problem Obvious?

### Check 1.1: Run the Scorer on Known Good/Bad Swings

```python
# Terminal: Load 3 known PRO swings and 3 known BEGINNER swings

from src.biomechanics.phase_scorer import PhaseScorer
import pandas as pd

# Load a pro swing
pro_poses = pd.read_csv('data/extracted_poses/golf_swing_001_cleaned_poses.csv')  # Known pro
beginner_poses = pd.read_csv('data/extracted_poses/0_cleaned_poses.csv')  # Known beginner

scorer = PhaseScorer()

# Score both
pro_result = scorer.score_swing(pro_poses)
beginner_result = scorer.score_swing(beginner_poses)

print("PRO SWING SCORES:")
for phase, score in pro_result['phase_scores'].items():
    print(f"  {phase:20s}: {score:.1f}")
print(f"  OVERALL: {pro_result['overall_score']:.1f}")

print("\nBEGINNER SWING SCORES:")
for phase, score in beginner_result['phase_scores'].items():
    print(f"  {phase:20s}: {score:.1f}")
print(f"  OVERALL: {beginner_result['overall_score']:.1f}")

print(f"\nDifference: {abs(pro_result['overall_score'] - beginner_result['overall_score']):.1f}")
```

**DIAGNOSIS:**
- Difference > 15 pts? → Weighting or thresholds might be okay
- Difference < 5 pts? → **STOP** — Major problem here, continue to Check 1.2
- Both scores 80–95? → **RED FLAG** — Metrics or thresholds too loose

### Check 1.2: Inspect Raw Metric Values

Do pro and beginner even have different metric values, or are the metrics themselves broken?

```python
from src.biomechanics.angles import AngleCalculator
import pandas as pd

calc = AngleCalculator()

# Load pro and beginner swings
pro_poses = pd.read_csv('data/extracted_poses/golf_swing_001_cleaned_poses.csv')
beginner_poses = pd.read_csv('data/extracted_poses/0_cleaned_poses.csv')

# Extract same metric from both
def get_sample_metrics(poses_df, label):
    metrics = {}
    # Sample 5 frames from middle of swing (takeaway → top)
    for frame_idx in poses_df['frame'].iloc[20:30]:
        row = poses_df[poses_df['frame'] == frame_idx].iloc[0]
        
        metrics[frame_idx] = {
            'spine_angle': calc.get_spine_angle(row),
            'lead_arm_angle': calc.get_lead_arm_angle(row),
            'trail_elbow_angle': calc.get_trail_elbow_angle(row),
            'lead_knee_flex': calc.get_lead_knee_flex(row),
        }
    return metrics

pro_metrics = get_sample_metrics(pro_poses, 'PRO')
beginner_metrics = get_sample_metrics(beginner_poses, 'BEGINNER')

print("SPINE ANGLE (frames 20-30):")
print(f"  Pro:      {[m['spine_angle'] for m in pro_metrics.values()]}")
print(f"  Beginner: {[m['spine_angle'] for m in beginner_metrics.values()]}")

print("\nLEAD ARM ANGLE (frames 20-30):")
print(f"  Pro:      {[m['lead_arm_angle'] for m in pro_metrics.values()]}")
print(f"  Beginner: {[m['lead_arm_angle'] for m in beginner_metrics.values()]}")

print("\nTRAIL ELBOW ANGLE (frames 20-30):")
print(f"  Pro:      {[m['trail_elbow_angle'] for m in pro_metrics.values()]}")
print(f"  Beginner: {[m['trail_elbow_angle'] for m in beginner_metrics.values()]}")
```

**DIAGNOSIS:**
- Pro and beginner values are clearly different? → **Metrics are okay**, problem is weighting or thresholds
- Pro and beginner values are nearly identical? → **PROBLEM: Metrics are broken or view-dependent**
- One metric is identical, others differ? → That metric is useless, remove it

---

## Part 2: Threshold Analysis — Are Cutoffs Too Loose?

### Check 2.1: Score Distribution Across All Videos

Do beginner and pro form **two distinct populations** or do they **overlap completely**?

```python
import pandas as pd
import os
from src.biomechanics.phase_scorer import PhaseScorer

scorer = PhaseScorer()
results = []

# Score all available swings
poses_dir = 'data/extracted_poses'
for pose_file in os.listdir(poses_dir)[:100]:  # First 100 to start
    if not pose_file.endswith('.csv'):
        continue
    
    swing_id = pose_file.replace('_cleaned_poses.csv', '')
    try:
        poses = pd.read_csv(os.path.join(poses_dir, pose_file))
        score_result = scorer.score_swing(poses)
        
        results.append({
            'swing_id': swing_id,
            'overall_score': score_result['overall_score'],
        })
    except Exception as e:
        print(f"Error processing {swing_id}: {e}")

score_df = pd.DataFrame(results)

# Show distribution
print("SCORE DISTRIBUTION:")
print(score_df['overall_score'].describe())
print(f"\nMin: {score_df['overall_score'].min():.1f}")
print(f"Max: {score_df['overall_score'].max():.1f}")
print(f"Mean: {score_df['overall_score'].mean():.1f}")
print(f"Std:  {score_df['overall_score'].std():.1f}")

# Plot histogram (text-based)
import numpy as np
bins = np.arange(0, 105, 10)
hist, _ = np.histogram(score_df['overall_score'], bins=bins)
for i, count in enumerate(hist):
    print(f"  {bins[i]:3.0f}-{bins[i+1]:3.0f}: {'█' * count} ({count})")
```

**DIAGNOSIS:**
- Most scores 80–100? → **THRESHOLDS ARE TOO LOOSE** (accepting poor form)
- Scores 20–40 common? → **WEIGHTING IS WRONG** (penalizing too much)
- Spread across 0–100 with two peaks? → **OKAY**, check separation between pro/beginner clusters

### Check 2.2: Per-Phase Threshold Reality Check

Are thresholds in `scoring_config.py` even reasonable?

```python
from src.biomechanics.scoring_config import PHASE_THRESHOLDS

# Look at what thresholds actually are
for phase, thresholds in PHASE_THRESHOLDS.items():
    print(f"\n{phase.upper()}:")
    for metric, thresh in thresholds.items():
        print(f"  {metric:30s}: {thresh}")
        
        # Sanity check: are these realistic ranges?
        # E.g., "if spine_angle > 40, excellent" — is that actually achievable?
```

**Red flags in thresholds:**
- Excellent range too wide (e.g., "80–100 is excellent" → almost everything is excellent)
- All metrics have same weight (e.g., all 0.125 for 8 phases) → doesn't reflect reality
- No differentiation between views (front vs side use same thresholds) → 2D vs 3D issue
- Thresholds don't match biomechanics knowledge (e.g., lag angle 80–100° for excellent when 150–180° is better)

---

## Part 3: Metric Importance Analysis — Which Metrics Actually Matter?

### Check 3.1: Disable Each Metric One-by-One

Does removing one metric change the score much, or does it stay the same?

```python
from src.biomechanics.phase_scorer import PhaseScorer
import pandas as pd

scorer = PhaseScorer()
test_swing = pd.read_csv('data/extracted_poses/golf_swing_001_cleaned_poses.csv')

# Baseline score
baseline = scorer.score_swing(test_swing)
print(f"BASELINE SCORE: {baseline['overall_score']:.1f}")

# Now disable each metric by setting its weight to 0
# (You'll need to modify scoring_config.py temporarily for this)

# Save original config
import src.biomechanics.scoring_config as config
original_weights = config.PHASE_WEIGHTS.copy()

# Test each metric
for metric in config.PHASE_THRESHOLDS['address'].keys():
    # Disable this metric for all phases
    for phase in config.PHASE_WEIGHTS:
        config.PHASE_WEIGHTS[phase][metric] = 0
    
    # Re-score
    modified = scorer.score_swing(test_swing)
    difference = abs(baseline['overall_score'] - modified['overall_score'])
    
    print(f"Remove {metric:20s}: Score changes by {difference:.1f} pts")
    
    # Restore
    config.PHASE_WEIGHTS = original_weights.copy()

print(f"\n⚠️ If most metrics cause <1pt change, they're not contributing → REMOVE THEM")
```

**DIAGNOSIS:**
- Removing a metric changes score by <1 pt → **Metric is noise, remove it**
- Removing a metric changes score by 2–5 pts → **Metric has some signal**
- Removing a metric changes score by >10 pts → **Metric carries too much weight, rebalance**

### Check 3.2: Correlation Between Metrics and Skill Level

Do metrics actually correlate with being "pro" vs "beginner"?

```python
import pandas as pd
import os
import numpy as np
from src.biomechanics.angles import AngleCalculator

calc = AngleCalculator()

# Build dataset: (metric_value, skill_level)
# You need to know which swings are pro vs beginner (from naming or metadata)

data = []

# Assume naming convention: pro swings are in certain folder or have certain pattern
pro_swings = [f for f in os.listdir('data/extracted_poses') if 'golf_swing' in f]  # Example
beginner_swings = [f for f in os.listdir('data/extracted_poses') if not 'golf_swing' in f][:30]

for swing_file in pro_swings[:20]:
    try:
        poses = pd.read_csv(f'data/extracted_poses/{swing_file}')
        # Sample a metric
        spine_angles = []
        for idx, row in poses.iloc[20:30].iterrows():
            spine_angles.append(calc.get_spine_angle(row))
        
        data.append({
            'skill': 'pro',
            'spine_angle_mean': np.mean(spine_angles),
            'spine_angle_std': np.std(spine_angles)
        })
    except:
        pass

for swing_file in beginner_swings[:20]:
    try:
        poses = pd.read_csv(f'data/extracted_poses/{swing_file}')
        spine_angles = []
        for idx, row in poses.iloc[20:30].iterrows():
            spine_angles.append(calc.get_spine_angle(row))
        
        data.append({
            'skill': 'beginner',
            'spine_angle_mean': np.mean(spine_angles),
            'spine_angle_std': np.std(spine_angles)
        })
    except:
        pass

df = pd.DataFrame(data)

# Check for separation
print("SPINE ANGLE MEAN BY SKILL:")
print(df.groupby('skill')['spine_angle_mean'].describe())

print("\nSPINE ANGLE STD BY SKILL:")
print(df.groupby('skill')['spine_angle_std'].describe())

# Calculate effect size (Cohen's d)
pro_mean = df[df['skill'] == 'pro']['spine_angle_mean'].mean()
pro_std = df[df['skill'] == 'pro']['spine_angle_mean'].std()
beginner_mean = df[df['skill'] == 'beginner']['spine_angle_mean'].mean()
beginner_std = df[df['skill'] == 'beginner']['spine_angle_mean'].std()

cohens_d = (pro_mean - beginner_mean) / np.sqrt((pro_std**2 + beginner_std**2) / 2)
print(f"\nCohen's d (effect size): {cohens_d:.2f}")
print(f"  < 0.2: Negligible effect (metric is useless)")
print(f"  0.2-0.5: Small effect (metric has weak signal)")
print(f"  0.5-0.8: Medium effect (metric is okay)")
print(f"  > 0.8: Large effect (metric is strong)")
```

**DIAGNOSIS:**
- Cohen's d < 0.2 for metric → **Metric doesn't discriminate, remove it**
- Cohen's d 0.2–0.5 → **Metric has signal but noisy**
- Cohen's d > 0.8 → **Metric works well**

---

## Part 4: Weighting Analysis — Are Phases Weighted Fairly?

### Check 4.1: Phase Importance Distribution

Are all phases weighted equally, or do some phases dominate?

```python
from src.biomechanics.scoring_config import PHASE_WEIGHTS

# Print current weights
print("CURRENT PHASE WEIGHTS:")
for phase, metrics in PHASE_WEIGHTS.items():
    total_weight = sum(metrics.values())
    print(f"  {phase:20s}: Total = {total_weight:.3f}")

# Check metric weights within each phase
print("\nWEIGHTS WITHIN EACH PHASE:")
for phase, metrics in PHASE_WEIGHTS.items():
    print(f"  {phase}:")
    for metric, weight in sorted(metrics.items(), key=lambda x: x[1], reverse=True):
        pct = (weight / sum(metrics.values())) * 100 if sum(metrics.values()) > 0 else 0
        print(f"    {metric:30s}: {weight:.3f} ({pct:.1f}%)")
```

**Red flags:**
- All phases have exactly equal weight → Wrong, impact should count more than address
- All metrics within a phase have equal weight → Wrong, some metrics are more important
- Mid-downswing/impact weighted same as address/takeaway → Wrong, focus should be on strike phases

### Check 4.2: What Should Weights Actually Be?

Based on coaching, which phases matter most?

```
# Golf coaching priority (intuitive):
Address:         10% (setup, important but not impact-determining)
Takeaway:        8%  (initiates sequence)
Mid-backswing:   5%  (less critical than top)
Top:            12%  (coil/loading, power prep)
Mid-downswing:  20%  (CRITICAL — sequencing and acceleration)
Impact:         35%  (MOST CRITICAL — where power and accuracy are determined)
Follow-through:  5%  (less important, but shows control)
Finish:          5%  (shows balance, less critical)

# Current weights (check your config):
# ?????

# If your weights don't roughly match above, adjust them
```

---

## Part 5: View-Awareness Check — Is the Camera View the Problem?

### Check 5.1: Does Scoring Differ by View?

Front-view and side-view cameras see different things. Are you treating them the same?

```python
import pandas as pd
from src.biomechanics.phase_scorer import PhaseScorer

# Score same swing from side view
side_view = pd.read_csv('data/extracted_poses/golf_swing_001_cleaned_poses.csv')  # Assume side view
scorer = PhaseScorer()
side_score = scorer.score_swing(side_view)

print(f"SIDE VIEW SCORE: {side_score['overall_score']:.1f}")

# If you have front-view extracted poses:
# front_view = pd.read_csv('data/extracted_poses/front_golf_swing_001_cleaned_poses.csv')
# front_score = scorer.score_swing(front_view)
# print(f"FRONT VIEW SCORE: {front_score['overall_score']:.1f}")
```

**Key insight:**
- If both views get ~90 score for same swing → Scorer is view-naive (may be 2D projection issue)
- If front/side differ by 30+ pts → Scorer can't generalize across views (major problem)

### Check 5.2: Which Metrics Are View-Dependent?

Some metrics only work from certain camera angles.

```
SIDE VIEW RELIABILITY:
✓ Spine angle — Sagittal plane, works well from side
✓ Lead arm angle — Mostly planar from side
✓ Trail elbow — Can see from side
? Hip rotation — Partially visible from side (depth hidden)
✗ Shoulder rotation — Depth hidden from side
✗ X-factor — Both hips and shoulders need depth visibility
✗ Club path — Club not tracked

FRONT VIEW RELIABILITY:
✓ Head displacement — Can see side-to-side movement
✓ Stance width — Both feet visible
? Lead/trail knee flex — Some depth ambiguity
✗ Spine angle — Forward tilt harder to see from front
✗ Lead arm angle — Ambiguous from front
✗ Hip rotation — Mostly hidden from front

# Current problem: are you using shoulder_rotation from side view?
# That's the 2D projection problem the external advisors mentioned
```

---

## Part 6: Quick Root Cause Analysis

Answer these questions in order:

### Question 1: Are Raw Metrics Different for Pro vs Beginner?

```python
# From Check 1.2 above:
# Yes  → Go to Question 2
# No   → PROBLEM: Metrics are broken or view-dependent
#         FIX: Remove problematic metrics (shoulder_rotation, hip_rotation, x_factor)
```

### Question 2: Do Final Scores Separate Pro from Beginner?

```python
# From Check 1.1 above:
# Difference > 15 pts  → Go to Question 3
# Difference < 5 pts   → PROBLEM: Either weighting or thresholds are broken
#                        FIX: Check Question 3
```

### Question 3: Are Thresholds Too Loose?

```python
# From Check 2.1 above:
# Most scores 80-100  → PROBLEM: Thresholds accept too much
#                       FIX: Tighten excellent/good cutoffs
# 
# Wide spread 0-100  → Go to Question 4
```

### Question 4: Are Phase Weights Reasonable?

```python
# From Check 4.1 above:
# Impact gets 35%+  → Weighting looks reasonable
#                     FIX: Revisit metric choice (Part 3)
#
# All phases equal  → PROBLEM: Weighting is broken
#                     FIX: Increase impact/mid-downswing weight
```

---

## Diagnosis Summary Template

After running all checks, fill in:

```
DIAGNOSIS REPORT
================

1. Raw metrics separate pro/beginner?
   YES / NO / PARTIALLY
   
   Evidence: [from Check 1.2]

2. Final scores separate pro/beginner?
   YES (>15 pts) / SMALL (<5 pts) / NO
   
   Pro average: ___
   Beginner average: ___
   Difference: ___

3. Thresholds too loose?
   YES / NO
   
   Evidence: [histogram from Check 2.1]

4. Phase weights reasonable?
   YES / NO
   
   Current weights:
   - Address: __%
   - Mid-downswing: __%
   - Impact: __%

5. View-dependent problem?
   YES / NO
   
   Problem metrics: [list]

ROOT CAUSE:
===========

[ ] Metric choice — Some metrics don't work from your camera view
    → FIX: Remove shoulder_rotation, hip_rotation, x_factor
    → ADD: trajectory smoothness, velocity-based features

[ ] Weighting — Phases not weighted by importance
    → FIX: Increase impact/mid-downswing to 50%+ total
    → Decrease address/takeaway to 10% total

[ ] Thresholds — Cutoffs too loose, accepting poor form
    → FIX: Use actual pro/beginner data to set realistic cutoffs
    → Tighten "excellent" range

[ ] Multiple issues — Some combination of above
    → FIX: Start with whichever is easiest to validate

NEXT STEP:
==========
[ ] If metric choice → Remove 5 worst metrics
[ ] If weighting → Rebalance to 35% impact, 20% mid-downswing
[ ] If thresholds → Retune using pro/beginner distributions
```

---

## Quick Test After Each Fix

After you make a change, re-run Check 1.1:

```python
# Test on 3 known pro + 3 known beginner swings
# GOAL: Difference should increase from current <5 to >20

pro_score = scorer.score_swing(pro_poses)
beginner_score = scorer.score_swing(beginner_poses)
difference = abs(pro_score - beginner_score)

print(f"Difference after fix: {difference:.1f}")
if difference > 15:
    print("✓ PROGRESS!")
elif difference > 10:
    print("⚠ Some progress, keep going")
else:
    print("✗ No improvement, try different fix")
```

---

## Expected Improvement Path

| Change | Expected Impact |
|--------|-----------------|
| Remove 10 low-value metrics | +5–10 pt separation |
| Reweight phases (35% impact) | +10–15 pt separation |
| Tighten thresholds 10% | +5–8 pt separation |
| All three combined | +20–30 pt separation |

**Target:** Reach >20 pt difference between pro and beginner, consistently.

