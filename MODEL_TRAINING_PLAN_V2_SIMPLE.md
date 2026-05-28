# Golf Swing Scorer - Neural Network Training Plan (Simplified)

**Goal:** Collect 150 labeled swings → train multi-output phase scorer → achieve 15pt pro/beginner discrimination

**Timeline:** 3 weeks (7 days expert labeling + 14 days model development)

---

## PART 1: EXPERT ANNOTATION GUIDE (For Your Golf Pro)

### What Your Pro Needs to Do

Your pro will rate **150 golf swings** — about **2-3 per video** using this format.

**Time estimate:** 10-15 swings per hour = 10-15 hours total

### Annotation Format: CSV Template

Create a Google Sheet or Excel file with this structure. Send **this exact template** to your pro:

```
swing_id,camera_angle,skill_level,confidence,address_score,takeaway_score,mid_backswing_score,top_score,mid_downswing_score,impact_score,follow_through_score,finish_score,feedback
B90_swing_001,front,pro,5,85,82,88,90,85,92,88,86,Good tempo and lag management
akh_swing_001,front,beginner,4,65,58,62,68,55,48,58,72,Slow takeaway; early wrist break at top
me_swing_001,front,intermediate,4,72,70,75,78,72,65,68,75,Good address but loses lag in downswing
```

### Column Definitions

| Column | Values | Example | Notes |
|--------|--------|---------|-------|
| `swing_id` | Video filename (no extension) | `B90_swing_001` | Use exact file name from data/extracted_poses/ |
| `camera_angle` | `front` \| `side` \| `45deg` | `front` | Which direction is camera pointing |
| `skill_level` | `pro` \| `advanced` \| `intermediate` \| `beginner` | `pro` | Based on handicap/skill |
| `confidence` | 1-5 | 4 | How confident in these ratings (1=guess, 5=very sure) |
| `address_score` | 0-100 | 85 | See scoring guide below |
| `takeaway_score` | 0-100 | 82 | " |
| `mid_backswing_score` | 0-100 | 88 | " |
| `top_score` | 0-100 | 90 | " |
| `mid_downswing_score` | 0-100 | 85 | " |
| `impact_score` | 0-100 | 92 | " |
| `follow_through_score` | 0-100 | 88 | " |
| `finish_score` | 0-100 | 86 | " |
| `feedback` | Short text (1-2 sentences) | "Good lag at impact; slight early finish" | What needs fixing (if score <70) |

---

## PHASE SCORING GUIDE FOR PRO

For **each swing**, rate each of these 8 phases on a **0-100 scale**:

### 1. ADDRESS (Starting position)
**Score 80-100 if:**
- Feet shoulder-width apart, knees slightly bent
- Spine angle good (not too vertical, not too bent forward)
- Grip pressure neutral
- Head still, eyes on ball

**Score 50-70 if:** Stance too narrow, knees locked, or posture compromised

**Score <50 if:** Major postural issues

---

### 2. TAKEAWAY (First 3 feet of backswing)
**Score 80-100 if:**
- Club moves straight back low (no inside/outside)
- Wrists stay flat (no early hinge)
- Body and arms move together (synchronized)
- Head stays still

**Score 50-70 if:** Wrists hinge early or club path slightly off

**Score <50 if:** Club moves way inside or wrists break immediately

---

### 3. MID-BACKSWING (Club parallel to ground, going back)
**Score 80-100 if:**
- Club on plane (shaft line matches spine angle)
- Hips rotated ~45°, shoulders ~90°
- Lead arm straight, trail arm bent ~90°
- No sway or reverse pivot

**Score 50-70 if:** Club slightly off-plane or hip/shoulder ratio poor

**Score <50 if:** Severe off-plane swing or excessive swaying

---

### 4. TOP (Club at highest point)
**Score 80-100 if:**
- Wrists set (lag angle ~90°)
- Hips rotated ~45°, shoulders ~90°
- Trail elbow stays close to body
- Shaft on-plane (not across the line)

**Score 50-70 if:** Wrists overcooked (lag >100°) or shaft slightly across line

**Score <50 if:** Severe lag loss (<70°) or shaft way across the line

---

### 5. MID-DOWNSWING (Club parallel to ground, coming down)
**Score 80-100 if:**
- Club on plane
- Lead hip clearing back (open 20-30°)
- Lag maintained (wrist angle still 60-80°)
- Trail heel down or lifting naturally

**Score 50-70 if:** Lag releasing early or lead hip not clearing enough

**Score <50 if:** Major lag loss or severe loss of plane

---

### 6. IMPACT (Club hits ball)
**Score 80-100 if:**
- X-factor unwind (lag angle 10-20° at impact) ← **MOST CRITICAL**
- Lead arm extended but not hyperextended
- Head behind ball, eyes on strike
- Square or slightly closed clubface

**Score 50-70 if:** Lag released too early (>30°) or arm not extended

**Score <50 if:** Complete lag loss (<5°) or early release

---

### 7. FOLLOW-THROUGH (Right after impact)
**Score 80-100 if:**
- Club continues upward on plane
- Body rotating (hips/shoulders continuing turn)
- Trail arm folding naturally
- Head still tracking target

**Score 50-70 if:** Club slightly off-plane or follow-through cramped

**Score <50 if:** Severe deceleration or blocked follow-through

---

### 8. FINISH (End of swing)
**Score 80-100 if:**
- Weight fully transferred to lead foot
- Club wrapped around back of neck/shoulder
- Hips and shoulders fully rotated (open ~90°)
- Balanced, no stumbling

**Score 50-70 if:** Finish slightly cramped or weight not fully transferred

**Score <50 if:** Off-balance or severe finish fault

---

## HOW TO DELIVER THIS TO YOUR PRO

### Email Template

Subject: **Need Your Help: Rate 150 Golf Swings (10 hours work)**

---

Hi [Pro Name],

I'm building an AI golf swing scorer and need an expert to rate 150 swings across our video library. This will train a neural network to understand what separates good form from bad.

**What I need:**
1. Rate each swing's 8 phases (0-100 scale)
2. Fill in attached CSV template
3. Add 1-2 sentence feedback for phases scoring <70

**Time:** ~10-15 hours total (2-3 swings/hour)

**Format:** Use the attached guide + CSV template. I'll send you the first batch of videos via folder link.

**Timeline:** Can you complete in 1-2 weeks?

**Template:** See attached `pro_annotation_template.csv`

**Scoring Guide:** See attached `PHASE_SCORING_GUIDE.md`

Let me know if you have questions!

---

### CSV Template File to Send

Save this and send to pro:

```csv
swing_id,camera_angle,skill_level,confidence,address_score,takeaway_score,mid_backswing_score,top_score,mid_downswing_score,impact_score,follow_through_score,finish_score,feedback
[PRO: FILL IN BELOW]
,,,,,,,,,,,,
,,,,,,,,,,,,
```

---

## PART 2: PARALLEL WORK - FIX IMPACT SCORING BUG

**While waiting for pro labels, fix the scoring issue:**

Your May 20 notes show all impact scores are uniformly 85.0 → debug this now.

**Check in `src/biomechanics/phase_scorer.py` line ~474-492:**
- Verify x_factor_unwind calculation is correct
- Check if all 5 impact components are being evaluated
- Add debug prints to see which component is dominating

**Expected flow:**
1. Extract x_factor from top vs impact
2. Score based on lag retained (0-20°) = high score
3. Weight with arm extension + lead wrist angle
4. Return result in 50-95 range (not flat 85)

---

## PART 3: MODEL TRAINING (Once labels arrive)

**Days 1-7:** Collect expert labels (PRO TASK)

**Days 8-10:** Prepare training dataset
- Combine expert labels + pose extractions + phase boundaries
- Output: `datasets/training_150_phases.csv`

**Days 11-16:** Train multi-output phase scorer
- Use pose sequences (LSTM) or CNN features
- Predict 8 phase scores (0-100) + feedback
- Minimize MAE per phase

**Days 17-21:** Evaluate & integrate
- Cross-validation on test split
- Update pipeline.py to use neural scorer
- Deploy to production

---

## Success Criteria

- [ ] Pro delivers 150 labeled swings with >85% inter-rater agreement
- [ ] MAE per phase < 8 points
- [ ] Pro vs Beginner overall scores differ by ≥15 points
- [ ] Impact scoring bug fixed (scores vary by swing, not flat 85)
- [ ] Model generalizes to unseen videos

---

## Notes

**If pro can't label all 150:** Start with 50 swings (balanced: 30 pro, 20 beginner/intermediate). Test model performance. If discriminator ≥5pts improvement, expand to 150.

**Video folder structure for pro:**
```
videos_to_label/
  B90_swing_001/
    frames/ (keyframe images for each phase)
    video.mp4 (full swing clip)
  akh_swing_001/
    ...
```

Provide folder link so pro can review video + keyframes while filling in scores.

