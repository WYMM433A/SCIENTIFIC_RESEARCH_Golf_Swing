# Pro Annotation Guide - 100 ds_videos Swings

**Goal:** You will rate 100 golf swings (8 phases each) using a structured format.


**What You'll Use:**
- Video player (to watch swings frame-by-frame)
- Keyframe images (visual reference for each phase)
- CSV template (to record your ratings)
- Scoring guide (phase criteria)

---

## SETUP (5 minutes)

### What You'll Receive

A folder organized like this:
```
ds_videos_annotation_pack/
├── README.txt (this file)
├── PHASE_SCORING_GUIDE.md (scoring criteria for each phase)
├── pro_annotation_template.csv (WHERE YOU FILL IN YOUR RATINGS)
├── videos/
│   ├── B12_F1_I.mov
│   ├── B12_F2_I.mov
│   └── ... (100 videos total)
└── keyframes/
    ├── B12_F1_I_nn/
    │   ├── B12_F1_I_cleaned_Address.jpg
    │   ├── B12_F1_I_cleaned_Takeaway.jpg
    │   ├── B12_F1_I_cleaned_Mid-backswing.jpg
    │   ├── B12_F1_I_cleaned_Top.jpg
    │   ├── B12_F1_I_cleaned_Mid-downswing.jpg
    │   ├── B12_F1_I_cleaned_Impact.jpg
    │   ├── B12_F1_I_cleaned_Follow-through.jpg
    │   ├── B12_F1_I_cleaned_Finish.jpg
    │   └── B12_F1_I_cleaned_8phases.csv
    ├── B12_F2_I_nn/
    │   └── ... (same structure)
    └── ... (100 folders total)
```

---

## STEP 1: Install Video Player (5 minutes)

### Option A: VLC Media Player (Recommended - Free)

**Download:** https://www.videolan.org/vlc/

**Why VLC?**
- ✅ Free
- ✅ Frame-by-frame playback (arrow keys)
- ✅ Slow-motion playback
- ✅ Works on Windows/Mac/Linux
- ✅ Easy to use

**How to use for frame-by-frame:**
1. Open video in VLC
2. Press **E** key to go frame-by-frame (forward)
3. Press **Shift+E** to go backward one frame
4. Press **Space** to pause/play
5. Press **[** to slow down, **]** to speed up

### Option B: Windows Media Player

Built-in on Windows, but slower controls.

### Option C: ffmpeg (Command Line - Advanced)

For extracting specific frames, but VLC is easier.

---

## STEP 2: Open The Scoring Template

### File: `pro_annotation_template.csv`

**Open with:** Excel or Google Sheets

**Structure:**
```
swing_id,camera_angle,skill_level,confidence,address_score,takeaway_score,mid_backswing_score,top_score,mid_downswing_score,impact_score,follow_through_score,finish_score,feedback
B12_F1_I,front,pro,5,85,82,88,90,85,92,88,86,Good tempo and lag management
B12_F2_I,front,intermediate,4,72,70,75,78,72,65,68,75,Good address but loses lag in downswing
[FILL BELOW THIS]
```

**Columns to fill for each video:**

| Column | How to Fill | Example |
|--------|---|---|
| `swing_id` | Copy exact folder name (no .mov) | B12_F1_I |
| `camera_angle` | Look at video direction | front, side |
| `skill_level` | Your assessment | pro, advanced, intermediate, beginner |
| `confidence` | How sure are you? | 1-5 (1=guess, 5=very sure) |
| `address_score` | Rate 0-100 | 85 |
| `takeaway_score` | Rate 0-100 | 82 |
| ... (6 more phases) | Rate 0-100 each | ... |
| `feedback` | What needs fixing (if <70) | "Slow takeaway; early wrist break" |

---

## STEP 3: Rate Each Swing (Per Video: ~2-3 minutes)

### Workflow for ONE Swing:

**Example: Rating B12_F1_I.mov**

#### 3.1: Open Keyframes Folder

Go to: `keyframes/B12_F1_I_nn/`

You'll see 8 images: might not be 100% accurate so check the keyframe range.csv
```
B12_F1_I_cleaned_Address.jpg      (start of swing)
B12_F1_I_cleaned_Takeaway.jpg
B12_F1_I_cleaned_Mid-backswing.jpg
B12_F1_I_cleaned_Top.jpg
B12_F1_I_cleaned_Mid-downswing.jpg
B12_F1_I_cleaned_Impact.jpg
B12_F1_I_cleaned_Follow-through.jpg
B12_F1_I_cleaned_Finish.jpg       (end of swing)
```

**Open in image viewer or browser** to see visual reference of each phase.

#### 3.2: Open Video in VLC

`videos/B12_F1_I.mov`

**Watch the full swing first** to get a feel for it.

#### 3.3: Frame-by-Frame Analysis

Using VLC controls:
- Press **E** to advance one frame at a time
- Watch through each phase
- Compare to keyframe images for reference

#### 3.4: Score Each Phase

Using **PHASE_SCORING_GUIDE.md**, rate each phase:

```
ADDRESS (frames 1-10):
  Looking at keyframe: B12_F1_I_cleaned_Address.jpg
  ✓ Good feet position, knees bent
  ✓ Posture looks good
  → Score: 85/100

TAKEAWAY (frames 11-25):
  ✓ Club moves straight back
  ✓ Wrists stay flat
  ✓ Body/arms synchronized
  → Score: 82/100

... (repeat for 6 more phases)
```

#### 3.5: Note Feedback

If ANY phase scores < 70, add feedback:
```
feedback: "Slow takeaway; early wrist break at top"
```

#### 3.6: Fill CSV Row

```
swing_id,camera_angle,skill_level,confidence,address_score,takeaway_score,mid_backswing_score,top_score,mid_downswing_score,impact_score,follow_through_score,finish_score,feedback
B12_F1_I,front,pro,5,85,82,88,90,85,92,88,86,Good tempo and lag management
```

---

## STEP 4: Repeat For All 100 Videos

Process:
1. **Day 1:** Videos 1-30 (B12_F1_I through B24_F3_O)
2. **Day 2:** Videos 31-65 (B24_F4_I through B68_S4_O)
3. **Day 3:** Videos 66-100 (B68_S5_I through B80_S6_O)

**Pacing:**
- 2-3 minutes per video
- 30-40 videos per day
- 3-4 days total

---

## KEY TIPS

### ✅ DO:
- Take breaks every 20 videos (rest your eyes)
- Be consistent in your ratings
- Rate honestly (no need to be generous)
- Save CSV after every 10 videos

### ❌ DON'T:
- Rush through videos (quality matters)
- Rate without watching full video
- Forget to save CSV frequently
- Score based on feeling, use criteria from guide

---

## PHASE SCORING QUICK REFERENCE

| Phase | Key Metric | High Score (80-100) | Low Score (<50) |
|-------|---|---|---|
| **Address** | Posture & stance | Feet shoulder-width, bent knees | Slouched, locked knees |
| **Takeaway** | Club path & sync | Straight back, flat wrists | Inside, early hinge |
| **Mid-backswing** | Plane & rotation | On-plane, hips 45°/shoulders 90° | Off-plane, excessive sway |
| **Top** | Lag & position | Lag ~90°, shaft on-plane | Lag >100°, across line |
| **Mid-downswing** | Lag retention | Lag 60-80°, hip clear | Early release, hip stuck |
| **Impact** ⭐ | X-factor unwind | Lag 10-20° at impact | Lag <5° (early release) |
| **Follow-through** | Continuity | Club up, body rotating | Cramped, deceleration |
| **Finish** | Balance | Weight forward, open hips | Off-balance, stuck |

---

## SAVING & SUBMITTING

### Save Frequently
- After every 10 videos: **Ctrl+S**
- Use format: `pro_annotation_template.csv` (keep this name)

### When Done
- Make sure all 100 rows filled
- Check CSV opens in Excel without errors
- Send back via: [EMAIL/FOLDER LINK]

---

## Questions?

**Common Issues:**

**Q: Can I see phase boundaries in the video?**
A: Yes! Check `B12_F1_I_cleaned_8phases.csv` (same folder as keyframes). It shows frame numbers for each phase.

**Q: The video is too fast. How do I slow down?**
A: In VLC, press **[** to slow down, **]** to speed up. Or use arrow keys for frame-by-frame.

**Q: What if I can't see phase clearly?**
A: Rate based on what you observe + guide criteria. Confidence = lower number (3-4 instead of 5).

**Q: Should I score each phase separately or overall?**
A: **Separately.** Each of 8 phases gets its own 0-100 score. Overall score is calculated later.

---

## Timeline

| Task | Time |
|------|------|
| Setup (install VLC, understand guide) | 15 min |
| Day 1: Videos 1-30 | 2-3 hours |
| Day 2: Videos 31-65 | 2-3 hours |
| Day 3: Videos 66-100 | 2-3 hours |
| **Total** | **~10 hours** |

---

## Example: Full Annotation for ONE Video

**Video:** `B12_F1_I.mov`

**Steps:**
1. Open `keyframes/B12_F1_I_nn/` (see 8 phase images)
2. Open `videos/B12_F1_I.mov` in VLC
3. Watch full swing
4. Go frame-by-frame through each phase
5. Compare to keyframe images
6. Score each phase 0-100 using guide
7. Fill one row in CSV:

```
B12_F1_I,front,pro,5,85,82,88,90,85,92,88,86,Good tempo and lag management
```

**That's it!** Repeat for 99 more videos.

---

## THANK YOU!

Your ratings will train an AI model to understand what separates professional from amateur golf swings.

**Questions during annotation?** Message me anytime.

Good luck! ⛳
