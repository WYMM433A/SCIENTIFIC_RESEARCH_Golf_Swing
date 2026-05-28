# QUICK REFERENCE - Pro Annotation

## TOOLS NEEDED
- **Video Player:** VLC (free, download from videolan.org)
- **Spreadsheet:** Excel or Google Sheets
- **Folder:** ds_videos_annotation_pack/ (you'll receive this)

---

## VLC CONTROLS (Frame-by-Frame)
| Key | Action |
|-----|--------|
| **E** | Next frame |
| **Shift+E** | Previous frame |
| **Space** | Play/Pause |
| **[** | Slow down |
| **]** | Speed up |

---

## SCORING SCALE (0-100)

```
80-100  = Professional quality
50-70   = Average (needs work)
<50     = Major faults
```

---

## 8 PHASES TO RATE

1. **Address** - Starting position
2. **Takeaway** - First 3 feet back
3. **Mid-backswing** - Halfway back
4. **Top** - Club at highest point
5. **Mid-downswing** - Halfway down
6. **Impact** ⭐ - Ball strike (MOST CRITICAL)
7. **Follow-through** - After impact
8. **Finish** - End position

---

## PER VIDEO: 2-3 MINUTES

1. Open `keyframes/<video_name>_nn/` folder (see 8 phase ranges and key frames they are all extracted using a machine learning model so might not be 100% accurate)
2. Open `videos/<video_name>.mov` in VLC
3. Watch full swing
4. Go frame-by-frame through each phase (go direct to the key frame range for each phase if u want it quick)
5. Check to keyframes with range
6. Score each phase 0-100
7. Add feedback if score < 70
8. Fill one row in CSV

---

## CSV COLUMNS

```
swing_id,camera_angle,skill_level,confidence,address_score,takeaway_score,mid_backswing_score,top_score,mid_downswing_score,impact_score,follow_through_score,finish_score,feedback
```

**Example Row:**
```
B12_F1_I,front,pro,5,85,82,88,90,85,92,88,86,Good tempo and lag management
```

---

## KEY METRICS

| Phase | GOOD (80+) | BAD (<50) |
|-------|--------|-------|
| Address | Feet apart, knees bent | Slouched, locked |
| Takeaway | Straight back, flat wrists | Inside, early hinge |
| Top | Lag 90°, on-plane | Lag 100°+, across line |
| Impact | Lag 10-20° | Lag <5° (early release) |
| Finish | Balanced, weight forward | Off-balance |

---

## CONFIDENCE SCALE

- **5** = Very sure (clear phase)
- **4** = Pretty sure
- **3** = Somewhat unsure
- **2** = Uncertain (hard to see)
- **1** = Guess

---

## TIMELINE

**Total: ~10 hours across 3 days**
- Day 1: 30 videos (2-3 hrs)
- Day 2: 35 videos (2-3 hrs)
- Day 3: 35 videos (2-3 hrs)

**Rate:** 2-3 minutes per video

---

## SAVE FREQUENTLY

- Save CSV every 10 videos
- Use **Ctrl+S** in Excel
- Keep filename: `pro_annotation_template.csv`

---

## QUESTIONS?

See: `PRO_ANNOTATION_INSTRUCTIONS.md` (full guide)

Or: Message if confused about any phase
