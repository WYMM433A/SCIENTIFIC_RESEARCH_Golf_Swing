# Golf Biomechanics Metrics - Complete Formula Reference

This document provides comprehensive mathematical formulas and implementation details for every biomechanical metric used in phase scoring.

**Related files:**
- `src/biomechanics/angles.py` - Core calculations
- `src/biomechanics/scoring_config.py` - Thresholds and weights
- `src/biomechanics/phase_scorer.py` - Scoring logic

---

## Table of Contents

1. [Posture Metrics](#posture-metrics)
2. [Rotation Metrics](#rotation-metrics)
3. [Arm & Wrist Metrics](#arm--wrist-metrics)
4. [Lower Body Metrics](#lower-body-metrics)
5. [Stability & Head Movement](#stability--head-movement)
6. [Kinematic Sequence (Advanced)](#kinematic-sequence-advanced)

---

## Posture Metrics

### 1. Spine Angle (Forward Tilt)

**Description:** Forward lean of spine from vertical axis

**Formula:**
```
spine_angle = atan2(shoulder_x - hip_x, hip_y - shoulder_y) × 180/π
```

**Calculation Steps:**
1. Get midpoint of left and right hips: `hip_mid = (left_hip + right_hip) / 2`
2. Get midpoint of left and right shoulders: `shoulder_mid = (left_shoulder + right_shoulder) / 2`
3. Calculate angle from vertical (hip_mid is base, shoulder_mid is apex)
4. Use `atan2` to get angle in degrees

**Code Implementation:** [`angles.py` line 220](src/biomechanics/angles.py#L220)
```python
def get_spine_angle(self, lmList: List = None, frame: int = None) -> float:
    # Get landmarks
    hip_mid = (left_hip + right_hip) / 2
    shoulder_mid = (left_shoulder + right_shoulder) / 2
    
    # Calculate angle from vertical
    dx = shoulder_mid[0] - hip_mid[0]
    dy = hip_mid[1] - shoulder_mid[1]  # Inverted Y (image coords)
    angle_from_vertical = atan2(dx, dy)
    return degrees(angle_from_vertical)
```

**Target Ranges (by phase):**
| Phase | Ideal | Acceptable |
|-------|-------|-----------|
| Address | -5 to 10° | -20 to 35° |
| Impact | -3 to 8° | -20 to 35° |
| Finish | -10 to 12° | -20 to 35° |

**Interpretation:**
- Positive = forward lean (golfer bent toward ball)
- Negative = back lean (unusual)
- Optimal: ~5° forward lean at address

---

### 2. Spine Lateral Tilt (Side Bend)

**Description:** Right shoulder lower than left (for RH golfer)

**Formula:**
```
spine_lateral_tilt = atan2(right_shoulder_y - left_shoulder_y, |right_shoulder_x - left_shoulder_x|) × 180/π
```

**Calculation Steps:**
1. Get right shoulder Y position - left shoulder Y position: `dy = right_shoulder[1] - left_shoulder[1]`
2. Get horizontal distance between shoulders: `dx = |right_shoulder[0] - left_shoulder[0]|`
3. Calculate angle using `atan2(dy, dx)`

**Code Implementation:** [`angles.py` line 420](src/biomechanics/angles.py#L420)

**Target Ranges (by phase):**
| Phase | Ideal | Acceptable |
|-------|-------|-----------|
| Address | 0° | -5 to 5° |
| Top | 10° | 5 to 20° |
| Impact | 15° | 10 to 25° |

**Interpretation:**
- Positive = right shoulder lower (correct for RH backswing)
- At address: should be minimal (upright)
- At top: increases (behind ball tilt)
- At impact: maximum (driving down and through)

---

## Rotation Metrics

### 3. Shoulder Rotation

**Description:** Rotation of shoulder line from target line (horizontal)

**Formula:**
```
shoulder_rotation = |atan2(right_shoulder_y - left_shoulder_y, right_shoulder_x - left_shoulder_x)| × 180/π
```

**Calculation Steps:**
1. Calculate line angle of shoulder line: `angle = atan2(Δy, Δx)`
2. Convert to degrees
3. Take absolute value to normalize (0-180°)

**Code Implementation:** [`angles.py` line 265](src/biomechanics/angles.py#L265)
```python
def get_shoulder_rotation(self, lmList=None, frame=None) -> float:
    left_shoulder = [x1, y1]
    right_shoulder = [x2, y2]
    angle = atan2(y2 - y1, x2 - x1)
    return abs(degrees(angle))  # 0-180 normalized
```

**Target Ranges (by phase):**
| Phase | Ideal |
|-------|-------|
| Address | 0-5° (square) |
| Takeaway | 5-25° (initiating turn) |
| Mid-backswing | 155-185° (full turn) |
| Top | 90-110° (maximum turn) |
| Impact | 20-35° (rotated through) |

**Interpretation:**
- 0° = square to target (address)
- 90° = perpendicular to target (top of backswing)
- 180° = facing away from target (finish)

---

### 4. Hip Rotation

**Description:** Rotation of hip line from target line

**Formula:**
```
hip_rotation = |atan2(right_hip_y - left_hip_y, right_hip_x - left_hip_x)| × 180/π
```

**Calculation Steps:**
Same as shoulder rotation but using hip landmarks instead

**Code Implementation:** [`angles.py` line 290](src/biomechanics/angles.py#L290)

**Target Ranges (by phase):**
| Phase | Ideal |
|-------|-------|
| Address | 0-5° |
| Top | 40-55° (less than shoulders) |
| Impact | 40-50° |
| Finish | 160-185° |

**Interpretation:**
- Hips rotate less than shoulders (creates X-factor)
- ~45° at top is typical
- More rotation = better hip drive

---

### 5. X-Factor (Shoulder-Hip Separation)

**Description:** Difference between shoulder and hip rotation - key power indicator

**Formula:**
```
x_factor = |shoulder_rotation - hip_rotation|
(with circular angular difference to avoid 355° instead of 5°)

x_factor = min(diff, 360 - diff)  where diff = |shoulder_rot - hip_rot| % 360
```

**Calculation Steps:**
1. Calculate shoulder rotation: `s_rot = get_shoulder_rotation()`
2. Calculate hip rotation: `h_rot = get_hip_rotation()`
3. Find difference: `diff = |s_rot - h_rot| % 360`
4. Apply circular distance: `x_factor = min(diff, 360 - diff)`

**Code Implementation:** [`angles.py` line 300](src/biomechanics/angles.py#L300)
```python
def get_x_factor(self, lmList=None, frame=None) -> float:
    shoulder_rot = self.get_shoulder_rotation(lmList, frame)
    hip_rot = self.get_hip_rotation(lmList, frame)
    diff = abs(shoulder_rot - hip_rot) % 360
    return min(diff, 360 - diff)  # Circular distance
```

**Target Ranges (by phase):**
| Phase | Ideal |
|-------|-------|
| Address | 0-5° |
| Mid-backswing | 2-14° |
| Top | 2-18° (max separation) |
| Mid-downswing | 3-15° (x-factor stretch) |
| Finish | 0-6° (release) |

**Interpretation:**
- Higher X-factor = more coil = more power potential
- At top: 40-50° is professional level
- Critical for storing elastic energy in backswing

**X-Factor Stretch (Dynamic):**
```
x_factor_stretch = percentile_90(x_factor_series) - percentile_10(x_factor_series)
(capped at 40.0)
```
Measures the range of X-factor variation during the sequence.

---

## Arm & Wrist Metrics

### 6. Lead Arm Angle

**Description:** Straightness of left arm at elbow (180° = perfectly straight)

**Formula:**
```
lead_arm_angle = arccos((v1 · v2) / (||v1|| × ||v2||)) × 180/π

where:
  v1 = left_shoulder - left_elbow
  v2 = left_wrist - left_elbow
```

**Calculation Steps (3-Point Angle):**
1. Create vectors from elbow to shoulder and elbow to wrist
2. Calculate dot product: `v1 · v2 = v1_x × v2_x + v1_y × v2_y`
3. Calculate magnitudes: `||v1|| = √(v1_x² + v1_y²)`
4. Use `arccos` to get angle: `angle = arccos(dot_product / (mag1 × mag2))`
5. Convert to degrees

**Code Implementation:** [`angles.py` line 135](src/biomechanics/angles.py#L135)
```python
def calculate_angle_3points(self, p1, p2, p3) -> float:
    v1 = p1[:2] - p2[:2]  # Vector from p2 to p1
    v2 = p3[:2] - p2[:2]  # Vector from p2 to p3
    
    cos_angle = dot(v1, v2) / (norm(v1) * norm(v2) + 1e-8)
    angle = arccos(clip(cos_angle, -1, 1))
    return degrees(angle)
```

**Target Ranges (by phase):**
| Phase | Ideal |
|-------|-------|
| Address | 160-180° (extended) |
| Top | 165-180° (maintained) |
| Impact | 160-180° (extended) |

**Interpretation:**
- 180° = fully extended
- 170° = slight bend
- Lower = bent arm (less efficient)

---

### 7. Trail Elbow Angle

**Description:** Right elbow bend angle

**Formula:**
```
trail_elbow_angle = arccos((v1 · v2) / (||v1|| × ||v2||)) × 180/π

where:
  v1 = right_shoulder - right_elbow
  v2 = right_wrist - right_elbow
```

**Calculation Steps:** Same as lead arm angle but using right side

**Code Implementation:** [`angles.py` line 350](src/biomechanics/angles.py#L350)

**Target Ranges (by phase):**
| Phase | Ideal |
|-------|-------|
| Address | 160-180° (extended) |
| Top | 75-105° (bent) |
| Impact | 135-170° (extending) |

**Interpretation:**
- At address: extended (160-180°)
- At top: sharp bend ~90° (classic "chicken wing" in backswing)
- At impact: straightening as club hits ball

---

### 8. Wrist Hinge (Lag Angle)

**Description:** Wrist cock angle - critical for power and lag retention

**Formula:**
```
wrist_hinge = arccos((v1 · v2) / (||v1|| × ||v2||)) × 180/π

where:
  v1 = left_elbow - left_wrist
  v2 = left_index - left_wrist
  (measures angle between forearm and hand)
```

**Calculation Steps:**
1. Create vectors from wrist to elbow and wrist to index finger
2. Calculate angle between them (same 3-point method)
3. Smaller angle = more hinged = more lag

**Code Implementation:** [`angles.py` line 375](src/biomechanics/angles.py#L375)

**Target Ranges (by phase):**
| Phase | Ideal |
|-------|-------|
| Address | 140-170° (neutral) |
| Takeaway | 160-180° (slight hinge initiation) |
| Mid-backswing | 150-180° (increasing hinge) |
| Top | 145-180° (strong hinge) |
| Mid-downswing | 150-180° (lag retained) |
| Impact | 150-180° (lag release) |

**Interpretation:**
- 180° = fully extended (no lag)
- 150° = 30° of lag (good lag retention)
- 90° = maximum hinge (extreme cock)
- **Critical for power:** More lag at impact = longer drive

**LAG ANGLE = WRIST HINGE** (same metric, different name in different contexts)

---

## Lower Body Metrics

### 9. Lead Knee Flex

**Description:** Left knee bend angle (180° = straight leg)

**Formula:**
```
lead_knee_flex = arccos((v1 · v2) / (||v1|| × ||v2||)) × 180/π

where:
  v1 = left_hip - left_knee
  v2 = left_ankle - left_knee
```

**Calculation Steps:** 3-point angle calculation using hip-knee-ankle

**Code Implementation:** [`angles.py` line 395](src/biomechanics/angles.py#L395)

**Target Ranges (by phase):**
| Phase | Ideal |
|-------|-------|
| Address | 145-170° (slight flex) |
| Top | 140-165° |
| Impact | 160-180° (straightening) |

**Interpretation:**
- 180° = fully extended (straight)
- 150° = 30° bend
- Should straighten through impact

---

### 10. Trail Knee Flex

**Description:** Right knee bend angle

**Formula:** Same as lead knee but using right side

**Code Implementation:** [`angles.py` line 410](src/biomechanics/angles.py#L410)

**Target Ranges (by phase):**
| Phase | Ideal |
|-------|-------|
| Address | 145-170° |
| Top | 145-170° (maintains flex) |
| Impact | 130-160° |

**Interpretation:**
- Should maintain flex throughout backswing (power source)
- Straightens during downswing

---

### 11. Stance Width Ratio

**Description:** Ankle-to-ankle distance as ratio of shoulder width

**Formula:**
```
stance_width = distance(left_ankle, right_ankle)
shoulder_width = distance(left_shoulder, right_shoulder)

stance_width_ratio = stance_width / shoulder_width
```

**Calculation Steps:**
1. Calculate stance width: `stance = √((x2-x1)² + (y2-y1)²)`
2. Calculate shoulder width: `shoulder = √((x2-x1)² + (y2-y1)²)`
3. Divide: `ratio = stance / shoulder`

**Code Implementation:** [`angles.py` line 465](src/biomechanics/angles.py#L465)
```python
def get_stance_width_ratio(self, lmList=None, frame=None) -> float:
    stance = self.get_stance_width(lmList, frame)
    shoulder = self.get_shoulder_width(lmList, frame)
    return stance / (shoulder + 1e-8)
```

**Target Ranges:**
| Phase | Ideal |
|-------|-------|
| Address | 0.9-1.2 (shoulder-width stance) |

**Interpretation:**
- 1.0 = same width as shoulders (ideal)
- < 0.8 = too narrow (less stable)
- > 1.5 = too wide (less mobile)

---

## Stability & Head Movement

### 12. Head Displacement

**Description:** How much head moves from address reference position

**Formula:**
```
head_displacement = √((x_current - x_ref)² + (y_current - y_ref)²)

where (x_ref, y_ref) = head position at address (set as reference)
```

**Calculation Steps:**
1. Store reference head position (nose landmark) at address: `ref_nose = get_point_at_frame(address_frame)`
2. For each frame, get current nose position: `current_nose = get_point_at_frame(current_frame)`
3. Calculate Euclidean distance: `distance = √((Δx)² + (Δy)²)`

**Code Implementation:** [`angles.py` line 545](src/biomechanics/angles.py#L545)
```python
def get_head_movement(self, lmList=None, frame=None) -> Tuple[float, float]:
    # Returns (lateral_movement, vertical_movement)
    current_nose = get_point(...)
    ref_nose = self._reference_positions['nose']
    
    lateral = current_nose[0] - ref_nose[0]
    vertical = current_nose[1] - ref_nose[1]
    return (lateral, vertical)
```

**To get displacement:** `displacement = √(lateral² + vertical²)`

**Target Ranges (by phase):**
| Phase | Ideal |
|-------|-------|
| Takeaway | 0-8 cm |
| Top | 0-3 cm (very still) |
| Impact | 0-3 cm (strike consistency) |
| Finish | 0-4 cm |

**Interpretation:**
- Lower = better (minimal head movement = consistent strike)
- Professional golfers keep head very still

---

## Kinematic Sequence (Advanced)

### 13. Kinematic Sequence Timing

**Description:** The onset timing of hip → torso → arm → club rotation during downswing

This is the **MOST CRITICAL** metric for power generation.

#### A. Angular Velocity Calculation

**Formula:**
```
angular_velocity[i] = (angle[i+1] - angle[i]) / time_step

where angle[i] is rotation angle of body segment at frame i
```

**Steps:**
1. Extract rotation angles for each frame in downswing window
2. Calculate frame-to-frame rotation: `Δangle = angle[i+1] - angle[i]`
3. Convert to velocity: `velocity = Δangle (degrees per frame)`

**Code Implementation:** [`angles.py` line 800](src/biomechanics/angles.py#L800)
```python
def compute_angular_velocity_sequence(self, df, start_frame, end_frame):
    # 1. Extract rotation angles
    hip_rotations = []
    torso_rotations = []
    arm_rotations = []
    
    for frame in range(start_frame, end_frame):
        hip_angle = atan2(right_hip_x - left_hip_x, ...)
        torso_angle = atan2(right_shoulder_x - left_shoulder_x, ...)
        arm_angle = atan2(right_wrist_x - left_wrist_x, ...)
        
        hip_rotations.append(hip_angle)
        torso_rotations.append(torso_angle)
        arm_rotations.append(arm_angle)
    
    # 2. Unwrap angles to avoid 180/-180 discontinuity
    hip_unwrapped = np.degrees(np.unwrap(np.radians(hip_rotations)))
    torso_unwrapped = np.degrees(np.unwrap(np.radians(torso_rotations)))
    arm_unwrapped = np.degrees(np.unwrap(np.radians(arm_rotations)))
    
    # 3. Smooth velocities
    hip_velocities = smooth_velocity(np.abs(np.diff(hip_unwrapped)), window=3)
    torso_velocities = smooth_velocity(np.abs(np.diff(torso_unwrapped)), window=3)
    arm_velocities = smooth_velocity(np.abs(np.diff(arm_unwrapped)), window=3)
```

#### B. Motion Onset Detection (Persistent Activation)

**Formula:**
```
is_active[i] = velocity[i] >= max(percentile_65(velocity), 0.25 * max(velocity))

onset_frame = first_frame where N_consecutive(is_active) >= 2
(requires 2+ consecutive frames of activity, not 1-frame spikes)
```

**Steps:**
1. Calculate threshold: `threshold = max(percentile_65(velocities), 0.25 * max(velocities))`
2. Create active mask: `active[i] = velocity[i] >= threshold`
3. Find first run of 2+ consecutive True values
4. Onset = frame index of that run

**Code Implementation:** [`angles.py` line 850](src/biomechanics/angles.py#L850)
```python
def find_motion_start(velocities, threshold_percentile=65, min_consecutive=2):
    if len(velocities) == 0:
        return 0
    
    percentile_threshold = np.percentile(velocities, threshold_percentile)
    absolute_floor = 0.25 * np.max(velocities)
    threshold = max(percentile_threshold, absolute_floor)
    
    active = velocities >= threshold
    
    # Find first run of min_consecutive True values
    consecutive = 0
    for i, is_active in enumerate(active):
        if is_active:
            consecutive += 1
            if consecutive >= min_consecutive:
                return i - min_consecutive + 1  # Return start of run
        else:
            consecutive = 0
    
    return 0
```

#### C. Sequence Efficiency Scoring

**Formula:**
```
ideal_sequence = [hip_start, torso_start, arm_start, club_start]

sequence_efficiency = 1.0 - (total_deviation / ideal_range)

where:
  total_deviation = sum of actual delays - ideal delays
  ideal_range = maximum possible deviation
```

**Target Delays (ideal kinematic sequence):**
- Hip to Torso: 0-50ms (hip leads slightly)
- Torso to Arm: 50-150ms (arms lag torso)
- Arm to Club: 50-150ms (club lags arms)

**Score Interpretation:**
- 1.0 = perfect sequence
- 0.8-1.0 = excellent (professional level)
- 0.5-0.8 = good
- < 0.5 = poor

#### D. X-Factor Stretch (Dynamic Measurement)

**Formula:**
```
x_factor_series[i] = |torso_rotation[i] - hip_rotation[i]|

x_factor_stretch = percentile_90(x_factor_series) - percentile_10(x_factor_series)
(capped at 40.0 degrees)
```

**Steps:**
1. Calculate X-factor at each frame: `xf[i] = |torso_rot[i] - hip_rot[i]|`
2. Get 90th percentile: `p90 = percentile(x_factor_series, 90)`
3. Get 10th percentile: `p10 = percentile(x_factor_series, 10)`
4. Stretch = `p90 - p10` (uses inter-quartile range to ignore outlier spikes)

**Interpretation:**
- Higher stretch = more coil variation = better power generation
- 20-30° is typical
- Capped at 40° to penalize unrealistic values

---

## Summary Scoring Table

| Metric | Category | Formula Type | Data Points | Key Weight |
|--------|----------|--------------|------------|------------|
| Spine Angle | Posture | Angle from vertical | 2 (hips, shoulders) | 0.25 (Address) |
| Spine Lateral Tilt | Posture | Angle from horizontal | 2 (shoulders) | 0.20 (Top) |
| Shoulder Rotation | Rotation | Line angle | 2 (shoulders) | 0.30 (Takeaway) |
| Hip Rotation | Rotation | Line angle | 2 (hips) | 0.20 (Mid-downswing) |
| X-Factor | Rotation | Difference | 4 (hips, shoulders) | 0.35 (Top) |
| Lead Arm Angle | Arm | 3-point angle | 3 (shoulder, elbow, wrist) | 0.20 (Impact) |
| Trail Elbow Angle | Arm | 3-point angle | 3 (shoulder, elbow, wrist) | 0.15 (Mid-downswing) |
| Wrist Hinge (Lag) | Arm/Wrist | 3-point angle | 3 (elbow, wrist, index) | 0.40 (Mid-downswing) |
| Lead Knee Flex | Lower Body | 3-point angle | 3 (hip, knee, ankle) | 0.10 (Impact) |
| Trail Knee Flex | Lower Body | 3-point angle | 3 (hip, knee, ankle) | 0.10 (Impact) |
| Stance Width Ratio | Lower Body | Distance ratio | 4 (ankles, shoulders) | 0.20 (Address) |
| Head Displacement | Stability | Euclidean distance | 2 (reference, current nose) | 0.25 (Top) |
| Kinematic Sequence | Timing | Velocity onset analysis | Full sequence | 0.40 (Mid-downswing) |

---

## Implementation Notes

### Robustness Features

1. **Angle Unwrapping:**
   - Problem: Angles wrap at ±180°, causing discontinuity spikes
   - Solution: `np.unwrap()` creates continuous angle series
   - Applied to: hip, torso, arm rotations in kinematic sequence

2. **Velocity Smoothing:**
   - Problem: Single-frame noise spikes trigger false motion onset
   - Solution: 3-frame convolve kernel for moving average
   - Applied to: hip/torso/arm velocities

3. **Persistent Activation:**
   - Problem: Outlier spike can falsely trigger onset
   - Solution: Require 2+ consecutive active frames
   - Applied to: motion onset detection

4. **Circular Angular Difference:**
   - Problem: 355° is actually 5° in circular space
   - Solution: `min(diff, 360 - diff)` for true circular distance
   - Applied to: X-factor calculation

---

## Data Sources

All metrics are derived from **33 MediaPipe pose landmarks:**

```
Head:       0: nose, 1-3: left eye, 4-6: right eye, 7-8: ears
Torso:      11-12: shoulders, 23-24: hips
Arms:       13-14: elbows, 15-16: wrists, 17-22: hands
Legs:       25-26: knees, 27-28: ankles, 29-32: feet
```

---

## References

- Neal & Wilson (1985): 3D Kinematics of Golf Swing
- Hume et al. (2005): Role of Biomechanics in Maximizing Distance and Accuracy
- Chu et al. (2010): Biomechanical Comparison Between Elite Female and Male Golfers
