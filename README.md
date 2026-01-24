# 🏌️ DataStorm - Golf Swing Analysis Pipeline

A computer vision pipeline for analyzing golf swings using pose estimation and phase detection. This project extracts body landmarks from golf swing videos, detects the 8 phases of a golf swing, and provides biomechanical analysis.

---

## 🎯 Project Goal

Build a **Motion-to-Text** and **Text-to-Motion** golf coaching system:

1. **Motion → Text**: Input golf video → Extract poses → Detect phases → Analyze biomechanics → Generate coaching feedback
2. **Text → Motion**: (Future) Generate 3D skeleton animations from text descriptions

---

## 📁 Project Structure

```
DataStorm/
├── pipeline.py                 # Main entry point - runs full pipeline
├── README.md                   # This file
├── ANNOTATION_AND_TRAINING_GUIDE.md
│
├── src/                        # Source code modules
│   ├── __init__.py
│   ├── config.py               # Project configuration
│   │
│   ├── pose/                   # Pose detection
│   │   ├── detector.py         # MediaPipe pose extraction
│   │   └── analyzer.py         # Swing analysis & visualization
│   │
│   ├── phase/                  # Phase detection
│   │   ├── rule_based.py       # Wrist trajectory-based detection
│   │   ├── neural.py           # Bi-LSTM model (for training)
│   │   └── adapter.py          # Switch between methods
│   │
│   ├── video/                  # Video processing
│   │   └── cleaner.py          # Auto-crop to swing bounds
│   │
│   └── biomechanics/           # Angle calculations
│       ├── angles.py           # 15+ golf-critical angles
│       ├── benchmarks.py       # Pro golfer reference values
│       └── comparator.py       # Compare user vs ideal
│
├── models/                     # ML models
│   └── pose_landmarker_lite.task   # MediaPipe pose model
│
├── data/                       # Data directory
│   ├── raw_videos/             # Input: Original golf videos
│   ├── cleaned_videos/         # Auto-cropped videos
│   ├── extracted_poses/        # CSV: 33 landmarks per frame
│   ├── keyframes/              # 8 key frame images per video
│   └── metrics/                # Biomechanics measurements
│
├── tools/                      # Utility scripts
│   ├── inspect_mediapipe.py
│   ├── inspect_videos.py
│   └── visualize_trajectories.py
│
├── notebooks/                  # Jupyter notebooks
└── tests/                      # Unit tests
```

---

## 🔧 Setup Instructions

### 1. Prerequisites

- **Python 3.10+** (tested with 3.13)
- **Anaconda** (recommended) or pip
- **Windows/Mac/Linux**

### 2. Create Conda Environment

```bash
# Create new environment
conda create -n DataStorm python=3.13

# Activate environment
conda activate DataStorm
```

### 3. Install Dependencies

```bash
# Core dependencies
pip install opencv-python numpy pandas matplotlib

# MediaPipe for pose detection
pip install mediapipe

# PyTorch for neural network (phase classifier training)
pip install torch torchvision

# Optional: Jupyter for notebooks
pip install jupyter
```

### 4. Download MediaPipe Model

The pose model should already be in `models/`. If missing:

```bash
# Download MediaPipe Pose Landmarker
curl -o models/pose_landmarker_lite.task https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task
```

### 5. Verify Installation

```bash
# Test that everything loads
python -c "from src.pose.detector import PoseDetector; print('✓ Setup complete!')"
```

---

## How to Use

### Run Full Pipeline (Recommended)

Process a single video through all steps:

```bash
python pipeline.py data/raw_videos/your_video.mp4
```

This will:
1. **Clean** the video (auto-crop to swing motion)
2. **Extract** 33 body landmarks per frame
3. **Detect** 8 swing phases
4. **Save** key frames and data

### Output Files

After running the pipeline, you'll find:

| Output | Location | Description |
|--------|----------|-------------|
| Cleaned video | `data/cleaned_videos/{name}_cleaned.mp4` | Cropped to swing only |
| Pose data | `data/extracted_poses/{name}_poses.csv` | 33 landmarks × 4 values per frame |
| Phase info | `data/keyframes/{name}/{name}_8phases.csv` | Frame ranges for each phase |
| Key frames | `data/keyframes/{name}/*.jpg` | 8 images (one per phase) |
| Metrics | `data/metrics/{name}_metrics.csv` | Biomechanics angles per frame |

---

##  The 8 Golf Swing Phases

| # | Phase | Description |
|---|-------|-------------|
| 1 | **Address** | Setup position, ready to swing |
| 2 | **Takeaway** | Club moves away from ball |
| 3 | **Mid-backswing** | Arms at waist height going back |
| 4 | **Top** | Highest point of backswing |
| 5 | **Mid-downswing** | Arms at waist height coming down |
| 6 | **Impact** | Club hits the ball |
| 7 | **Follow-through** | After impact, arms extending |
| 8 | **Finish** | Final pose, club over shoulder |

---

## 📊 Data Format

### Pose CSV (`extracted_poses/*.csv`)

Each row = 1 frame. Columns:

```
frame, nose_x, nose_y, nose_z, nose_visibility, 
       left_eye_x, left_eye_y, ...
       (33 landmarks × 4 values = 132 features)
```

### Phase CSV (`keyframes/*_8phases.csv`)

```csv
Video,Phase,Start_Frame,End_Frame,Duration,Key_Frame,Image_Path
golf_swing_001,Address,1,50,50,25,data/keyframes/.../Address.jpg
golf_swing_001,Takeaway,50,75,25,62,data/keyframes/.../Takeaway.jpg
...
```

---

## 🧠 Training the Phase Classifier

The current phase detection uses **rule-based** logic (wrist Y trajectory). To train the **neural network** version:

### 1. Prepare Training Data

Process multiple videos through the pipeline to build dataset:

```bash
# Process all videos in raw_videos folder
python pipeline.py data/raw_videos/video1.mp4
python pipeline.py data/raw_videos/video2.mp4
# ... (need 50-100+ videos for good results)
```

### 2. Training Data Structure

```
Training Input:  data/extracted_poses/*.csv    → X (132 features per frame)
Training Labels: data/keyframes/*_8phases.csv  → y (phase label per frame)
```

### 3. Model Architecture

- **Input**: 132 pose features per frame
- **Model**: Bi-LSTM (2 layers, 128 hidden units)
- **Output**: 8 phase classes

See `src/phase/neural.py` for implementation.

---

## 🔬 Biomechanics Analysis

The `src/biomechanics/` module calculates golf-critical angles:

| Metric | Description |
|--------|-------------|
| `spine_angle` | Forward tilt of spine |
| `shoulder_rotation` | Shoulder turn (degrees) |
| `hip_rotation` | Hip turn (degrees) |
| `x_factor` | Shoulder - Hip rotation (power indicator) |
| `lead_arm_angle` | Lead arm straightness |
| `trail_elbow_angle` | Trail elbow bend |
| `wrist_hinge` | Wrist cock angle |
| `knee_flex` | Knee bend angles |

---

## 🗺️ Roadmap

- [x] Video cleaning (auto-crop)
- [x] Pose extraction (MediaPipe)
- [x] Rule-based phase detection
- [x] 8 key frame extraction
- [x] Biomechanics module
- [ ] Neural phase classifier training
- [ ] Text feedback generation
- [ ] Visual overlay system
- [ ] Text → Motion (3D skeleton synthesis)

---

## 👥 Team

- WAI YAN MOE MYINT
- AUNG KAUNG HTET
- NGUYEN THI TUYET NHUNG

---

## 📚 References

- [MediaPipe Pose](https://developers.google.com/mediapipe/solutions/vision/pose_landmarker)
- [Golf Swing Biomechanics Research](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6413833/)
