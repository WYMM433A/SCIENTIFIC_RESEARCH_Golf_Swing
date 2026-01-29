# 🏌️ DataStorm - Golf Swing Analysis Pipeline

A computer vision pipeline for analyzing golf swings using pose estimation and phase detection. This project extracts body landmarks from golf swing videos, detects the 8 phases of a golf swing, and provides biomechanical analysis.

---

## ⚡ Quick Command Reference

| Task | Command |
|------|---------|
| **Full pipeline (rule-based)** | `python pipeline.py data/raw_videos/video.mp4` |
| **Full pipeline (neural network)** | `python pipeline.py data/raw_videos/video.mp4 --method neural-network` |
| **Test neural network** | `python tools/test_neural_network.py --video golf_swing_007` |
| **Compare RB vs NN** | `python tools/test_neural_network.py --video golf_swing_007 --compare` |
| **Train model** | `python tools/train_with_golfdb.py --epochs 50` |
| **Extract GolfDB poses** | `python tools/extract_poses_range.py --start 0 --end 50` |

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
│   │   ├── neural.py           # Bi-LSTM model (PoseSwingNet)
│   │   └── adapter.py          # Switch between rule-based & neural
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
│   ├── pose_landmarker_lite.task   # MediaPipe pose model
│   ├── pose_swingnet_trained.pth   # Trained Bi-LSTM (after training)
│   └── pose_swingnet_best.pth      # Best validation checkpoint
│
├── data/                       # Data directory
│   ├── raw_videos/             # Input: Original golf videos
│   ├── cleaned_videos/         # Auto-cropped videos
│   ├── extracted_poses/        # CSV: 33 landmarks per frame
│   ├── keyframes/              # 8 key frame images per video
│   ├── metrics/                # Biomechanics measurements
│   ├── videos_160/             # GolfDB preprocessed videos
│   ├── golfdb_poses/           # Extracted poses from GolfDB
│   └── golfDB.pkl              # GolfDB annotations
│
├── tools/                      # Utility scripts
│   ├── train_with_golfdb.py    # Train neural network with GolfDB
│   ├── extract_poses_range.py  # Extract poses for video range for example to extract poses from video 50 to 100
│   ├── test_neural_network.py  # Test trained model
│   ├── inspect_mediapipe.py    # Debug MediaPipe output
│   ├── inspect_videos.py       # Video inspection utility
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
pip install opencv-python numpy pandas matplotlib tqdm

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

## 🚀 Quick Start

### Run Full Pipeline

Process a single video through all steps:

```bash
# Using rule-based phase detection (default)
python pipeline.py data/raw_videos/your_video.mp4

# Using neural network phase detection
python pipeline.py data/raw_videos/your_video.mp4 --method neural-network

# Using neural network with custom model
python pipeline.py data/raw_videos/your_video.mp4 -m neural-network --model models/pose_swingnet_best.pth

# With live preview
python pipeline.py data/raw_videos/your_video.mp4 --preview
```

### Pipeline Options

| Flag | Description | Default |
|------|-------------|---------|
| `--method`, `-m` | Phase detection: `rule-based` or `neural-network` | `rule-based` |
| `--model` | Path to trained model (for neural-network) | `models/pose_swingnet_trained.pth` |
| `--preview`, `-p` | Show live preview during processing | Off |

### Pipeline Steps

1. **Clean** the video (auto-crop to swing motion)
2. **Extract** 33 body landmarks per frame
3. **Detect** 8 swing phases (rule-based or neural network)
4. **Save** key frames and data

### Output Files

After running the pipeline, you'll find:

| Output | Location | Description |
|--------|----------|-------------|
| Cleaned video | `data/cleaned_videos/{name}_cleaned.mp4` | Cropped to swing only |
| Pose data | `data/extracted_poses/{name}_cleaned_poses.csv` | 33 landmarks × 4 values per frame |
| Phase info | `data/keyframes/{name}_rb/` or `{name}_nn/` | Folder suffix indicates method used |
| Phase CSV | `data/keyframes/{name}_rb/{name}_cleaned_8phases.csv` | Frame ranges for each phase |
| Key frames | `data/keyframes/{name}_rb/*.jpg` | 8 images (one per phase) |
| Metrics | `data/metrics/{name}_cleaned_metrics.csv` | Biomechanics angles per frame |

> **Note**: The keyframes folder uses `_rb` suffix for rule-based and `_nn` suffix for neural-network detection, so you can easily compare both methods on the same video.

---

## 📋 Usage Examples

### 1. Full Pipeline with Rule-Based Detection

```bash
python pipeline.py data/raw_videos/golf_swing_001.mp4
```

**Output:**
- `data/keyframes/golf_swing_001_rb/` - 8 key frame images + phase CSV

### 2. Full Pipeline with Neural Network Detection

```bash
python pipeline.py data/raw_videos/golf_swing_001.mp4 --method neural-network
```

**Output:**
- `data/keyframes/golf_swing_001_nn/` - 8 key frame images + phase CSV

### 3. Compare Both Methods on Same Video

Run both methods on the same video to compare results:

```bash
# First run with rule-based
python pipeline.py data/raw_videos/golf_swing_001.mp4 --method rule-based

# Then run with neural network
python pipeline.py data/raw_videos/golf_swing_001.mp4 --method neural-network
```

This creates two separate folders:
- `data/keyframes/golf_swing_001_rb/` - Rule-based results
- `data/keyframes/golf_swing_001_nn/` - Neural network results

### 4. Test Neural Network Only (Without Full Pipeline)

Use the test script to quickly test the neural network on already-processed videos:

```bash
# Test on a video (uses existing pose CSV)
python tools/test_neural_network.py --video golf_swing_007

# Test with a specific model
python tools/test_neural_network.py --video golf_swing_007 --model models/pose_swingnet_best.pth

# Compare rule-based vs neural network side-by-side
python tools/test_neural_network.py --video golf_swing_007 --compare
```

### 5. Extract Poses Only (No Phase Detection)

Use the SwingAnalyzer directly for pose extraction:

```python
from src.pose import SwingAnalyzer

analyzer = SwingAnalyzer()

# Process video and get poses
df, metrics = analyzer.process_video(
    video_path='data/cleaned_videos/golf_swing_001_cleaned.mp4',
    output_csv='data/extracted_poses/golf_swing_001_poses.csv',
    show_preview=False
)

print(f"Extracted {len(df)} frames with {len(df.columns)} features")
```

Or use the pipeline with early exit:

```python
from pipeline import GolfSwingPipeline

pipeline = GolfSwingPipeline()
result = pipeline.run('data/raw_videos/golf_swing_001.mp4')

# Access pose data
poses_csv = result['poses_csv']  # Path to extracted poses
metrics_csv = result['metrics_csv']  # Path to biomechanics metrics
```

### 6. Batch Process Multiple Videos

```bash
# Process all videos in a folder
for video in data/raw_videos/*.mp4; do
    python pipeline.py "$video" --method neural-network
done
```

PowerShell version:
```powershell
Get-ChildItem data/raw_videos/*.mp4 | ForEach-Object {
    python pipeline.py $_.FullName --method neural-network
}
```

### 7. Extract Poses for GolfDB Videos (Training Data)

```bash
# Extract poses for videos 0-50
python tools/extract_poses_range.py --start 0 --end 50

# Extract specific range
python tools/extract_poses_range.py --start 100 --end 150
```

---

## 🏋️ Training the Neural Network (PoseSwingNet)

The neural network uses **GolfDB** dataset for training. It's a Bi-LSTM that predicts golf swing phases from pose sequences.

### Prerequisites

1. Download GolfDB videos to `data/videos_160/`
2. Place `golfDB.pkl` annotations in `data/`

### Step 1: Extract Poses from GolfDB Videos

```bash
# Extract poses for videos 0-50
python tools/extract_poses_range.py --start 0 --end 50

# Extract more videos (46-100)
python tools/extract_poses_range.py --start 46 --end 100
```

### Step 2: Train the Model

```bash
# Train with first 50 videos
python tools/train_with_golfdb.py --max-videos 50 --epochs 50

# Or specify a range
python tools/train_with_golfdb.py --start-video 0 --end-video 50 --epochs 50
```

### Step 3: Continue Training (Fine-tuning)

```bash
# Continue training with videos 46-100, loading existing model
python tools/train_with_golfdb.py \
    --start-video 46 \
    --end-video 100 \
    --epochs 50 \
    --resume models/pose_swingnet_trained.pth \
    --skip-extraction \
    --lr 0.0005
```

### Training Options

| Flag | Description | Default |
|------|-------------|---------|
| `--max-videos` | Limit total videos | All |
| `--start-video` | Start from video index | 0 |
| `--end-video` | End at video index | End |
| `--epochs` | Training epochs | 50 |
| `--batch-size` | Batch size | 16 |
| `--lr` | Learning rate | 0.001 |
| `--resume` | Resume from model path | None |
| `--skip-extraction` | Skip pose extraction | False |

### Output Models

| Model | Description |
|-------|-------------|
| `models/pose_swingnet_trained.pth` | Final trained model |
| `models/pose_swingnet_best.pth` | Best validation accuracy |
| `models/pose_swingnet_epoch_N.pth` | Checkpoint every 10 epochs |

---

## 🧪 Testing & Comparison

### Test Neural Network on a Video

```bash
# Basic test - shows detected phases and confidence
python tools/test_neural_network.py --video golf_swing_007

# Test with a specific model checkpoint
python tools/test_neural_network.py --video golf_swing_007 --model models/pose_swingnet_best.pth
```

### Compare Rule-Based vs Neural Network

```bash
# Side-by-side comparison of both methods
python tools/test_neural_network.py --video golf_swing_007 --compare
```

This outputs a comparison table showing:
- Which frames each method selected for each phase
- The difference between the two methods

Example output:
```
======================================================================
COMPARISON: Rule-Based vs Neural Network
======================================================================
Phase              Rule-Based    Neural-Net       Diff
----------------------------------------------------------------------
Address                   107           199        +92
Takeaway                  215           211         -4
Mid-backswing             221           216         -5
Top                       228           236         +8
...
```

### Full Pipeline Comparison

To fully compare both methods with all outputs:

```bash
# Run both methods on the same video
python pipeline.py data/raw_videos/golf_swing_004.mp4 --method rule-based
python pipeline.py data/raw_videos/golf_swing_004.mp4 --method neural-network

# Check the results
ls data/keyframes/golf_swing_004_rb/  # Rule-based output
ls data/keyframes/golf_swing_004_nn/  # Neural network output
```

You can then visually compare the extracted key frames from both folders.

---

## 🔄 Switching Detection Methods

The `adapter.py` provides a unified interface to switch between methods:

```python
from src.phase.adapter import create_predictor

# Rule-based (default, no model required)
predictor = create_predictor('rule-based')

# Neural network (requires trained model)
predictor = create_predictor('neural-network', 'models/pose_swingnet_trained.pth')

# Process a video
results = predictor.process(
    csv_path='data/extracted_poses/video_poses.csv',
    video_path='data/cleaned_videos/video.mp4',
    output_dir='data/keyframes/video'
)
```

---

## 🛠️ Tools Reference

All utility scripts are in the `tools/` directory:

| Tool | Description | Usage |
|------|-------------|-------|
| `train_with_golfdb.py` | Train neural network with GolfDB | `python tools/train_with_golfdb.py --epochs 50` |
| `test_neural_network.py` | Test trained model on videos | `python tools/test_neural_network.py --video golf_swing_007` |
| `extract_poses_range.py` | Extract poses for GolfDB videos | `python tools/extract_poses_range.py --start 0 --end 50` |
| `visualize_trajectories.py` | Visualize wrist trajectories | `python tools/visualize_trajectories.py` |
| `inspect_mediapipe.py` | Debug MediaPipe pose output | `python tools/inspect_mediapipe.py` |
| `inspect_videos.py` | Video inspection utility | `python tools/inspect_videos.py` |

### Test Neural Network Options

```bash
python tools/test_neural_network.py [OPTIONS]
```

| Flag | Description | Default |
|------|-------------|---------|
| `--video` | Video ID to test (e.g., `0`, `119`, `golf_swing_001`) | `0` |
| `--model` | Path to trained model | `models/pose_swingnet_trained.pth` |
| `--compare` | Compare rule-based vs neural network | Off |

### Extract Poses Range Options

```bash
python tools/extract_poses_range.py [OPTIONS]
```

| Flag | Description | Default |
|------|-------------|---------|
| `--start` | Start video index | `0` |
| `--end` | End video index | `10` |

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

## 🧠 Model Architecture (PoseSwingNet)

The neural network is a **Bidirectional LSTM** that processes pose sequences:

```
Input (132 features: 33 landmarks × 4 values)
          ↓
┌─────────────────────────┐
│  Input Projection       │
│  Linear(132 → 256)      │
│  ReLU + Dropout(0.2)    │
│  Linear(256 → 128)      │
└─────────────────────────┘
          ↓
┌─────────────────────────┐
│  Bi-LSTM (2 layers)     │
│  hidden_size=128        │
│  bidirectional=True     │
└─────────────────────────┘
          ↓
┌─────────────────────────┐
│  Classifier             │
│  Linear(256 → 64)       │
│  Linear(64 → 9)         │
└─────────────────────────┘
          ↓
Output (9 classes per frame)
  - Classes 0-7: 8 swing phases
  - Class 8: no-event (between phases)
```

**Why Bi-LSTM?**
- Processes sequences in both directions
- Can use future context when predicting current frame
- Better for golf swings where phases depend on before AND after

---

## 📊 Data Format

### Pose CSV (`extracted_poses/*.csv`)

Each row = 1 frame. Columns:

```
frame, nose_x, nose_y, nose_z, nose_visibility, 
       left_eye_x, left_eye_y, ...
       (33 landmarks × 4 values = 132 features)
```

### Phase CSV (`keyframes/{name}_rb/*.csv` or `keyframes/{name}_nn/*.csv`)

```csv
Video,Phase,Start_Frame,End_Frame,Duration,Key_Frame,Image_Path
golf_swing_001,Address,1,50,50,25,data/keyframes/golf_swing_001_rb/Address.jpg
golf_swing_001,Takeaway,50,75,25,62,data/keyframes/golf_swing_001_rb/Takeaway.jpg
...
```

### Folder Naming Convention

| Suffix | Detection Method |
|--------|------------------|
| `_rb` | Rule-based (wrist trajectory analysis) |
| `_nn` | Neural network (Bi-LSTM classifier) |

Example:
```
data/keyframes/
├── golf_swing_001_rb/     # Rule-based detection
│   ├── Address.jpg
│   ├── Takeaway.jpg
│   └── golf_swing_001_cleaned_8phases.csv
├── golf_swing_001_nn/     # Neural network detection
│   ├── Address.jpg
│   ├── Takeaway.jpg
│   └── golf_swing_001_cleaned_8phases.csv
```

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
- [x] Neural phase classifier (PoseSwingNet)
- [x] GolfDB training pipeline
- [x] Confidence-based phase detection
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
- [GolfDB Dataset](https://github.com/wmcnally/golfdb)
- [Golf Swing Biomechanics Research](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6413833/)
