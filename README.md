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

## 📌 Current Status (May 2026)

The project is now running an end-to-end swing evaluation pipeline with both phase detection and phase scoring in production.

### What is working now

- Full pipeline from raw video to scored 8-phase output
- Rule-based and neural-network phase segmentation via a unified adapter
- Per-phase biomechanical scoring with weighted metrics
- Human-readable coaching feedback in score output
- Detailed component-level feedback export for diagnostics and UI

### Latest completed improvements

- Actionable feedback upgrade:
    - Uses raw component scores for accurate issue labels
    - Includes measured value vs target range
    - Adds fix cue and drill prescription per issue
- New detailed diagnostics file:
    - `data/metrics/{video_name}_feedback_detailed.csv`
- Mid-downswing robustness improvements:
    - More stable kinematic sequence extraction in `src/biomechanics/angles.py`
    - Broader sequence window (Top-start to Impact-end) in scoring path
    - Phase/keyframe lookup normalization in `pipeline.py` to avoid silent fallback errors

### Current known direction

- Front-view scoring quality and feedback are the current priority.
- View-aware front/side normalization is intentionally deferred for a later milestone.

---

## 🧭 Technical Deep Dive

### 1. End-to-End Pipeline Architecture

Main entry: `pipeline.py`

Processing flow:

1. Video cleaning (`src/video/cleaner.py`)
2. Pose extraction (`src/pose/analyzer.py`, `src/pose/detector.py`)
3. Phase segmentation (`src/phase/adapter.py` -> rule-based or neural)
4. Biomechanical scoring (`src/biomechanics/angles.py`, `src/biomechanics/phase_scorer.py`)
5. Feedback generation (summary in scores CSV + detailed diagnostics CSV)
6. Keyframe export and summary report

Primary outputs:

- `data/cleaned_videos/{name}_cleaned.mp4`
- `data/extracted_poses/{name}_cleaned_poses.csv`
- `data/metrics/{name}_cleaned_metrics.csv`
- `data/metrics/{name}_scores.csv`
- `data/metrics/{name}_feedback_detailed.csv`
- `data/keyframes/{name}_rb/` or `data/keyframes/{name}_nn/`

### 2. How Phase Segmentation Is Done

`src/phase/adapter.py` exposes one interface and dispatches to two implementations.

#### Rule-based segmentation (`src/phase/rule_based.py`)

- Uses right-wrist Y trajectory over time
- Smooths wrist signal and computes velocity magnitude
- Detects swing boundaries from motion thresholding
- Builds 8 phases with a hybrid strategy:
    - Address and Finish: stability windows
    - Top: peak-based event
    - Remaining phases: time-proportional partitions between landmarks

Best for deterministic behavior and no model dependency.

#### Neural-network segmentation (`src/phase/neural.py` + loaded model)

- Uses normalized per-frame pose features (132 dims: 33 landmarks x 4)
- Bi-LSTM predicts class probabilities per frame
- Peak-confidence frame is chosen per phase
- Each phase gets a small window around the peak
- Temporal order correction enforces Address -> ... -> Finish chronology

Best for learned timing patterns and noisy real-world swings.

### 3. How Analysis and Scoring Work Per Phase

Scoring engine: `src/biomechanics/phase_scorer.py`
Configuration: `src/biomechanics/scoring_config.py`

Scoring structure:

- Each phase has component metrics and weights (`METRIC_WEIGHTS`)
- Each metric is evaluated against target ranges (`SCORING_THRESHOLDS`)
- Component scores are combined into a normalized phase score (0-100)
- Overall score uses phase importance weights (`PHASE_WEIGHTS`)

#### Phase-by-phase factors

| Phase | Main factors used for scoring |
|------|-------------------------------|
| Address | posture, grip indicator, weight distribution |
| Takeaway | shoulder rotation initiation, hip lag, wrist position, club path/head stability |
| Mid-backswing | coil (x-factor), shoulder rotation, wrist hinge, shaft/lead-arm plane |
| Top | coil, posture retention, head stability, wrist set |
| Mid-downswing | kinematic sequence, lag retention, hip rotation drive, upper-body lag |
| Impact | lag release timing, x-factor unwind, arm extension, wrist structure, stability |
| Follow-through | deceleration pattern, posture, arm swing shape, rotation completion |
| Finish | balance, posture, final rotation, symmetry |

#### Mid-downswing sequence specifics

- Sequence scoring is the highest-weight component inside mid-downswing.
- Sequence extraction is computed from a broader transition window:
    - Top phase start -> Impact phase end
- Robustness logic in `angles.py` includes:
    - Angle unwrapping
    - Velocity smoothing
    - Persistent-onset detection (not one-frame spikes)
    - Stabilized x-factor stretch estimation

### 4. Feedback System Design

Summary feedback (`{name}_scores.csv`):

- One readable block per phase
- Prioritized issues only
- Includes fix cue + drill suggestion

Detailed feedback (`{name}_feedback_detailed.csv`):

- One row per phase component
- Raw score and weighted score
- Measured value and target range
- Delta from target, severity, priority
- Coaching cue and drill text

This dual-output design supports both:

- Coach/user-facing concise reports
- Developer/debugger-facing diagnostic analysis

### 5. Data and Control Flow in `pipeline.py`

At scoring time:

1. Set biomechanical reference frame at Address keyframe
2. For each phase, select keyframe and compute metrics
3. For Mid-downswing, additionally compute kinematic sequence data
4. Score phase through `PhaseScorer`
5. Generate summary feedback and detailed component diagnostics
6. Compute full-swing overall score and save all artifacts

Implementation notes:

- Phase dictionary keys are normalized (case/hyphen/space tolerant) before lookup
- This prevents bad fallback behavior when detector naming style differs

---

## 🗺️ Architecture Diagram

### System Flow

```mermaid
flowchart LR
    A[Raw Video<br/>data/raw_videos/*.mp4] --> B[Video Cleaner<br/>src/video/cleaner.py]
    B --> C[Cleaned Video<br/>data/cleaned_videos/*_cleaned.mp4]
    C --> D[Pose Extraction<br/>src/pose/detector.py + analyzer.py]
    D --> E[Pose CSV<br/>data/extracted_poses/*_cleaned_poses.csv]
    D --> F[Frame Metrics CSV<br/>data/metrics/*_cleaned_metrics.csv]

    E --> G[Phase Adapter<br/>src/phase/adapter.py]
    G --> H[Rule-Based Detector<br/>src/phase/rule_based.py]
    G --> I[Neural Detector (Bi-LSTM)<br/>src/phase/neural.py]
    H --> J[Phase Ranges + Keyframes]
    I --> J

    E --> K[Biomechanics Engine<br/>src/biomechanics/angles.py]
    J --> L[Phase Scorer<br/>src/biomechanics/phase_scorer.py]
    K --> L
    L --> M[Scores CSV<br/>data/metrics/*_scores.csv]
    L --> N[Detailed Feedback CSV<br/>data/metrics/*_feedback_detailed.csv]

    J --> O[Keyframe Export<br/>data/keyframes/*_rb or *_nn]
```

### Module Responsibility Map

| Layer | Main modules | Responsibility |
|------|--------------|----------------|
| Orchestration | `pipeline.py` | Runs all stages, tracks artifacts, prints summary |
| Video preprocessing | `src/video/cleaner.py` | Crops video to swing motion boundaries |
| Pose extraction | `src/pose/detector.py`, `src/pose/analyzer.py` | Extracts 33 landmarks/frame and derived frame metrics |
| Phase segmentation | `src/phase/adapter.py`, `src/phase/rule_based.py`, `src/phase/neural.py` | Produces 8 ordered phase ranges and keyframes |
| Biomechanics | `src/biomechanics/angles.py` | Computes golf-specific angles and kinematic sequence signals |
| Scoring + feedback | `src/biomechanics/phase_scorer.py`, `src/biomechanics/scoring_config.py` | Converts metrics to phase scores, overall score, and coaching feedback |
| Artifacts | `data/metrics/`, `data/keyframes/`, `data/extracted_poses/` | Stores analysis outputs for downstream UI/reporting/training |

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
- [x] Text feedback generation (summary + detailed diagnostics)
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
