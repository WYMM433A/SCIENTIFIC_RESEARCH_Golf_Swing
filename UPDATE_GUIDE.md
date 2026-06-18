# Scientific Research Guide: SwingAI Coach v2.0

This document provides technical details and execution instructions for the SwingAI Coach system, designed for high-accuracy biomechanical analysis of golf swings.

## 1. System Architecture
The system utilizes a hybrid approach combining Computer Vision and Deep Learning:
- **Pose Estimation:** MediaPipe Pose Landmarker for 33 high-fidelity body keypoints.
- **Phase Classification:** A Bidirectional LSTM (Bi-LSTM) trained on the GolfDB dataset to identify 8 critical swing events (Address to Finish).
- **Biomechanics:** 3D vector geometry projected onto the X-Z horizontal plane to ensure rotation accuracy independent of camera perspective.

## 2. Key Methodologies
### 3.D Kinematics
Rotations (Shoulder Turn, Hip Turn) are calculated relative to a baseline vector captured at the **Address** phase. This removes errors introduced by non-perpendicular camera placement.

### Stability Metric
**Head Stability** is quantified as the Euclidean norm of the standard deviation of the head (nose) landmark positions across all frames:
$$S_{head} = \sqrt{\sigma_{x}^2 + \sigma_{y}^2 + \sigma_{z}^2}$$

## 3. Running the Pipeline

### End-to-End Analysis
To process a raw video through motion cleaning, pose extraction, phase detection, and evaluation:
```bash
python pipeline.py path/to/video.mp4 --method neural-network --level amateur
```

### Generating Visualization
Once the pipeline completes, generate the academic-grade Radar Chart:
```bash
python tools/visualize_report.py
```

## 4. Output Data Structure
- `data/extracted_poses/`: CSV files containing normalized 3D coordinates.
- `data/metrics/`: 
    - `*_evaluation.json`: Raw metric deviations and priority fix logs.
    - `*_evaluation_radar.png`: The visual biomechanical performance profile.
- `data/keyframes/`: Individual frame captures for each detected swing phase.

