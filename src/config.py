# config.py
"""
Configuration file for SwingAI Coach
Modify paths and parameters here
"""

import os

# Project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Data directories
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
RAW_VIDEOS_DIR = os.path.join(DATA_DIR, 'raw_videos')
EXTRACTED_POSES_DIR = os.path.join(DATA_DIR, 'extracted_poses')
VISUALIZATIONS_DIR = os.path.join(DATA_DIR, 'visualizations')
METRICS_DIR = os.path.join(DATA_DIR, 'metrics')

# Model directories
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')

# MediaPipe Parameters
MEDIAPIPE_CONFIG = {
    'model_complexity': 1,  # 0=Lite, 1=Full, 2=Heavy
    'smooth_landmarks': True,
    'min_detection_confidence': 0.5,
    'min_tracking_confidence': 0.5
}

# Visualization Settings
SHOW_LIVE_PREVIEW = True
SAVE_VISUALIZATION = True
DRAW_LANDMARKS = True

# Video Processing
FRAME_SKIP = 1  # Process every Nth frame (1 = process all)

# Output Settings
SAVE_CSV = True
SAVE_METRICS = True

# Create directories if they don't exist
for directory in [RAW_VIDEOS_DIR, EXTRACTED_POSES_DIR, 
                  VISUALIZATIONS_DIR, METRICS_DIR, MODELS_DIR]:
    os.makedirs(directory, exist_ok=True)