"""
Shared Constants for DataStorm Golf Swing Analysis
===================================================
Centralized definitions used across the project.
"""

# 8 Golf Swing Phases
PHASE_NAMES = [
    "Address",
    "Takeaway",
    "Mid-backswing",
    "Top",
    "Mid-downswing",
    "Impact",
    "Follow-through",
    "Finish"
]

# Phase colors for visualization (BGR format for OpenCV)
PHASE_COLORS_BGR = [
    (255, 0, 0),      # Blue - Address
    (255, 128, 0),    # Light Blue - Takeaway
    (255, 255, 0),    # Cyan - Mid-backswing
    (0, 255, 0),      # Green - Top
    (0, 255, 128),    # Yellow-Green - Mid-downswing
    (0, 255, 255),    # Yellow - Impact
    (128, 0, 255),    # Orange - Follow-through
    (255, 0, 255)     # Pink - Finish
]

# Phase colors for matplotlib (RGB format)
PHASE_COLORS_RGB = [
    (0, 0, 1),        # Blue - Address
    (0, 0.5, 1),      # Light Blue - Takeaway
    (0, 1, 1),        # Cyan - Mid-backswing
    (0, 1, 0),        # Green - Top
    (0.5, 1, 0),      # Yellow-Green - Mid-downswing
    (1, 1, 0),        # Yellow - Impact
    (1, 0.5, 0),      # Orange - Follow-through
    (1, 0, 1)         # Pink - Finish
]

# MediaPipe Pose Landmark indices
LANDMARK_NAMES = [
    'nose', 'left_eye_inner', 'left_eye', 'left_eye_outer',
    'right_eye_inner', 'right_eye', 'right_eye_outer',
    'left_ear', 'right_ear', 'mouth_left', 'mouth_right',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_pinky', 'right_pinky',
    'left_index', 'right_index', 'left_thumb', 'right_thumb',
    'left_hip', 'right_hip', 'left_knee', 'right_knee',
    'left_ankle', 'right_ankle', 'left_heel', 'right_heel',
    'left_foot_index', 'right_foot_index'
]

# Number of pose features (33 landmarks × 4 values each)
NUM_LANDMARKS = 33
NUM_FEATURES = NUM_LANDMARKS * 4  # x, y, z, visibility = 132

# Model defaults
DEFAULT_MODEL_PATH = 'models/pose_swingnet_trained.pth'
POSE_MODEL_PATH = 'models/pose_landmarker_lite.task'
