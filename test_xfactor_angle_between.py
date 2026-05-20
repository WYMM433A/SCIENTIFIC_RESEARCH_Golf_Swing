"""
Test alternative x_factor calculation using angle between shoulder and hip lines.
"""

import numpy as np
import math
from pathlib import Path
import pandas as pd

def angle_between_vectors(v1: np.ndarray, v2: np.ndarray) -> float:
    """Calculate angle between two 2D vectors in degrees."""
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
    angle = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))
    return angle

poses_path = Path("data/extracted_poses/min_indoor_cleaned_poses.csv")
phases_path = Path("data/keyframes/min_indoor_nn/min_indoor_cleaned_8phases.csv")

poses_df = pd.read_csv(poses_path)
phases_df = pd.read_csv(phases_path)

phase_frames = {}
for _, row in phases_df.iterrows():
    phase_name = row['Phase'].lower().replace('-', '_')
    phase_frames[phase_name] = row['Key_Frame']

phases_to_check = ["address", "takeaway", "mid_backswing", "top"]

print("COMPARING X_FACTOR CALCULATION METHODS")
print("=" * 100)

for phase_name in phases_to_check:
    if phase_name not in phase_frames:
        continue
    
    frame = phase_frames[phase_name]
    row = poses_df.iloc[frame]
    
    # Get coordinates
    left_shoulder = np.array([row['left_shoulder_x'], row['left_shoulder_y']])
    right_shoulder = np.array([row['right_shoulder_x'], row['right_shoulder_y']])
    left_hip = np.array([row['left_hip_x'], row['left_hip_y']])
    right_hip = np.array([row['right_hip_x'], row['right_hip_y']])
    
    # Current method: difference of line angles
    def line_angle(p1, p2):
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        return math.degrees(math.atan2(dy, dx))
    
    shoulder_angle = abs(line_angle(left_shoulder, right_shoulder))
    hip_angle = abs(line_angle(left_hip, right_hip))
    current_xfactor = abs(shoulder_angle - hip_angle)
    
    # NEW method: angle between the two line vectors
    shoulder_vector = right_shoulder - left_shoulder
    hip_vector = right_hip - left_hip
    new_xfactor = angle_between_vectors(shoulder_vector, hip_vector)
    
    print(f"\n{phase_name.upper()} (Frame {frame})")
    print(f"  Current X-factor (angle difference):  {current_xfactor:7.2f}°")
    print(f"  NEW X-factor (angle between lines):   {new_xfactor:7.2f}°")
    print(f"  Improvement: {new_xfactor - current_xfactor:+7.2f}°")
    
    # Show vectors
    print(f"  Shoulder vector: ({shoulder_vector[0]:7.4f}, {shoulder_vector[1]:7.4f})")
    print(f"  Hip vector:      ({hip_vector[0]:7.4f}, {hip_vector[1]:7.4f})")

print("\n" + "=" * 100)
print("If NEW method shows 0.01° → larger number at TOP, this approach works better!")
