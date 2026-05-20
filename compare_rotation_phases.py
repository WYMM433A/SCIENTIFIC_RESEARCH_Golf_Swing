"""
Compare rotation values across all phases to understand why they're different early but same at TOP.
"""

import pandas as pd
import numpy as np
import math
from pathlib import Path

def calculate_line_angle(p1: np.ndarray, p2: np.ndarray) -> float:
    """Calculate angle of line from horizontal."""
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    angle = math.degrees(math.atan2(dy, dx))
    return angle

poses_path = Path("data/extracted_poses/min_indoor_cleaned_poses.csv")
phases_path = Path("data/keyframes/min_indoor_nn/min_indoor_cleaned_8phases.csv")

poses_df = pd.read_csv(poses_path)
phases_df = pd.read_csv(phases_path)

# Get phase frames
phase_frames = {}
for _, row in phases_df.iterrows():
    phase_name = row['Phase'].lower().replace('-', '_')
    phase_frames[phase_name] = row['Key_Frame']

phases_to_check = ["address", "takeaway", "mid_backswing", "top"]

print("ROTATION METRICS ACROSS PHASES")
print("=" * 100)

for phase_name in phases_to_check:
    if phase_name not in phase_frames:
        continue
    
    frame = phase_frames[phase_name]
    row = poses_df.iloc[frame]
    
    # Get landmark coordinates
    left_shoulder = np.array([row['left_shoulder_x'], row['left_shoulder_y']])
    right_shoulder = np.array([row['right_shoulder_x'], row['right_shoulder_y']])
    left_hip = np.array([row['left_hip_x'], row['left_hip_y']])
    right_hip = np.array([row['right_hip_x'], row['right_hip_y']])
    
    # Calculate line angles
    shoulder_angle = abs(calculate_line_angle(left_shoulder, right_shoulder))
    hip_angle = abs(calculate_line_angle(left_hip, right_hip))
    
    # Vertical differences (dy)
    shoulder_dy = right_shoulder[1] - left_shoulder[1]
    hip_dy = right_hip[1] - left_hip[1]
    
    print(f"\n{phase_name.upper()} (Frame {frame})")
    print(f"  Shoulder line: {shoulder_angle:7.2f}°  (dy = {shoulder_dy:7.4f})")
    print(f"  Hip line:      {hip_angle:7.2f}°  (dy = {hip_dy:7.4f})")
    print(f"  X-factor:      {abs(shoulder_angle - hip_angle):7.2f}°")
    
    print(f"\n  Coordinates:")
    print(f"    L-Shoulder: ({left_shoulder[0]:.3f}, {left_shoulder[1]:.3f})")
    print(f"    R-Shoulder: ({right_shoulder[0]:.3f}, {right_shoulder[1]:.3f})")
    print(f"    L-Hip:      ({left_hip[0]:.3f}, {left_hip[1]:.3f})")
    print(f"    R-Hip:      ({right_hip[0]:.3f}, {right_hip[1]:.3f})")

print("\n" + "=" * 100)
print("KEY INSIGHT:")
print("When dy (vertical difference) ≈ 0 for BOTH lines, both angles collapse to ~180°")
print("This happens when rotation is perpendicular to camera (most at TOP)")
