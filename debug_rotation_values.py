"""
Debug script to check actual landmark coordinates and why rotation values are the same.
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

# Get TOP frame
phase_frames = {}
for _, row in phases_df.iterrows():
    phase_name = row['Phase'].lower().replace('-', '_')
    phase_frames[phase_name] = row['Key_Frame']

top_frame = phase_frames['top']
row = poses_df.iloc[top_frame]

print(f"TOP PHASE - Frame {top_frame}")
print("=" * 80)

# Get landmark coordinates
left_shoulder = np.array([row['left_shoulder_x'], row['left_shoulder_y']])
right_shoulder = np.array([row['right_shoulder_x'], row['right_shoulder_y']])
left_hip = np.array([row['left_hip_x'], row['left_hip_y']])
right_hip = np.array([row['right_hip_x'], row['right_hip_y']])

print("\nLANDMARK COORDINATES (pixel positions):")
print(f"  Left Shoulder:  ({left_shoulder[0]:7.2f}, {left_shoulder[1]:7.2f})")
print(f"  Right Shoulder: ({right_shoulder[0]:7.2f}, {right_shoulder[1]:7.2f})")
print(f"  Left Hip:       ({left_hip[0]:7.2f}, {left_hip[1]:7.2f})")
print(f"  Right Hip:      ({right_hip[0]:7.2f}, {right_hip[1]:7.2f})")

# Calculate line angles
shoulder_line_angle = calculate_line_angle(left_shoulder, right_shoulder)
hip_line_angle = calculate_line_angle(left_hip, right_hip)

print("\nLINE ANGLES FROM HORIZONTAL:")
print(f"  Shoulder line angle: {shoulder_line_angle:7.2f}°")
print(f"  Hip line angle:      {hip_line_angle:7.2f}°")

# Calculate absolute values (what the code does)
abs_shoulder = abs(shoulder_line_angle)
abs_hip = abs(hip_line_angle)

print("\nABSOLUTE VALUES (what's reported):")
print(f"  shoulder_rotation: {abs_shoulder:7.2f}°")
print(f"  hip_rotation:      {abs_hip:7.2f}°")
print(f"  x_factor (difference): {abs(abs_shoulder - abs_hip):7.2f}°")

# Distance between shoulders vs hips
shoulder_distance = np.linalg.norm(right_shoulder - left_shoulder)
hip_distance = np.linalg.norm(right_hip - left_hip)

print("\nDISTANCES:")
print(f"  Shoulder width: {shoulder_distance:7.2f} pixels")
print(f"  Hip width:      {hip_distance:7.2f} pixels")

print("\n" + "=" * 80)
print("ANALYSIS:")
print("If shoulder_rotation ≈ hip_rotation, the line angles are similar.")
print("This happens when BOTH rotate together (expected in front-view camera).")
print("To detect hip-shoulder separation, we need to measure 3D body rotation,")
print("which we can't do from a single 2D front-view camera.")
