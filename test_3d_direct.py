"""Quick test of 3D rotation calculation on actual data."""
import pandas as pd
import math
import numpy as np

# Load raw pose data
poses_df = pd.read_csv("data/extracted_poses/min2_cleaned_poses.csv")

# Get a frame (frame 42 which should be TOP)
frame = 42
row = poses_df[poses_df['frame'] == frame].iloc[0]

print(f"Frame {frame} - TOP phase")
print(f"Left Shoulder:  x={row['left_shoulder_x']:.4f}, y={row['left_shoulder_y']:.4f}, z={row['left_shoulder_z']:.4f}")
print(f"Right Shoulder: x={row['right_shoulder_x']:.4f}, y={row['right_shoulder_y']:.4f}, z={row['right_shoulder_z']:.4f}")
print(f"Left Hip:       x={row['left_hip_x']:.4f}, y={row['left_hip_y']:.4f}, z={row['left_hip_z']:.4f}")
print(f"Right Hip:      x={row['right_hip_x']:.4f}, y={row['right_hip_y']:.4f}, z={row['right_hip_z']:.4f}")

# 2D calculation (using Y)
dx_shoulder_2d = row['right_shoulder_x'] - row['left_shoulder_x']
dy_shoulder_2d = row['right_shoulder_y'] - row['left_shoulder_y']
shoulder_2d = abs(math.degrees(math.atan2(dy_shoulder_2d, dx_shoulder_2d)))

dx_hip_2d = row['right_hip_x'] - row['left_hip_x']
dy_hip_2d = row['right_hip_y'] - row['left_hip_y']
hip_2d = abs(math.degrees(math.atan2(dy_hip_2d, dx_hip_2d)))

xfactor_2d = abs(shoulder_2d - hip_2d)

print(f"\n2D Calculations (using Y):")
print(f"  Shoulder 2D: {shoulder_2d:.2f}°")
print(f"  Hip 2D:      {hip_2d:.2f}°")
print(f"  X-factor 2D: {xfactor_2d:.2f}°")

# 3D calculation (using Z)
dx_shoulder_3d = row['right_shoulder_x'] - row['left_shoulder_x']
dz_shoulder_3d = row['right_shoulder_z'] - row['left_shoulder_z']
shoulder_3d = abs(math.degrees(math.atan2(dz_shoulder_3d, dx_shoulder_3d)))

dx_hip_3d = row['right_hip_x'] - row['left_hip_x']
dz_hip_3d = row['right_hip_z'] - row['left_hip_z']
hip_3d = abs(math.degrees(math.atan2(dz_hip_3d, dx_hip_3d)))

xfactor_3d = abs(shoulder_3d - hip_3d)

print(f"\n3D Calculations (using Z):")
print(f"  Shoulder 3D: {shoulder_3d:.2f}°  (dx={dx_shoulder_3d:.4f}, dz={dz_shoulder_3d:.4f})")
print(f"  Hip 3D:      {hip_3d:.2f}°  (dx={dx_hip_3d:.4f}, dz={dz_hip_3d:.4f})")
print(f"  X-factor 3D: {xfactor_3d:.2f}°")

print(f"\nImprovement: {xfactor_3d - xfactor_2d:+.2f}°")
