"""Debug 3D rotations - detailed analysis"""
import pandas as pd
from src.biomechanics.angles import GolfBiomechanics
import math
import numpy as np

# Load pose data
df = pd.read_csv("data/extracted_poses/min_indoor_cleaned_poses.csv")

bio = GolfBiomechanics()
bio.df = df

# Test at frame 50 (early swing)
frame = 50
row = df[df['frame'] == frame].iloc[0]

print(f"Frame {frame}:")
print(f"  left_shoulder_x: {row['left_shoulder_x']:.4f}")
print(f"  left_shoulder_z: {row['left_shoulder_z']:.4f}")
print(f"  right_shoulder_x: {row['right_shoulder_x']:.4f}")
print(f"  right_shoulder_z: {row['right_shoulder_z']:.4f}")
print(f"  left_hip_x: {row['left_hip_x']:.4f}")
print(f"  left_hip_z: {row['left_hip_z']:.4f}")
print(f"  right_hip_x: {row['right_hip_x']:.4f}")
print(f"  right_hip_z: {row['right_hip_z']:.4f}")
print()

# Get points (should be in pixel space with Z scaled by 640)
left_shoulder = bio._get_point_from_df(frame, 'left_shoulder')
right_shoulder = bio._get_point_from_df(frame, 'right_shoulder')
left_hip = bio._get_point_from_df(frame, 'left_hip')
right_hip = bio._get_point_from_df(frame, 'right_hip')

print(f"Extracted points (pixel-space with Z scaled):")
print(f"  left_shoulder: x={left_shoulder[0]:.2f}, y={left_shoulder[1]:.2f}, z={left_shoulder[2]:.2f}")
print(f"  right_shoulder: x={right_shoulder[0]:.2f}, y={right_shoulder[1]:.2f}, z={right_shoulder[2]:.2f}")
print(f"  left_hip: x={left_hip[0]:.2f}, y={left_hip[1]:.2f}, z={left_hip[2]:.2f}")
print(f"  right_hip: x={right_hip[0]:.2f}, y={right_hip[1]:.2f}, z={right_hip[2]:.2f}")
print()

# Manual calculation for shoulder
dx_shoulder = right_shoulder[0] - left_shoulder[0]
dz_shoulder = right_shoulder[2] - left_shoulder[2]
angle_shoulder_rad = math.atan2(dz_shoulder, dx_shoulder)
angle_shoulder_deg = math.degrees(angle_shoulder_rad)

print(f"Shoulder rotation (manual 3D calc):")
print(f"  dx (right - left) = {dx_shoulder:.2f}")
print(f"  dz (right - left) = {dz_shoulder:.2f}")
print(f"  atan2(dz, dx) radians = {angle_shoulder_rad:.4f}")
print(f"  atan2(dz, dx) degrees = {angle_shoulder_deg:.2f}°")
print(f"  abs(angle) = {abs(angle_shoulder_deg):.2f}°")
print()

# Manual calculation for hip
dx_hip = right_hip[0] - left_hip[0]
dz_hip = right_hip[2] - left_hip[2]
angle_hip_rad = math.atan2(dz_hip, dx_hip)
angle_hip_deg = math.degrees(angle_hip_rad)

print(f"Hip rotation (manual 3D calc):")
print(f"  dx (right - left) = {dx_hip:.2f}")
print(f"  dz (right - left) = {dz_hip:.2f}")
print(f"  atan2(dz, dx) radians = {angle_hip_rad:.4f}")
print(f"  atan2(dz, dx) degrees = {angle_hip_deg:.2f}°")
print(f"  abs(angle) = {abs(angle_hip_deg):.2f}°")
print()

# Calculate metrics using methods
shoulder_rot_3d = bio.get_shoulder_rotation_3d(frame=frame)
hip_rot_3d = bio.get_hip_rotation_3d(frame=frame)
x_factor_3d = bio.get_x_factor_3d(frame=frame)

print(f"3D Rotations (from methods):")
print(f"  shoulder_rotation_3d: {shoulder_rot_3d:.2f}°")
print(f"  hip_rotation_3d: {hip_rot_3d:.2f}°")
print(f"  x_factor_3d: {x_factor_3d:.2f}°")
print()

# 2D for comparison
shoulder_rot_2d = bio.get_shoulder_rotation(frame=frame)
hip_rot_2d = bio.get_hip_rotation(frame=frame)
x_factor_2d = bio.get_x_factor(frame=frame)

print(f"2D Rotations:")
print(f"  shoulder_rotation: {shoulder_rot_2d:.2f}°")
print(f"  hip_rotation: {hip_rot_2d:.2f}°")
print(f"  x_factor: {x_factor_2d:.2f}°")
