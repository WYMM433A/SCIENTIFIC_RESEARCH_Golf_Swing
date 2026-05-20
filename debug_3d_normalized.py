# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from src.biomechanics.angles import GolfBiomechanics

# Load the min2 pose data
df = pd.read_csv("data/extracted_poses/min_indoor_cleaned_poses.csv")

bio = GolfBiomechanics()
bio.df = df

# Test at frame 50 (early swing)
frame = 50
row = df[df['frame'] == frame].iloc[0]

print(f"Frame {frame}:")
print(f"  left_shoulder_x: {row.get('left_shoulder_x', 'N/A')}")
print(f"  left_shoulder_z: {row.get('left_shoulder_z', 'N/A')}")
print(f"  right_shoulder_x: {row.get('right_shoulder_x', 'N/A')}")
print(f"  right_shoulder_z: {row.get('right_shoulder_z', 'N/A')}")
print()

# Get points to verify extraction
left_shoulder = bio._get_point_from_df(frame, 'left_shoulder')
right_shoulder = bio._get_point_from_df(frame, 'right_shoulder')
left_hip = bio._get_point_from_df(frame, 'left_hip')
right_hip = bio._get_point_from_df(frame, 'right_hip')

print(f"Extracted points:")
print(f"  left_shoulder: {left_shoulder}")
print(f"  right_shoulder: {right_shoulder}")
print(f"  left_hip: {left_hip}")
print(f"  right_hip: {right_hip}")
print()

# Calculate metrics
shoulder_rot_3d = bio.get_shoulder_rotation_3d(frame=frame)
hip_rot_3d = bio.get_hip_rotation_3d(frame=frame)
x_factor_3d = bio.get_x_factor_3d(frame=frame)

print(f"3D Rotations:")
print(f"  shoulder_rotation_3d: {shoulder_rot_3d:.2f}")
print(f"  hip_rotation_3d: {hip_rot_3d:.2f}")
print(f"  x_factor_3d: {x_factor_3d:.2f}")
print()

# Also test 2D for comparison
shoulder_rot_2d = bio.get_shoulder_rotation(frame=frame)
hip_rot_2d = bio.get_hip_rotation(frame=frame)
x_factor_2d = bio.get_x_factor(frame=frame)

print(f"2D Rotations (for comparison):")
print(f"  shoulder_rotation: {shoulder_rot_2d:.2f}")
print(f"  hip_rotation: {hip_rot_2d:.2f}")
print(f"  x_factor: {x_factor_2d:.2f}")
