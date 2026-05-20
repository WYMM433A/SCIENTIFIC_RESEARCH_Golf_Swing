"""Trace hip_rotation_3d step by step"""
import pandas as pd
from src.biomechanics.angles import GolfBiomechanics
import numpy as np
import math

df = pd.read_csv("data/extracted_poses/min_indoor_cleaned_poses.csv")

bio = GolfBiomechanics()
bio.df = df

frame = 50

# Get the points
left_hip = bio._get_point_from_df(frame, 'left_hip')
right_hip = bio._get_point_from_df(frame, 'right_hip')

print(f"Points from _get_point_from_df:")
print(f"  left_hip: {left_hip}")
print(f"  right_hip: {right_hip}")
print(f"  len(left_hip): {len(left_hip)}")
print(f"  len(right_hip): {len(right_hip)}")
print()

# Manual step-by-step calc
dx = right_hip[0] - left_hip[0]
dz = right_hip[2] - left_hip[2]

print(f"Calculation steps:")
print(f"  dx = right_hip[0] - left_hip[0]")
print(f"  dx = {right_hip[0]} - {left_hip[0]}")
print(f"  dx = {dx}")
print()

print(f"  dz = right_hip[2] - left_hip[2]")
print(f"  dz = {right_hip[2]} - {left_hip[2]}")
print(f"  dz = {dz}")
print()

angle_rad = math.atan2(dz, dx)
print(f"  angle_rad = atan2(dz, dx)")
print(f"  angle_rad = atan2({dz}, {dx})")
print(f"  angle_rad = {angle_rad}")
print()

angle_deg = math.degrees(angle_rad)
print(f"  angle_deg = degrees(angle_rad)")
print(f"  angle_deg = {angle_deg}")
print()

clipped = np.clip(angle_deg, 0, 180)
print(f"  clipped = np.clip(angle_deg, 0, 180)")
print(f"  clipped = np.clip({angle_deg}, 0, 180)")
print(f"  clipped = {clipped}")
print()

final = abs(clipped)
print(f"  final = abs(clipped)")
print(f"  final = abs({clipped})")
print(f"  final = {final}")
print()

# Now call the actual method
result = bio.get_hip_rotation_3d(frame=frame)
print(f"Method result: {result}°")
