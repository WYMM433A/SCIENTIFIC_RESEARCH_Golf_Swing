"""Debug why hip_rotation_3d returns 0.00 when manual calc gives 90.10"""
import pandas as pd
from src.biomechanics.angles import GolfBiomechanics
import numpy as np

df = pd.read_csv("data/extracted_poses/min_indoor_cleaned_poses.csv")

bio = GolfBiomechanics()
bio.df = df

frame = 50

# Get the points manually
left_hip = bio._get_point_from_df(frame, 'left_hip')
right_hip = bio._get_point_from_df(frame, 'right_hip')

print(f"Left hip point: {left_hip}")
print(f"Right hip point: {right_hip}")
print(f"len(left_hip): {len(left_hip)}")
print(f"len(right_hip): {len(right_hip)}")
print()

# Check the fallback condition
print(f"Fallback check: len(left_hip) < 3 or len(right_hip) < 3")
print(f"  {len(left_hip)} < 3 or {len(right_hip)} < 3 = {len(left_hip) < 3 or len(right_hip) < 3}")
print()

# Call the methods
print(f"hip_rotation_3d (should be ~90): {bio.get_hip_rotation_3d(frame=frame):.2f}°")
print(f"hip_rotation (2D fallback): {bio.get_hip_rotation(frame=frame):.2f}°")
