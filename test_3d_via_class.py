"""Test 3D rotation using GolfBiomechanics class to verify it's working end-to-end."""
import pandas as pd
from src.biomechanics.angles import GolfBiomechanics

# Load raw pose data
poses_df = pd.read_csv("data/extracted_poses/min2_cleaned_poses.csv")

# Initialize biomechanics
biomechanics = GolfBiomechanics()
biomechanics.df = poses_df

# Test at frame 42 (TOP phase)
frame = 42

print(f"Testing 3D Rotations via GolfBiomechanics class (Frame {frame})")
print("=" * 70)

# Get 2D values
shoulder_2d = biomechanics.get_shoulder_rotation(frame=frame)
hip_2d = biomechanics.get_hip_rotation(frame=frame)
xfactor_2d = biomechanics.get_x_factor(frame=frame)

print(f"2D METHOD (using Y-height):")
print(f"  Shoulder: {shoulder_2d:.2f}°")
print(f"  Hip:      {hip_2d:.2f}°")
print(f"  X-factor: {xfactor_2d:.2f}°")

# Get 3D values
shoulder_3d = biomechanics.get_shoulder_rotation_3d(frame=frame)
hip_3d = biomechanics.get_hip_rotation_3d(frame=frame)
xfactor_3d = biomechanics.get_x_factor_3d(frame=frame)

print(f"\n3D METHOD (using Z-depth):")
print(f"  Shoulder: {shoulder_3d:.2f}°")
print(f"  Hip:      {hip_3d:.2f}°")
print(f"  X-factor: {xfactor_3d:.2f}°")

print(f"\nIMPROVEMENT:")
print(f"  X-factor improved by: {xfactor_3d - xfactor_2d:.2f}° ({((xfactor_3d / max(xfactor_2d, 0.01)) - 1) * 100:.0f}% better)")

if xfactor_3d > 5:
    print("\n✓ 3D rotations are working! Z-coordinates are being properly extracted.")
else:
    print("\n✗ 3D rotations still returning low values - Z extraction may still be broken.")
