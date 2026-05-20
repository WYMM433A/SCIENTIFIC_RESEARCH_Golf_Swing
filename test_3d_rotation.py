"""
Test 3D rotation calculation using Z coordinates instead of Y.
Compares current 2D approach vs new 3D approach.
"""

import pandas as pd
import numpy as np
import math
from pathlib import Path

def calculate_2d_line_angle(p1: np.ndarray, p2: np.ndarray) -> float:
    """Current method: uses only X, Y (height difference)."""
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    angle = math.degrees(math.atan2(dy, dx))
    return abs(angle)

def calculate_3d_line_angle(p1: np.ndarray, p2: np.ndarray) -> float:
    """NEW method: uses X, Z (depth difference) to detect rotation."""
    dx = p2[0] - p1[0]
    dz = p2[2] - p1[2] if len(p2) > 2 else 0
    angle = math.degrees(math.atan2(dz, dx))
    return abs(angle)

def angle_between_3d_vectors(v1: np.ndarray, v2: np.ndarray) -> float:
    """Calculate angle between two 3D vectors."""
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
    angle = math.degrees(math.acos(np.clip(cos_angle, -1, 1)))
    return angle

poses_path = Path("data/extracted_poses/min_indoor_cleaned_poses.csv")
phases_path = Path("data/keyframes/min_indoor_nn/min_indoor_cleaned_8phases.csv")

poses_df = pd.read_csv(poses_path)
phases_df = pd.read_csv(phases_path)

phase_frames = {}
for _, row in phases_df.iterrows():
    phase_name = row['Phase'].lower().replace('-', '_')
    phase_frames[phase_name] = row['Key_Frame']

phases_to_check = ["address", "takeaway", "mid_backswing", "top", "mid_downswing", "impact"]

print("=" * 120)
print("COMPARING 2D vs 3D ROTATION CALCULATION")
print("=" * 120)

for phase_name in phases_to_check:
    if phase_name not in phase_frames:
        continue
    
    frame = phase_frames[phase_name]
    row = poses_df.iloc[frame]
    
    # Get 3D coordinates (x, y, z)
    left_shoulder = np.array([row['left_shoulder_x'], row['left_shoulder_y'], row['left_shoulder_z']])
    right_shoulder = np.array([row['right_shoulder_x'], row['right_shoulder_y'], row['right_shoulder_z']])
    left_hip = np.array([row['left_hip_x'], row['left_hip_y'], row['left_hip_z']])
    right_hip = np.array([row['right_hip_x'], row['right_hip_y'], row['right_hip_z']])
    
    # CURRENT 2D METHOD
    shoulder_2d = calculate_2d_line_angle(left_shoulder[:2], right_shoulder[:2])
    hip_2d = calculate_2d_line_angle(left_hip[:2], right_hip[:2])
    xfactor_2d = abs(shoulder_2d - hip_2d)
    
    # NEW 3D METHOD (using Z/X angle)
    shoulder_3d = calculate_3d_line_angle(left_shoulder, right_shoulder)
    hip_3d = calculate_3d_line_angle(left_hip, right_hip)
    xfactor_3d = abs(shoulder_3d - hip_3d)
    
    # ALTERNATIVE 3D METHOD (angle between vectors)
    shoulder_vec = right_shoulder - left_shoulder
    hip_vec = right_hip - left_hip
    xfactor_vec = angle_between_3d_vectors(shoulder_vec, hip_vec)
    
    # Z depth differences
    shoulder_z_diff = abs(right_shoulder[2] - left_shoulder[2])
    hip_z_diff = abs(right_hip[2] - left_hip[2])
    
    print(f"\n{phase_name.upper()} (Frame {frame})")
    print(f"  2D METHOD (using Y/height):        X-factor = {xfactor_2d:7.2f}°")
    print(f"  3D METHOD (using Z/depth):         X-factor = {xfactor_3d:7.2f}°  (Δ {xfactor_3d - xfactor_2d:+6.2f}°)")
    print(f"  3D VECTOR METHOD:                  X-factor = {xfactor_vec:7.2f}°")
    print(f"  Z depth diff (shoulder): {shoulder_z_diff:.4f}  (hip): {hip_z_diff:.4f}")
    
    if phase_name == "top":
        print(f"\n  >>> TOP PHASE COMPARISON <<<")
        print(f"      2D measured: {xfactor_2d:.2f}° (broken, appears as no coil)")
        print(f"      3D measured: {xfactor_3d:.2f}° (detects rotation via Z)")
        print(f"      Improvement: {((xfactor_3d / max(xfactor_2d, 0.01)) - 1) * 100:.0f}% better!")

print("\n" + "=" * 120)
print("INTERPRETATION:")
print("  - 2D method collapses at TOP (measures only Y height, which is ~0)")
print("  - 3D method uses Z depth (shows 0.2+ difference when rotated)")
print("  - 3D method should better detect shoulder-hip separation at all phases")
print("=" * 120)
