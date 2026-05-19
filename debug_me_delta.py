import pandas as pd
from src.biomechanics.angles import GolfBiomechanics

# Load pose data
pose_csv = 'data/extracted_poses/golf_swing_001_cleaned_poses.csv'
df = pd.read_csv(pose_csv)

# Use correct frames from pipeline
phases_csv = 'data/keyframes/golf_swing_001_nn/golf_swing_001_cleaned_8phases.csv'
phases_df = pd.read_csv(phases_csv)
phases_dict = dict(zip(phases_df['Phase'], phases_df['Key_Frame']))

biomech = GolfBiomechanics(df)

top_frame = phases_dict['Top']
impact_frame = phases_dict['Impact']

print(f"ARM EXTENSION DELTA - me.mp4")
print("="*70)
print(f"Top frame:    {top_frame}")
print(f"Impact frame: {impact_frame}")

# Get lead arm angles
top_arm = biomech.get_lead_arm_angle(frame=top_frame)
impact_arm = biomech.get_lead_arm_angle(frame=impact_frame)
delta = impact_arm - top_arm

print(f"\nLead arm angles:")
print(f"  At top ({top_frame}):    {top_arm:.2f}°")
print(f"  At impact ({impact_frame}): {impact_arm:.2f}°")
print(f"  Delta:                  {delta:.2f}°")

# Calculate quality using ideal delta = 28
ideal_delta = 28
delta_error = abs(delta - ideal_delta)
quality = max(0, 100 - (delta_error * 5))

print(f"\nArm extension delta quality:")
print(f"  Ideal delta: {ideal_delta}°")
print(f"  Actual delta: {delta:.2f}°")
print(f"  Error: {delta_error:.2f}°")
print(f"  Quality score: {quality:.2f}/100")
print(f"\nWeighted component:")
print(f"  Weight: 0.75 (combined x_factor_unwind + arm_extension)")
print(f"  Weighted score: {quality * 0.75:.2f}")
