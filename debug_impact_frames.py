import pandas as pd
from src.biomechanics.angles import GolfBiomechanics

# Load pose data
pose_csv = 'data/extracted_poses/B80O_cleaned_poses.csv'
df = pd.read_csv(pose_csv)

# Use correct frames from pipeline
phases_csv = 'data/keyframes/B80O_nn/B80O_cleaned_8phases.csv'
phases_df = pd.read_csv(phases_csv)
phases_dict = dict(zip(phases_df['Phase'], phases_df['Key_Frame']))

biomech = GolfBiomechanics(df)

top_frame = phases_dict['Top']
impact_frame = phases_dict['Impact']

print("ARM EXTENSION DELTA ANALYSIS")
print("="*70)
print(f"\nFrames being used:")
print(f"  Top frame:    {top_frame}")
print(f"  Impact frame: {impact_frame}")

# Get lead arm angles
top_arm = biomech.get_lead_arm_angle(frame=top_frame)
impact_arm = biomech.get_lead_arm_angle(frame=impact_frame)
delta = impact_arm - top_arm

print(f"\nLead arm angles:")
print(f"  At top ({top_frame}):    {top_arm:.2f}°")
print(f"  At impact ({impact_frame}): {impact_arm:.2f}°")
print(f"  Delta:                  {delta:.2f}°")
print(f"  Ideal delta:            15°")

# Check if frames overlap (which would be wrong)
if top_frame >= impact_frame:
    print(f"\n⚠️  WARNING: Top frame ({top_frame}) >= Impact frame ({impact_frame})")
    print(f"  This will give negative or zero delta!")

# Check all arm angles in the sequence
print(f"\nArm angles across the swing:")
for phase in ['Top', 'Mid-downswing', 'Impact', 'Follow-through']:
    if phase in phases_dict:
        frame = phases_dict[phase]
        angle = biomech.get_lead_arm_angle(frame=frame)
        print(f"  {phase:15} (frame {frame:3}): {angle:.2f}°")
