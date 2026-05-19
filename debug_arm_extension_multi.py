import pandas as pd
from src.biomechanics.angles import GolfBiomechanics
import sys

# Test videos
test_videos = {
    'min2': 'data/extracted_poses/min2_cleaned_poses.csv',
    'golf_swing_001': 'data/extracted_poses/golf_swing_001_cleaned_poses.csv',
    'me': 'data/extracted_poses/me_cleaned_poses.csv',
}

ideal_delta = 28

for video_name, pose_csv in test_videos.items():
    try:
        df = pd.read_csv(pose_csv)
        
        # Get phases
        phases_csv = f'data/keyframes/{video_name}_nn/{video_name}_cleaned_8phases.csv'
        phases_df = pd.read_csv(phases_csv)
        phases_dict = dict(zip(phases_df['Phase'], phases_df['Key_Frame']))
        
        biomech = GolfBiomechanics(df)
        
        top_frame = phases_dict['Top']
        impact_frame = phases_dict['Impact']
        
        # Get lead arm angles
        top_arm = biomech.get_lead_arm_angle(frame=top_frame)
        impact_arm = biomech.get_lead_arm_angle(frame=impact_frame)
        delta = impact_arm - top_arm
        
        # Calculate quality
        delta_error = abs(delta - ideal_delta)
        quality = max(0, 100 - (delta_error * 5))
        
        print(f"{video_name.upper()}")
        print("="*70)
        print(f"Top frame:    {top_frame} (arm: {top_arm:.2f}°)")
        print(f"Impact frame: {impact_frame} (arm: {impact_arm:.2f}°)")
        print(f"Delta:        {delta:.2f}°  |  Ideal: {ideal_delta}°  |  Error: {delta_error:.2f}°")
        print(f"Quality:      {quality:.2f}/100  |  Weighted (0.75): {quality * 0.75:.2f}")
        print()
        
    except FileNotFoundError as e:
        print(f"{video_name.upper()}: File not found - {e}")
        print()
