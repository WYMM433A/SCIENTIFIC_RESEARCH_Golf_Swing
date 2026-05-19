import pandas as pd
import numpy as np
from math import sqrt

def get_arm_extension(pose):
    try:
        ls_x, ls_y = pose['left_shoulder_x'], pose['left_shoulder_y']
        rs_x, rs_y = pose['right_shoulder_x'], pose['right_shoulder_y']
        lw_x, lw_y = pose['left_wrist_x'], pose['left_wrist_y']
        rw_x, rw_y = pose['right_wrist_x'], pose['right_wrist_y']

        ms_x, ms_y = (ls_x + rs_x) / 2, (ls_y + rs_y) / 2
        mw_x, mw_y = (lw_x + rw_x) / 2, (lw_y + rw_y) / 2
        
        return sqrt((mw_x - ms_x)**2 + (mw_y - ms_y)**2)
    except Exception as e:
        return 0

def calculate_arm_extension_delta(poses_df, top_frame_idx, impact_frame_idx):
    try:
        top_pose = poses_df[poses_df['frame'] == top_frame_idx].iloc[0]
        impact_pose = poses_df[poses_df['frame'] == impact_frame_idx].iloc[0]
        
        top_arm = get_arm_extension(top_pose)
        impact_arm = get_arm_extension(impact_pose)
        
        delta = impact_arm - top_arm
        quality = "Good" if delta > 0 else "Poor"
        
        return top_arm, impact_arm, delta, quality
    except Exception as e:
        return 0, 0, 0, f"Error: {e}"

df = pd.read_csv('data/extracted_poses/B80O_cleaned_poses.csv')

# Find top frame (peak of mid-wrist y) in the first 60 frames (standard swing range)
df_swing = df[df['frame'] < 60]
df_swing['mw_y'] = (df_swing['left_wrist_y'] + df_swing['right_wrist_y']) / 2
top_frame = int(df_swing.loc[df_swing['mw_y'].idxmin(), 'frame'])
impact_frame = 51

t_arm, i_arm, delta, qual = calculate_arm_extension_delta(df, top_frame, impact_frame)

print(f"Top Arm Extension (Frame {top_frame}): {t_arm}")
print(f"Impact Arm Extension (Frame {impact_frame}): {i_arm}")
print(f"Delta: {delta}")
print(f"Quality: {qual}")
