import pandas as pd

df_004 = pd.read_csv('data/extracted_poses/golf_swing_004_cleaned_poses.csv')
df_min2 = pd.read_csv('data/extracted_poses/min2_cleaned_poses.csv')

print('GOLF_SWING_004 FINISH (264-267):')
for frame in [264, 265, 266, 267]:
    row = df_004[df_004['frame'] == frame].iloc[0]
    print(f'  Frame {frame}: nose_x={row["nose_x"]:.3f}, L_shoulder_x={row["left_shoulder_x"]:.3f}, R_shoulder_x={row["right_shoulder_x"]:.3f}')

print()
print('MIN2 FINISH (81-87):')
for frame in [85, 86, 87]:
    row = df_min2[df_min2['frame'] == frame].iloc[0]
    print(f'  Frame {frame}: nose_x={row["nose_x"]:.3f}, L_shoulder_x={row["left_shoulder_x"]:.3f}, R_shoulder_x={row["right_shoulder_x"]:.3f}')
