import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import uniform_filter1d
import glob

# Load all pose CSVs
csv_files = sorted(glob.glob('data/extracted_poses/*_poses.csv'))
print(f"Found {len(csv_files)} videos:")
for f in csv_files:
    print(f"  - {f}")

# Extract wrist Y from each
trajectories = {}
for csv_path in csv_files:
    video_name = csv_path.split('/')[-1].replace('_poses.csv', '')
    df = pd.read_csv(csv_path)
    wrist_y = df['right_wrist_y'].values
    smoothed = uniform_filter1d(wrist_y, size=5, mode='nearest')
    trajectories[video_name] = smoothed
    print(f"{video_name}: {len(smoothed)} frames, range {smoothed.min():.3f} to {smoothed.max():.3f}")

# Plot all trajectories
fig, axes = plt.subplots(len(trajectories), 1, figsize=(14, 3*len(trajectories)))
if len(trajectories) == 1:
    axes = [axes]

colors = plt.cm.tab10(np.linspace(0, 1, len(trajectories)))

for idx, (video_name, traj) in enumerate(trajectories.items()):
    axes[idx].plot(traj, color=colors[idx], linewidth=2)
    axes[idx].set_title(f"{video_name} - Wrist Y Trajectory", fontsize=12, fontweight='bold')
    axes[idx].set_ylabel('Wrist Y')
    axes[idx].grid(True, alpha=0.3)
    axes[idx].set_ylim([min(traj.min() - 0.02, 0.35), max(traj.max() + 0.02, 0.70)])

axes[-1].set_xlabel('Frame Index')
plt.tight_layout()
plt.savefig('data/visualizations/wrist_trajectories_all_videos.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nSaved visualization to data/visualizations/wrist_trajectories_all_videos.png")
