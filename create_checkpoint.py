"""
Create checkpoint file by scanning existing outputs
Detects which videos have already been processed
Allows seamless continuation with the updated batch script
"""

import json
import os
from pathlib import Path

def create_checkpoint_from_existing_outputs(keyframes_dir='data/keyframes', poses_dir='data/extracted_poses'):
    """
    Scan existing output directories and create checkpoint file.
    
    Detects already-processed videos by looking for:
    - Pose CSV files in data/extracted_poses/
    - Keyframe folders in data/keyframes/
    """
    
    processed = set()
    failed = []
    
    # Scan extracted poses
    poses_path = Path(poses_dir)
    if poses_path.exists():
        for csv_file in poses_path.glob('*_cleaned_poses.csv'):
            # Extract video name: B46_F5_I from "B46_F5_I_cleaned_poses.csv"
            video_name = csv_file.stem.replace('_cleaned_poses', '')
            processed.add(video_name)
    
    # Save checkpoint
    checkpoint_file = 'data/batch_process_checkpoint.json'
    checkpoint = {'processed': sorted(list(processed)), 'failed': failed}
    
    with open(checkpoint_file, 'w') as f:
        json.dump(checkpoint, f, indent=2)
    
    print(f"\n{'=' * 70}")
    print(f"CHECKPOINT CREATED")
    print(f"{'=' * 70}")
    print(f"✅ Detected {len(processed)} already-processed videos:")
    
    # Show first and last few
    sorted_videos = sorted(list(processed))
    for video in sorted_videos[:5]:
        print(f"   • {video}")
    if len(sorted_videos) > 10:
        print(f"   ...")
    for video in sorted_videos[-5:]:
        print(f"   • {video}")
    
    print(f"\n💾 Checkpoint saved: {checkpoint_file}")
    print(f"\nNext run will skip these {len(processed)} videos automatically:")
    print(f"  python batch_process_phase_only.py --limit 50")
    print(f"{'=' * 70}\n")

if __name__ == '__main__':
    create_checkpoint_from_existing_outputs()
