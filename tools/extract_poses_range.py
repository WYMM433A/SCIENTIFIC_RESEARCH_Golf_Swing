"""
Extract poses for a specific range of GolfDB videos.

Usage:
    python tools/extract_poses_range.py --start 46 --end 100
"""

import sys
import argparse
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.pose import SwingAnalyzer


def extract_poses_range(start_idx, end_idx):
    """Extract poses for videos in the specified range."""
    
    golfdb_path = PROJECT_ROOT / 'data'
    output_dir = PROJECT_ROOT / 'data' / 'golfdb_poses'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    videos_dir = golfdb_path / "videos_160"
    pkl_path = golfdb_path / "golfDB.pkl"
    
    # Check paths
    if not videos_dir.exists():
        print(f"❌ Videos not found: {videos_dir}")
        return
    if not pkl_path.exists():
        print(f"❌ golfDB.pkl not found: {pkl_path}")
        return
    
    # Load annotations
    df = pd.read_pickle(pkl_path)
    print(f"✓ Loaded {len(df)} annotations from golfDB.pkl")
    
    # Select range
    df = df.iloc[start_idx:end_idx]
    print(f"✓ Processing videos {start_idx} to {end_idx} ({len(df)} videos)")
    
    # Initialize pose analyzer
    analyzer = SwingAnalyzer()
    
    processed = 0
    skipped = 0
    failed = 0
    
    print(f"\n{'='*60}")
    print("EXTRACTING POSES")
    print(f"{'='*60}\n")
    
    for idx in tqdm(range(len(df)), desc="Extracting poses"):
        row = df.iloc[idx]
        video_id = row['id']
        video_path = videos_dir / f"{video_id}.mp4"
        output_csv = output_dir / f"{video_id}_poses.csv"
        
        # Skip if already processed
        if output_csv.exists():
            skipped += 1
            continue
        
        if not video_path.exists():
            failed += 1
            continue
        
        try:
            pose_df = analyzer.processVideo(
                video_path=str(video_path),
                output_csv=str(output_csv),
                show_preview=False
            )
            
            if pose_df is not None and len(pose_df) > 0:
                processed += 1
            else:
                failed += 1
                
        except Exception as e:
            print(f"\n  ✗ Error processing {video_id}: {e}")
            failed += 1
    
    print(f"\n{'='*60}")
    print("EXTRACTION COMPLETE")
    print(f"{'='*60}")
    print(f"  ✓ Processed: {processed}")
    print(f"  ⏭ Skipped (already done): {skipped}")
    print(f"  ✗ Failed: {failed}")
    print(f"\nPoses saved to: {output_dir}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract poses for a range of GolfDB videos')
    parser.add_argument('--start', type=int, default=0, help='Start video index (0-based)')
    parser.add_argument('--end', type=int, default=100, help='End video index (exclusive)')
    args = parser.parse_args()
    
    extract_poses_range(args.start, args.end)
