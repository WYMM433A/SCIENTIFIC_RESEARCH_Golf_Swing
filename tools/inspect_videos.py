"""
Inspect video properties: fps, resolution, duration, frame count
"""

import cv2
from pathlib import Path
import pandas as pd

def inspect_video(video_path):
    """Get video properties"""
    cap = cv2.VideoCapture(video_path)
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = frame_count / fps if fps > 0 else 0
    
    cap.release()
    
    return {
        'fps': fps,
        'resolution': f"{width}x{height}",
        'width': width,
        'height': height,
        'frame_count': frame_count,
        'duration_sec': duration_sec
    }

# Scan all videos
video_dir = Path('data/raw_videos')
results = []

print("\n" + "="*80)
print("VIDEO INSPECTION REPORT")
print("="*80 + "\n")

for video_file in sorted(video_dir.glob('**/*.mp4')):
    try:
        props = inspect_video(str(video_file))
        results.append({
            'Video': video_file.stem,
            'FPS': props['fps'],
            'Resolution': props['resolution'],
            'Frames': props['frame_count'],
            'Duration (sec)': f"{props['duration_sec']:.2f}"
        })
        print(f"✓ {video_file.stem}")
        print(f"  FPS: {props['fps']}, Resolution: {props['resolution']}, "
              f"Frames: {props['frame_count']}, Duration: {props['duration_sec']:.2f}s")
    except Exception as e:
        print(f"✗ {video_file.stem}: {str(e)}")

# Summary
df = pd.DataFrame(results)
if len(results) > 0:
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Total videos: {len(results)}")
    print(f"\nFPS Range: {df['FPS'].min():.1f} - {df['FPS'].max():.1f}")
    print(f"Resolution Range: {df['Resolution'].unique()}")
    print(f"Frame Count Range: {df['Frames'].min()} - {df['Frames'].max()}")
    print(f"Duration Range: {df['Duration (sec)'].min()} - {df['Duration (sec)'].max()}")
    print("\nDetailed Table:")
    print(df.to_string(index=False))
    print("="*80 + "\n")
