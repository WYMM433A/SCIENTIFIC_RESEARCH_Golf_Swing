"""
Video Cleaner: Auto-crop to swing bounds, remove pre/post-swing padding
Uses motion detection to find swing start/end, keeps only the swing data
"""

import cv2
import numpy as np
from pathlib import Path
import os

def detect_swing_bounds(video_path, motion_threshold=0.5, buffer_frames=0):
    """
    Detect swing start/end using motion detection
    
    Args:
        video_path: Path to video
        motion_threshold: Threshold for motion detection (0-1)
        buffer_frames: Extra frames to keep before/after motion (default 0 = no buffer)
    
    Returns:
        (swing_start, swing_end): Frame indices
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Read first frame to get dimensions
    ret, prev_frame = cap.read()
    if not ret:
        return 0, total_frames
    
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    motion_scores = []
    
    print(f"  Analyzing motion... ", end='', flush=True)
    
    # Calculate motion for each frame
    for frame_idx in range(1, total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate frame difference (optical flow approximation)
        diff = cv2.absdiff(prev_gray, gray)
        motion_score = np.mean(diff)
        motion_scores.append(motion_score)
        
        prev_gray = gray
    
    cap.release()
    
    motion_scores = np.array(motion_scores)
    
    # Normalize scores
    motion_threshold_value = motion_threshold * np.max(motion_scores) if len(motion_scores) > 0 else 0
    
    # Find motion start/end
    motion_frames = np.where(motion_scores > motion_threshold_value)[0]
    
    if len(motion_frames) == 0:
        print(f"No motion detected, keeping full video")
        return 0, total_frames
    
    swing_start = max(0, motion_frames[0] - buffer_frames)
    swing_end = min(total_frames - 1, motion_frames[-1] + buffer_frames + 1)
    
    swing_duration = swing_end - swing_start
    print(f"Motion detected: frames {swing_start}-{swing_end} ({swing_duration} frames)")
    
    return swing_start, swing_end

def crop_video(input_path, output_path, swing_start, swing_end, target_width=720, target_height=1280):
    """
    Crop and save video to swing bounds
    
    Args:
        input_path: Input video path
        output_path: Output video path
        swing_start: Start frame index
        swing_end: End frame index
        target_width, target_height: Output resolution
    """
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (target_width, target_height))
    
    total_frames = swing_end - swing_start
    frames_written = 0
    
    # Skip to swing_start
    cap.set(cv2.CAP_PROP_POS_FRAMES, swing_start)
    
    print(f"  Writing video... ", end='', flush=True)
    
    for i in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        
        # Resize if needed
        if frame.shape[1] != target_width or frame.shape[0] != target_height:
            frame = cv2.resize(frame, (target_width, target_height))
        
        out.write(frame)
        frames_written += 1
    
    cap.release()
    out.release()
    
    print(f"Saved {frames_written} frames")
    
    return frames_written

def clean_videos(input_dir='data/raw_videos', output_dir='data/cleaned_videos', motion_threshold=0.5):
    """
    Clean all videos in directory: detect swing bounds, crop, standardize
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*80)
    print("VIDEO CLEANING: AUTO-CROP TO SWING BOUNDS")
    print("="*80 + "\n")
    
    results = []
    
    for video_file in sorted(input_path.glob('**/*.mp4')):
        video_name = video_file.stem
        output_file = output_path / f"{video_name}_cleaned.mp4"
        
        print(f"\n{video_name}:")
        
        # Detect swing bounds
        swing_start, swing_end = detect_swing_bounds(str(video_file), motion_threshold=motion_threshold, buffer_frames=0)
        swing_duration = swing_end - swing_start
        
        # Crop and save
        frames_written = crop_video(str(video_file), str(output_file), swing_start, swing_end)
        
        results.append({
            'Video': video_name,
            'Original_Frames': int(video_file.stat().st_size),
            'Swing_Start': swing_start,
            'Swing_End': swing_end,
            'Cropped_Frames': frames_written,
            'Output': output_file.name
        })
    
    # Summary
    print("\n" + "="*80)
    print("CLEANING SUMMARY")
    print("="*80)
    
    for result in results:
        print(f"\n✓ {result['Video']}")
        print(f"  Swing bounds: {result['Swing_Start']}-{result['Swing_End']}")
        print(f"  Frames kept: {result['Cropped_Frames']}")
        print(f"  Output: {result['Output']}")
    
    print("\n" + "="*80)
    print(f"✓ Cleaned videos saved to: {output_dir}")
    print("="*80 + "\n")

if __name__ == '__main__':
    # Clean all videos (using 30% motion threshold for swing detection)
    clean_videos(
        input_dir='data/raw_videos',
        output_dir='data/cleaned_videos',
        motion_threshold=0.3  # 30% of max motion = swing threshold (catches more motion)
    )
