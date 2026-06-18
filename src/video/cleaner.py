"""
Video Cleaner: Auto-crop to swing bounds, remove pre/post-swing padding
Uses localized motion detection to bypass pillarbox black bars and keep the full swing.
"""

import cv2
import numpy as np
from pathlib import Path
import os

def find_active_video_bounds(video_path):
    """
    Find where the actual vertical video is by using a slightly higher 
    threshold tolerance to completely discard noisy outdoor compression artifacts.
    """
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        return None

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Increased threshold to 25 to safely ignore compression artifacts in dark borders
    _, thresh = cv2.threshold(gray, 25, 255, cv2.THRESH_BINARY)
    
    x, y, w, h = cv2.boundingRect(thresh)
    
    if w == 0 or h == 0:
        return 0, 0, frame.shape[1], frame.shape[0]
        
    return x, y, w, h

def detect_swing_bounds(video_path, motion_threshold=0.15, buffer_frames=30):
    """
    Robust swing detector using clamped percentile normalization to neutralize
    massive background artifacts and camera shake.
    """
    bounds = find_active_video_bounds(video_path)
    if bounds is None:
        return 0, 0
    bx, by, bw, bh = bounds

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    ret, prev_frame = cap.read()
    if not ret:
        return 0, total_frames
    
    prev_active = prev_frame[by:by+bh, bx:bx+bw]
    prev_gray = cv2.cvtColor(prev_active, cv2.COLOR_BGR2GRAY)
    
    # Apply a slight Gaussian Blur to smooth out pixel noise/grass blowing in the wind
    prev_gray = cv2.GaussianBlur(prev_gray, (5, 5), 0)
    
    motion_scores = []
    
    print(f"  Analyzing active pixels across {total_frames} frames... ", end='', flush=True)
    
    for frame_idx in range(1, total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        
        active_zone = frame[by:by+bh, bx:bx+bw]
        gray = cv2.cvtColor(active_zone, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        
        diff = cv2.absdiff(prev_gray, gray)
        motion_score = np.mean(diff)
        motion_scores.append(motion_score)
        
        prev_gray = gray
    
    cap.release()
    
    motion_scores = np.array(motion_scores)
    if len(motion_scores) == 0:
        return 0, total_frames
        
    # --- CRITICAL FIX: SQUASH OUTLIER SPIKES ---
    # Instead of using np.max() which can be ruined by a single bad frame artifact,
    # we normalize against the 95th percentile of motion.
    motion_peak = np.percentile(motion_scores, 95)
    if motion_peak == 0: 
        motion_peak = np.max(motion_scores)
        
    motion_threshold_value = motion_threshold * motion_peak
    
    # Find all frames matching baseline motion criteria
    motion_frames = np.where(motion_scores > motion_threshold_value)[0]
    
    if len(motion_frames) == 0:
        print(f"Low delta threshold, keeping full frame tracking.")
        return 0, total_frames
    
    # Expand the padding safely (30 frames = roughly 1 entire second of buffer room)
    swing_start = max(0, motion_frames[0] - buffer_frames)
    swing_end = min(total_frames - 1, motion_frames[-1] + buffer_frames + 1)
    
    swing_duration = swing_end - swing_start
    print(f"Success! Captured: frames {swing_start}-{swing_end} ({swing_duration} frames)")
    
    return swing_start, swing_end

def crop_video(input_path, output_path, swing_start, swing_end, target_width=720, target_height=1280):
    """
    Crop and save video to swing bounds
    """
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (target_width, target_height))
    
    total_frames = swing_end - swing_start
    frames_written = 0
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, swing_start)
    
    print(f"  Writing video... ", end='', flush=True)
    
    for i in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame.shape[1] != target_width or frame.shape[0] != target_height:
            frame = cv2.resize(frame, (target_width, target_height))
        
        out.write(frame)
        frames_written += 1
    
    cap.release()
    out.release()
    
    print(f"Saved {frames_written} frames")
    return frames_written

def clean_videos(input_dir='data/raw_videos', output_dir='data/cleaned_videos', motion_threshold=0.2, buffer_frames=25):
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
        
        # Using updated parameters for safe tracking
        swing_start, swing_end = detect_swing_bounds(
            str(video_file), 
            motion_threshold=motion_threshold, 
            buffer_frames=buffer_frames
        )
        
        frames_written = crop_video(str(video_file), str(output_file), swing_start, swing_end)
        
        results.append({
            'Video': video_name,
            'Swing_Start': swing_start,
            'Swing_End': swing_end,
            'Cropped_Frames': frames_written,
            'Output': output_file.name
        })
    
    print("\n" + "="*80)
    print("CLEANING SUMMARY")
    print("="*80)
    for result in results:
        print(f"\n✓ {result['Video']}")
        print(f"  Swing bounds: {result['Swing_Start']}-{result['Swing_End']}")
        print(f"  Frames kept: {result['Cropped_Frames']}")
        print(f"  Output: {result['Output']}")

if __name__ == '__main__':
    # Lowered baseline threshold + added a 25-frame buffer pad to prevent clipping
    clean_videos(
        input_dir='data/raw_videos',
        output_dir='data/cleaned_videos',
        motion_threshold=0.20,  # 20% threshold over active region pixels
        buffer_frames=25        # Keep ~1 second of padding on both sides of the detected motion
    )