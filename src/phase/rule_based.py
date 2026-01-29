"""
8-Phase Golf Swing Detector (Hybrid Approach)
Combines WristTrajectory detection methods with time-proportional segmentation
Phases: Address, Takeaway, Mid-backswing, Top, Mid-downswing, Impact, Follow-through, Finish
"""

import cv2
import pandas as pd
import numpy as np
from scipy.ndimage import uniform_filter1d
import os
from pathlib import Path
from ..constants import PHASE_NAMES


class EightPhaseDetector:
    """
    Detect golf swing phases using wrist Y trajectory analysis
    8-phase version: Address, Takeaway, Mid-backswing, Top, Mid-downswing, Impact, Follow-through, Finish
    """
    
    PHASE_NAMES = PHASE_NAMES  # Use shared constant
    
    def __init__(self, csv_path, video_path, output_dir='phase_frames'):
        """
        Initialize detector
        
        Args:
            csv_path: Path to pose CSV file (must have 'right_wrist_y' column)
            video_path: Path to video file
            output_dir: Directory to save output frames
        """
        self.csv_path = csv_path
        self.video_path = video_path
        self.output_dir = output_dir
        
        # Extract video name for file naming
        self.video_name = Path(csv_path).stem.replace('_poses', '')
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        self.phase_ranges = {}
        self.extracted_data = {}
    
    def detect_phases(self, smoothing_window=5, precheck_window=30, threshold_percentile=90):
        """
        Detect 8 golf swing phases from wrist Y trajectory
        
        Uses hybrid approach:
        - Address & Finish: Stability detection
        - Takeaway & Mid-backswing: Time-proportional percentages
        - Top: Peak detection
        - Mid-downswing: From end of Top to start of Impact
        - Impact: Plateau detection
        - Follow-through: From Impact end to Finish start
        
        Returns:
            dict: Phase ranges with phase names as keys
        """
        # Load pose data
        df = pd.read_csv(self.csv_path)
        wrist_y = df['right_wrist_y'].values
        
        # Smooth and calculate velocity
        smoothed = uniform_filter1d(wrist_y, size=smoothing_window, mode='nearest')
        velocity = np.gradient(smoothed)
        velocity_mag = np.abs(velocity)
        
        # === FIND SWING BOUNDARIES ===
        early_motion = np.nanmean(velocity_mag[:precheck_window])
        motion_std = np.nanstd(velocity_mag[:precheck_window])
        buffer_start = precheck_window if (early_motion < 0.001 and motion_std < 0.001) else 0
        threshold = np.nanpercentile(velocity_mag[buffer_start:], threshold_percentile)
        motion_indices = np.where(velocity_mag > threshold)[0]
        motion_indices = motion_indices[motion_indices > buffer_start]
        
        if len(motion_indices) > 0:
            swing_start = int(motion_indices[0])
            peak_idx = motion_indices[np.argmax(velocity_mag[motion_indices])]
            target_y = smoothed[swing_start]
            post_peak_range = smoothed[peak_idx+1:]
            swing_end = peak_idx + 1 + int(np.nanargmin(np.abs(post_peak_range - target_y))) if len(post_peak_range) > 0 else len(wrist_y) - 1
        else:
            swing_start = 0
            swing_end = len(wrist_y) - 1
        
        # === PHASE 0: ADDRESS (backward search for stability) ===
        flat_std_thresh = 1.0
        min_flat_frames = 10
        address_start = 0
        for i in range(swing_start - min_flat_frames, 0, -1):
            window = smoothed[i:swing_start]
            if np.count_nonzero(~np.isnan(window)) < min_flat_frames:
                continue
            if np.nanstd(window) < flat_std_thresh:
                address_start = i
            else:
                break
        address_end = swing_start
        address_range = (address_start, address_end)
        
        # === FIND TOP OF BACKSWING ===
        full_swing_segment = smoothed[swing_start:swing_end]
        diff = np.diff(full_swing_segment)
        rising_indices = np.where(diff > 0)[0]
        top_candidate_max = swing_start + rising_indices[0] if len(rising_indices) > 0 else min(swing_start + 20, len(smoothed) - 1)
        top_search_end = min(top_candidate_max, swing_end)
        top_idx = swing_start + int(np.nanargmin(smoothed[swing_start:top_search_end])) if top_search_end > swing_start else swing_start
        
        # === PHASE 3: TOP (±5 frames around peak) ===
        top_range = (max(0, top_idx - 5), min(len(wrist_y) - 1, top_idx + 5))
        
        # Calculate backswing duration for time-proportional phases
        backswing_duration = top_idx - swing_start

        # === PHASE 1: TAKEAWAY (0-20% of backswing) ===
        takeaway_end = swing_start + int(0.2 * backswing_duration)
        takeaway_range = (swing_start, max(swing_start + 5, takeaway_end))
        
        # === PHASE 2: MID-BACKSWING (25-75% of backswing) ===
        midback_start = takeaway_range[1]
        midback_end = swing_start + int(0.75 * backswing_duration)
        midback_range = (midback_start, max(midback_start + 5, midback_end))
        
        # === DETECT IMPACT (plateau detection - WristTrajectory method) ===
        post_top = smoothed[top_range[1]:swing_end]
        
        if len(post_top) == 0 or np.all(np.isnan(post_top)):
            # Edge case: no post-top data
            impact_range = (top_range[1], top_range[1] + 1)
            impact_end = top_range[1] + 1
        else:
            # Find maximum wrist Y value in post-top segment
            local_max_idx = int(np.nanargmax(post_top))
            max_val = post_top[local_max_idx]
            local_min = np.nanmin(post_top)
            local_max = np.nanmax(post_top)
            
            # Dynamic tolerance based on range (5% of range)
            dynamic_tol = 0.05 * (local_max - local_min) if (local_max - local_min) > 0 else 0.01
            
            # Expand left/right around peak while values stay within tolerance
            left = local_max_idx
            right = local_max_idx
            while left > 0 and abs(post_top[left - 1] - max_val) < dynamic_tol:
                left -= 1
            while right < len(post_top) - 1 and abs(post_top[right + 1] - max_val) < dynamic_tol:
                right += 1
            
            # Impact range is the plateau around peak
            impact_range = (top_range[1] + left, top_range[1] + right)
            impact_end = impact_range[1]
        
        # === PHASE 4: MID-DOWNSWING (from end of Top to start of Impact) ===
        middown_start = top_range[1]
        middown_end = impact_range[0]
        
        # Midpoint fallback if Impact is very close to Top
        if middown_end <= middown_start:
            middown_end = middown_start + int((impact_end - middown_start) / 2)
        
        middown_range = (middown_start, max(middown_start + 3, middown_end))
        
        # === DETECT FINISH (minimum Y after impact + velocity end) ===
        # After impact, find where wrist reaches lowest point and velocity ends
        post_impact = smoothed[impact_end:]
        
        if len(post_impact) > 0 and not np.all(np.isnan(post_impact)):
            # Find minimum Y value after impact
            finish_min_idx = impact_end + int(np.nanargmin(post_impact))
            
            # Calculate velocity after impact
            post_impact_velocity = np.gradient(post_impact)
            post_impact_velocity_mag = np.abs(post_impact_velocity)
            
            # Find where velocity becomes minimal (motion ends)
            # Look for where velocity drops below a low threshold or stays low
            velocity_threshold = np.nanpercentile(post_impact_velocity_mag, 25)  # Lower quartile
            
            # Find the last significant motion after minimum Y
            finish_velocity_end = impact_end
            for i in range(finish_min_idx, min(len(post_impact) - 1, finish_min_idx + 40)):
                if post_impact_velocity_mag[i - impact_end] < velocity_threshold:
                    finish_velocity_end = i
                    break
            else:
                # If no low velocity found, use a point past minimum
                finish_velocity_end = min(finish_min_idx + 20, len(wrist_y) - 1)
        else:
            # Edge case: no post-impact data
            finish_min_idx = impact_end + 1
            finish_velocity_end = min(impact_end + 15, len(wrist_y) - 1)
        
        # === PHASE 5: IMPACT (already detected above) ===
        # impact_range already calculated
        
        # Calculate post-impact duration for time-proportional phases
        post_impact_duration = finish_velocity_end - impact_end
        
        # === PHASE 6: FOLLOW-THROUGH (25-75% of post-impact duration) ===
        followthrough_start = impact_end
        followthrough_end = impact_end + int(0.32 * post_impact_duration)
        followthrough_range = (followthrough_start, max(followthrough_start + 3, followthrough_end))
        
        # === PHASE 7: FINISH (from minimum Y to where velocity ends) ===
        finish_start = finish_min_idx
        finish_end = finish_velocity_end
        finish_range = (finish_start, max(finish_start + 1, finish_end))
        
        self.phase_ranges = {
            "Address": address_range,
            "Takeaway": takeaway_range,
            "Mid-backswing": midback_range,
            "Top": top_range,
            "Mid-downswing": middown_range,
            "Impact": impact_range,
            "Follow-through": followthrough_range,
            "Finish": finish_range
        }
        
        return self.phase_ranges
    
    def extract_8_frames(self):
        """Extract one representative frame per phase"""
        
        if not hasattr(self, 'phase_ranges') or not self.phase_ranges:
            self.detect_phases()
        
        if not os.path.exists(self.video_path):
            print(f"✗ Video not found: {self.video_path}")
            return None
        
        cap = cv2.VideoCapture(self.video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        extracted_data = {}
        print(f"\nExtracting 8 frames from {self.video_name}:")
        print(f"{'Phase':<20} {'Frame Range':<25} {'Key Frame':<15} {'Status':<10}")
        print("-" * 70)
        
        for phase_name in self.PHASE_NAMES:
            if phase_name not in self.phase_ranges:
                continue
                
            phase_range = self.phase_ranges[phase_name]
            
            # Get middle frame of phase
            frame_idx = int((phase_range[0] + phase_range[1]) / 2)
            frame_idx = max(0, min(frame_idx, total_frames - 1))
            
            # Read frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if ret:
                # Save frame
                output_path = os.path.join(self.output_dir, f'{self.video_name}_{phase_name}.jpg')
                cv2.imwrite(output_path, frame)
                extracted_data[phase_name] = {
                    'phase': phase_name,
                    'start_frame': phase_range[0],
                    'end_frame': phase_range[1],
                    'duration': phase_range[1] - phase_range[0] + 1,
                    'key_frame': frame_idx,
                    'image_path': output_path
                }
                status = "✓"
                print(f"{phase_name:<20} {str(phase_range):<25} {frame_idx:<15} {status:<10}")
            else:
                print(f"{phase_name:<20} {str(phase_range):<25} {'N/A':<15} {'✗':<10}")
        
        cap.release()
        
        self.extracted_data = extracted_data
        return extracted_data
    
    def save_phase_csv(self, csv_output_path=None):
        """Save phase information to CSV"""
        
        if not hasattr(self, 'extracted_data') or not self.extracted_data:
            self.extract_8_frames()
        
        if csv_output_path is None:
            csv_output_path = os.path.join(self.output_dir, f'{self.video_name}_8phases.csv')
        
        # Create DataFrame
        rows = []
        for phase_name in self.PHASE_NAMES:
            if phase_name in self.extracted_data:
                data = self.extracted_data[phase_name]
                rows.append({
                    'Video': self.video_name,
                    'Phase': data['phase'],
                    'Start_Frame': data['start_frame'],
                    'End_Frame': data['end_frame'],
                    'Duration': data['duration'],
                    'Key_Frame': data['key_frame'],
                    'Image_Path': data['image_path']
                })
        
        df = pd.DataFrame(rows)
        df.to_csv(csv_output_path, index=False)
        
        print(f"\n✓ 8-Phase CSV saved: {csv_output_path}")
        return csv_output_path
    
    def process(self):
        """Complete pipeline: detect phases, extract frames, save CSV"""
        print(f"\n{'='*70}")
        print(f"8-PHASE GOLF SWING DETECTION (Hybrid Method)")
        print(f"{'='*70}")
        print(f"Processing: {self.video_name}")
        print(f"{'='*70}")
        
        # Detect phases
        self.detect_phases()
        print("✓ Phases detected:")
        for phase_name in self.PHASE_NAMES:
            if phase_name in self.phase_ranges:
                start, end = self.phase_ranges[phase_name]
                duration = end - start + 1
                print(f"  • {phase_name:<20} [{start:>4d}:{end:>4d}] duration: {duration:>3d} frames")
        
        # Extract frames
        self.extract_8_frames()
        print("✓ 8 frames extracted")
        
        # Save CSV
        csv_path = self.save_phase_csv()
        
        print(f"{'='*70}\n")
        
        return csv_path


if __name__ == '__main__':
    import sys
    
    # Accept csv_path and video_path as command-line arguments
    if len(sys.argv) >= 3:
        # User provided both paths: python script.py <csv_path> <video_path>
        csv_path = sys.argv[1]
        video_path = sys.argv[2]
        output_dir = sys.argv[3] if len(sys.argv) > 3 else 'phase_frames'
    elif len(sys.argv) == 2:
        # User provided just a video number: python script.py 005
        video_num = sys.argv[1]
        csv_path = f'data/extracted_poses/golf_swing_{video_num}_cleaned_poses.csv'
        video_path = None
        output_dir = 'phase_frames'
    else:
        # Default example usage
        csv_path = 'data/extracted_poses/golf_swing_002_cleaned_poses.csv'
        video_path = 'data/cleaned_videos/golf_swing_002_cleaned.mp4'
        output_dir = 'phase_frames'
    
    # If video_path not specified, try to auto-find it
    if video_path is None:
        import os
        from pathlib import Path
        video_dir = Path('data/cleaned_videos')
        video_name = Path(csv_path).stem.replace('_cleaned_poses', '')
        video_files = list(video_dir.glob(f'{video_name}_cleaned.mp4'))
        if video_files:
            video_path = str(video_files[0])
        else:
            print(f"Video not found for {video_name}")
            sys.exit(1)
    
    # Create and run detector
    if os.path.exists(csv_path) and os.path.exists(video_path):
        detector = EightPhaseDetector(csv_path, video_path, output_dir)
        detector.process()
    else:
        if not os.path.exists(csv_path):
            print(f"✗ CSV not found: {csv_path}")
        if not os.path.exists(video_path):
            print(f"✗ Video not found: {video_path}")
