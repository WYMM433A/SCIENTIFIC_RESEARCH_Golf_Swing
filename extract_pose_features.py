"""
Extract pose landmark sequences as features for neural network training.

Converts pose CSV files into numpy arrays:
- Shape: (num_frames, 33, 4) for each swing
- Ready for LSTM input
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def extract_pose_landmarks(poses_csv, start_frame=None, end_frame=None):
    """
    Load pose CSV and extract landmark array.
    
    Args:
        poses_csv: Path to *_cleaned_poses.csv
        start_frame: Start frame (inclusive), None = first frame
        end_frame: End frame (inclusive), None = last frame
    
    Returns:
        landmarks: np.array of shape (num_frames, 33, 4)
                   where 4 = [x, y, z, visibility]
    """
    
    df = pd.read_csv(poses_csv)
    
    if start_frame is None:
        start_frame = 0
    if end_frame is None:
        end_frame = len(df) - 1
    
    # Filter to frame range
    df = df.iloc[start_frame:end_frame+1]
    
    if len(df) == 0:
        return None
    
    # Extract 33 landmarks (MediaPipe format)
    landmarks = []
    
    for frame_idx, row in df.iterrows():
        frame_lms = []
        
        for lm_idx in range(33):
            x = row.get(f'landmark_{lm_idx}_x', 0.0)
            y = row.get(f'landmark_{lm_idx}_y', 0.0)
            z = row.get(f'landmark_{lm_idx}_z', 0.0)
            v = row.get(f'landmark_{lm_idx}_visibility', 0.0)
            
            # Handle NaN
            if pd.isna(x): x = 0.0
            if pd.isna(y): y = 0.0
            if pd.isna(z): z = 0.0
            if pd.isna(v): v = 0.0
            
            frame_lms.append([float(x), float(y), float(z), float(v)])
        
        landmarks.append(frame_lms)
    
    return np.array(landmarks, dtype=np.float32)

def extract_full_swing_features(swing_id, poses_dir, phases_dir, output_dir):
    """
    Extract pose features for full swing (all frames from address to finish).
    
    Returns:
        features: np.array (num_frames, 33, 4)
        num_frames: int
    """
    
    poses_csv = os.path.join(poses_dir, f"{swing_id}_cleaned_poses.csv")
    
    if not os.path.exists(poses_csv):
        return None, 0
    
    # Extract all frames
    features = extract_pose_landmarks(poses_csv)
    
    if features is None or len(features) == 0:
        return None, 0
    
    # Save
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{swing_id}_landmarks.npy")
    np.save(output_path, features)
    
    return features, len(features)

def extract_phase_window_features(swing_id, start_frame, end_frame, 
                                  poses_dir, output_dir, phase_name=None):
    """
    Extract pose features for a specific phase window.
    
    Used for phase-specific training (if needed later).
    
    Returns:
        features: np.array (window_length, 33, 4)
    """
    
    poses_csv = os.path.join(poses_dir, f"{swing_id}_cleaned_poses.csv")
    
    if not os.path.exists(poses_csv):
        return None
    
    features = extract_pose_landmarks(poses_csv, start_frame, end_frame)
    
    if features is not None and phase_name:
        os.makedirs(output_dir, exist_ok=True)
        phase_output = os.path.join(output_dir, f"{swing_id}_{phase_name}_landmarks.npy")
        np.save(phase_output, features)
    
    return features

def normalize_landmarks(landmarks, method='z-score'):
    """
    Normalize landmarks to handle different body scales/positions.
    
    Args:
        landmarks: np.array (num_frames, 33, 4)
        method: 'z-score', 'minmax', or None
    
    Returns:
        normalized: np.array (num_frames, 33, 4)
    """
    
    if method == 'z-score':
        # Z-score normalize per landmark dimension
        for lm_idx in range(33):
            for dim in range(3):  # x, y, z (not visibility)
                values = landmarks[:, lm_idx, dim]
                mean = np.nanmean(values)
                std = np.nanstd(values)
                if std > 0:
                    landmarks[:, lm_idx, dim] = (values - mean) / std
    
    elif method == 'minmax':
        # Min-max scale per landmark
        for lm_idx in range(33):
            for dim in range(3):
                values = landmarks[:, lm_idx, dim]
                min_val = np.nanmin(values)
                max_val = np.nanmax(values)
                if max_val > min_val:
                    landmarks[:, lm_idx, dim] = (values - min_val) / (max_val - min_val)
    
    return landmarks

def batch_extract_features(training_csv, poses_dir, phases_dir, output_dir,
                          normalize=True, normalize_method='z-score'):
    """
    Extract pose features for all swings in training dataset.
    
    Args:
        training_csv: Path to datasets/training_150_labeled.csv
        poses_dir: Path to data/extracted_poses
        phases_dir: Path to data/keyframes
        output_dir: Output directory for .npy files
        normalize: Whether to normalize landmarks
        normalize_method: 'z-score' or 'minmax'
    """
    
    df = pd.read_csv(training_csv)
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nExtracting pose features for {len(df)} swings...")
    print(f"Output dir: {output_dir}")
    
    success_count = 0
    failed_swings = []
    stats = {
        'total_frames': [],
        'num_swings': 0
    }
    
    for idx, row in df.iterrows():
        swing_id = row['swing_id']
        
        try:
            features, num_frames = extract_full_swing_features(
                swing_id, poses_dir, phases_dir, output_dir
            )
            
            if features is not None:
                if normalize:
                    features = normalize_landmarks(features, method=normalize_method)
                    # Re-save normalized
                    output_path = os.path.join(output_dir, f"{swing_id}_landmarks.npy")
                    np.save(output_path, features)
                
                success_count += 1
                stats['total_frames'].append(num_frames)
                
                if (idx + 1) % 20 == 0:
                    print(f"  ✓ {idx + 1}/{len(df)} extracted")
            else:
                failed_swings.append((swing_id, "no landmarks"))
        
        except Exception as e:
            failed_swings.append((swing_id, str(e)))
    
    # Summary
    print(f"\n" + "="*70)
    print(f"FEATURE EXTRACTION SUMMARY")
    print(f"="*70)
    print(f"✓ Successfully extracted: {success_count}/{len(df)}")
    print(f"  Total frames processed: {sum(stats['total_frames'])}")
    print(f"  Avg frames per swing: {np.mean(stats['total_frames']):.0f}")
    print(f"  Min frames: {min(stats['total_frames'])}")
    print(f"  Max frames: {max(stats['total_frames'])}")
    
    if failed_swings:
        print(f"\n⚠ Failed to extract {len(failed_swings)} swings:")
        for swing_id, reason in failed_swings[:10]:
            print(f"  - {swing_id}: {reason}")
        if len(failed_swings) > 10:
            print(f"  ... and {len(failed_swings) - 10} more")
    
    print(f"\n✓ Features saved to: {output_dir}")
    print(f"="*70 + "\n")
    
    return success_count, failed_swings

def verify_extracted_features(features_dir, training_csv):
    """
    Verify that all training swings have extracted features.
    """
    
    df = pd.read_csv(training_csv)
    
    print(f"\nVerifying extracted features...")
    
    missing = []
    for idx, row in df.iterrows():
        swing_id = row['swing_id']
        feature_path = os.path.join(features_dir, f"{swing_id}_landmarks.npy")
        
        if not os.path.exists(feature_path):
            missing.append(swing_id)
        else:
            # Load and check shape
            features = np.load(feature_path)
            if features.shape[1:] != (33, 4):
                print(f"⚠ {swing_id}: unexpected shape {features.shape}")
    
    if missing:
        print(f"⚠ Missing features for {len(missing)} swings:")
        for swing_id in missing[:10]:
            print(f"  - {swing_id}")
    else:
        print(f"✓ All {len(df)} swings have extracted features")
    
    return len(missing) == 0

if __name__ == '__main__':
    
    # Step 1: Extract features for all training swings
    print("\n" + "="*70)
    print("EXTRACTING POSE LANDMARKS FOR NEURAL NETWORK TRAINING")
    print("="*70)
    
    success, failed = batch_extract_features(
        training_csv='datasets/training_150_labeled.csv',
        poses_dir='data/extracted_poses',
        phases_dir='data/keyframes',
        output_dir='datasets/pose_features',
        normalize=True,
        normalize_method='z-score'
    )
    
    # Step 2: Verify
    all_good = verify_extracted_features(
        features_dir='datasets/pose_features',
        training_csv='datasets/training_150_labeled.csv'
    )
    
    if all_good:
        print("\n✓ READY FOR NEURAL NETWORK TRAINING!")
    else:
        print("\n⚠ Some features are missing. Fix errors above before training.")
