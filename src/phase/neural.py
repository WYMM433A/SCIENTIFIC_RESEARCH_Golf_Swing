"""
SwingNet Phase Detection - Extract 8 Key Frames
Simplified version that finds the most confident frame for each phase

Output:
- 8 images (one per phase) saved to disk
- CSV with frame numbers and confidence scores
- Visual summary showing all 8 phases
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import cv2
import os
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


# ============================================
# SIMPLIFIED SWINGNET MODEL
# ============================================

class PoseSwingNet(nn.Module):
    """
    Bi-LSTM model for phase detection from pose sequences
    """
    
    def __init__(self, input_size=132, hidden_size=128, num_layers=2, num_classes=8):
        super(PoseSwingNet, self).__init__()
        
        # Input projection
        self.input_projection = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, hidden_size),
            nn.ReLU()
        )
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            hidden_size, 
            hidden_size, 
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.2 if num_layers > 1 else 0
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, input_size)
        Returns:
            logits: (batch, seq_len, num_classes)
        """
        x = self.input_projection(x)
        lstm_out, _ = self.lstm(x)
        logits = self.classifier(lstm_out)
        return logits


# ============================================
# DATA LOADING
# ============================================

def load_pose_csv(csv_path):
    """
    Load pose CSV and prepare features
    
    Returns:
        poses: numpy array (num_frames, 132)
        frames: frame numbers
    """
    df = pd.read_csv(csv_path)
    
    # Extract feature columns (all except 'frame')
    feature_cols = [col for col in df.columns if col != 'frame']
    poses = df[feature_cols].values
    frames = df['frame'].values
    
    print(f"✓ Loaded {len(poses)} frames with {poses.shape[1]} features")
    
    # Normalize
    poses = (poses - poses.mean(axis=0)) / (poses.std(axis=0) + 1e-8)
    
    return poses, frames


def create_sequences(poses, sequence_length=30, stride=5):
    """
    Create overlapping sequences
    
    Args:
        poses: (num_frames, 132)
        sequence_length: frames per sequence
        stride: step size between sequences
        
    Returns:
        sequences: (num_seq, seq_len, 132)
        center_frames: center frame index for each sequence
    """
    sequences = []
    center_frames = []
    
    for i in range(0, len(poses) - sequence_length + 1, stride):
        seq = poses[i:i + sequence_length]
        sequences.append(seq)
        # Center frame of this sequence
        center_frames.append(i + sequence_length // 2)
    
    sequences = np.array(sequences)
    print(f"✓ Created {len(sequences)} sequences of length {sequence_length}")
    
    return sequences, center_frames


# ============================================
# PHASE PREDICTION
# ============================================

def predict_phases(model, sequences, device='cpu'):
    """
    Predict phases for all sequences
    
    Returns:
        all_probs: (num_sequences, seq_len, 8) - probabilities
        all_preds: (num_sequences, seq_len) - predicted classes
    """
    model.eval()
    model.to(device)
    
    sequences_tensor = torch.FloatTensor(sequences).to(device)
    
    all_probs = []
    all_preds = []
    
    with torch.no_grad():
        batch_size = 16
        for i in range(0, len(sequences_tensor), batch_size):
            batch = sequences_tensor[i:i + batch_size]
            logits = model(batch)  # (batch, seq_len, 8)
            probs = torch.softmax(logits, dim=-1)  # Convert to probabilities
            preds = torch.argmax(probs, dim=-1)
            
            all_probs.append(probs.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
    
    all_probs = np.concatenate(all_probs, axis=0)
    all_preds = np.concatenate(all_preds, axis=0)
    
    return all_probs, all_preds


def find_best_frames_per_phase(probs, preds, center_frames, num_phases=8):
    """
    Find the frame with highest confidence for each phase
    
    Args:
        probs: (num_seq, seq_len, 8) - probabilities
        preds: (num_seq, seq_len) - predictions
        center_frames: center frame index for each sequence
        
    Returns:
        best_frames: dict {phase_id: (frame_num, confidence)}
    """
    
    best_frames = {}
    
    for phase_id in range(num_phases):
        max_conf = 0
        best_frame = None
        
        # Check all sequences
        for seq_idx in range(len(probs)):
            # Get center frame of sequence
            frame_num = center_frames[seq_idx]
            
            # Get probabilities for center frame
            center_idx = probs.shape[1] // 2
            frame_probs = probs[seq_idx, center_idx, :]  # (8,)
            
            # Check if this phase has highest probability
            if frame_probs[phase_id] > max_conf:
                max_conf = frame_probs[phase_id]
                best_frame = frame_num
        
        if best_frame is not None:
            best_frames[phase_id] = {
                'frame': best_frame,
                'confidence': float(max_conf),
                'phase_name': PHASE_NAMES[phase_id]
            }
    
    return best_frames


# ============================================
# VISUALIZATION
# ============================================

PHASE_NAMES = [
    "Address",
    "Toe-up",
    "Mid-backswing",
    "Top",
    "Mid-downswing",
    "Impact",
    "Mid-follow-through",
    "Finish"
]

PHASE_COLORS = [
    (255, 0, 0),      # Blue
    (255, 128, 0),    # Light Blue
    (255, 255, 0),    # Cyan
    (0, 255, 0),      # Green
    (0, 255, 128),    # Yellow-Green
    (0, 255, 255),    # Yellow
    (128, 0, 255),    # Orange
    (255, 0, 255)     # Pink
]


def extract_frames_from_video(video_path, frame_numbers):
    """
    Extract specific frames from video
    
    Args:
        video_path: path to video
        frame_numbers: list of frame numbers to extract
        
    Returns:
        frames: dict {frame_num: image}
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"✗ Cannot open video: {video_path}")
        return None
    
    frames = {}
    
    for frame_num in frame_numbers:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        
        if ret:
            frames[frame_num] = frame
        else:
            print(f"⚠ Could not read frame {frame_num}")
    
    cap.release()
    return frames


def save_phase_images(video_path, best_frames, output_dir='phase_frames'):
    """
    Save individual images for each phase
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Get frame numbers
    frame_numbers = [info['frame'] for info in best_frames.values()]
    
    # Extract frames
    frames = extract_frames_from_video(video_path, frame_numbers)
    
    if frames is None:
        return
    
    saved_files = []
    
    # Save each phase
    for phase_id, info in best_frames.items():
        frame_num = info['frame']
        phase_name = info['phase_name']
        confidence = info['confidence']
        
        if frame_num in frames:
            img = frames[frame_num].copy()
            
            # Add label
            label = f"{phase_name} ({confidence:.1%})"
            color = PHASE_COLORS[phase_id]
            
            cv2.rectangle(img, (10, 10), (400, 60), (0, 0, 0), -1)
            cv2.putText(img, label, (20, 45), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            
            # Save
            filename = f"{output_dir}/phase_{phase_id}_{phase_name.replace(' ', '_')}.jpg"
            cv2.imwrite(filename, img)
            saved_files.append(filename)
            
            print(f"✓ Saved: {filename}")
    
    return saved_files


def create_summary_image(video_path, best_frames, output_path='phase_summary.jpg'):
    """
    Create a single image showing all 8 phases in a grid
    """
    # Get frame numbers
    frame_numbers = [info['frame'] for info in best_frames.values()]
    
    # Extract frames
    frames = extract_frames_from_video(video_path, frame_numbers)
    
    if frames is None:
        return
    
    # Create figure with 2x4 grid
    fig = plt.figure(figsize=(20, 10))
    gs = GridSpec(2, 4, figure=fig, hspace=0.3, wspace=0.2)
    
    for phase_id in range(8):
        if phase_id not in best_frames:
            continue
        
        info = best_frames[phase_id]
        frame_num = info['frame']
        phase_name = info['phase_name']
        confidence = info['confidence']
        
        if frame_num not in frames:
            continue
        
        # Get subplot position
        row = phase_id // 4
        col = phase_id % 4
        ax = fig.add_subplot(gs[row, col])
        
        # Convert BGR to RGB for matplotlib
        img_rgb = cv2.cvtColor(frames[frame_num], cv2.COLOR_BGR2RGB)
        
        # Display
        ax.imshow(img_rgb)
        ax.set_title(f"{phase_id}: {phase_name}\nConfidence: {confidence:.1%}", 
                    fontsize=14, fontweight='bold')
        ax.axis('off')
    
    plt.suptitle('Golf Swing - 8 Key Phases', fontsize=18, fontweight='bold')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Summary image saved: {output_path}")


# ============================================
# MAIN PIPELINE
# ============================================

def extract_key_phase_frames(csv_path, video_path=None, model_path=None):
    """
    Main function: Extract 8 key frames (one per phase)
    
    Args:
        csv_path: Path to extracted pose CSV
        video_path: Path to original video (for frame extraction)
        model_path: Path to trained model (optional)
        
    Returns:
        best_frames: dict with best frame for each phase
    """
    
    print("="*70)
    print("SwingNet Phase Detection - Extract Key Frames")
    print("="*70)
    print()
    
    # Step 1: Load poses
    print("Step 1: Loading pose data...")
    poses, frame_nums = load_pose_csv(csv_path)
    print()
    
    # Step 2: Create sequences
    print("Step 2: Creating sequences...")
    sequences, center_frames = create_sequences(poses, sequence_length=30, stride=5)
    print()
    
    # Step 3: Initialize model
    print("Step 3: Initializing model...")
    model = PoseSwingNet(input_size=132, hidden_size=128, num_layers=2, num_classes=8)
    
    if model_path and os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        print(f"✓ Loaded model from: {model_path}")
    else:
        print("⚠ No pre-trained model found - using random weights")
        print("  Results will be random! You need to train or load a model.")
    print()
    
    # Step 4: Predict
    print("Step 4: Predicting phases...")
    probs, preds = predict_phases(model, sequences)
    print(f"✓ Generated predictions with shape: {probs.shape}")
    print()
    
    # Step 5: Find best frame per phase
    print("Step 5: Finding best frame for each phase...")
    best_frames = find_best_frames_per_phase(probs, preds, center_frames)
    print()
    
    # Print results
    print("="*70)
    print("DETECTED KEY FRAMES")
    print("="*70)
    for phase_id in range(8):
        if phase_id in best_frames:
            info = best_frames[phase_id]
            print(f"Phase {phase_id} - {info['phase_name']:20} "
                  f"Frame: {info['frame']:4d}  Confidence: {info['confidence']:.1%}")
        else:
            print(f"Phase {phase_id} - {PHASE_NAMES[phase_id]:20} NOT DETECTED")
    print("="*70)
    print()
    
    # Step 6: Save results to CSV
    results_df = pd.DataFrame([
        {
            'phase_id': phase_id,
            'phase_name': info['phase_name'],
            'frame_number': info['frame'],
            'confidence': info['confidence']
        }
        for phase_id, info in best_frames.items()
    ])
    
    csv_output = 'key_phase_frames.csv'
    results_df.to_csv(csv_output, index=False)
    print(f"✓ Results saved to: {csv_output}")
    print()
    
    # Step 7: Extract and save images (if video provided)
    if video_path and os.path.exists(video_path):
        print("Step 6: Extracting frames from video...")
        saved_files = save_phase_images(video_path, best_frames)
        print()
        
        print("Step 7: Creating summary image...")
        create_summary_image(video_path, best_frames)
        print()
    else:
        print("⚠ Video path not provided - skipping image extraction")
        print()
    
    print("="*70)
    print("✅ COMPLETE!")
    print("="*70)
    print()
    print("Output files:")
    print("  - key_phase_frames.csv (frame numbers + confidence)")
    print("  - phase_frames/ (8 individual images)")
    print("  - phase_summary.jpg (all 8 phases in one image)")
    print()
    
    return best_frames


# ============================================
# USAGE
# ============================================

if __name__ == "__main__":
    
    
    # Your extracted pose CSV
    csv_path = "data/extracted_poses/golf_swing_001_poses.csv"
    
    # Original video (for extracting actual frames)
    video_path = "data/raw_videos/golf_swing_001.mp4"
    
    # Trained model (set to None if not available yet)
    model_path = None  # or "models/swingnet_trained.pth"
    
    # Run extraction
    best_frames = extract_key_phase_frames(
        csv_path=csv_path,
        video_path=video_path,
        model_path=model_path
    )
    
    print("\nNote: Since no trained model was provided, predictions are random.")
    print("Next steps:")
    print("  1. Train the model on labeled data, OR")
    print("  2. Load pre-trained weights")