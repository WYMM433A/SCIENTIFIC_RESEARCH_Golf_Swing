"""
Train PoseSwingNet with GolfDB Dataset
======================================
Uses your existing SwingAnalyzer to extract poses from GolfDB videos,
combines with GolfDB labels, and trains the neural network.

Prerequisites:
1. GolfDB preprocessed videos in: data/videos_160/
2. GolfDB annotations: data/golfDB.pkl

Usage:
    cd "C:\Code Related\DataStorm"
    
    # Quick test (10 videos, 5 epochs)
    python tools/train_with_golfdb.py --max-videos 10 --epochs 5
    
    # Full training
    python tools/train_with_golfdb.py --epochs 50
    
    # Skip pose extraction if already done
    python tools/train_with_golfdb.py --skip-extraction --epochs 50
"""

import os
import sys
import argparse
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.pose import SwingAnalyzer
from src.phase.neural_model import PoseSwingNet
from src.constants import PHASE_NAMES


# ============================================
# PHASE MAPPING: GolfDB → DataStorm
# ============================================

# Use shared constants (PHASE_NAMES imported from src.constants)
# GolfDB events[1-8] map to our 8 phases


# ============================================
# STEP 1: EXTRACT POSES FROM GOLFDB VIDEOS
# ============================================

def extract_poses_from_golfdb(golfdb_path, output_dir, max_videos=None):
    """
    Extract poses from GolfDB videos using your SwingAnalyzer.
    
    Args:
        golfdb_path: Path to golfdb data folder (contains videos_160/, golfDB.pkl)
        output_dir: Where to save extracted poses
        max_videos: Limit for testing (None = all)
    
    Returns:
        List of successfully processed video IDs
    """
    golfdb_path = Path(golfdb_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    videos_dir = golfdb_path / "videos_160"
    pkl_path = golfdb_path / "golfDB.pkl"
    
    # Check paths
    if not videos_dir.exists():
        raise FileNotFoundError(f"Videos not found: {videos_dir}")
    if not pkl_path.exists():
        raise FileNotFoundError(f"golfDB.pkl not found: {pkl_path}. Run generate_splits.py first!")
    
    # Load annotations
    df = pd.read_pickle(pkl_path)
    print(f"✓ Loaded {len(df)} annotations from golfDB.pkl")
    
    if max_videos:
        df = df.head(max_videos)
        print(f"  Limited to {max_videos} videos for testing")
    
    # Initialize your pose analyzer
    analyzer = SwingAnalyzer()
    
    processed_ids = []
    failed_ids = []
    
    print(f"\nExtracting poses from {len(df)} videos...")
    
    for idx in tqdm(range(len(df)), desc="Extracting poses"):
        row = df.iloc[idx]
        video_id = row['id']
        video_path = videos_dir / f"{video_id}.mp4"
        output_csv = output_dir / f"{video_id}_poses.csv"
        
        # Skip if already processed
        if output_csv.exists():
            processed_ids.append(video_id)
            continue
        
        if not video_path.exists():
            failed_ids.append(video_id)
            continue
        
        try:
            # Use YOUR existing SwingAnalyzer
            pose_df = analyzer.processVideo(
                video_path=str(video_path),
                output_csv=str(output_csv),
                show_preview=False
            )
            
            if pose_df is not None and len(pose_df) > 0:
                processed_ids.append(video_id)
            else:
                failed_ids.append(video_id)
                
        except Exception as e:
            print(f"\n  ✗ Error processing {video_id}: {e}")
            failed_ids.append(video_id)
    
    print(f"\n✓ Poses extracted: {len(processed_ids)}/{len(df)}")
    if failed_ids:
        print(f"✗ Failed: {len(failed_ids)} videos")
    
    return processed_ids


# ============================================
# STEP 2: CREATE TRAINING DATASET
# ============================================

class GolfDBPoseDataset(Dataset):
    """
    Dataset that loads poses from CSVs and GolfDB labels.
    """
    
    def __init__(self, annotations_df, poses_dir, seq_length=64, train=True):
        """
        Args:
            annotations_df: DataFrame from golfDB.pkl
            poses_dir: Directory containing pose CSVs
            seq_length: Number of frames per training sample
            train: If True, random sampling; if False, full video
        """
        self.poses_dir = Path(poses_dir)
        self.seq_length = seq_length
        self.train = train
        
        # Filter to only videos with extracted poses
        valid_ids = []
        valid_rows = []
        
        for idx in range(len(annotations_df)):
            row = annotations_df.iloc[idx]
            video_id = row['id']
            pose_csv = self.poses_dir / f"{video_id}_poses.csv"
            
            if pose_csv.exists():
                valid_ids.append(video_id)
                valid_rows.append(row)
        
        self.annotations = pd.DataFrame(valid_rows).reset_index(drop=True)
        print(f"  Dataset: {len(self.annotations)} videos with poses")
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        row = self.annotations.iloc[idx]
        video_id = row['id']
        events = row['events'].copy()
        
        # Normalize events (so first frame = 0)
        events = events - events[0]
        
        # Load poses
        pose_csv = self.poses_dir / f"{video_id}_poses.csv"
        pose_df = pd.read_csv(pose_csv)
        
        # Get feature columns (all except 'frame')
        feature_cols = [c for c in pose_df.columns if c != 'frame']
        poses = pose_df[feature_cols].values  # (num_frames, 132)
        
        # Normalize poses
        poses = (poses - poses.mean(axis=0)) / (poses.std(axis=0) + 1e-8)
        
        num_frames = len(poses)
        
        if self.train:
            # Random window sampling
            if num_frames > self.seq_length:
                start = np.random.randint(0, num_frames - self.seq_length)
            else:
                start = 0
            
            end = start + self.seq_length
            
            # Handle short videos
            if end > num_frames:
                # Pad with last frame
                poses_seq = np.vstack([
                    poses[start:],
                    np.tile(poses[-1:], (end - num_frames, 1))
                ])
                labels = self._create_labels(events, num_frames, start, end)
                # Pad labels too
                labels = np.concatenate([labels, np.full(end - num_frames, 8)])
            else:
                poses_seq = poses[start:end]
                labels = self._create_labels(events, num_frames, start, end)
        else:
            # Full video for validation
            poses_seq = poses
            labels = self._create_labels(events, num_frames, 0, num_frames)
        
        return {
            'poses': torch.FloatTensor(poses_seq),
            'labels': torch.LongTensor(labels),
            'video_id': video_id
        }
    
    def _create_labels(self, events, num_frames, start, end):
        """
        Create per-frame labels from events array.
        events[1] through events[8] are the 8 phases.
        """
        labels = np.full(end - start, 8, dtype=np.int32)  # Default: no-event (class 8)
        
        for phase_id in range(8):
            event_idx = phase_id + 1  # events[1] = Address, events[2] = Takeaway, etc.
            
            if event_idx < len(events):
                event_frame = int(events[event_idx])
                
                # Map to position in our window
                rel_pos = event_frame - start
                
                if 0 <= rel_pos < len(labels):
                    labels[rel_pos] = phase_id
        
        return labels


# ============================================
# STEP 3: TRAINING LOOP
# ============================================

def train_model(
    train_dataset,
    val_dataset,
    model,
    epochs=50,
    batch_size=16,
    learning_rate=0.001,
    save_dir='models'
):
    """
    Train the PoseSwingNet model.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n✓ Using device: {device}")
    
    model = model.to(device)
    
    # DataLoader - adjust batch size if dataset is small
    effective_batch_size = min(batch_size, len(train_dataset))
    train_loader = DataLoader(
        train_dataset, 
        batch_size=effective_batch_size, 
        shuffle=True,
        num_workers=0,  # Windows compatibility
        drop_last=False  # Keep all samples for small datasets
    )
    
    # Loss with class weighting (8 phases + no-event)
    # Events are rare (~1:35 ratio), so weight them higher
    weights = torch.FloatTensor([1/8]*8 + [1/35]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training
    print(f"\n{'='*60}")
    print("TRAINING POSESWINGNET")
    print(f"{'='*60}")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Training samples: {len(train_dataset)}")
    print(f"{'='*60}\n")
    
    best_val_acc = 0
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}', leave=False):
            poses = batch['poses'].to(device)      # (B, seq_len, 132)
            labels = batch['labels'].to(device)    # (B, seq_len)
            
            # Forward
            logits = model(poses)                  # (B, seq_len, 9)
            
            # Reshape for loss
            logits_flat = logits.view(-1, 9)       # (B*seq_len, 9)
            labels_flat = labels.view(-1)          # (B*seq_len,)
            
            loss = criterion(logits_flat, labels_flat)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Accuracy (on event frames only)
            preds = torch.argmax(logits_flat, dim=1)
            event_mask = labels_flat < 8
            if event_mask.sum() > 0:
                correct += (preds[event_mask] == labels_flat[event_mask]).sum().item()
                total += event_mask.sum().item()
        
        avg_loss = total_loss / len(train_loader)
        train_acc = correct / total if total > 0 else 0
        
        # Validation
        if val_dataset is not None and len(val_dataset) > 0:
            val_acc = evaluate_model(model, val_dataset, device)
            print(f"Epoch {epoch+1:3d}: Loss={avg_loss:.4f}, Train Acc={train_acc:.1%}, Val Acc={val_acc:.1%}")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc
                }, save_dir / 'pose_swingnet_best.pth')
        else:
            print(f"Epoch {epoch+1:3d}: Loss={avg_loss:.4f}, Train Acc={train_acc:.1%}")
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, save_dir / f'pose_swingnet_epoch_{epoch+1}.pth')
    
    # Save final model
    torch.save(model.state_dict(), save_dir / 'pose_swingnet_trained.pth')
    print(f"\n✓ Model saved to: {save_dir / 'pose_swingnet_trained.pth'}")
    
    return model


def evaluate_model(model, dataset, device):
    """Evaluate model accuracy on event frames."""
    model.eval()
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for i in range(len(dataset)):
            sample = dataset[i]
            poses = sample['poses'].unsqueeze(0).to(device)
            labels = sample['labels'].to(device)
            
            logits = model(poses).squeeze(0)
            preds = torch.argmax(logits, dim=1)
            
            # Only count event frames
            event_mask = labels < 8
            if event_mask.sum() > 0:
                correct += (preds[event_mask] == labels[event_mask]).sum().item()
                total += event_mask.sum().item()
    
    return correct / total if total > 0 else 0


# ============================================
# MAIN
# ============================================

def main():
    parser = argparse.ArgumentParser(description='Train PoseSwingNet with GolfDB')
    parser.add_argument('--golfdb', default=None, help='Path to GolfDB data folder (default: data/)')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--seq-length', type=int, default=64, help='Sequence length for training')
    parser.add_argument('--split', type=int, default=1, help='Validation split (1-4)')
    parser.add_argument('--max-videos', type=int, default=None, help='Limit videos (for testing)')
    parser.add_argument('--start-video', type=int, default=0, help='Start from video index (0-based)')
    parser.add_argument('--end-video', type=int, default=None, help='End at video index (exclusive)')
    parser.add_argument('--skip-extraction', action='store_true', help='Skip pose extraction (use existing)')
    parser.add_argument('--resume', type=str, default=None, help='Resume from existing model (e.g., models/pose_swingnet_trained.pth)')
    args = parser.parse_args()
    
    # Use local data/ folder by default
    if args.golfdb:
        golfdb_path = Path(args.golfdb)
    else:
        golfdb_path = PROJECT_ROOT / 'data'
    
    poses_dir = PROJECT_ROOT / 'data' / 'golfdb_poses'
    
    print("\n" + "="*70)
    print("🏌️ TRAIN POSESWINGNET WITH GOLFDB")
    print("="*70)
    print(f"GolfDB path: {golfdb_path}")
    print(f"Poses dir: {poses_dir}")
    if args.resume:
        print(f"Resume from: {args.resume}")
    if args.start_video > 0 or args.end_video:
        print(f"Video range: {args.start_video} - {args.end_video or 'end'}")
    print("="*70 + "\n")
    
    # ========== STEP 1: Extract Poses ==========
    if not args.skip_extraction:
        print("STEP 1: Extracting poses from GolfDB videos...")
        print("-"*50)
        extract_poses_from_golfdb(golfdb_path, poses_dir, args.max_videos)
    else:
        print("STEP 1: Skipping pose extraction (--skip-extraction)")
    
    # ========== STEP 2: Load Annotations & Create Datasets ==========
    print("\nSTEP 2: Creating training datasets...")
    print("-"*50)
    
    pkl_path = golfdb_path / "golfDB.pkl"
    df = pd.read_pickle(pkl_path)
    
    # Apply video range filtering
    if args.end_video:
        df = df.iloc[args.start_video:args.end_video]
        print(f"  Using videos {args.start_video} to {args.end_video}")
    elif args.max_videos:
        df = df.head(args.max_videos)
        print(f"  Using first {args.max_videos} videos")
    
    # Split into train/val
    train_df = df[df['split'] != args.split].reset_index(drop=True)
    val_df = df[df['split'] == args.split].reset_index(drop=True)
    
    print(f"  Train annotations: {len(train_df)}")
    print(f"  Val annotations: {len(val_df)} (split {args.split})")
    
    train_dataset = GolfDBPoseDataset(train_df, poses_dir, seq_length=args.seq_length, train=True)
    val_dataset = GolfDBPoseDataset(val_df, poses_dir, seq_length=args.seq_length, train=False)
    
    if len(train_dataset) == 0:
        print("\n✗ ERROR: No training data! Make sure poses are extracted.")
        print(f"  Check: {poses_dir}")
        return
    
    # ========== STEP 3: Initialize Model ==========
    print("\nSTEP 3: Initializing PoseSwingNet...")
    print("-"*50)
    
    model = PoseSwingNet(
        input_size=132,      # 33 landmarks × 4 values
        hidden_size=128,
        num_layers=2,
        num_classes=9        # 8 phases + no-event
    )
    
    # Resume from existing model if specified
    if args.resume:
        resume_path = PROJECT_ROOT / args.resume if not Path(args.resume).is_absolute() else Path(args.resume)
        if resume_path.exists():
            checkpoint = torch.load(resume_path, map_location='cpu')
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                print(f"  ✓ Resumed from checkpoint: {resume_path}")
            else:
                model.load_state_dict(checkpoint)
                print(f"  ✓ Resumed from model: {resume_path}")
        else:
            print(f"  ✗ Resume model not found: {resume_path}")
            return
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {total_params:,}")
    
    # ========== STEP 4: Train ==========
    print("\nSTEP 4: Training...")
    print("-"*50)
    
    model = train_model(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        model=model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        save_dir=PROJECT_ROOT / 'models'
    )
    
    # ========== DONE ==========
    print("\n" + "="*70)
    print("✅ TRAINING COMPLETE!")
    print("="*70)
    print(f"\nModel saved to: {PROJECT_ROOT / 'models' / 'pose_swingnet_trained.pth'}")
    print("\nTo use in your pipeline, update adapter.py:")
    print("  predictor = create_predictor('neural-network', 'models/pose_swingnet_trained.pth')")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
