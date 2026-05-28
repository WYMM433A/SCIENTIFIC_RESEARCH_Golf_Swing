"""
Train PoseSwingNet with FULL GolfDB Dataset (1,300 videos)

Usage:
    python train_full_golfdb.py --epochs 100
    python train_full_golfdb.py --epochs 50 --max-videos 500  (test with subset)
"""

import sys
from pathlib import Path

# Run the standard training script but with full dataset path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import the training function
from tools.train_with_golfdb import main
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train PoseSwingNet with FULL GolfDB (1,300 videos)')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs (default: 100)')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size (default: 32)')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--max-videos', type=int, default=None, help='Limit to first N videos (for testing)')
    parser.add_argument('--skip-extraction', action='store_true', help='Skip pose extraction if already done')
    
    args = parser.parse_args()
    
    # Build command args for the standard script
    sys.argv = [
        'train_with_golfdb.py',
        '--golfdb', str(PROJECT_ROOT / 'data'),  # Expects golfDB.pkl here
        '--epochs', str(args.epochs),
        '--batch-size', str(args.batch_size),
        '--lr', str(args.lr),
    ]
    
    if args.max_videos:
        sys.argv.extend(['--max-videos', str(args.max_videos)])
    
    if args.skip_extraction:
        sys.argv.append('--skip-extraction')
    
    print("\n" + "="*70)
    print("🏌️ TRAINING POSESWINGNET WITH FULL GOLFDB (1,300 VIDEOS)")
    print("="*70)
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    if args.max_videos:
        print(f"Limit: First {args.max_videos} videos (testing)")
    print("="*70 + "\n")
    
    # Run the training
    main()
