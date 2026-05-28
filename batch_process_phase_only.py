"""
Batch process all videos in data/ds_videos/ 
Extract: keyframes + landmarks (poses) + phase segmentation ONLY (no scoring)

Tracks progress in: data/batch_process_checkpoint.txt
Automatically skips already-processed videos
"""

import os
import json
from pathlib import Path
from pipeline import GolfSwingPipeline

def load_checkpoint(checkpoint_file='data/batch_process_checkpoint.json'):
    """Load list of already-processed videos"""
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            return json.load(f)
    return {'processed': [], 'failed': []}

def save_checkpoint(processed, failed, checkpoint_file='data/batch_process_checkpoint.json'):
    """Save progress checkpoint"""
    checkpoint = {'processed': processed, 'failed': failed}
    with open(checkpoint_file, 'w') as f:
        json.dump(checkpoint, f, indent=2)

def batch_process_phase_segmentation(input_folder='data/ds_videos', method='neural-network', model_path='models/pose_swingnet_trained.pth', limit=None):
    """
    Process videos for PHASE SEGMENTATION ONLY (skip scoring).
    
    Tracks progress and automatically skips already-processed videos.
    
    Outputs:
    - Cleaned video
    - Pose landmarks CSV
    - 8 keyframes (one per phase)
    - Phase frame numbers CSV
    
    Args:
        input_folder: Path to folder with videos
        method: 'rule-based' or 'neural-network'
        model_path: Path to trained NN model
        limit: Process only first N videos (None = all)
    """
    
    input_path = Path(input_folder)
    
    if not input_path.exists():
        print(f"❌ Folder not found: {input_folder}")
        return
    
    # Find all video files
    video_extensions = ['.mp4', '.mov', '.avi', '.mkv']
    videos = []
    for ext in video_extensions:
        videos.extend(input_path.glob(f'*{ext}'))
    
    if not videos:
        print(f"❌ No videos found in {input_folder}")
        return
    
    videos = sorted(videos)
    
    # Limit to N videos if specified
    if limit and limit > 0:
        videos = videos[:limit]
    
    # Load checkpoint
    checkpoint_file = 'data/batch_process_checkpoint.json'
    checkpoint = load_checkpoint(checkpoint_file)
    processed_videos = set(checkpoint['processed'])
    failed_videos_list = checkpoint['failed']
    print(f"\n{'=' * 70}")
    print(f"PHASE SEGMENTATION BATCH: {len(videos)} videos with {method}")
    print(f"(Skipping scoring step)")
    print(f"{'=' * 70}")
    
    # Show checkpoint status
    print(f"\n📊 CHECKPOINT STATUS:")
    print(f"   Already processed: {len(processed_videos)}")
    print(f"   Failed previously: {len(failed_videos_list)}")
    print(f"   New to process:    {len(videos) - len([v for v in videos if v.stem in processed_videos])}\n")
    
    successful = 0
    failed = 0
    
    for idx, video_path in enumerate(videos, 1):
        video_stem = video_path.stem
        
        # Skip if already processed
        if video_stem in processed_videos:
            print(f"[{idx}/{len(videos)}] ⏭️  SKIP (already processed): {video_path.name}")
            continue
        
        print(f"\n[{idx}/{len(videos)}] Processing: {video_path.name}")
        print("-" * 70)
        
        try:
            # Create pipeline
            pipeline = GolfSwingPipeline(
                output_base_dir='data',
                phase_method=method,
                model_path=model_path if method == 'neural-network' else None
            )
            
            # Set video name from path (CRITICAL - prevents None output folder)
            pipeline.video_name = video_path.stem
            
            # Step 1: Clean video
            print("   [1/3] Cleaning video...")
            pipeline._clean_video(str(video_path))
            
            # Step 2: Extract poses
            print("   [2/3] Extracting poses and landmarks...")
            pipeline._extract_poses()
            
            # Step 3: Detect phases (includes keyframe extraction)
            print("   [3/3] Detecting 8 phases with neural network...")
            pipeline._detect_phases()
            
            # SKIP Step 4 (scoring) - user doesn't need it
            
            print(f"✅ SUCCESS - Phase segmentation complete")
            print(f"   Keyframes: {pipeline.keyframes_dir / pipeline.output_folder_name}")
            print(f"   Poses:     {pipeline.poses_csv_path}")
            print(f"   Phases:    {pipeline.phases_csv_path}")
            
            # Update checkpoint
            processed_videos.add(video_stem)
            save_checkpoint(list(processed_videos), failed_videos_list, checkpoint_file)
            
            successful += 1
            
        except Exception as e:
            print(f"❌ FAILED: {e}")
            
            # Add to failed list
            if video_stem not in failed_videos_list:
                failed_videos_list.append(video_stem)
            save_checkpoint(list(processed_videos), failed_videos_list, checkpoint_file)
            
            failed += 1
    
    # Summary
    print(f"\n{'=' * 70}")
    print(f"PHASE SEGMENTATION COMPLETE")
    print(f"{'=' * 70}")
    print(f"✅ Successful in this run: {successful}")
    print(f"❌ Failed in this run:     {failed}")
    print(f"\n📊 CUMULATIVE PROGRESS:")
    print(f"   Total processed: {len(processed_videos)}")
    print(f"   Total failed:    {len(failed_videos_list)}")
    
    if failed_videos_list:
        print(f"\n⚠️  Failed videos (saved for retry):")
        for video_name in failed_videos_list[-10:]:  # Show last 10
            print(f"   • {video_name}")
        if len(failed_videos_list) > 10:
            print(f"   ... and {len(failed_videos_list) - 10} more")
    
    print(f"\n✅ CHECKPOINT: {checkpoint_file}")
    print(f"   Automatically saves progress after each video")
    print(f"   Next run will skip already-processed videos\n")
    
    print(f"📂 Outputs saved to:")
    print(f"   • Keyframes (8 per video): data/keyframes/")
    print(f"   • Pose landmarks:          data/extracted_poses/")
    print(f"   • Phase frame numbers:     data/keyframes/<video>_nn/")
    print(f"\n📊 NOT generated (skipped):")
    print(f"   • Phase scores (0-100)")
    print(f"   • Feedback/recommendations")
    print(f"{'=' * 70}\n")

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Batch process golf swing videos - phase segmentation only',
        epilog="""
Examples:
  python batch_process_phase_only.py --limit 50        # First 50 videos (autoresumes)
  python batch_process_phase_only.py                   # All remaining videos
  python batch_process_phase_only.py --reset          # Delete checkpoint and restart
        """
    )
    parser.add_argument('--input', '-i', default='data/ds_videos', help='Input folder (default: data/ds_videos)')
    parser.add_argument('--method', '-m', choices=['rule-based', 'neural-network'], 
                        default='neural-network', help='Phase detection method')
    parser.add_argument('--model', type=str, default='models/pose_swingnet_trained.pth',
                        help='Path to trained model (for neural-network)')
    parser.add_argument('--limit', '-l', type=int, default=None,
                        help='Process only first N videos (default: all unprocessed)')
    parser.add_argument('--reset', action='store_true',
                        help='Delete checkpoint and restart from beginning')
    
    args = parser.parse_args()
    
    # Reset checkpoint if requested
    if args.reset:
        checkpoint_file = 'data/batch_process_checkpoint.json'
        if os.path.exists(checkpoint_file):
            os.remove(checkpoint_file)
            print(f"✅ Checkpoint deleted. Starting fresh.\n")
    
    batch_process_phase_segmentation(
        input_folder=args.input,
        method=args.method,
        model_path=args.model,
        limit=args.limit
    )
