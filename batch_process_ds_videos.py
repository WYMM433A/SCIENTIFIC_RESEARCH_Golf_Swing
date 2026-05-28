"""
Batch process all videos in data/ds_videos/ 
Extract: keyframes + landmarks (poses) + phase segmentation using neural network
"""

import os
from pathlib import Path
from pipeline import GolfSwingPipeline

def batch_process_videos(input_folder='data/ds_videos', method='neural-network', model_path='models/pose_swingnet_trained.pth'):
    """
    Process all videos in a folder.
    
    Args:
        input_folder: Path to folder with videos
        method: 'rule-based' or 'neural-network'
        model_path: Path to trained NN model
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
    print(f"\n{'=' * 70}")
    print(f"BATCH PROCESSING: {len(videos)} videos with {method}")
    print(f"{'=' * 70}\n")
    
    # Create pipeline
    pipeline = GolfSwingPipeline(
        output_base_dir='data',
        phase_method=method,
        model_path=model_path if method == 'neural-network' else None
    )
    
    # Process each video
    successful = 0
    failed = 0
    failed_videos = []
    
    for idx, video_path in enumerate(videos, 1):
        print(f"\n[{idx}/{len(videos)}] Processing: {video_path.name}")
        print("-" * 70)
        
        try:
            results = pipeline.run(str(video_path), show_preview=False)
            
            print(f"✅ SUCCESS")
            print(f"   Keyframes: {results['keyframes_dir']}")
            print(f"   Poses CSV: {results['poses_csv']}")
            print(f"   Phases CSV: {results['phases_csv']}")
            
            successful += 1
            
        except Exception as e:
            print(f"❌ FAILED: {e}")
            failed += 1
            failed_videos.append(video_path.name)
    
    # Summary
    print(f"\n{'=' * 70}")
    print(f"BATCH PROCESSING COMPLETE")
    print(f"{'=' * 70}")
    print(f"✅ Successful: {successful}")
    print(f"❌ Failed:     {failed}")
    
    if failed_videos:
        print(f"\nFailed videos:")
        for video_name in failed_videos:
            print(f"  • {video_name}")
    
    print(f"\n📂 All outputs saved to: data/")
    print(f"   • Keyframes: data/keyframes/")
    print(f"   • Poses:     data/extracted_poses/")
    print(f"   • Phases:    data/keyframes/<video>_nn/")
    print(f"{'=' * 70}\n")

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Batch process golf swing videos')
    parser.add_argument('--input', '-i', default='data/ds_videos', help='Input folder (default: data/ds_videos)')
    parser.add_argument('--method', '-m', choices=['rule-based', 'neural-network'], 
                        default='neural-network', help='Phase detection method')
    parser.add_argument('--model', type=str, default='models/pose_swingnet_trained.pth',
                        help='Path to trained model (for neural-network)')
    
    args = parser.parse_args()
    
    batch_process_videos(
        input_folder=args.input,
        method=args.method,
        model_path=args.model
    )
