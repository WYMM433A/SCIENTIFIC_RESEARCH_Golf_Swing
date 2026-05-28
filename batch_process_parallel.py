"""
Parallel batch processing - processes multiple videos simultaneously
Uses multiprocessing to utilize all CPU cores
Approximately 2-3x faster than sequential processing
"""

import os
import multiprocessing as mp
from pathlib import Path
from pipeline import GolfSwingPipeline

def process_single_video(args):
    """Worker function for multiprocessing"""
    video_path, method, model_path, idx, total = args
    
    try:
        pipeline = GolfSwingPipeline(
            output_base_dir='data',
            phase_method=method,
            model_path=model_path if method == 'neural-network' else None
        )
        
        print(f"[{idx}/{total}] Processing: {video_path.name}")
        results = pipeline.run(str(video_path), show_preview=False)
        
        return {
            'status': 'success',
            'video': video_path.name,
            'keyframes_dir': results['keyframes_dir']
        }
        
    except Exception as e:
        return {
            'status': 'failed',
            'video': video_path.name,
            'error': str(e)
        }

def batch_process_parallel(input_folder='data/ds_videos', method='neural-network', 
                          model_path='models/pose_swingnet_trained.pth', num_workers=None):
    """
    Process all videos in parallel using multiprocessing.
    
    Args:
        input_folder: Path to folder with videos
        method: 'rule-based' or 'neural-network'
        model_path: Path to trained NN model
        num_workers: Number of parallel processes (default: CPU count - 1)
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
    
    # Set number of workers
    if num_workers is None:
        num_workers = max(1, mp.cpu_count() - 1)  # Leave 1 core free
    
    print(f"\n{'=' * 70}")
    print(f"PARALLEL BATCH PROCESSING: {len(videos)} videos")
    print(f"Method: {method}")
    print(f"Workers: {num_workers} (CPU count: {mp.cpu_count()})")
    print(f"{'=' * 70}\n")
    
    # Prepare arguments for workers
    worker_args = [
        (video, method, model_path, idx + 1, len(videos))
        for idx, video in enumerate(videos)
    ]
    
    # Process in parallel
    successful = 0
    failed = 0
    failed_videos = []
    
    with mp.Pool(processes=num_workers) as pool:
        results = pool.map(process_single_video, worker_args)
    
    # Collect results
    for result in results:
        if result['status'] == 'success':
            print(f"✅ {result['video']}")
            successful += 1
        else:
            print(f"❌ {result['video']}: {result['error']}")
            failed += 1
            failed_videos.append(result['video'])
    
    # Summary
    print(f"\n{'=' * 70}")
    print(f"PARALLEL BATCH PROCESSING COMPLETE")
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
    print(f"{'=' * 70}\n")

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Parallel batch process golf swing videos')
    parser.add_argument('--input', '-i', default='data/ds_videos', help='Input folder')
    parser.add_argument('--method', '-m', choices=['rule-based', 'neural-network'], 
                        default='neural-network', help='Phase detection method')
    parser.add_argument('--model', type=str, default='models/pose_swingnet_trained.pth',
                        help='Path to trained model')
    parser.add_argument('--workers', '-w', type=int, default=None,
                        help='Number of parallel workers (default: CPU count - 1)')
    
    args = parser.parse_args()
    
    batch_process_parallel(
        input_folder=args.input,
        method=args.method,
        model_path=args.model,
        num_workers=args.workers
    )
