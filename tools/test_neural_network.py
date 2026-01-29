"""
Test the trained PoseSwingNet model on videos.

Usage:
    python tools/test_neural_network.py --video 0
    python tools/test_neural_network.py --video golf_swing_007
    python tools/test_neural_network.py --video 0 --compare
"""

import sys
import argparse
from pathlib import Path

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.phase.adapter import create_predictor


def test_model(video_id='0', model_path='models/pose_swingnet_trained.pth'):
    """Test the neural network model on a video."""
    
    # Paths
    pose_csv = PROJECT_ROOT / 'data' / 'extracted_poses' / f'{video_id}_cleaned_poses.csv'
    video_path = PROJECT_ROOT / 'data' / 'cleaned_videos' / f'{video_id}_cleaned.mp4'
    output_dir = PROJECT_ROOT / 'data' / 'keyframes' / f'{video_id}_nn'
    
    # Check files exist
    if not pose_csv.exists():
        print(f"❌ Pose CSV not found: {pose_csv}")
        return
    if not video_path.exists():
        print(f"❌ Video not found: {video_path}")
        return
    
    print("=" * 60)
    print("🧠 TESTING NEURAL NETWORK PHASE DETECTOR")
    print("=" * 60)
    print(f"Model: {model_path}")
    print(f"Video: {video_path}")
    print(f"Poses: {pose_csv}")
    print(f"Output: {output_dir}")
    print("=" * 60)
    
    # Load model
    predictor = create_predictor('neural-network', str(PROJECT_ROOT / model_path))
    
    # Run prediction
    results = predictor.process(
        csv_path=str(pose_csv),
        video_path=str(video_path),
        output_dir=str(output_dir)
    )
    
    # Print results
    print("\n✅ PHASE DETECTION RESULTS")
    print("-" * 40)
    
    for phase_name in predictor.PHASE_NAMES:
        if phase_name in results['phase_ranges']:
            start, end = results['phase_ranges'][phase_name]
            duration = end - start + 1 if end > start else 0
            print(f"  {phase_name:15s}: frames {start:4d} - {end:4d} ({duration:3d} frames)")
    
    print("-" * 40)
    print(f"\n📁 Output saved to: {output_dir}")
    print(f"📄 CSV: {results['phases_csv']}")
    
    # List extracted keyframes
    print("\n🖼️ Extracted keyframes:")
    for phase, data in results['keyframes'].items():
        print(f"  {phase}: {data['image_path']}")
    
    return results


def compare_methods(video_id='0'):
    """Compare rule-based vs neural network on same video."""
    
    pose_csv = PROJECT_ROOT / 'data' / 'extracted_poses' / f'{video_id}_cleaned_poses.csv'
    video_path = PROJECT_ROOT / 'data' / 'cleaned_videos' / f'{video_id}_cleaned.mp4'
    
    if not pose_csv.exists() or not video_path.exists():
        print(f"❌ Video {video_id} files not found")
        return
    
    print("\n" + "=" * 70)
    print("📊 COMPARING RULE-BASED vs NEURAL NETWORK")
    print("=" * 70)
    
    # Rule-based
    print("\n[1] Rule-Based Detector")
    rb_predictor = create_predictor('rule-based')
    rb_results = rb_predictor.process(
        csv_path=str(pose_csv),
        video_path=str(video_path),
        output_dir=str(PROJECT_ROOT / 'data' / 'keyframes' / f'{video_id}_rule')
    )
    
    # Neural network
    print("\n[2] Neural Network Detector")
    nn_predictor = create_predictor('neural-network', str(PROJECT_ROOT / 'models/pose_swingnet_trained.pth'))
    nn_results = nn_predictor.process(
        csv_path=str(pose_csv),
        video_path=str(video_path),
        output_dir=str(PROJECT_ROOT / 'data' / 'keyframes' / f'{video_id}_nn')
    )
    
    # Compare
    print("\n" + "=" * 70)
    print("COMPARISON: Key Frame Selection")
    print("=" * 70)
    print(f"{'Phase':<15} {'Rule-Based':>15} {'Neural Net':>15} {'Diff':>10}")
    print("-" * 55)
    
    for phase in rb_predictor.PHASE_NAMES:
        rb_range = rb_results['phase_ranges'].get(phase, (0, 0))
        nn_range = nn_results['phase_ranges'].get(phase, (0, 0))
        
        rb_mid = (rb_range[0] + rb_range[1]) // 2
        nn_mid = (nn_range[0] + nn_range[1]) // 2
        diff = abs(rb_mid - nn_mid)
        
        print(f"{phase:<15} {rb_mid:>15} {nn_mid:>15} {diff:>10}")
    
    print("=" * 70)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test neural network phase detector')
    parser.add_argument('--video', default='0', help='Video ID to test (e.g., 0, 119, golf_swing_001)')
    parser.add_argument('--model', default='models/pose_swingnet_trained.pth', help='Model path')
    parser.add_argument('--compare', action='store_true', help='Compare rule-based vs neural network')
    args = parser.parse_args()
    
    if args.compare:
        compare_methods(args.video)
    else:
        test_model(args.video, args.model)
