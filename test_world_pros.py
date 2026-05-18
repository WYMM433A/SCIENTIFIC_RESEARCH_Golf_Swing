import sys
import os
sys.path.insert(0, '.')

import pandas as pd
from src.biomechanics.phase_scorer import PhaseScorer
from src.biomechanics.angles import GolfBiomechanics
from src.phase import create_predictor

def score_swing(poses_csv_path, video_path):
    """Score a single swing."""
    poses_df = pd.read_csv(poses_csv_path)
    
    # Get phase detection
    predictor = create_predictor('neural-network', 'models/pose_swingnet_trained.pth')
    result = predictor.process(poses_csv_path, video_path, 'temp_world_pro')
    phase_ranges = result['phase_ranges']
    
    # Initialize biomechanics and scorer
    biomechanics = GolfBiomechanics()
    scorer = PhaseScorer()
    
    phase_scores = {}
    
    # Score each phase
    for phase_name in ['address', 'takeaway', 'mid_backswing', 'top', 'mid_downswing', 'impact', 'follow_through', 'finish']:
        phase_key = None
        for key in phase_ranges.keys():
            if key.lower().replace('-', '_') == phase_name:
                phase_key = key
                break
        
        if not phase_key:
            continue
        
        start, end = phase_ranges[phase_key]
        phase_frames = poses_df[(poses_df['frame'] >= start) & (poses_df['frame'] <= end)]
        
        if len(phase_frames) == 0:
            continue
        
        # Set reference at address
        if phase_name == 'address':
            address_frame = phase_frames.iloc[0]
            biomechanics.set_reference_position(address_frame)
        
        # Get metrics for the key frame of this phase
        key_frame = phase_frames.iloc[len(phase_frames)//2]  # Middle frame
        metrics = biomechanics.calculate_all_metrics(key_frame)
        
        # For mid_downswing, add kinematic sequence
        if phase_name == 'mid_downswing':
            kinematic_data = biomechanics.compute_angular_velocity_sequence(poses_df, start, end)
        else:
            kinematic_data = None
        
        # Score the phase
        score, details = scorer.score_phase_with_metrics(phase_name, metrics, kinematic_data)
        phase_scores[phase_name] = score
    
    # Calculate overall
    overall = sum(phase_scores.values()) / len(phase_scores) if phase_scores else 0
    
    return phase_scores, overall

# Test world-level pros
print("WORLD-LEVEL PROS vs BEGINNER")
print("="*60)

test_pairs = [
    ("206_cleaned_poses.csv", "206"),
    ("72_cleaned_poses.csv", "72"),
    ("me_cleaned_poses.csv", "me (beginner)"),
]

for csv_file, label in test_pairs:
    csv_path = f"data/extracted_poses/{csv_file}"
    
    # Try to find corresponding video
    video_name = csv_file.replace("_cleaned_poses.csv", "_cleaned.mp4")
    video_path = f"data/videos_160/{video_name}"
    
    if not os.path.exists(csv_path):
        print(f"❌ {label}: File not found")
        continue
    
    if not os.path.exists(video_path):
        print(f"⚠️  {label}: Video not found, using dummy path")
    
    try:
        phases, overall = score_swing(csv_path, video_path)
        print(f"\n{label} (Overall: {overall:.1f})")
        for phase, score in sorted(phases.items()):
            print(f"  {phase:20s}: {score:6.1f}")
    except Exception as e:
        print(f"❌ {label}: ERROR - {str(e)[:60]}")

print("\n" + "="*60)
print("WORLD PRO vs BEGINNER COMPARISON")
print("="*60)
