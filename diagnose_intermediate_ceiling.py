"""
Diagnose which metrics are causing intermediate ceiling effect.
Show metric values across skill levels (1-2 beginner, 6-8 intermediate, 9-10 pro).
"""

import pandas as pd
import numpy as np
from pathlib import Path
from src.biomechanics.angles import GolfBiomechanics
from src.biomechanics.scoring_config import SCORING_THRESHOLDS

# Define test swings by skill level
SWINGS = {
    "beginner_1-2": ["me_cleaned_poses.csv", "B12_cleaned_poses.csv"],
    "intermediate_6-8": ["B68_cleaned_poses.csv", "B68O_cleaned_poses.csv"],
    "pro_9-10": ["B90_cleaned_poses.csv", "B80O_cleaned_poses.csv"],
}

DATA_DIR = Path("data/extracted_poses")

def extract_metrics_at_phase(csv_file, phase_name):
    """Extract all metrics for a specific phase from a swing."""
    csv_path = DATA_DIR / csv_file
    if not csv_path.exists():
        return None
    
    df = pd.read_csv(csv_path)
    
    # Use middle frame of the phase for analysis
    phase_start = max(0, len(df) // 4)
    phase_end = min(len(df), len(df) * 3 // 4)
    frame_idx = (phase_start + phase_end) // 2
    
    if frame_idx >= len(df):
        return None
    
    biomech = GolfBiomechanics()
    biomech.set_reference_position(df.iloc[0])
    
    metrics = biomech.calculate_all_metrics(df.iloc[frame_idx])
    return metrics

def show_metric_thresholds(phase_name):
    """Show thresholds and metric values across skill levels for a phase."""
    if phase_name not in SCORING_THRESHOLDS:
        return
    
    thresholds = SCORING_THRESHOLDS[phase_name]
    print(f"\n{'='*100}")
    print(f"PHASE: {phase_name.upper()}")
    print(f"{'='*100}\n")
    
    print(f"{'Metric':<20} {'Ideal Range':<20} {'Beginner Val':<15} {'Intermediate Val':<15} {'Pro Val':<15}")
    print(f"{'-'*100}")
    
    # Collect values across skill levels
    values_by_level = {level: {} for level in SWINGS.keys()}
    
    for level, files in SWINGS.items():
        for csv_file in files:
            metrics = extract_metrics_at_phase(csv_file, phase_name)
            if metrics:
                for metric_name, value in metrics.items():
                    if metric_name not in values_by_level[level]:
                        values_by_level[level][metric_name] = []
                    values_by_level[level][metric_name].append(value)
    
    # Average values and show
    for metric_name in thresholds.keys():
        threshold = thresholds[metric_name]
        
        # Get ideal_range or acceptable_range
        ideal_range = threshold.get("ideal_range", (0, 360))
        acceptable_range = threshold.get("acceptable_range", ideal_range)
        
        beg_val = np.mean(values_by_level["beginner_1-2"].get(metric_name, [0]))
        int_val = np.mean(values_by_level["intermediate_6-8"].get(metric_name, [0]))
        pro_val = np.mean(values_by_level["pro_9-10"].get(metric_name, [0]))
        
        # Check which ones are in ideal vs acceptable
        beg_status = "✓" if ideal_range[0] <= beg_val <= ideal_range[1] else "~"
        int_status = "✓" if ideal_range[0] <= int_val <= ideal_range[1] else "~"
        pro_status = "✓" if ideal_range[0] <= pro_val <= ideal_range[1] else "~"
        
        print(f"{metric_name:<20} {str(ideal_range):<20} {beg_val:>6.1f} {beg_status}  {int_val:>6.1f} {int_status}  {pro_val:>6.1f} {pro_status}")

# Analyze problematic phases
PHASES_TO_CHECK = ["address", "takeaway", "mid_downswing", "impact", "follow_through", "finish"]

print("\n" + "█"*100)
print("█  INTERMEDIATE CEILING EFFECT DIAGNOSIS")
print("█"*100)

for phase in PHASES_TO_CHECK:
    show_metric_thresholds(phase)

print("\n" + "█"*100)
print("█  SUMMARY: Look for metrics where INTERMEDIATE ✓ score same as PRO ✓")
print("█  Those metrics have thresholds that are TOO LOOSE")
print("█"*100 + "\n")
