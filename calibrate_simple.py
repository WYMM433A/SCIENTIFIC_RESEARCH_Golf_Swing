"""
Direct Threshold Calibration
=============================
Simplest approach: Load pose CSV files, calculate metrics, analyze distributions.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from src.biomechanics import GolfBiomechanics

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 10)

def calibrate_from_pose_files(num_files=30):
    """Extract and analyze metrics from pose CSV files."""
    
    print("\n" + "="*80)
    print("DIRECT THRESHOLD CALIBRATION")
    print("="*80)
    
    # Get pose files
    extracted_dir = Path('data/extracted_poses')
    csv_files = sorted([f for f in extracted_dir.glob('*_cleaned_poses.csv')])[:num_files]
    
    print(f"\nLoading {len(csv_files)} pose files...\n")
    
    all_metrics_data = []
    
    for i, csv_path in enumerate(csv_files, 1):
        try:
            print(f"[{i}/{len(csv_files)}] {csv_path.name}", end=" ... ")
            
            # Load pose data and ensure it has a 'frame' column
            pose_df = pd.read_csv(csv_path, index_col=0)
            
            if 'frame' not in pose_df.columns:
                pose_df.reset_index(drop=True, inplace=True)
                pose_df['frame'] = pose_df.index
            
            if len(pose_df) < 20:
                print(f"SKIP (only {len(pose_df)} frames)")
                continue
            
            # Initialize GolfBiomechanics with the DataFrame
            biomechanics = GolfBiomechanics(pose_data=pose_df)
            biomechanics.set_reference_position(frame=int(pose_df.iloc[0]['frame']))
            
            # Calculate metrics for different frame ranges to get phase-like data
            # Address: first 5% of swing
            # Top: around 30-40% of swing  
            # Impact: around 70-80% of swing
            # Follow-through: final 20% of swing
            
            n_frames = len(pose_df)
            phases_data = {
                'address': (0, max(1, int(0.05 * n_frames))),
                'top': (int(0.30 * n_frames), int(0.45 * n_frames)),
                'mid_downswing': (int(0.45 * n_frames), int(0.70 * n_frames)),
                'impact': (int(0.70 * n_frames), int(0.85 * n_frames)),
                'follow_through': (int(0.85 * n_frames), n_frames - 1),
            }
            
            for phase_name, (start, end) in phases_data.items():
                if end <= start:
                    continue
                
                # Get middle frame of phase
                mid_frame = start + (end - start) // 2
                frame_num = int(pose_df.iloc[mid_frame]['frame'])
                
                # Calculate metrics for this frame
                metrics = biomechanics.calculate_all_metrics(frame=frame_num)
                
                if isinstance(metrics, dict) and len(metrics) > 0:
                    record = {
                        'video': csv_path.stem.replace('_cleaned_poses', ''),
                        'phase': phase_name,
                        **metrics
                    }
                    all_metrics_data.append(record)
            
            print(f"OK ({n_frames} frames)")
            
        except Exception as e:
            print(f"ERROR: {e}")
            continue
    
    if not all_metrics_data:
        print("\nERROR: No metrics extracted!")
        return None
    
    metrics_df = pd.DataFrame(all_metrics_data)
    
    print(f"\n{'='*80}")
    print(f"Extracted {len(metrics_df)} phase metric records from {len(csv_files)} videos")
    print(f"{'='*80}\n")
    
    # Analyze distributions by phase
    print("METRIC DISTRIBUTIONS BY PHASE")
    print("="*80)
    
    calibration_report = {
        'timestamp': pd.Timestamp.now().isoformat(),
        'num_videos': len(csv_files),
        'phases': {}
    }
    
    for phase in sorted(metrics_df['phase'].unique()):
        phase_data = metrics_df[metrics_df['phase'] == phase]
        n_records = len(phase_data)
        
        print(f"\n{phase.upper()} (n={n_records} records):")
        print("-" * 80)
        
        # Get numeric columns
        numeric_cols = [col for col in phase_data.columns 
                       if col not in ['video', 'phase'] 
                       and pd.api.types.is_numeric_dtype(phase_data[col])]
        
        phase_calibration = {}
        
        for col in sorted(numeric_cols):
            values = phase_data[col].dropna()
            
            if len(values) == 0:
                continue
            
            # Calculate statistics
            q1 = values.quantile(0.25)
            q3 = values.quantile(0.75)
            median = values.median()
            mean = values.mean()
            std = values.std()
            min_val = values.min()
            max_val = values.max()
            
            # Store calibration
            phase_calibration[col] = {
                'count': len(values),
                'mean': round(mean, 2),
                'std': round(std, 2),
                'min': round(min_val, 2),
                'q25': round(q1, 2),
                'median': round(median, 2),
                'q75': round(q3, 2),
                'max': round(max_val, 2),
                'ideal_min': round(q1, 2),
                'ideal_max': round(q3, 2),
                'acceptable_min': round(min_val, 2),
                'acceptable_max': round(max_val, 2),
            }
            
            print(f"\n  {col}:")
            print(f"    n={len(values)}")
            print(f"    mean ± std:         {mean:8.2f} ± {std:6.2f}")
            print(f"    range (min-max):    [{min_val:8.2f}, {max_val:8.2f}]")
            print(f"    IQR (Q1-Q3):        [{q1:8.2f}, {q3:8.2f}]  ← IDEAL RANGE")
            
        calibration_report['phases'][phase] = phase_calibration
    
    # Save report
    report_path = Path('data/metrics/calibration_report.json')
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(report_path, 'w') as f:
        json.dump(calibration_report, f, indent=2, default=str)
    
    print(f"\n{'='*80}")
    print(f"Calibration report saved to: {report_path}")
    
    # Save metrics
    metrics_csv = Path('data/metrics/extracted_metrics.csv')
    metrics_df.to_csv(metrics_csv, index=False)
    print(f"Raw metrics saved to: {metrics_csv}")
    
    # Create visualization
    create_visualization(metrics_df, phase_calibration)
    
    return metrics_df, calibration_report

def create_visualization(metrics_df, phase_calibration):
    """Create distribution plots for each phase."""
    
    print("\nGenerating visualizations...")
    
    phases = sorted(metrics_df['phase'].unique())
    fig, axes = plt.subplots(len(phases), 1, figsize=(16, 5*len(phases)))
    
    if len(phases) == 1:
        axes = [axes]
    
    for ax, phase in zip(axes, phases):
        phase_data = metrics_df[metrics_df['phase'] == phase]
        numeric_cols = [col for col in phase_data.columns 
                       if col not in ['video', 'phase'] 
                       and pd.api.types.is_numeric_dtype(phase_data[col])][:10]  # First 10 metrics
        
        # Create box plot
        plot_data = [phase_data[col].dropna().values for col in numeric_cols]
        
        bp = ax.boxplot(plot_data, labels=[col[:20] for col in numeric_cols], patch_artist=True)
        
        # Color the boxes
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
        
        ax.set_title(f'{phase.upper()} - Metric Distributions', fontsize=14, fontweight='bold')
        ax.set_ylabel('Value')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    viz_path = Path('data/metrics/calibration_distributions.png')
    viz_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(viz_path, dpi=100, bbox_inches='tight')
    print(f"Distribution plot saved to: {viz_path}")
    
    plt.close()

if __name__ == '__main__':
    metrics_df, report = calibrate_from_pose_files(num_files=30)
    
    if metrics_df is not None:
        print("\n" + "="*80)
        print("CALIBRATION COMPLETE!")
        print("="*80)
        print("\nNext steps:")
        print("1. Review data/metrics/calibration_report.json")
        print("2. Update src/biomechanics/scoring_config.py with new ideal/acceptable ranges")
        print("3. Run pipeline on full dataset to validate")
