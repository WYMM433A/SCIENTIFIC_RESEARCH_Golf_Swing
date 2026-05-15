"""
Threshold Calibration Script
=============================
Analyzes phase metrics across GolfDB pose data to calibrate scoring thresholds.

Steps:
1. Load pre-extracted pose CSV files from data/extracted_poses
2. Calculate phase metrics for each pose file using GolfBiomechanics
3. Detect phases and score each phase
4. Aggregate metrics by phase
5. Analyze distributions (min/max/percentiles/quartiles)
6. Plot distributions to identify ideal vs acceptable ranges
7. Output calibration recommendations
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import traceback

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.biomechanics import PhaseScorer, GolfBiomechanics
from src.phase import create_predictor

# Set style for better plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)

class ThresholdCalibrator:
    """Calibrates scoring thresholds based on real pose data."""
    
    def __init__(self, num_videos=30, output_dir='data/metrics'):
        self.num_videos = num_videos
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.phase_names = ['address', 'takeaway', 'mid_backswing', 'top', 
                           'mid_downswing', 'impact', 'follow_through', 'finish']
        self.all_metrics = []
        self.phase_metrics = {phase: [] for phase in self.phase_names}
        
    def get_sample_pose_files(self):
        """Get list of pose CSV files to analyze."""
        extracted_dir = Path('data/extracted_poses')
        
        # Get cleaned_poses.csv files
        csv_files = sorted([f for f in extracted_dir.glob('*_cleaned_poses.csv')])
        
        if len(csv_files) < self.num_videos:
            print(f"Warning: Only {len(csv_files)} pose files found, using all")
            return csv_files
        
        # Take first 30 to ensure consistency
        return csv_files[:self.num_videos]
    
    def extract_metrics_from_pose_file(self, csv_path):
        """Extract phase metrics from a pose CSV file."""
        try:
            csv_path = Path(csv_path)
            print(f"  Processing: {csv_path.name}")
            
            # Load pose data
            pose_df = pd.read_csv(csv_path, index_col=0)
            
            if len(pose_df) < 10:
                print(f"    Skipping: too few frames ({len(pose_df)})")
                return None
            
            # Calculate biomechanics metrics
            biomechanics = GolfBiomechanics()
            biomechanics.set_reference_position(pose_df.iloc[0])
            
            # Detect phases using rule-based detector
            phase_detector = create_predictor('rule-based')
            phase_frames = phase_detector.detect_phases(pose_df)
            
            # Score each phase
            scorer = PhaseScorer()
            phase_metrics_dict = {}
            
            for phase_name, frame_range in phase_frames.items():
                if frame_range is None:
                    continue
                
                start_idx = int(frame_range[0])
                end_idx = int(frame_range[1])
                
                if end_idx <= start_idx or start_idx >= len(pose_df):
                    continue
                
                phase_df = pose_df.iloc[start_idx:end_idx]
                
                # Calculate all metrics for this phase
                metrics = biomechanics.calculate_all_metrics(phase_df)
                
                if metrics:
                    # Get phase score and collect metrics
                    score = scorer.score_phase_with_metrics(
                        phase_name, metrics, phase_df, phase_start_idx=start_idx
                    )
                    
                    # Store all metrics with phase label
                    metrics_with_phase = {
                        'video': csv_path.stem.replace('_cleaned_poses', ''),
                        'phase': phase_name,
                        'score': score,
                        **metrics
                    }
                    phase_metrics_dict[phase_name] = metrics_with_phase
            
            return phase_metrics_dict
            
        except Exception as e:
            print(f"    Error processing {csv_path.name}: {e}")
            traceback.print_exc()
            return None
    
    def run_calibration(self):
        """Run calibration across all sampled pose files."""
        print("\n" + "="*80)
        print("THRESHOLD CALIBRATION - RUNNING ANALYSIS")
        print("="*80)
        
        pose_files = self.get_sample_pose_files()
        print(f"\nProcessing {len(pose_files)} pose files...")
        
        for i, pose_path in enumerate(pose_files, 1):
            print(f"\n[{i}/{len(pose_files)}]")
            metrics = self.extract_metrics_from_pose_file(pose_path)
            
            if metrics:
                for phase_name, phase_data in metrics.items():
                    self.phase_metrics[phase_name].append(phase_data)
                    self.all_metrics.append(phase_data)
        
        print(f"\n{'='*80}")
        print(f"Collected metrics for {sum(len(m) for m in self.phase_metrics.values())} phase instances")
        return len(self.all_metrics) > 0
    
    def analyze_distributions(self):
        """Analyze and print metric distributions by phase."""
        print("\n" + "="*80)
        print("METRIC DISTRIBUTIONS BY PHASE")
        print("="*80)
        
        results = {}
        
        for phase_name in self.phase_names:
            metrics_list = self.phase_metrics[phase_name]
            
            if not metrics_list:
                print(f"\n{phase_name.upper()}: No data")
                continue
            
            metrics_df = pd.DataFrame(metrics_list)
            print(f"\n{phase_name.upper()} (n={len(metrics_df)} instances):")
            print("-" * 80)
            
            # Get numeric columns (exclude video, phase, score)
            numeric_cols = [col for col in metrics_df.columns 
                          if col not in ['video', 'phase', 'score'] 
                          and metrics_df[col].dtype in ['float64', 'int64']]
            
            phase_results = {}
            for col in sorted(numeric_cols):
                values = metrics_df[col].dropna()
                
                if len(values) == 0:
                    continue
                
                stats = {
                    'count': len(values),
                    'mean': values.mean(),
                    'std': values.std(),
                    'min': values.min(),
                    'q25': values.quantile(0.25),
                    'median': values.quantile(0.50),
                    'q75': values.quantile(0.75),
                    'max': values.max(),
                }
                
                phase_results[col] = stats
                
                print(f"\n  {col}:")
                print(f"    Count:   {stats['count']}")
                print(f"    Mean:    {stats['mean']:8.2f} ± {stats['std']:6.2f}")
                print(f"    Range:   [{stats['min']:8.2f}, {stats['max']:8.2f}]")
                print(f"    IQR:     [{stats['q25']:8.2f}, {stats['q75']:8.2f}]  (Q1-Q3)")
            
            results[phase_name] = phase_results
        
        return results
    
    def generate_calibration_report(self, distributions):
        """Generate calibration recommendations based on distributions."""
        print("\n" + "="*80)
        print("CALIBRATION RECOMMENDATIONS")
        print("="*80)
        
        report = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'num_videos': self.num_videos,
            'phases': {}
        }
        
        for phase_name in self.phase_names:
            if phase_name not in distributions:
                continue
            
            phase_dist = distributions[phase_name]
            print(f"\n{phase_name.upper()}:")
            print("-" * 80)
            
            recommendations = {}
            
            for metric, stats in phase_dist.items():
                # Suggested thresholds:
                # - Ideal: Q1 to Q3 (middle 50%, represents good golfers)
                # - Acceptable: Min to Max (full range observed)
                # - Flexible: ±1 std from mean
                
                ideal_min = stats['q25']
                ideal_max = stats['q75']
                acceptable_min = stats['min']
                acceptable_max = stats['max']
                
                recommendation = {
                    'metric': metric,
                    'ideal_min': round(ideal_min, 2),
                    'ideal_max': round(ideal_max, 2),
                    'acceptable_min': round(acceptable_min, 2),
                    'acceptable_max': round(acceptable_max, 2),
                    'mean': round(stats['mean'], 2),
                    'std': round(stats['std'], 2),
                    'range': round(acceptable_max - acceptable_min, 2),
                }
                
                recommendations[metric] = recommendation
                
                print(f"\n  {metric}:")
                print(f"    Ideal (Q1-Q3):      [{ideal_min:8.2f}, {ideal_max:8.2f}]")
                print(f"    Acceptable (min-max): [{acceptable_min:8.2f}, {acceptable_max:8.2f}]")
                print(f"    Mean ± Std:         {stats['mean']:8.2f} ± {stats['std']:6.2f}")
            
            report['phases'][phase_name] = recommendations
        
        return report
    
    def plot_distributions(self, output_file='data/metrics/threshold_analysis.png'):
        """Create visualization of metric distributions by phase."""
        print(f"\nGenerating distribution plots...")
        
        # Convert to DataFrame for easier plotting
        metrics_df = pd.DataFrame(self.all_metrics)
        
        if len(metrics_df) == 0:
            print("No data to plot")
            return
        
        # Create subplots for each phase
        phases_with_data = [p for p in self.phase_names if p in metrics_df['phase'].values]
        
        fig, axes = plt.subplots(len(phases_with_data), 1, figsize=(16, 4*len(phases_with_data)))
        if len(phases_with_data) == 1:
            axes = [axes]
        
        for ax, phase_name in zip(axes, phases_with_data):
            phase_data = metrics_df[metrics_df['phase'] == phase_name]
            
            # Get numeric columns
            numeric_cols = [col for col in phase_data.columns 
                          if col not in ['video', 'phase', 'score'] 
                          and phase_data[col].dtype in ['float64', 'int64']]
            
            # Plot each metric as a box plot
            plot_data = phase_data[numeric_cols].describe().T
            
            ax.set_title(f'{phase_name.upper()} - Metric Distributions (n={len(phase_data)})', fontsize=14, fontweight='bold')
            ax.axis('off')
            
            # Create text summary instead
            text = f"Metrics in {phase_name}:\n\n"
            for col in sorted(numeric_cols)[:8]:  # Show first 8 metrics
                vals = phase_data[col].dropna()
                if len(vals) > 0:
                    text += f"{col}:\n"
                    text += f"  Mean: {vals.mean():.2f} ± {vals.std():.2f}\n"
                    text += f"  Range: [{vals.min():.2f}, {vals.max():.2f}]\n\n"
            
            ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=10,
                   verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(output_path), dpi=100, bbox_inches='tight')
        print(f"Saved to: {output_path}")
        
        return output_path
    
    def save_report(self, report, output_file='data/metrics/calibration_report.json'):
        """Save calibration report as JSON."""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"Calibration report saved to: {output_path}")
        return output_path
    
    def execute(self):
        """Run complete calibration pipeline."""
        try:
            # Run calibration
            success = self.run_calibration()
            if not success:
                print("ERROR: No valid videos processed")
                return
            
            # Analyze distributions
            distributions = self.analyze_distributions()
            
            # Generate recommendations
            report = self.generate_calibration_report(distributions)
            
            # Plot and save
            self.plot_distributions()
            self.save_report(report)
            
            print("\n" + "="*80)
            print("CALIBRATION COMPLETE")
            print("="*80)
            print("\nNext steps:")
            print("1. Review calibration_report.json for metric recommendations")
            print("2. Update src/biomechanics/scoring_config.py with new thresholds")
            print("3. Test on full pipeline")
            
        except Exception as e:
            print(f"ERROR: {e}")
            traceback.print_exc()


if __name__ == '__main__':
    calibrator = ThresholdCalibrator(num_videos=30)
    calibrator.execute()
