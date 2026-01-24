"""
Golf Swing Benchmarks - Pro golfer reference values

Ideal values based on research and analysis of professional golfers.
"""

import json
import os
from typing import Dict, Optional

# Default benchmark values (research-based)
DEFAULT_BENCHMARKS = {
    "address": {
        "spine_angle": {"min": 35, "max": 55, "ideal": 45},
        "spine_lateral_tilt": {"min": -5, "max": 5, "ideal": 0},
        "shoulder_rotation": {"min": -10, "max": 10, "ideal": 0},
        "hip_rotation": {"min": -10, "max": 10, "ideal": 0},
        "x_factor": {"min": 0, "max": 10, "ideal": 0},
        "lead_arm_angle": {"min": 160, "max": 180, "ideal": 170},
        "trail_elbow_angle": {"min": 160, "max": 180, "ideal": 170},
        "wrist_hinge": {"min": 140, "max": 170, "ideal": 150},
        "lead_knee_flex": {"min": 145, "max": 170, "ideal": 155},
        "trail_knee_flex": {"min": 145, "max": 170, "ideal": 155},
        "stance_width_ratio": {"min": 1.0, "max": 1.5, "ideal": 1.2}
    },
    "top": {
        "spine_angle": {"min": 30, "max": 50, "ideal": 40},
        "spine_lateral_tilt": {"min": 5, "max": 20, "ideal": 10},
        "shoulder_rotation": {"min": 75, "max": 110, "ideal": 90},
        "hip_rotation": {"min": 35, "max": 55, "ideal": 45},
        "x_factor": {"min": 40, "max": 60, "ideal": 50},
        "lead_arm_angle": {"min": 165, "max": 180, "ideal": 175},
        "trail_elbow_angle": {"min": 75, "max": 105, "ideal": 90},
        "wrist_hinge": {"min": 70, "max": 110, "ideal": 90},
        "lead_knee_flex": {"min": 140, "max": 165, "ideal": 150},
        "trail_knee_flex": {"min": 145, "max": 170, "ideal": 155}
    },
    "mid_downswing": {
        "spine_angle": {"min": 35, "max": 55, "ideal": 45},
        "x_factor": {"min": 45, "max": 65, "ideal": 55},
        "wrist_hinge": {"min": 60, "max": 100, "ideal": 80},
        "lead_knee_flex": {"min": 150, "max": 175, "ideal": 160},
        "trail_elbow_angle": {"min": 100, "max": 140, "ideal": 120}
    },
    "impact": {
        "spine_angle": {"min": 35, "max": 55, "ideal": 45},
        "spine_lateral_tilt": {"min": 10, "max": 25, "ideal": 15},
        "shoulder_rotation": {"min": 10, "max": 35, "ideal": 20},
        "hip_rotation": {"min": 30, "max": 50, "ideal": 40},
        "x_factor": {"min": 25, "max": 45, "ideal": 35},
        "lead_arm_angle": {"min": 160, "max": 180, "ideal": 170},
        "trail_elbow_angle": {"min": 135, "max": 170, "ideal": 150},
        "lead_knee_flex": {"min": 160, "max": 180, "ideal": 170},
        "trail_knee_flex": {"min": 130, "max": 160, "ideal": 145}
    }
}


class GolfBenchmarks:
    """
    Manages benchmark values for golf swing analysis.
    
    Supports both default (research-based) and custom (pro golfer) benchmarks.
    """
    
    def __init__(self, benchmarks_path: Optional[str] = None):
        """
        Initialize with benchmark data.
        
        Args:
            benchmarks_path: Path to custom benchmarks JSON file, or None for defaults
        """
        if benchmarks_path and os.path.exists(benchmarks_path):
            with open(benchmarks_path, 'r') as f:
                self.benchmarks = json.load(f)
        else:
            self.benchmarks = DEFAULT_BENCHMARKS
    
    def get_phase_benchmarks(self, phase: str) -> Dict:
        """
        Get benchmark values for a specific phase.
        
        Args:
            phase: Phase name (e.g., 'address', 'top', 'impact')
            
        Returns:
            Dictionary of benchmark values for that phase
        """
        # Normalize phase name
        phase_key = phase.lower().replace('-', '_').replace(' ', '_')
        
        # Map variations to standard keys
        phase_mapping = {
            'address': 'address',
            'takeaway': 'address',
            'mid_backswing': 'top',
            'midbackswing': 'top',
            'top': 'top',
            'top_of_backswing': 'top',
            'mid_downswing': 'mid_downswing',
            'middownswing': 'mid_downswing',
            'impact': 'impact',
            'follow_through': 'impact',
            'followthrough': 'impact',
            'finish': 'impact'
        }
        
        standard_phase = phase_mapping.get(phase_key, 'address')
        return self.benchmarks.get(standard_phase, {})
    
    def get_metric_benchmark(self, phase: str, metric: str) -> Optional[Dict]:
        """
        Get benchmark for a specific metric in a phase.
        
        Args:
            phase: Phase name
            metric: Metric name (e.g., 'x_factor', 'spine_angle')
            
        Returns:
            Dictionary with 'min', 'max', 'ideal' or None if not found
        """
        phase_benchmarks = self.get_phase_benchmarks(phase)
        return phase_benchmarks.get(metric)
    
    def is_in_range(self, phase: str, metric: str, value: float) -> bool:
        """
        Check if a value is within acceptable range for a metric.
        
        Args:
            phase: Phase name
            metric: Metric name
            value: Actual measured value
            
        Returns:
            True if within range, False otherwise
        """
        benchmark = self.get_metric_benchmark(phase, metric)
        if benchmark is None:
            return True  # No benchmark = no restriction
        
        return benchmark['min'] <= value <= benchmark['max']
    
    def get_deviation(self, phase: str, metric: str, value: float) -> Optional[Dict]:
        """
        Calculate deviation from ideal for a metric.
        
        Args:
            phase: Phase name
            metric: Metric name
            value: Actual measured value
            
        Returns:
            Dictionary with deviation info or None if no benchmark
        """
        benchmark = self.get_metric_benchmark(phase, metric)
        if benchmark is None:
            return None
        
        deviation = value - benchmark['ideal']
        
        return {
            'actual': value,
            'ideal': benchmark['ideal'],
            'deviation': deviation,
            'deviation_pct': (deviation / benchmark['ideal']) * 100 if benchmark['ideal'] != 0 else 0,
            'in_range': benchmark['min'] <= value <= benchmark['max'],
            'severity': self._calculate_severity(value, benchmark)
        }
    
    def _calculate_severity(self, value: float, benchmark: Dict) -> str:
        """
        Calculate severity of deviation.
        
        Returns:
            'good', 'minor', 'moderate', or 'major'
        """
        ideal = benchmark['ideal']
        min_val = benchmark['min']
        max_val = benchmark['max']
        
        # Within range
        if min_val <= value <= max_val:
            if abs(value - ideal) < (max_val - min_val) * 0.2:
                return 'good'
            return 'minor'
        
        # Outside range - calculate how far
        if value < min_val:
            overshoot = min_val - value
            range_size = max_val - min_val
        else:
            overshoot = value - max_val
            range_size = max_val - min_val
        
        if overshoot < range_size * 0.5:
            return 'moderate'
        return 'major'
    
    def save_benchmarks(self, path: str):
        """Save current benchmarks to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.benchmarks, f, indent=2)
    
    def update_benchmark(self, phase: str, metric: str, min_val: float, max_val: float, ideal: float):
        """
        Update or add a benchmark value.
        
        Args:
            phase: Phase name
            metric: Metric name
            min_val, max_val, ideal: Benchmark values
        """
        if phase not in self.benchmarks:
            self.benchmarks[phase] = {}
        
        self.benchmarks[phase][metric] = {
            'min': min_val,
            'max': max_val,
            'ideal': ideal
        }
