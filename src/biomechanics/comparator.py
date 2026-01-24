"""
Swing Comparator - Compare user swing to benchmarks

Analyzes biomechanical metrics and identifies areas for improvement.
"""

from typing import Dict, List, Tuple, Optional
import pandas as pd
from .angles import GolfBiomechanics, GOLF_CRITICAL_ANGLES
from .benchmarks import GolfBenchmarks


class SwingComparator:
    """
    Compare user's swing metrics to benchmarks and identify issues.
    """
    
    def __init__(self, benchmarks: GolfBenchmarks = None):
        """
        Initialize with benchmark data.
        
        Args:
            benchmarks: GolfBenchmarks instance or None for defaults
        """
        self.benchmarks = benchmarks or GolfBenchmarks()
    
    def compare_phase(self, phase: str, metrics: Dict[str, float]) -> Dict:
        """
        Compare metrics for a single phase to benchmarks.
        
        Args:
            phase: Phase name (e.g., 'top', 'impact')
            metrics: Dictionary of metric name -> value
            
        Returns:
            Comparison results with deviations and issues
        """
        results = {
            'phase': phase,
            'metrics': {},
            'issues': [],
            'strengths': [],
            'overall_score': 0
        }
        
        total_metrics = 0
        good_metrics = 0
        
        for metric_name, value in metrics.items():
            if metric_name in ['frame', 'phase']:
                continue
                
            deviation = self.benchmarks.get_deviation(phase, metric_name, value)
            
            if deviation:
                results['metrics'][metric_name] = deviation
                total_metrics += 1
                
                if deviation['severity'] == 'good':
                    good_metrics += 1
                    results['strengths'].append({
                        'metric': metric_name,
                        'value': value,
                        'ideal': deviation['ideal']
                    })
                elif deviation['severity'] in ['moderate', 'major']:
                    # Get description from GOLF_CRITICAL_ANGLES
                    angle_info = GOLF_CRITICAL_ANGLES.get(metric_name, {})
                    results['issues'].append({
                        'metric': metric_name,
                        'value': value,
                        'ideal': deviation['ideal'],
                        'deviation': deviation['deviation'],
                        'severity': deviation['severity'],
                        'description': angle_info.get('description', ''),
                        'importance': angle_info.get('importance', '')
                    })
        
        # Calculate overall score (0-100)
        if total_metrics > 0:
            results['overall_score'] = int((good_metrics / total_metrics) * 100)
        
        # Sort issues by severity
        severity_order = {'major': 0, 'moderate': 1, 'minor': 2}
        results['issues'].sort(key=lambda x: severity_order.get(x['severity'], 3))
        
        return results
    
    def compare_full_swing(self, phase_metrics: Dict[str, Dict[str, float]]) -> Dict:
        """
        Compare all phases of a swing to benchmarks.
        
        Args:
            phase_metrics: Dictionary of phase -> metrics dictionary
            
        Returns:
            Full swing comparison with per-phase and overall analysis
        """
        results = {
            'phases': {},
            'summary': {
                'total_issues': 0,
                'major_issues': 0,
                'overall_score': 0,
                'priority_fixes': [],
                'strengths': []
            }
        }
        
        total_score = 0
        phase_count = 0
        all_issues = []
        all_strengths = []
        
        for phase, metrics in phase_metrics.items():
            phase_comparison = self.compare_phase(phase, metrics)
            results['phases'][phase] = phase_comparison
            
            total_score += phase_comparison['overall_score']
            phase_count += 1
            
            # Collect issues with phase context
            for issue in phase_comparison['issues']:
                issue['phase'] = phase
                all_issues.append(issue)
            
            for strength in phase_comparison['strengths']:
                strength['phase'] = phase
                all_strengths.append(strength)
        
        # Calculate overall score
        if phase_count > 0:
            results['summary']['overall_score'] = int(total_score / phase_count)
        
        # Count issues by severity
        results['summary']['total_issues'] = len(all_issues)
        results['summary']['major_issues'] = len([i for i in all_issues if i['severity'] == 'major'])
        
        # Get top priority fixes (major issues first, then by phase importance)
        phase_priority = {
            'address': 1,
            'top': 2, 
            'impact': 3,
            'mid_downswing': 4,
            'takeaway': 5,
            'mid_backswing': 6,
            'follow_through': 7,
            'finish': 8
        }
        
        all_issues.sort(key=lambda x: (
            0 if x['severity'] == 'major' else 1,
            phase_priority.get(x['phase'].lower().replace('-', '_'), 10)
        ))
        
        results['summary']['priority_fixes'] = all_issues[:3]  # Top 3 issues
        results['summary']['strengths'] = all_strengths[:3]  # Top 3 strengths
        
        return results
    
    def generate_report(self, comparison: Dict) -> str:
        """
        Generate a text report from comparison results.
        
        Args:
            comparison: Results from compare_full_swing()
            
        Returns:
            Formatted text report
        """
        lines = []
        
        lines.append("=" * 60)
        lines.append("GOLF SWING ANALYSIS REPORT")
        lines.append("=" * 60)
        lines.append("")
        
        # Overall score
        score = comparison['summary']['overall_score']
        lines.append(f"Overall Score: {score}/100")
        lines.append(f"Total Issues: {comparison['summary']['total_issues']}")
        lines.append(f"Major Issues: {comparison['summary']['major_issues']}")
        lines.append("")
        
        # Priority fixes
        lines.append("-" * 40)
        lines.append("PRIORITY IMPROVEMENTS:")
        lines.append("-" * 40)
        
        for i, issue in enumerate(comparison['summary']['priority_fixes'], 1):
            lines.append(f"\n{i}. {issue['metric'].replace('_', ' ').title()} [{issue['phase']}]")
            lines.append(f"   Current: {issue['value']:.1f}° | Ideal: {issue['ideal']:.1f}°")
            lines.append(f"   Issue: {issue['description']}")
            lines.append(f"   Why it matters: {issue['importance']}")
        
        # Strengths
        if comparison['summary']['strengths']:
            lines.append("")
            lines.append("-" * 40)
            lines.append("STRENGTHS:")
            lines.append("-" * 40)
            
            for strength in comparison['summary']['strengths']:
                lines.append(f"• {strength['metric'].replace('_', ' ').title()} "
                           f"[{strength['phase']}]: {strength['value']:.1f}°")
        
        # Per-phase breakdown
        lines.append("")
        lines.append("-" * 40)
        lines.append("PHASE-BY-PHASE BREAKDOWN:")
        lines.append("-" * 40)
        
        for phase, phase_data in comparison['phases'].items():
            lines.append(f"\n{phase.upper()} - Score: {phase_data['overall_score']}/100")
            
            if phase_data['issues']:
                for issue in phase_data['issues']:
                    severity_icon = "⚠️" if issue['severity'] == 'major' else "⚡"
                    lines.append(f"  {severity_icon} {issue['metric']}: {issue['value']:.1f}° "
                               f"(ideal: {issue['ideal']:.1f}°)")
        
        lines.append("")
        lines.append("=" * 60)
        
        return "\n".join(lines)
    
    def get_feedback_for_phase(self, phase: str, metrics: Dict[str, float]) -> List[str]:
        """
        Get concise feedback strings for a phase.
        
        Args:
            phase: Phase name
            metrics: Dictionary of metric values
            
        Returns:
            List of feedback strings
        """
        comparison = self.compare_phase(phase, metrics)
        feedback = []
        
        for issue in comparison['issues']:
            metric_name = issue['metric'].replace('_', ' ').title()
            deviation = issue['deviation']
            
            if deviation > 0:
                feedback.append(f"{metric_name}: {abs(deviation):.0f}° too high")
            else:
                feedback.append(f"{metric_name}: {abs(deviation):.0f}° too low")
        
        return feedback
