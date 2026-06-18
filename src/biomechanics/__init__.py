"""
Biomechanics Evaluation Module v2.0
==================================
Scientific-grade golf swing analysis based on professional benchmarks.
"""

from .angles import GolfBiomechanics, GOLF_CRITICAL_ANGLES
from .benchmarks import GolfBenchmarks
from .comparator import SwingComparator, SwingBiomechanicsEvaluator
from .phase_scorer import PhaseScorer
from .scoring_config import (
    PHASE_WEIGHTS, METRIC_WEIGHTS, SCORING_THRESHOLDS,
    SCORE_RANGES, FEEDBACK_TEMPLATES, KINEMATIC_SEQUENCE
)

__all__ = [
    'GolfBiomechanics',
    'GOLF_CRITICAL_ANGLES',
    'GolfBenchmarks',
    'SwingComparator',
    'SwingBiomechanicsEvaluator',
    'PhaseScorer',
    'PHASE_WEIGHTS',
    'METRIC_WEIGHTS',
    'SCORING_THRESHOLDS',
    'SCORE_RANGES',
    'FEEDBACK_TEMPLATES',
    'KINEMATIC_SEQUENCE'
]
