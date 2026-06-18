"""
Biomechanics Evaluation Module v2.0
==================================
Scientific-grade golf swing analysis based on professional benchmarks.
"""

<<<<<<< HEAD
from .angles import GolfBiomechanics, GOLF_CRITICAL_ANGLES
from .benchmarks import GolfBenchmarks
from .comparator import SwingComparator
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
    'PhaseScorer',
    'PHASE_WEIGHTS',
    'METRIC_WEIGHTS',
    'SCORING_THRESHOLDS',
    'SCORE_RANGES',
    'FEEDBACK_TEMPLATES',
    'KINEMATIC_SEQUENCE'
]
=======
from .comparator import SwingBiomechanicsEvaluator
from .angles import GolfBiomechanics
>>>>>>> origin/main
