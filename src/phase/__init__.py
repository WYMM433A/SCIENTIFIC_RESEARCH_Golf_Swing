"""
Phase Detection Module
======================
Golf swing phase detection using rule-based and neural approaches.
"""

from .rule_based import EightPhaseDetector
from .adapter import PhasePredictor, create_predictor

__all__ = ['EightPhaseDetector', 'PhasePredictor', 'create_predictor']
