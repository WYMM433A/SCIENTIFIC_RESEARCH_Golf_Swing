"""
Phase Detection Module
======================
Golf swing phase detection using rule-based and neural approaches.
"""

from .rule_based import EightPhaseDetector
from .adapter import PhasePredictor, create_predictor
from .neural_model import PoseSwingNet

__all__ = ['EightPhaseDetector', 'PhasePredictor', 'create_predictor', 'PoseSwingNet']
