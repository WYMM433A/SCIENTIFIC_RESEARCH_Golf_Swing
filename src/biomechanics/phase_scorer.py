"""
Phase-specific biomechanical scoring system.
Evaluates golf swing quality for each phase based on metrics and benchmarks.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, List, Optional
from .scoring_config import (
    PHASE_WEIGHTS, METRIC_WEIGHTS, SCORING_THRESHOLDS, SCORE_RANGES,
    FEEDBACK_TEMPLATES, FEEDBACK_PLAYBOOK, KINEMATIC_SEQUENCE, PROFESSIONAL_BENCHMARKS
)


class PhaseScorer:
    """Scores individual phases and full swing based on biomechanical metrics."""
    
    def __init__(self, confidence_threshold: float = 0.7):
        """
        Initialize the phase scorer.
        
        Args:
            confidence_threshold: Minimum confidence (0-1) to report scores
        """
        self.confidence_threshold = confidence_threshold
        self.phase_scores = {}
        self.metric_scores = {}
        self.feedback = {}

    COMPONENT_METRIC_MAP = {
        "address": {
            "posture": ("spine_angle", "address_ideal"),
            "grip": ("wrist_angle", "address_ideal"),
            "weight_distribution": ("stance_width_ratio", "address_ideal"),
        },
        "takeaway": {
            "shoulder_rotation": ("shoulder_rotation", "takeaway_ideal"),
            "hip_lag": ("x_factor", "takeaway_ideal"),
            "wrist_position": ("wrist_angle", "takeaway_ideal"),
            "club_path": ("head_displacement", "takeaway_ideal"),
        },
        "mid_backswing": {
            "coil": ("x_factor", "mid_backswing_ideal"),
            "shoulder_rotation": ("shoulder_rotation", "backswing_ideal"),
            "wrist_hinge": ("wrist_angle", "mid_backswing_ideal"),
            "shaft_plane": ("lead_arm_angle", "mid_backswing_ideal"),
        },
        "top": {
            "coil": ("shoulder_rotation", "backswing_ideal"),
            "posture": ("spine_angle", "top_ideal"),
            "stability": ("head_displacement", "top_ideal"),
            "wrist_angle": ("wrist_angle", "top_ideal"),
        },
        "mid_downswing": {
            "kinematic_sequence": ("kinematic_sequence", "ideal"),
            "lag": ("lag_angle", "mid_downswing_ideal"),
            "hip_rotation": ("hip_rotation", "downswing_ideal"),
            "upper_body_lag": ("x_factor", "mid_downswing_ideal"),
        },
        "impact": {
            "lag_release": ("lag_angle", "impact_ideal"),
            "x_factor_unwind": ("x_factor", "impact_ideal"),
            "arm_extension": ("lead_arm_angle", "impact_ideal"),
            "wrist_angle": ("wrist_angle", "impact_ideal"),
            "stability": ("head_displacement", "impact_ideal"),
        },
        "follow_through": {
            "deceleration": ("x_factor", "follow_through_ideal"),
            "posture": ("spine_angle", "follow_through_ideal"),
            "arm_swing": ("lead_arm_angle", "follow_through_ideal"),
            "rotation": ("hip_rotation", "follow_through_ideal"),
        },
        "finish": {
            "balance": ("head_displacement", "finish_ideal"),
            "posture": ("spine_angle", "finish_ideal"),
            "rotation": ("hip_rotation", "finish_ideal"),
            "symmetry": ("x_factor", "finish_ideal"),
        },
    }
        
    # ========================================================================
    # PHASE SCORING METHODS
    # ========================================================================
    
    def score_address(self, metrics: Dict[str, float]) -> Tuple[float, Dict]:
        """
        Score the address phase (setup position).
        
        Evaluates:
        - Posture (spine angle)
        - Stance width and alignment
        - Weight distribution
        - Grip indicator (wrist angle)
        
        Args:
            metrics: Dictionary with 'spine_angle', 'hip_angle', 'wrist_angle', etc.
            
        Returns:
            (score, details_dict)
        """
        phase = "address"
        score_components = {}
        
        # Posture evaluation
        if "spine_angle" in metrics:
            spine_score = self._evaluate_metric(
                metrics["spine_angle"],
                SCORING_THRESHOLDS["spine_angle"]["address_ideal"],
                metric_name="spine_angle"
            )
            score_components["posture"] = spine_score * METRIC_WEIGHTS[phase]["posture"]
        
        # Wrist angle / grip indicator
        wrist_value = self._get_metric(metrics, ["wrist_angle", "wrist_hinge"])
        if wrist_value is not None:
            wrist_score = self._evaluate_metric(
                wrist_value,
                SCORING_THRESHOLDS["wrist_angle"]["address_ideal"],
                metric_name="wrist_angle"
            )
            score_components["grip"] = wrist_score * METRIC_WEIGHTS[phase]["grip"]
        
        # Weight distribution via stance width ratio
        stance_ratio = self._get_metric(metrics, ["stance_width_ratio"])
        if stance_ratio is not None:
            weight_score = self._evaluate_metric(
                stance_ratio,
                SCORING_THRESHOLDS["stance_width_ratio"]["address_ideal"],
                metric_name="stance_width_ratio"
            )
            score_components["weight_distribution"] = weight_score * METRIC_WEIGHTS[phase]["weight_distribution"]
        
        phase_score = self._normalize_phase_score(phase, score_components)
        
        # FIX 3/4: Apply window-based penalties if available
        context = getattr(self, '_window_context', {})
        if context.get('use_window'):
            phase_score, penalty_details = self._apply_window_metrics(phase_score, phase)
        else:
            penalty_details = {}
        
        return phase_score, {
            "components": score_components,
            "confidence": len(score_components) / len(METRIC_WEIGHTS[phase]),
            "penalty_details": penalty_details,
        }
    
    def score_takeaway(self, metrics: Dict[str, float]) -> Tuple[float, Dict]:
        """
        Score the takeaway phase (initial motion from address).
        
        Evaluates:
        - Shoulder rotation initiation (small, controlled turn)
        - Hip lag (hips stay quiet, minimal rotation early on)
        - Wrist hinge initiation (slight hinge, not forced)
        - Head stability (minimal lateral movement)
        
        Args:
            metrics: Dictionary with 'shoulder_rotation', 'x_factor', 'wrist_angle', 'head_displacement'
            
        Returns:
            (score, details_dict)
        """
        phase = "takeaway"
        score_components = {}
        
        # Shoulder rotation initiation
        shoulder_value = self._get_metric(metrics, ["shoulder_rotation"])
        if shoulder_value is not None:
            shoulder_score = self._evaluate_metric(
                shoulder_value,
                SCORING_THRESHOLDS["shoulder_rotation"]["takeaway_ideal"],
                metric_name="shoulder_rotation"
            )
            score_components["shoulder_rotation"] = shoulder_score * METRIC_WEIGHTS[phase]["shoulder_rotation"]
        
        # Hip lag evaluation (minimal hip turn at takeaway)
        # Using x_factor as proxy: small x_factor means hips not yet turning (good at takeaway)
        x_factor_value = self._get_metric(metrics, ["x_factor"])
        if x_factor_value is not None:
            hip_lag_score = self._evaluate_metric(
                x_factor_value,
                SCORING_THRESHOLDS["x_factor"]["takeaway_ideal"],
                metric_name="x_factor"
            )
            score_components["hip_lag"] = hip_lag_score * METRIC_WEIGHTS[phase]["hip_lag"]
        
        # Wrist hinge initiation (slight, not too much)
        wrist_value = self._get_metric(metrics, ["wrist_angle", "wrist_hinge"])
        if wrist_value is not None:
            wrist_score = self._evaluate_metric(
                wrist_value,
                SCORING_THRESHOLDS["wrist_angle"]["takeaway_ideal"],
                metric_name="wrist_angle"
            )
            score_components["wrist_position"] = wrist_score * METRIC_WEIGHTS[phase]["wrist_position"]
        
        # Head stability during takeaway
        head_disp = self._get_head_displacement(metrics)
        if head_disp is not None:
            head_score = self._evaluate_metric(
                head_disp,
                SCORING_THRESHOLDS["head_displacement"]["takeaway_ideal"],
                metric_name="head_displacement"
            )
            score_components["club_path"] = head_score * METRIC_WEIGHTS[phase]["club_path"]
        
        phase_score = self._normalize_phase_score(phase, score_components)
        
        # FIX 3/4: Apply window-based penalties if available
        context = getattr(self, '_window_context', {})
        if context.get('use_window'):
            phase_score, penalty_details = self._apply_window_metrics(phase_score, phase)
        else:
            penalty_details = {}
        
        return phase_score, {
            "components": score_components,
            "confidence": len(score_components) / len(METRIC_WEIGHTS[phase]),
            "penalty_details": penalty_details,
        }

    def score_mid_backswing(self, metrics: Dict[str, float]) -> Tuple[float, Dict]:
        """
        Score the mid-backswing phase (coil-building segment).

        Evaluates:
        - Coil development (x-factor growth)
        - Shoulder rotation progression
        - Wrist hinge progression
        - Shaft plane proxy via lead arm structure

        Args:
            metrics: Dictionary with backswing-related metrics

        Returns:
            (score, details_dict)
        """
        phase = "mid_backswing"
        score_components = {}

        # Coil development
        x_factor_value = self._get_metric(metrics, ["x_factor"])
        if x_factor_value is not None:
            coil_score = self._evaluate_metric(
                x_factor_value,
                SCORING_THRESHOLDS["x_factor"]["mid_backswing_ideal"],
                metric_name="x_factor"
            )
            score_components["coil"] = coil_score * METRIC_WEIGHTS[phase]["coil"]

        # Shoulder rotation progression
        shoulder_value = self._get_metric(metrics, ["shoulder_rotation"])
        if shoulder_value is not None:
            shoulder_score = self._evaluate_metric(
                shoulder_value,
                SCORING_THRESHOLDS["shoulder_rotation"]["backswing_ideal"],
                metric_name="shoulder_rotation"
            )
            score_components["shoulder_rotation"] = shoulder_score * METRIC_WEIGHTS[phase]["shoulder_rotation"]

        # Wrist hinge progression
        wrist_value = self._get_metric(metrics, ["wrist_angle", "wrist_hinge"])
        if wrist_value is not None:
            wrist_score = self._evaluate_metric(
                wrist_value,
                SCORING_THRESHOLDS["wrist_angle"]["mid_backswing_ideal"],
                metric_name="wrist_angle"
            )
            score_components["wrist_hinge"] = wrist_score * METRIC_WEIGHTS[phase]["wrist_hinge"]

        # Shaft plane proxy
        lead_arm_value = self._get_metric(metrics, ["lead_arm_angle", "arm_extension"])
        if lead_arm_value is not None:
            shaft_score = self._evaluate_metric(
                lead_arm_value,
                SCORING_THRESHOLDS["lead_arm_angle"]["mid_backswing_ideal"],
                metric_name="lead_arm_angle"
            )
            score_components["shaft_plane"] = shaft_score * METRIC_WEIGHTS[phase]["shaft_plane"]

        phase_score = self._normalize_phase_score(phase, score_components)

        # FIX 3/4: Apply window-based penalties if available
        context = getattr(self, '_window_context', {})
        if context.get('use_window'):
            phase_score, penalty_details = self._apply_window_metrics(phase_score, phase)
        else:
            penalty_details = {}

        return phase_score, {
            "components": score_components,
            "confidence": len(score_components) / len(METRIC_WEIGHTS[phase]),
            "penalty_details": penalty_details,
        }
    
    def score_top(self, metrics: Dict[str, float]) -> Tuple[float, Dict]:
        """
        Score the top of backswing phase.
        
        Evaluates:
        - Maximum coil (X-factor)
        - Posture maintained
        - Stability (minimal sway)
        - Wrist hinge at top
        
        Args:
            metrics: Dictionary with 'x_factor', 'spine_angle', 'wrist_angle', etc.
            
        Returns:
            (score, details_dict)
        """
        phase = "top"
        score_components = {}
        
        # Coil evaluation using shoulder rotation
        shoulder_value = self._get_metric(metrics, ["shoulder_rotation"])
        if shoulder_value is not None:
            coil_score = self._evaluate_metric(
                shoulder_value,
                SCORING_THRESHOLDS["shoulder_rotation"]["backswing_ideal"],
                metric_name="shoulder_rotation"
            )
            score_components["coil"] = coil_score * METRIC_WEIGHTS[phase]["coil"]
        
        # Posture check
        if "spine_angle" in metrics:
            posture_score = self._evaluate_metric(
                metrics["spine_angle"],
                SCORING_THRESHOLDS["spine_angle"]["top_ideal"],
                metric_name="spine_angle"
            )
            score_components["posture"] = posture_score * METRIC_WEIGHTS[phase]["posture"]
        
        # Stability (minimal head movement)
        head_disp = self._get_head_displacement(metrics)
        if head_disp is not None:
            stability_score = self._evaluate_metric(
                head_disp,
                SCORING_THRESHOLDS["head_displacement"]["top_ideal"],
                metric_name="head_displacement"
            )
            score_components["stability"] = stability_score * METRIC_WEIGHTS[phase]["stability"]
        
        # Wrist hinge
        wrist_value = self._get_metric(metrics, ["wrist_angle", "wrist_hinge"])
        if wrist_value is not None:
            wrist_score = self._evaluate_metric(
                wrist_value,
                SCORING_THRESHOLDS["wrist_angle"]["top_ideal"],
                metric_name="wrist_angle"
            )
            score_components["wrist_angle"] = wrist_score * METRIC_WEIGHTS[phase]["wrist_angle"]
        
        phase_score = self._normalize_phase_score(phase, score_components)
        
        # FIX 3/4: Apply window-based penalties if available
        context = getattr(self, '_window_context', {})
        if context.get('use_window'):
            phase_score, penalty_details = self._apply_window_metrics(phase_score, phase)
        else:
            penalty_details = {}
        
        return phase_score, {
            "components": score_components,
            "confidence": len(score_components) / len(METRIC_WEIGHTS[phase]),
            "penalty_details": penalty_details,
        }
    
    def score_mid_downswing(self, metrics: Dict[str, float], 
                            kinematic_data: Optional[Dict] = None) -> Tuple[float, Dict]:
        """
        Score the mid-downswing phase (most critical).
        
        Evaluates:
        - Kinematic sequence (hips → torso → arms → club) - CRITICAL
        - Lag angle maintained
        - Hip rotation rate
        - Upper body lag
        
        Args:
            metrics: Dictionary with 'lag_angle', 'hip_rotation', 'shoulder_rotation', etc.
            kinematic_data: Optional kinematic sequence timing data
            
        Returns:
            (score, details_dict)
        """
        phase = "mid_downswing"
        score_components = {}
        
        # Kinematic sequence (highest weight - 40%)
        if kinematic_data:
            sequence_score = self._evaluate_kinematic_sequence(kinematic_data)
            score_components["kinematic_sequence"] = sequence_score * METRIC_WEIGHTS[phase]["kinematic_sequence"]
        
        # Lag angle (should be high)
        lag_value = self._get_metric(metrics, ["lag_angle", "wrist_angle", "wrist_hinge"])
        if lag_value is not None:
            lag_score = self._evaluate_metric(
                lag_value,
                SCORING_THRESHOLDS["lag_angle"]["mid_downswing_ideal"],
                metric_name="lag_angle"
            )
            score_components["lag"] = lag_score * METRIC_WEIGHTS[phase]["lag"]
        
        # Hip rotation driving sequence
        hip_value = self._get_metric(metrics, ["hip_rotation", "hip_angle"])
        if hip_value is not None:
            hip_score = self._evaluate_metric(
                hip_value,
                SCORING_THRESHOLDS["hip_rotation"]["downswing_ideal"],
                metric_name="hip_rotation"
            )
            score_components["hip_rotation"] = hip_score * METRIC_WEIGHTS[phase]["hip_rotation"]
        
        # Upper body lag (shoulders follow hips)
        upper_body_scores = []
        if "x_factor" in metrics:
            x_factor_score = self._evaluate_metric(
                metrics["x_factor"],
                SCORING_THRESHOLDS["x_factor"]["mid_downswing_ideal"],
                metric_name="x_factor"
            )
            upper_body_scores.append(x_factor_score)

        # Dynamic X-factor stretch from kinematic window when available
        if kinematic_data and "x_factor_stretch" in kinematic_data:
            stretch_score = self._evaluate_metric(
                kinematic_data["x_factor_stretch"],
                SCORING_THRESHOLDS["x_factor_stretch"]["ideal"],
                metric_name="x_factor_stretch"
            )
            upper_body_scores.append(stretch_score)

        if upper_body_scores:
            upper_body_score = float(np.mean(upper_body_scores))
            score_components["upper_body_lag"] = upper_body_score * METRIC_WEIGHTS[phase]["upper_body_lag"]
        
        phase_score = self._normalize_phase_score(phase, score_components)
        
        return phase_score, {
            "components": score_components,
            "confidence": len(score_components) / len(METRIC_WEIGHTS[phase]),
        }
    
    def score_impact(self, metrics: Dict[str, float]) -> Tuple[float, Dict]:
        """
        Score the impact phase (most performance-critical).
        
        Evaluates:
        - Lag release timing
        - X-factor unwind (torso-hip differential) — DEPRECATED in Fix 1
        - Arm extension delta (top vs impact) — NEW in Fix 1
        - Wrist angle at contact
        - Head/body stability
        
        Args:
            metrics: Dictionary with impact phase metrics
            
        Returns:
            (score, details_dict)
        """
        phase = "impact"
        score_components = {}
        
        # Lag release timing
        lag_value = self._get_metric(metrics, ["lag_angle", "wrist_angle", "wrist_hinge"])
        if lag_value is not None:
            lag_release_score = self._evaluate_metric(
                lag_value,
                SCORING_THRESHOLDS["lag_angle"]["impact_ideal"],
                metric_name="lag_angle"
            )
            score_components["lag_release"] = lag_release_score * METRIC_WEIGHTS[phase]["lag_release"]
        
        # FIX 1: Replace x_factor_unwind with arm_extension_delta
        context = getattr(self, '_window_context', {})
        if context.get('use_window'):
            # Use arm_extension_delta when window analysis is available
            delta_result = self._apply_arm_extension_delta(metrics)
            arm_ext_delta_score = delta_result.get('score', 75)
            score_components["arm_extension_delta"] = arm_ext_delta_score * METRIC_WEIGHTS[phase].get("x_factor_unwind", 0.45)
        else:
            # Fallback to x_factor_unwind for keyframe-only scoring
            if "x_factor" in metrics:
                unwind_score = self._evaluate_metric(
                    metrics["x_factor"],
                    SCORING_THRESHOLDS["x_factor"]["impact_ideal"],
                    metric_name="x_factor"
                )
                score_components["x_factor_unwind"] = unwind_score * METRIC_WEIGHTS[phase]["x_factor_unwind"]
        
        # Arm extension (static)
        arm_ext_value = self._get_metric(metrics, ["arm_extension", "lead_arm_angle"])
        if arm_ext_value is not None:
            arm_score = self._evaluate_metric(
                arm_ext_value,
                SCORING_THRESHOLDS["lead_arm_angle"]["impact_ideal"],
                metric_name="lead_arm_angle"
            )
            score_components["arm_extension"] = arm_score * METRIC_WEIGHTS[phase]["arm_extension"]
        
        # Wrist angle at impact
        wrist_value = self._get_metric(metrics, ["wrist_angle", "wrist_hinge"])
        if wrist_value is not None:
            wrist_score = self._evaluate_metric(
                wrist_value,
                SCORING_THRESHOLDS["wrist_angle"]["impact_ideal"],
                metric_name="wrist_angle"
            )
            score_components["wrist_angle"] = wrist_score * METRIC_WEIGHTS[phase]["wrist_angle"]
        
        # Stability
        head_disp = self._get_head_displacement(metrics)
        if head_disp is not None:
            stability_score = self._evaluate_metric(
                head_disp,
                SCORING_THRESHOLDS["head_displacement"]["impact_ideal"],
                metric_name="head_displacement"
            )
            score_components["stability"] = stability_score * METRIC_WEIGHTS[phase]["stability"]
        
        phase_score = self._normalize_phase_score(phase, score_components)
        
        # FIX 3/4: Apply window-based penalties if available
        if context.get('use_window'):
            phase_score, penalty_details = self._apply_window_metrics(phase_score, phase)
        else:
            penalty_details = {}
        
        return phase_score, {
            "components": score_components,
            "confidence": len(score_components) / len(METRIC_WEIGHTS[phase]),
            "penalty_details": penalty_details,
        }


    def score_follow_through(self, metrics: Dict[str, float]) -> Tuple[float, Dict]:
        """
        Score the follow-through phase (deceleration and exit quality).

        Evaluates:
        - Deceleration quality via x-factor unwind completion
        - Posture maintenance through release
        - Arm swing fold/recovery
        - Continued body rotation toward finish

        Args:
            metrics: Dictionary with follow-through phase metrics

        Returns:
            (score, details_dict)
        """
        phase = "follow_through"
        score_components = {}

        # Deceleration proxy: x-factor should be mostly unwound after impact
        x_factor_value = self._get_metric(metrics, ["x_factor"])
        if x_factor_value is not None:
            decel_score = self._evaluate_metric(
                x_factor_value,
                SCORING_THRESHOLDS["x_factor"]["follow_through_ideal"],
                metric_name="x_factor"
            )
            score_components["deceleration"] = decel_score * METRIC_WEIGHTS[phase]["deceleration"]

        # Posture maintenance
        if "spine_angle" in metrics:
            posture_score = self._evaluate_metric(
                metrics["spine_angle"],
                SCORING_THRESHOLDS["spine_angle"]["follow_through_ideal"],
                metric_name="spine_angle"
            )
            score_components["posture"] = posture_score * METRIC_WEIGHTS[phase]["posture"]

        # Arm swing quality (controlled fold and recovery)
        lead_arm_value = self._get_metric(metrics, ["lead_arm_angle", "arm_extension"])
        if lead_arm_value is not None:
            arm_swing_score = self._evaluate_metric(
                lead_arm_value,
                SCORING_THRESHOLDS["lead_arm_angle"]["follow_through_ideal"],
                metric_name="lead_arm_angle"
            )
            score_components["arm_swing"] = arm_swing_score * METRIC_WEIGHTS[phase]["arm_swing"]

        # Continued body rotation
        hip_value = self._get_metric(metrics, ["hip_rotation", "hip_angle"])
        if hip_value is not None:
            rotation_score = self._evaluate_metric(
                hip_value,
                SCORING_THRESHOLDS["hip_rotation"]["follow_through_ideal"],
                metric_name="hip_rotation"
            )
            score_components["rotation"] = rotation_score * METRIC_WEIGHTS[phase]["rotation"]

        phase_score = self._normalize_phase_score(phase, score_components)

        # FIX 3/4: Apply window-based penalties if available
        context = getattr(self, '_window_context', {})
        if context.get('use_window'):
            phase_score, penalty_details = self._apply_window_metrics(phase_score, phase)
        else:
            penalty_details = {}

        return phase_score, {
            "components": score_components,
            "confidence": len(score_components) / len(METRIC_WEIGHTS[phase]),
            "penalty_details": penalty_details,
        }

    def score_finish(self, metrics: Dict[str, float]) -> Tuple[float, Dict]:
        """
        Score the finish phase (balance and completion).

        Evaluates:
        - Balance (minimal late-swing sway)
        - Posture maintenance at finish
        - Rotation completion
        - Symmetry/unwind completion

        Args:
            metrics: Dictionary with finish phase metrics

        Returns:
            (score, details_dict)
        """
        phase = "finish"
        score_components = {}

        # Balance at finish
        head_disp = self._get_head_displacement(metrics)
        if head_disp is not None:
            balance_score = self._evaluate_metric(
                head_disp,
                SCORING_THRESHOLDS["head_displacement"]["finish_ideal"],
                metric_name="head_displacement"
            )
            score_components["balance"] = balance_score * METRIC_WEIGHTS[phase]["balance"]

        # Posture maintenance
        if "spine_angle" in metrics:
            posture_score = self._evaluate_metric(
                metrics["spine_angle"],
                SCORING_THRESHOLDS["spine_angle"]["finish_ideal"],
                metric_name="spine_angle"
            )
            score_components["posture"] = posture_score * METRIC_WEIGHTS[phase]["posture"]

        # Rotation completion
        hip_value = self._get_metric(metrics, ["hip_rotation", "hip_angle"])
        if hip_value is not None:
            rotation_score = self._evaluate_metric(
                hip_value,
                SCORING_THRESHOLDS["hip_rotation"]["finish_ideal"],
                metric_name="hip_rotation"
            )
            score_components["rotation"] = rotation_score * METRIC_WEIGHTS[phase]["rotation"]

        # Symmetry proxy: body has mostly unwound
        x_factor_value = self._get_metric(metrics, ["x_factor"])
        if x_factor_value is not None:
            symmetry_score = self._evaluate_metric(
                x_factor_value,
                SCORING_THRESHOLDS["x_factor"]["finish_ideal"],
                metric_name="x_factor"
            )
            score_components["symmetry"] = symmetry_score * METRIC_WEIGHTS[phase]["symmetry"]

        phase_score = self._normalize_phase_score(phase, score_components)

        # FIX 3/4: Apply window-based penalties if available
        context = getattr(self, '_window_context', {})
        if context.get('use_window'):
            phase_score, penalty_details = self._apply_window_metrics(phase_score, phase)
        else:
            penalty_details = {}

        return phase_score, {
            "components": score_components,
            "confidence": len(score_components) / len(METRIC_WEIGHTS[phase]),
            "penalty_details": penalty_details,
        }
    
    def score_full_swing(self, phase_scores: Dict[str, float]) -> Tuple[float, Dict]:
        """
        Calculate overall swing score from individual phase scores.
        
        Args:
            phase_scores: Dictionary of {phase_name: score}
            
        Returns:
            (overall_score, details_dict)
        """
        weighted_sum = 0
        total_weight = 0
        phase_details = {}
        
        for phase, score in phase_scores.items():
            # Normalize phase name
            phase_normalized = phase.replace("-", "_").lower()
            weight = PHASE_WEIGHTS.get(phase_normalized, 0.1)
            
            weighted_sum += score * weight
            total_weight += weight
            phase_details[phase_normalized] = {
                "score": score,
                "weight": weight,
                "contribution": score * weight,
            }
        
        overall_score = weighted_sum / total_weight if total_weight > 0 else 0
        
        return overall_score, {
            "phase_details": phase_details,
            "weighted_sum": weighted_sum,
            "total_weight": total_weight,
        }
    
    # ========================================================================
    # HELPER METHODS
    # ========================================================================
    
    def _evaluate_metric(self, value: float, ideal_range: Tuple[float, float],
                         metric_name: str = "", acceptable_range: Optional[Tuple] = None) -> float:
        """
        Evaluate a single metric against ideal and acceptable ranges.
        
        Scoring:
        - Within ideal range: 100 points
        - Within acceptable but outside ideal: 50-99 points
        - Beyond acceptable: 0-49 points
        
        Args:
            value: Measured value
            ideal_range: (min_ideal, max_ideal)
            metric_name: Name of metric (for logging)
            acceptable_range: (min_acceptable, max_acceptable), optional
            
        Returns:
            Score (0-100)
        """
        ideal_min, ideal_max = ideal_range
        
        # Get acceptable range from config if available
        if acceptable_range is None and metric_name in SCORING_THRESHOLDS:
            acceptable_min, acceptable_max = SCORING_THRESHOLDS[metric_name]["acceptable"]
        else:
            acceptable_min = ideal_min - 10
            acceptable_max = ideal_max + 10
        
        # Perfect score if within ideal
        if ideal_min <= value <= ideal_max:
            return 100.0
        
        # Partial credit if within acceptable
        if acceptable_min <= value <= acceptable_max:
            distance_to_ideal = min(
                abs(value - ideal_min) if value < ideal_min else abs(value - ideal_max),
                abs(ideal_min - acceptable_min) if value < ideal_min else abs(ideal_max - acceptable_max)
            )
            max_distance = min(
                abs(ideal_min - acceptable_min),
                abs(ideal_max - acceptable_max)
            )
            if max_distance == 0:
                score = 100 if distance_to_ideal == 0 else 0
            else:
                score = 50 + (50 * (1 - distance_to_ideal / max_distance))
            return max(50.0, min(99.0, score))
        
        # Low score if beyond acceptable
        distance_beyond = min(
            abs(value - acceptable_min) if value < acceptable_min else abs(value - acceptable_max),
            abs(value - acceptable_min) if value < acceptable_min else abs(value - acceptable_max)
        )
        return max(0.0, 49 - (distance_beyond * 5))
    
    def _get_metric(self, metrics: Dict[str, float], names: List[str], default: Optional[float] = None) -> Optional[float]:
        """Return the first metric value available from a list of aliases."""
        for name in names:
            if name in metrics:
                return metrics[name]
        return default
    
    def _get_head_displacement(self, metrics: Dict[str, float]) -> Optional[float]:
        """Compute head displacement magnitude from head_lateral/head_vertical."""
        if "head_displacement" in metrics:
            return metrics["head_displacement"]
        if "head_lateral" in metrics and "head_vertical" in metrics:
            return float(np.hypot(metrics["head_lateral"], metrics["head_vertical"]))
        return None

    def _normalize_phase_score(self, phase: str, score_components: Dict[str, float]) -> float:
        """Normalize a phase score by available metric weights."""
        if not score_components:
            return 0.0
        total_weight = sum(
            METRIC_WEIGHTS[phase].get(comp, 0.0)
            for comp in score_components
        )
        if total_weight <= 0:
            return 0.0
        return sum(score_components.values()) / total_weight

    def _weighted_to_raw_component_scores(self, phase: str, component_scores: Dict[str, float]) -> Dict[str, float]:
        """Convert weighted component scores back to raw 0-100 component scores."""
        raw_scores = {}
        phase_weights = METRIC_WEIGHTS.get(phase, {})
        for component, weighted_value in component_scores.items():
            weight = phase_weights.get(component, 0.0)
            if weight > 0:
                raw_scores[component] = float(weighted_value / weight)
        return raw_scores

    def _get_metric_value_for_component(self, metrics: Dict[str, float], metric_name: str) -> Optional[float]:
        """Resolve concrete metric value for a component metric key."""
        if metric_name == "head_displacement":
            return self._get_head_displacement(metrics)
        if metric_name == "lag_angle":
            return self._get_metric(metrics, ["lag_angle", "wrist_angle", "wrist_hinge"])
        if metric_name == "hip_rotation":
            return self._get_metric(metrics, ["hip_rotation", "hip_angle"])
        if metric_name == "wrist_angle":
            return self._get_metric(metrics, ["wrist_angle", "wrist_hinge"])
        if metric_name == "lead_arm_angle":
            return self._get_metric(metrics, ["lead_arm_angle", "arm_extension"])
        if metric_name == "kinematic_sequence":
            return None
        return metrics.get(metric_name)

    def _get_target_range(self, metric_name: str, target_key: str) -> Optional[Tuple[float, float]]:
        """Resolve ideal target range from config for feedback diagnostics."""
        if metric_name == "kinematic_sequence":
            return (75.0, 100.0)
        metric_cfg = SCORING_THRESHOLDS.get(metric_name)
        if not metric_cfg:
            return None
        if target_key in metric_cfg:
            return metric_cfg[target_key]
        return metric_cfg.get("ideal")

    def _delta_from_range(self, value: Optional[float], target_range: Optional[Tuple[float, float]]) -> Optional[float]:
        """Signed delta to nearest boundary; 0 when in range."""
        if value is None or target_range is None:
            return None
        low, high = target_range
        if low <= value <= high:
            return 0.0
        if value < low:
            return float(value - low)
        return float(value - high)

    def generate_feedback_details(self, phase: str, metrics: Dict, component_scores: Dict[str, float]) -> List[Dict]:
        """Generate actionable per-component diagnostics for a phase."""
        phase_key = phase.replace("-", "_").lower()
        raw_scores = self._weighted_to_raw_component_scores(phase_key, component_scores)
        phase_weight = PHASE_WEIGHTS.get(phase_key, 0.1)
        details = []

        for component, raw_score in raw_scores.items():
            metric_name, target_key = self.COMPONENT_METRIC_MAP.get(phase_key, {}).get(
                component, (component, "ideal")
            )
            measured_value = self._get_metric_value_for_component(metrics, metric_name)
            target_range = self._get_target_range(metric_name, target_key)
            delta = self._delta_from_range(measured_value, target_range)
            weight = METRIC_WEIGHTS.get(phase_key, {}).get(component, 0.0)
            priority = max(0.0, 100.0 - raw_score) * phase_weight * weight

            if raw_score < 60:
                severity = "critical"
            elif raw_score < 75:
                severity = "moderate"
            elif raw_score < 90:
                severity = "minor"
            else:
                severity = "strength"

            playbook = FEEDBACK_PLAYBOOK.get(f"{phase_key}.{component}", {})
            details.append({
                "phase": phase,
                "phase_key": phase_key,
                "component": component,
                "metric_name": metric_name,
                "raw_score": float(raw_score),
                "weighted_score": float(component_scores.get(component, 0.0)),
                "measured_value": measured_value,
                "target_min": None if target_range is None else float(target_range[0]),
                "target_max": None if target_range is None else float(target_range[1]),
                "delta_from_target": delta,
                "severity": severity,
                "priority_score": float(priority),
                "cue": playbook.get("cue", f"Improve {component.replace('_', ' ')} in {phase.lower()} phase."),
                "drill": playbook.get("drill", "Use slow-motion rehearsals with checkpoint pauses."),
            })

        return sorted(details, key=lambda x: x["priority_score"], reverse=True)
    
    def _evaluate_kinematic_sequence(self, kinematic_data: Dict) -> float:
        """
        Evaluate the kinematic sequence (hips → torso → arms → club).
        
        Args:
            kinematic_data: Dictionary with timing information
                {
                    "hip_start_ms": ms,
                    "torso_start_ms": ms,
                    "arm_start_ms": ms,
                    "club_start_ms": ms,
                }
        
        Returns:
            Score (0-100)
        """
        try:
            hip_start = kinematic_data.get("hip_start_ms", 0)
            torso_start = kinematic_data.get("torso_start_ms", 0)
            arm_start = kinematic_data.get("arm_start_ms", 0)
            club_start = kinematic_data.get("club_start_ms", 0)
            
            # Calculate sequence delays
            hip_to_torso = torso_start - hip_start
            torso_to_arm = arm_start - torso_start
            arm_to_club = club_start - arm_start
            
            sequence_config = KINEMATIC_SEQUENCE
            
            score = 0
            components = 0
            
            # Check hip-torso delay
            if 0 <= hip_to_torso <= sequence_config["hip_lead"]["ideal_ms"][1]:
                score += 30
            elif sequence_config["hip_lead"]["acceptable_ms"][0] <= hip_to_torso <= sequence_config["hip_lead"]["acceptable_ms"][1]:
                score += 15
            components += 30
            
            # Check torso-arm delay
            if sequence_config["torso_follow"]["ideal_ms_after_hips"][0] <= torso_to_arm <= sequence_config["torso_follow"]["ideal_ms_after_hips"][1]:
                score += 30
            elif sequence_config["torso_follow"]["acceptable_ms_after_hips"][0] <= torso_to_arm <= sequence_config["torso_follow"]["acceptable_ms_after_hips"][1]:
                score += 15
            components += 30
            
            # Check arm-club delay
            if sequence_config["arm_follow"]["ideal_ms_after_torso"][0] <= arm_to_club <= sequence_config["arm_follow"]["ideal_ms_after_torso"][1]:
                score += 30
            elif sequence_config["arm_follow"]["acceptable_ms_after_torso"][0] <= arm_to_club <= sequence_config["arm_follow"]["acceptable_ms_after_torso"][1]:
                score += 15
            components += 30
            
            # Proper sequence bonus
            if (0 <= hip_to_torso <= 100 and 50 <= torso_to_arm <= 150 and 50 <= arm_to_club <= 150):
                score += sequence_config["proper_sequence_bonus"]
            
            if components <= 0:
                return 0
            normalized = score / components * 100
            return float(max(0.0, min(100.0, normalized)))
            
        except Exception as e:
            print(f"Error evaluating kinematic sequence: {e}")
            return 0
    
    # ========================================================================
    # FIX 1/3/4: WINDOW-BASED ANALYSIS HELPERS
    # ========================================================================
    
    def _apply_window_metrics(self, phase_score: float, phase_name: str) -> Tuple[float, Dict]:
        """
        Apply consistency and jerk penalties for window-based analysis (Fixes 3 & 4).
        
        Called when window context is available (start_frame, end_frame, biomechanics_obj).
        
        Returns:
            (adjusted_score, penalty_details)
        """
        context = getattr(self, '_window_context', {})
        if not context.get('use_window') or not context.get('biomechanics'):
            return phase_score, {}
        
        biomechanics_obj = context['biomechanics']
        start_frame = context['start_frame']
        end_frame = context['end_frame']
        
        penalty_details = {}
        total_penalty = 0.0
        
        try:
            # Calculate consistency penalty for spine_angle (most important for posture)
            spine_window = biomechanics_obj.calculate_metrics_window(
                start_frame, end_frame, metric_name='spine_angle'
            )
            if spine_window:
                consistency_penalty = spine_window.get('consistency_penalty', 0)
                total_penalty += consistency_penalty * 1.0  # 100% weight on consistency (was 0.5)
                penalty_details['spine_consistency'] = {
                    'std': spine_window.get('std', 0),
                    'penalty': consistency_penalty
                }
            
            # Calculate wrist jerk penalty (smoothness)
            wrist_jerk_data = biomechanics_obj.calculate_wrist_jerk(start_frame, end_frame)
            if wrist_jerk_data:
                jerk_penalty = max(0, 100 - wrist_jerk_data.get('jerk_quality', 100))
                total_penalty += jerk_penalty * 0.8  # 80% weight on jerk (was 0.3)
                penalty_details['wrist_jerk'] = {
                    'jerk': wrist_jerk_data.get('wrist_jerk', 0),
                    'quality': wrist_jerk_data.get('jerk_quality', 100),
                    'penalty': jerk_penalty
                }
            
            # Apply penalties with higher ceiling (max -50 points instead of -20)
            total_penalty = min(50, total_penalty)
            adjusted_score = max(0, phase_score - total_penalty)
            penalty_details['total_penalty'] = total_penalty
            
            return adjusted_score, penalty_details
            
        except Exception as e:
            print(f"Error applying window metrics penalty: {e}")
            return phase_score, {'error': str(e)}
    
    def _apply_arm_extension_delta(self, metrics: Dict[str, float]) -> Dict[str, any]:
        """
        Apply Fix 1: arm_extension_delta component to IMPACT scoring.
        
        Replaces x_factor_unwind which has poor discrimination (0-5° range).
        
        Returns:
            {
                'component_name': 'arm_extension_delta',
                'score': 0-100,
                'details': {...}
            }
        """
        context = getattr(self, '_window_context', {})
        if not context.get('use_window') or not context.get('biomechanics') or not context.get('top_frame'):
            # Fallback: use static arm extension
            return {
                'component_name': 'arm_extension_delta',
                'score': self._evaluate_metric(
                    metrics.get('lead_arm_angle', 170),
                    SCORING_THRESHOLDS['lead_arm_angle']['impact_ideal'],
                    metric_name='lead_arm_angle'
                ),
                'fallback': True
            }
        
        try:
            biomechanics_obj = context['biomechanics']
            impact_frame = context['end_frame']  # Assume impact is at end of phase
            top_frame = context['top_frame']
            
            # Calculate arm extension change
            delta_data = biomechanics_obj.calculate_arm_extension_delta(top_frame, impact_frame)
            
            if not delta_data:
                # Fallback
                return {
                    'component_name': 'arm_extension_delta',
                    'score': self._evaluate_metric(
                        metrics.get('lead_arm_angle', 170),
                        SCORING_THRESHOLDS['lead_arm_angle']['impact_ideal'],
                        metric_name='lead_arm_angle'
                    ),
                    'fallback': True,
                    'reason': 'Could not calculate delta'
                }
            
            return {
                'component_name': 'arm_extension_delta',
                'score': delta_data.get('quality', 75),
                'top_angle': delta_data.get('top_lead_arm', 0),
                'impact_angle': delta_data.get('impact_lead_arm', 0),
                'delta': delta_data.get('delta', 0),
                'fallback': False
            }
            
        except Exception as e:
            print(f"Error applying arm extension delta: {e}")
            return {
                'component_name': 'arm_extension_delta',
                'score': self._evaluate_metric(
                    metrics.get('lead_arm_angle', 170),
                    SCORING_THRESHOLDS['lead_arm_angle']['impact_ideal'],
                    metric_name='lead_arm_angle'
                ),
                'fallback': True,
                'error': str(e)
            }
    
    def generate_feedback(self, phase: str, score: float, metrics: Dict,
                         component_scores: Dict) -> str:
        """
        Generate human-readable feedback for a phase.
        
        Args:
            phase: Phase name
            score: Phase score (0-100)
            metrics: Measured metrics
            component_scores: Scores for each component
            
        Returns:
            Feedback string
        """
        # Determine score range
        score_category = None
        for category, (min_score, max_score) in SCORE_RANGES.items():
            if min_score <= score <= max_score:
                score_category = category
                break
        
        if not score_category:
            score_category = "poor" if score < 40 else "excellent"
        
        phase_key = phase.replace("-", "_").lower()
        raw_component_scores = self._weighted_to_raw_component_scores(phase_key, component_scores)

        # Build feedback
        feedback_parts = [f"{phase.upper()}: {score:.0f}/100 ({score_category.upper()})"]

        actionable = [d for d in self.generate_feedback_details(phase, metrics, component_scores) if d["raw_score"] < 90]
        top_issues = actionable[:3]

        if not top_issues:
            feedback_parts.append("  ✓ No critical issues detected in this phase.")
            return "\n".join(feedback_parts)

        for item in top_issues:
            component = item["component"]
            raw_score = item["raw_score"]
            measured = item["measured_value"]
            tmin = item["target_min"]
            tmax = item["target_max"]

            if measured is not None and tmin is not None and tmax is not None:
                feedback_parts.append(
                    f"  ✗ {component}: {measured:.2f} vs target {tmin:.2f}-{tmax:.2f} (score {raw_score:.0f})"
                )
            else:
                feedback_parts.append(f"  ✗ {component}: score {raw_score:.0f}")

            feedback_parts.append(f"    Fix: {item['cue']}")
            feedback_parts.append(f"    Drill: {item['drill']}")
        
        return "\n".join(feedback_parts)
    
    def score_phase_with_metrics(self, phase: str, metrics: pd.DataFrame,
                                kinematic_data: Optional[Dict] = None,
                                biomechanics_obj: Optional[object] = None,
                                start_frame: Optional[int] = None,
                                end_frame: Optional[int] = None,
                                top_frame: Optional[int] = None) -> Tuple[float, Dict]:
        """
        Score a phase given a metrics DataFrame.
        
        Supports both keyframe-based (legacy) and window-based (Fix 3/4) scoring:
        - If start_frame/end_frame provided: Use window averaging + jerk + consistency penalties
        - Otherwise: Use single keyframe metrics (backward compatible)
        
        Args:
            phase: Phase name (e.g., "Address", "Top", "Impact")
            metrics: DataFrame with metrics for the phase
            kinematic_data: Optional kinematic sequence data
            biomechanics_obj: Optional GolfBiomechanics object for window analysis
            start_frame: Start frame of phase (for window analysis)
            end_frame: End frame of phase (for window analysis)
            top_frame: Frame number at top of swing (for arm_extension_delta in Fix 1)
            
        Returns:
            (score, details)
        """
        phase_normalized = phase.replace("-", "_").lower()
        if isinstance(metrics, dict):
            metrics_dict = metrics
        elif isinstance(metrics, pd.Series):
            metrics_dict = metrics.to_dict()
        elif isinstance(metrics, pd.DataFrame) and len(metrics) > 0:
            metrics_dict = metrics.iloc[0].to_dict()
        else:
            metrics_dict = {}
        
        # Store context for phase methods to use
        self._window_context = {
            'biomechanics': biomechanics_obj,
            'start_frame': start_frame,
            'end_frame': end_frame,
            'top_frame': top_frame,
            'use_window': biomechanics_obj is not None and start_frame is not None and end_frame is not None
        }
        
        if phase_normalized == "address":
            return self.score_address(metrics_dict)
        elif phase_normalized == "takeaway":
            return self.score_takeaway(metrics_dict)
        elif phase_normalized == "mid_backswing":
            return self.score_mid_backswing(metrics_dict)
        elif phase_normalized == "top":
            return self.score_top(metrics_dict)
        elif phase_normalized == "mid_downswing":
            return self.score_mid_downswing(metrics_dict, kinematic_data)
        elif phase_normalized == "impact":
            return self.score_impact(metrics_dict)
        elif phase_normalized == "follow_through":
            return self.score_follow_through(metrics_dict)
        elif phase_normalized == "finish":
            return self.score_finish(metrics_dict)
        else:
            # Default scoring for other phases
            default_score = 75.0
            return default_score, {"confidence": 0.5}
