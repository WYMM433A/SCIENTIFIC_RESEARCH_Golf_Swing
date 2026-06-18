"""
Golf Biomechanics - Comprehensive angle calculations

Research-backed angles that affect golf swing performance.
References:
- Hume et al. (2005) - The role of biomechanics in maximizing distance and accuracy of golf shots
- Chu et al. (2010) - Biomechanical comparison between elite female and male golfers
- Neal & Wilson (1985) - 3D kinematics of golf swing
"""

import numpy as np
import math
from typing import Dict, List, Tuple, Optional
import pandas as pd


# MediaPipe landmark indices for reference
LANDMARK_INDICES = {
    'nose': 0,
    'left_eye_inner': 1, 'left_eye': 2, 'left_eye_outer': 3,
    'right_eye_inner': 4, 'right_eye': 5, 'right_eye_outer': 6,
    'left_ear': 7, 'right_ear': 8,
    'mouth_left': 9, 'mouth_right': 10,
    'left_shoulder': 11, 'right_shoulder': 12,
    'left_elbow': 13, 'right_elbow': 14,
    'left_wrist': 15, 'right_wrist': 16,
    'left_pinky': 17, 'right_pinky': 18,
    'left_index': 19, 'right_index': 20,
    'left_thumb': 21, 'right_thumb': 22,
    'left_hip': 23, 'right_hip': 24,
    'left_knee': 25, 'right_knee': 26,
    'left_ankle': 27, 'right_ankle': 28,
    'left_heel': 29, 'right_heel': 30,
    'left_foot_index': 31, 'right_foot_index': 32
}


# Golf-critical angles with research-backed ideal values
GOLF_CRITICAL_ANGLES = {
    # ========== POSTURE ANGLES ==========
    'spine_angle': {
        'description': 'Forward tilt of spine from vertical at address',
        'landmarks': ['hip_midpoint', 'shoulder_midpoint'],
        'phases': {
            'address': {'ideal': 45, 'min': 35, 'max': 55},
            'impact': {'ideal': 45, 'min': 35, 'max': 55}  # Should maintain
        },
        'importance': 'Affects power transfer and consistency'
    },
    'spine_lateral_tilt': {
        'description': 'Side bend of spine (right shoulder lower than left for RH golfer)',
        'phases': {
            'address': {'ideal': 0, 'min': -5, 'max': 5},
            'top': {'ideal': 10, 'min': 5, 'max': 20},
            'impact': {'ideal': 15, 'min': 10, 'max': 25}
        },
        'importance': 'Creates proper angle of attack'
    },
    
    # ========== ROTATION ANGLES ==========
    'shoulder_rotation': {
        'description': 'Rotation of shoulders relative to target line',
        'landmarks': ['left_shoulder', 'right_shoulder'],
        'phases': {
            'address': {'ideal': 0, 'min': -5, 'max': 5},
            'top': {'ideal': 90, 'min': 80, 'max': 110},
            'impact': {'ideal': 20, 'min': 10, 'max': 35}
        },
        'importance': 'Primary power source in swing'
    },
    'hip_rotation': {
        'description': 'Rotation of hips relative to target line',
        'landmarks': ['left_hip', 'right_hip'],
        'phases': {
            'address': {'ideal': 0, 'min': -5, 'max': 5},
            'top': {'ideal': 45, 'min': 35, 'max': 55},
            'impact': {'ideal': 40, 'min': 30, 'max': 50}
        },
        'importance': 'Initiates downswing sequence'
    },
    'x_factor': {
        'description': 'Separation between shoulder and hip rotation',
        'landmarks': ['shoulders', 'hips'],
        'phases': {
            'address': {'ideal': 0, 'min': 0, 'max': 5},
            'top': {'ideal': 50, 'min': 40, 'max': 60},
            'mid_downswing': {'ideal': 55, 'min': 45, 'max': 65}  # X-factor stretch
        },
        'importance': 'Key power indicator - correlates with clubhead speed'
    },
    
    # ========== ARM ANGLES ==========
    'lead_arm_angle': {
        'description': 'Left arm straightness (180 = perfectly straight)',
        'landmarks': ['left_shoulder', 'left_elbow', 'left_wrist'],
        'phases': {
            'address': {'ideal': 170, 'min': 160, 'max': 180},
            'top': {'ideal': 175, 'min': 165, 'max': 180},
            'impact': {'ideal': 170, 'min': 160, 'max': 180}
        },
        'importance': 'Affects swing arc width and consistency'
    },
    'trail_elbow_angle': {
        'description': 'Right elbow bend angle',
        'landmarks': ['right_shoulder', 'right_elbow', 'right_wrist'],
        'phases': {
            'address': {'ideal': 170, 'min': 160, 'max': 180},
            'top': {'ideal': 90, 'min': 75, 'max': 105},
            'impact': {'ideal': 150, 'min': 135, 'max': 170}
        },
        'importance': 'Controls club path and lag retention'
    },
    'wrist_hinge': {
        'description': 'Wrist cock angle (lead wrist)',
        'landmarks': ['left_elbow', 'left_wrist', 'left_index'],
        'phases': {
            'address': {'ideal': 150, 'min': 140, 'max': 170},
            'top': {'ideal': 90, 'min': 70, 'max': 110},
            'mid_downswing': {'ideal': 80, 'min': 60, 'max': 100}  # Lag
        },
        'importance': 'Creates lag for power release'
    },
    
    # ========== LOWER BODY ANGLES ==========
    'lead_knee_flex': {
        'description': 'Left knee bend angle',
        'landmarks': ['left_hip', 'left_knee', 'left_ankle'],
        'phases': {
            'address': {'ideal': 155, 'min': 145, 'max': 170},
            'top': {'ideal': 150, 'min': 140, 'max': 165},
            'impact': {'ideal': 170, 'min': 160, 'max': 180}  # Straightening
        },
        'importance': 'Provides ground force and stability'
    },
    'trail_knee_flex': {
        'description': 'Right knee bend angle',
        'landmarks': ['right_hip', 'right_knee', 'right_ankle'],
        'phases': {
            'address': {'ideal': 155, 'min': 145, 'max': 170},
            'top': {'ideal': 155, 'min': 145, 'max': 170},  # Maintain
            'impact': {'ideal': 145, 'min': 130, 'max': 160}
        },
        'importance': 'Maintains stability and restricts sway'
    },
    'stance_width_ratio': {
        'description': 'Width of stance relative to shoulder width',
        'landmarks': ['left_ankle', 'right_ankle', 'left_shoulder', 'right_shoulder'],
        'phases': {
            'address': {'ideal': 1.2, 'min': 1.0, 'max': 1.5}
        },
        'importance': 'Affects balance and power generation'
    },
    
    # ========== HEAD/STABILITY ==========
    'head_movement_lateral': {
        'description': 'Lateral head movement from address (should be minimal)',
        'landmarks': ['nose'],
        'phases': {
            'top': {'ideal': 0, 'min': -2, 'max': 3},  # Inches
            'impact': {'ideal': 0, 'min': -1, 'max': 2}
        },
        'importance': 'Affects strike consistency'
    },
    'head_movement_vertical': {
        'description': 'Vertical head movement from address',
        'landmarks': ['nose'],
        'phases': {
            'top': {'ideal': 0, 'min': -2, 'max': 2},
            'impact': {'ideal': 0, 'min': -2, 'max': 2}
        },
        'importance': 'Affects strike consistency'
    }
}


class GolfBiomechanics:
    """
    Calculate comprehensive golf-critical angles from pose landmarks.
    
    Works with both:
    - Real-time: lmList from PoseDetector (pixel coordinates)
    - Batch: DataFrame from SwingAnalyzer (normalized coordinates)
    """
    
    def __init__(self, pose_data=None, image_dims: Tuple[int, int] = None):
        """
        Initialize with pose data.
        
        Args:
            pose_data: Either DataFrame with pose columns or None for real-time use
            image_dims: (width, height) of the video frame for coordinate conversion
        """
        self.df = pose_data
        self.image_dims = image_dims
        self._reference_positions = {}  # Store address position for movement tracking
    
    # ============================================
    # CORE ANGLE CALCULATIONS
    # ============================================
    
    def calculate_angle_3points(self, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
        """
        Calculate angle at p2 given three points.
        
        Args:
            p1, p2, p3: Points as [x, y] or [x, y, z] arrays
            
        Returns:
            Angle in degrees (0-180)
        """
        v1 = np.array(p1[:2]) - np.array(p2[:2])
        v2 = np.array(p3[:2]) - np.array(p2[:2])
        
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
        angle = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))
        
        return angle
    
    def calculate_line_angle(self, p1: np.ndarray, p2: np.ndarray) -> float:
        """
        Calculate angle of line from horizontal.
        
        Args:
            p1, p2: Endpoints as [x, y] arrays
            
        Returns:
            Angle in degrees (-180 to 180)
        """
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        return math.degrees(math.atan2(dy, dx))
    
    def calculate_angle_from_vertical(self, p1: np.ndarray, p2: np.ndarray) -> float:
        """
        Calculate angle of line from vertical axis.
        
        Args:
            p1: Lower point (e.g., hip midpoint)
            p2: Upper point (e.g., shoulder midpoint)
            
        Returns:
            Angle in degrees (0 = vertical, positive = forward lean)
        """
        dx = p2[0] - p1[0]
        dy = p1[1] - p2[1]  # Invert Y since image Y increases downward
        
        angle_from_vertical = math.degrees(math.atan2(dx, dy))
        return angle_from_vertical
    
    # ============================================
    # HELPER: GET POINTS FROM DATA
    # ============================================
    
    def _get_point_from_lmlist(self, lmList: List, landmark_name: str) -> np.ndarray:
        """Get point from real-time lmList [id, x, y]"""
        idx = LANDMARK_INDICES[landmark_name]
        if idx < len(lmList):
            return np.array([lmList[idx][1], lmList[idx][2]])
        return np.array([0, 0])
    
    def _get_point_from_df(self, frame: int, landmark_name: str) -> np.ndarray:
        """Get point from DataFrame at specific frame"""
        row = self.df[self.df['frame'] == frame].iloc[0]
        x = row[f'{landmark_name}_x']
        y = row[f'{landmark_name}_y']
        return np.array([x, y])
    
    def _get_midpoint(self, p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
        """Calculate midpoint between two points"""
        return (p1 + p2) / 2
    
    # ============================================
    # GOLF-SPECIFIC ANGLE CALCULATIONS
    # ============================================
    
    def get_spine_angle(self, lmList: List = None, frame: int = None) -> float:
        """
        Calculate spine angle (forward tilt from vertical).
        
        Returns:
            Angle in degrees (positive = forward lean)
        """
        if lmList:
            left_hip = self._get_point_from_lmlist(lmList, 'left_hip')
            right_hip = self._get_point_from_lmlist(lmList, 'right_hip')
            left_shoulder = self._get_point_from_lmlist(lmList, 'left_shoulder')
            right_shoulder = self._get_point_from_lmlist(lmList, 'right_shoulder')
        else:
            left_hip = self._get_point_from_df(frame, 'left_hip')
            right_hip = self._get_point_from_df(frame, 'right_hip')
            left_shoulder = self._get_point_from_df(frame, 'left_shoulder')
            right_shoulder = self._get_point_from_df(frame, 'right_shoulder')
        
        hip_mid = self._get_midpoint(left_hip, right_hip)
        shoulder_mid = self._get_midpoint(left_shoulder, right_shoulder)
        
        return self.calculate_angle_from_vertical(hip_mid, shoulder_mid)
    
    def get_shoulder_rotation(self, lmList: List = None, frame: int = None) -> float:
        """
        Calculate shoulder line rotation from horizontal.
        
        Returns:
            Angle in degrees
        """
        if lmList:
            left_shoulder = self._get_point_from_lmlist(lmList, 'left_shoulder')
            right_shoulder = self._get_point_from_lmlist(lmList, 'right_shoulder')
        else:
            left_shoulder = self._get_point_from_df(frame, 'left_shoulder')
            right_shoulder = self._get_point_from_df(frame, 'right_shoulder')
        
        # Normalize to [0, 180] so left/right camera orientation does not invert quality.
        return abs(self.calculate_line_angle(left_shoulder, right_shoulder))
    
    def get_hip_rotation(self, lmList: List = None, frame: int = None) -> float:
        """
        Calculate hip line rotation from horizontal.
        
        Returns:
            Angle in degrees (0-180, clamped to reasonable rotation range)
        """
        if lmList:
            left_hip = self._get_point_from_lmlist(lmList, 'left_hip')
            right_hip = self._get_point_from_lmlist(lmList, 'right_hip')
        else:
            left_hip = self._get_point_from_df(frame, 'left_hip')
            right_hip = self._get_point_from_df(frame, 'right_hip')
        
        # Normalize to [0, 180] so left/right camera orientation does not invert quality.
        angle = abs(self.calculate_line_angle(left_hip, right_hip))
        # Sanity check: hip rotation should be in reasonable range, clamp extreme outliers
        # to prevent measurement artifacts from breaking scoring
        return np.clip(angle, 0, 180)
    
    def get_x_factor(self, lmList: List = None, frame: int = None) -> float:
        """
        Calculate X-factor (shoulder-hip separation).
        
        Returns:
            Angle in degrees (absolute difference)
        """
        shoulder_rot = self.get_shoulder_rotation(lmList, frame)
        hip_rot = self.get_hip_rotation(lmList, frame)

        # Circular angular difference to avoid wrap artifacts (e.g., 355 instead of 5).
        diff = abs(shoulder_rot - hip_rot) % 360
        return min(diff, 360 - diff)
    
    def get_lead_arm_angle(self, lmList: List = None, frame: int = None) -> float:
        """
        Calculate lead (left) arm angle at elbow.
        180 = perfectly straight
        
        Returns:
            Angle in degrees
        """
        if lmList:
            shoulder = self._get_point_from_lmlist(lmList, 'left_shoulder')
            elbow = self._get_point_from_lmlist(lmList, 'left_elbow')
            wrist = self._get_point_from_lmlist(lmList, 'left_wrist')
        else:
            shoulder = self._get_point_from_df(frame, 'left_shoulder')
            elbow = self._get_point_from_df(frame, 'left_elbow')
            wrist = self._get_point_from_df(frame, 'left_wrist')
        
        return self.calculate_angle_3points(shoulder, elbow, wrist)
    
    def get_trail_elbow_angle(self, lmList: List = None, frame: int = None) -> float:
        """
        Calculate trail (right) elbow angle.
        
        Returns:
            Angle in degrees
        """
        if lmList:
            shoulder = self._get_point_from_lmlist(lmList, 'right_shoulder')
            elbow = self._get_point_from_lmlist(lmList, 'right_elbow')
            wrist = self._get_point_from_lmlist(lmList, 'right_wrist')
        else:
            shoulder = self._get_point_from_df(frame, 'right_shoulder')
            elbow = self._get_point_from_df(frame, 'right_elbow')
            wrist = self._get_point_from_df(frame, 'right_wrist')
        
        return self.calculate_angle_3points(shoulder, elbow, wrist)
    
    def get_wrist_hinge(self, lmList: List = None, frame: int = None) -> float:
        """
        Calculate wrist hinge/cock angle.
        
        Returns:
            Angle in degrees (smaller = more hinged)
        """
        if lmList:
            elbow = self._get_point_from_lmlist(lmList, 'left_elbow')
            wrist = self._get_point_from_lmlist(lmList, 'left_wrist')
            index = self._get_point_from_lmlist(lmList, 'left_index')
        else:
            elbow = self._get_point_from_df(frame, 'left_elbow')
            wrist = self._get_point_from_df(frame, 'left_wrist')
            index = self._get_point_from_df(frame, 'left_index')
        
        return self.calculate_angle_3points(elbow, wrist, index)
    
    def get_lead_knee_flex(self, lmList: List = None, frame: int = None) -> float:
        """
        Calculate lead (left) knee flexion angle.
        180 = straight leg
        
        Returns:
            Angle in degrees
        """
        if lmList:
            hip = self._get_point_from_lmlist(lmList, 'left_hip')
            knee = self._get_point_from_lmlist(lmList, 'left_knee')
            ankle = self._get_point_from_lmlist(lmList, 'left_ankle')
        else:
            hip = self._get_point_from_df(frame, 'left_hip')
            knee = self._get_point_from_df(frame, 'left_knee')
            ankle = self._get_point_from_df(frame, 'left_ankle')
        
        return self.calculate_angle_3points(hip, knee, ankle)
    
    def get_trail_knee_flex(self, lmList: List = None, frame: int = None) -> float:
        """
        Calculate trail (right) knee flexion angle.
        
        Returns:
            Angle in degrees
        """
        if lmList:
            hip = self._get_point_from_lmlist(lmList, 'right_hip')
            knee = self._get_point_from_lmlist(lmList, 'right_knee')
            ankle = self._get_point_from_lmlist(lmList, 'right_ankle')
        else:
            hip = self._get_point_from_df(frame, 'right_hip')
            knee = self._get_point_from_df(frame, 'right_knee')
            ankle = self._get_point_from_df(frame, 'right_ankle')
        
        return self.calculate_angle_3points(hip, knee, ankle)
    
    def get_stance_width(self, lmList: List = None, frame: int = None) -> float:
        """
        Calculate stance width (ankle to ankle distance).
        
        Returns:
            Distance in pixels or normalized units
        """
        if lmList:
            left_ankle = self._get_point_from_lmlist(lmList, 'left_ankle')
            right_ankle = self._get_point_from_lmlist(lmList, 'right_ankle')
        else:
            left_ankle = self._get_point_from_df(frame, 'left_ankle')
            right_ankle = self._get_point_from_df(frame, 'right_ankle')
        
        return np.linalg.norm(left_ankle - right_ankle)
    
    def get_shoulder_width(self, lmList: List = None, frame: int = None) -> float:
        """
        Calculate shoulder width for ratio calculations.
        
        Returns:
            Distance in pixels or normalized units
        """
        if lmList:
            left_shoulder = self._get_point_from_lmlist(lmList, 'left_shoulder')
            right_shoulder = self._get_point_from_lmlist(lmList, 'right_shoulder')
        else:
            left_shoulder = self._get_point_from_df(frame, 'left_shoulder')
            right_shoulder = self._get_point_from_df(frame, 'right_shoulder')
        
        return np.linalg.norm(left_shoulder - right_shoulder)
    
    def get_stance_width_ratio(self, lmList: List = None, frame: int = None) -> float:
        """
        Calculate stance width as ratio of shoulder width.
        
        Returns:
            Ratio (1.0 = same as shoulder width)
        """
        stance = self.get_stance_width(lmList, frame)
        shoulder = self.get_shoulder_width(lmList, frame)
        return stance / (shoulder + 1e-8)
    
    def get_spine_lateral_tilt(self, lmList: List = None, frame: int = None) -> float:
        """
        Calculate spine lateral tilt (side bend).
        Positive = right shoulder lower (for RH golfer in backswing)
        
        Returns:
            Angle in degrees
        """
        if lmList:
            left_shoulder = self._get_point_from_lmlist(lmList, 'left_shoulder')
            right_shoulder = self._get_point_from_lmlist(lmList, 'right_shoulder')
        else:
            left_shoulder = self._get_point_from_df(frame, 'left_shoulder')
            right_shoulder = self._get_point_from_df(frame, 'right_shoulder')
        
        # Calculate vertical difference between shoulders
        dy = right_shoulder[1] - left_shoulder[1]  # Positive if right is lower (lower Y in image)
        dx = right_shoulder[0] - left_shoulder[0]
        
        return math.degrees(math.atan2(dy, abs(dx) + 1e-8))
    
    # ============================================
    # HEAD MOVEMENT TRACKING
    # ============================================
    
    def set_reference_position(self, lmList: List = None, frame: int = None):
        """
        Store reference position (usually at address) for movement tracking.
        """
        if lmList:
            self._reference_positions['nose'] = self._get_point_from_lmlist(lmList, 'nose')
            self._reference_positions['hip_mid'] = self._get_midpoint(
                self._get_point_from_lmlist(lmList, 'left_hip'),
                self._get_point_from_lmlist(lmList, 'right_hip')
            )
        else:
            self._reference_positions['nose'] = self._get_point_from_df(frame, 'nose')
            left_hip = self._get_point_from_df(frame, 'left_hip')
            right_hip = self._get_point_from_df(frame, 'right_hip')
            self._reference_positions['hip_mid'] = self._get_midpoint(left_hip, right_hip)
    
    def get_head_movement(self, lmList: List = None, frame: int = None) -> Tuple[float, float]:
        """
        Calculate head movement from reference position.
        
        Returns:
            (lateral_movement, vertical_movement) in pixels/normalized units
        """
        if 'nose' not in self._reference_positions:
            return (0.0, 0.0)
        
        if lmList:
            current_nose = self._get_point_from_lmlist(lmList, 'nose')
        else:
            current_nose = self._get_point_from_df(frame, 'nose')
        
        ref_nose = self._reference_positions['nose']
        
        lateral = current_nose[0] - ref_nose[0]
        vertical = current_nose[1] - ref_nose[1]  # Positive = moved down
        
        return (lateral, vertical)
    
    # ============================================
    # COMPREHENSIVE ANALYSIS
    # ============================================
    
    def calculate_all_metrics(self, lmList: List = None, frame: int = None) -> Dict[str, float]:
        """
        Calculate all golf biomechanical metrics at once.
        
        Args:
            lmList: Real-time landmark list from PoseDetector
            frame: Frame number for DataFrame-based analysis
            
        Returns:
            Dictionary with all calculated metrics
        """
        metrics = {
            # Posture
            'spine_angle': self.get_spine_angle(lmList, frame),
            'spine_lateral_tilt': self.get_spine_lateral_tilt(lmList, frame),
            
            # Rotation
            'shoulder_rotation': self.get_shoulder_rotation(lmList, frame),
            'hip_rotation': self.get_hip_rotation(lmList, frame),
            'hip_angle': self.get_hip_rotation(lmList, frame),
            'x_factor': self.get_x_factor(lmList, frame),
            
            # Arms
            'lead_arm_angle': self.get_lead_arm_angle(lmList, frame),
            'trail_elbow_angle': self.get_trail_elbow_angle(lmList, frame),
            'arm_extension': self.get_lead_arm_angle(lmList, frame),
            'wrist_hinge': self.get_wrist_hinge(lmList, frame),
            'wrist_angle': self.get_wrist_hinge(lmList, frame),
            'lag_angle': self.get_wrist_hinge(lmList, frame),
            
            # Lower body
            'lead_knee_flex': self.get_lead_knee_flex(lmList, frame),
            'trail_knee_flex': self.get_trail_knee_flex(lmList, frame),
            'stance_width': self.get_stance_width(lmList, frame),
            'shoulder_width': self.get_shoulder_width(lmList, frame),
            'stance_width_ratio': self.get_stance_width_ratio(lmList, frame),
            
            # Head movement
            'head_lateral': self.get_head_movement(lmList, frame)[0],
            'head_vertical': self.get_head_movement(lmList, frame)[1],
            'head_displacement': np.linalg.norm(self.get_head_movement(lmList, frame)),
        }
        
        return metrics
    
    def analyze_phase(self, phase_name: str, lmList: List = None, frame: int = None) -> Dict:
        """
        Analyze metrics for a specific swing phase and compare to ideals.
        
        Args:
            phase_name: One of 'address', 'takeaway', 'mid_backswing', 'top', 
                       'mid_downswing', 'impact', 'follow_through', 'finish'
            lmList: Real-time landmark list
            frame: Frame number
            
        Returns:
            Dictionary with metrics and deviation from ideal
        """
        metrics = self.calculate_all_metrics(lmList, frame)
        
        # Map phase names to benchmark phases
        phase_mapping = {
            'address': 'address',
            'takeaway': 'address',  # Use address as baseline
            'mid_backswing': 'top',  # In between
            'top': 'top',
            'mid_downswing': 'mid_downswing',
            'impact': 'impact',
            'follow_through': 'impact',
            'finish': 'impact'
        }
        
        benchmark_phase = phase_mapping.get(phase_name.lower().replace('-', '_'), 'address')
        
        analysis = {
            'phase': phase_name,
            'metrics': metrics,
            'deviations': {},
            'issues': []
        }
        
        # Compare to benchmarks
        for angle_name, angle_info in GOLF_CRITICAL_ANGLES.items():
            if 'phases' in angle_info and benchmark_phase in angle_info['phases']:
                if angle_name in metrics:
                    ideal = angle_info['phases'][benchmark_phase]['ideal']
                    min_val = angle_info['phases'][benchmark_phase]['min']
                    max_val = angle_info['phases'][benchmark_phase]['max']
                    actual = metrics[angle_name]
                    
                    deviation = actual - ideal
                    analysis['deviations'][angle_name] = {
                        'actual': actual,
                        'ideal': ideal,
                        'deviation': deviation,
                        'in_range': min_val <= actual <= max_val
                    }
                    
                    if not (min_val <= actual <= max_val):
                        analysis['issues'].append({
                            'angle': angle_name,
                            'description': angle_info['description'],
                            'actual': actual,
                            'expected_range': (min_val, max_val),
                            'importance': angle_info.get('importance', '')
                        })
        
        return analysis
    
    def compute_angular_velocity_sequence(self, df: pd.DataFrame, 
                                          start_frame: int, end_frame: int) -> Dict[str, float]:
        """
        Calculate kinematic sequence during mid-downswing (hips → torso → arms → club).
        
        This is critical for golf performance as proper sequencing (hips lead, torso follows,
        then arms, then club) generates maximum power and consistency.
        
        Args:
            df: DataFrame with pose data
            start_frame: Start frame of mid-downswing phase
            end_frame: End frame of mid-downswing phase
            
        Returns:
            Dictionary with:
            {
                'hip_start_ms': ms when hips start rotating,
                'torso_start_ms': ms when torso starts rotating,
                'arm_start_ms': ms when arms start rotating,
                'club_start_ms': ms when club starts accelerating,
                'hip_velocity': degrees/ms,
                'torso_velocity': degrees/ms,
                'arm_velocity': degrees/ms,
                'sequence_efficiency': 0-1 score (1 = perfect),
            }
        """
        try:
            # Filter to phase frames
            phase_df = df[(df['frame'] >= start_frame) & (df['frame'] <= end_frame)].copy()
            
            if len(phase_df) < 3:
                return {}
            
            sequence = {
                'hip_start_ms': None,
                'torso_start_ms': None,
                'arm_start_ms': None,
                'club_start_ms': None,
                'hip_velocity': 0.0,
                'torso_velocity': 0.0,
                'arm_velocity': 0.0,
                'x_factor_stretch': 0.0,
                'sequence_efficiency': 0.0,
            }
            
            # Calculate rotation angles for each frame
            hip_rotations = []
            torso_rotations = []
            arm_rotations = []
            frames = []
            
            for idx, (_, row) in enumerate(phase_df.iterrows()):
                frame_num = row['frame']
                frames.append(frame_num)
                
                # Hip rotation
                try:
                    left_hip = np.array([row.get('left_hip_x', 0), row.get('left_hip_y', 0)])
                    right_hip = np.array([row.get('right_hip_x', 0), row.get('right_hip_y', 0)])
                    hip_mid = (left_hip + right_hip) / 2
                    
                    # Use a fixed reference point (target line assumed to be x=0)
                    # Rotation is angle of hip line from vertical
                    hip_angle = np.degrees(np.arctan2(right_hip[0] - left_hip[0], right_hip[1] - left_hip[1]))
                    hip_rotations.append(hip_angle)
                except:
                    hip_rotations.append(0)
                
                # Torso rotation (shoulders)
                try:
                    left_shoulder = np.array([row.get('left_shoulder_x', 0), row.get('left_shoulder_y', 0)])
                    right_shoulder = np.array([row.get('right_shoulder_x', 0), row.get('right_shoulder_y', 0)])
                    
                    torso_angle = np.degrees(np.arctan2(right_shoulder[0] - left_shoulder[0], 
                                                       right_shoulder[1] - left_shoulder[1]))
                    torso_rotations.append(torso_angle)
                except:
                    torso_rotations.append(0)
                
                # Arm rotation (wrist positions)
                try:
                    left_wrist = np.array([row.get('left_wrist_x', 0), row.get('left_wrist_y', 0)])
                    right_wrist = np.array([row.get('right_wrist_x', 0), row.get('right_wrist_y', 0)])
                    
                    arm_angle = np.degrees(np.arctan2(right_wrist[0] - left_wrist[0],
                                                      right_wrist[1] - left_wrist[1]))
                    arm_rotations.append(arm_angle)
                except:
                    arm_rotations.append(0)
            
            hip_rotations = np.array(hip_rotations, dtype=float)
            torso_rotations = np.array(torso_rotations, dtype=float)
            arm_rotations = np.array(arm_rotations, dtype=float)
            frames = np.array(frames)

            # Unwrap angle series to avoid 180/-180 discontinuity spikes.
            hip_unwrapped = np.degrees(np.unwrap(np.radians(hip_rotations)))
            torso_unwrapped = np.degrees(np.unwrap(np.radians(torso_rotations)))
            arm_unwrapped = np.degrees(np.unwrap(np.radians(arm_rotations)))

            # Dynamic X-factor stretch with circular angular difference.
            x_factor_series = np.abs((torso_rotations - hip_rotations + 180.0) % 360.0 - 180.0)
            if len(x_factor_series) > 0:
                if len(x_factor_series) >= 3:
                    kernel = np.ones(3, dtype=float) / 3.0
                    x_factor_smoothed = np.convolve(x_factor_series, kernel, mode='same')
                else:
                    x_factor_smoothed = x_factor_series

                # Use inter-percentile spread to suppress outlier spikes.
                p10 = np.percentile(x_factor_smoothed, 10)
                p90 = np.percentile(x_factor_smoothed, 90)
                stretch = max(0.0, float(p90 - p10))
                sequence['x_factor_stretch'] = min(stretch, 40.0)
            
            # Calculate velocities (rotation change over time)
            hip_velocities = np.abs(np.diff(hip_unwrapped))
            torso_velocities = np.abs(np.diff(torso_unwrapped))
            arm_velocities = np.abs(np.diff(arm_unwrapped))

            # Smooth velocities to suppress single-frame noise spikes.
            def smooth_velocity(v: np.ndarray, window: int = 3) -> np.ndarray:
                if len(v) < 3:
                    return v
                kernel = np.ones(window, dtype=float) / float(window)
                return np.convolve(v, kernel, mode='same')

            hip_velocities = smooth_velocity(hip_velocities)
            torso_velocities = smooth_velocity(torso_velocities)
            arm_velocities = smooth_velocity(arm_velocities)
            
            # Find robust motion onset requiring persistent activation.
            def find_motion_start(velocities, threshold_percentile=65, min_consecutive=2):
                """Find motion start from sustained velocity rise, not one-frame spikes."""
                if len(velocities) == 0:
                    return 0

                percentile_threshold = np.percentile(velocities, threshold_percentile)
                absolute_floor = 0.25 * np.max(velocities) if np.max(velocities) > 0 else 0
                threshold = max(percentile_threshold, absolute_floor)
                active = velocities >= threshold

                run = 0
                for i, is_active in enumerate(active):
                    run = run + 1 if is_active else 0
                    if run >= min_consecutive:
                        return i - min_consecutive + 1

                indices = np.where(active)[0]
                return int(indices[0]) if len(indices) > 0 else 0
            
            # Find start frames
            min_consecutive = 2 if len(frames) >= 6 else 1
            hip_start_idx = find_motion_start(hip_velocities, min_consecutive=min_consecutive)
            torso_start_idx = find_motion_start(torso_velocities, min_consecutive=min_consecutive)
            arm_start_idx = find_motion_start(arm_velocities, min_consecutive=min_consecutive)
            
            # Convert to milliseconds (assuming 30fps video)
            frame_to_ms = 1000 / 30.0  # ~33.33ms per frame
            
            sequence['hip_start_ms'] = hip_start_idx * frame_to_ms
            sequence['torso_start_ms'] = torso_start_idx * frame_to_ms
            sequence['arm_start_ms'] = arm_start_idx * frame_to_ms
            sequence['club_start_ms'] = (arm_start_idx + 1) * frame_to_ms  # club follows arms by ~1 frame
            
            # Calculate average velocities
            sequence['hip_velocity'] = float(np.mean(hip_velocities)) if len(hip_velocities) > 0 else 0
            sequence['torso_velocity'] = float(np.mean(torso_velocities)) if len(torso_velocities) > 0 else 0
            sequence['arm_velocity'] = float(np.mean(arm_velocities)) if len(arm_velocities) > 0 else 0
            
            # Calculate sequence efficiency
            # Perfect sequence: hips lead, torso follows ~50-150ms later, arms follow after that
            hip_to_torso_ms = sequence['torso_start_ms'] - sequence['hip_start_ms']
            torso_to_arm_ms = sequence['arm_start_ms'] - sequence['torso_start_ms']
            
            ideal_hip_to_torso = 75  # ms
            ideal_torso_to_arm = 100  # ms
            
            # Score based on how close to ideal
            hip_torso_error = abs(hip_to_torso_ms - ideal_hip_to_torso)
            torso_arm_error = abs(torso_to_arm_ms - ideal_torso_to_arm)
            
            max_error = 100  # ms
            efficiency = max(0, 1.0 - (hip_torso_error + torso_arm_error) / (2 * max_error))
            sequence['sequence_efficiency'] = float(efficiency)
            
            return sequence
            
        except Exception as e:
            print(f"Error computing kinematic sequence: {e}")
            return {}
    
    def analyze_full_swing(self, phase_frames: Dict[str, int]) -> pd.DataFrame:
        """
        Analyze all phases of a swing.
        
        Args:
            phase_frames: Dictionary mapping phase names to frame numbers
            
        Returns:
            DataFrame with metrics for each phase
        """
        results = []
        
        # Set reference at address
        if 'address' in phase_frames or 'Address' in phase_frames:
            address_frame = phase_frames.get('address', phase_frames.get('Address'))
            self.set_reference_position(frame=address_frame)
        
        for phase_name, frame_num in phase_frames.items():
            metrics = self.calculate_all_metrics(frame=frame_num)
            metrics['phase'] = phase_name
            metrics['frame'] = frame_num
            results.append(metrics)
        
        return pd.DataFrame(results)
