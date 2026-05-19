import numpy as np
from typing import Dict, Tuple, List

def get_vector(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    return p2 - p1

def vector_angle(v1: np.ndarray, v2: np.ndarray) -> float:
    """Tính góc giữa 2 vector trong không gian 3D (độ)."""
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 < 1e-6 or norm2 < 1e-6:
        return 0.0
    cos_ang = np.dot(v1, v2) / (norm1 * norm2)
    return np.degrees(np.arccos(np.clip(cos_ang, -1.0, 1.0)))

def calculate_joint_angle(p_top: np.ndarray, p_mid: np.ndarray, p_bot: np.ndarray) -> float:
    """Tính góc tại khớp (p_mid) giữa 3 điểm."""
    v1 = get_vector(p_mid, p_top)
    v2 = get_vector(p_mid, p_bot)
    return vector_angle(v1, v2)

def calculate_rotation_xz(p_l: np.ndarray, p_r: np.ndarray) -> float:
    """
    Tính góc xoay của một trục (ví dụ: vai/hông) trên mặt phẳng nằm ngang.
    Giả định hệ tọa độ MediaPipe: X (ngang), Z (sâu).
    """
    vec = p_r - p_l
    # Chiếu lên mặt phẳng X-Z
    vec_xz = np.array([vec[0], vec[2]])
    # So với trục X (target line giả định)
    target_line = np.array([1, 0])
    norm = np.linalg.norm(vec_xz)
    if norm < 1e-6: return 0.0
    
    angle = np.degrees(np.arccos(np.dot(vec_xz, target_line) / norm))
    return angle

def calculate_spine_angle(sh_mid: np.ndarray, hip_mid: np.ndarray) -> float:
    """Tính Forward Tilt của cột sống so với trục đứng (Y-axis)."""
    spine_vec = sh_mid - hip_mid
    vertical = np.array([0, -1, 0]) # MediaPipe Y hướng xuống, nên vector đứng là (0, -1, 0)
    return vector_angle(spine_vec, vertical)

def extract_metrics_from_row(row: Dict) -> Dict[str, float]:
    """Trích xuất các góc quan trọng từ một frame dữ liệu (row từ CSV)."""
    def to_np(prefix):
        return np.array([row[f"{prefix}_x"], row[f"{prefix}_y"], row[f"{prefix}_z"]])

    sh_l, sh_r = to_np("left_shoulder"), to_np("right_shoulder")
    hip_l, hip_r = to_np("left_hip"), to_np("right_hip")
    k_l, a_l = to_np("left_knee"), to_np("left_ankle")
    k_r, a_r = to_np("right_knee"), to_np("right_ankle")
    
    sh_mid = (sh_l + sh_r) / 2
    hip_mid = (hip_l + hip_r) / 2
    
    # 1. Rotations
    sh_turn = calculate_rotation_xz(sh_l, sh_r)
    hip_turn = calculate_rotation_xz(hip_l, hip_r)
    
    # 2. Knee Flex
    lead_knee_flex = calculate_joint_angle(hip_l, k_l, a_l)
    trail_knee_flex = calculate_joint_angle(hip_r, k_r, a_r)
    
    # 3. Spine & Head
    spine_tilt = calculate_spine_angle(sh_mid, hip_mid)
    
    # Head position (normalized)
    head_pos = to_np("nose")
    
    return {
        "shoulder_turn": sh_turn,
        "hip_turn": hip_turn,
        "x_factor": abs(sh_turn - hip_turn),
        "spine_angle": spine_tilt,
        "lead_knee_flex": lead_knee_flex,
        "trail_knee_flex": trail_knee_flex,
        "head_pos": head_pos
    }