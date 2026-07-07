"""
train_phase_scorer.py
=====================
Trains an XGBoost-based phase scorer from expert annotations.

WORKFLOW
--------
Step 1 – Prepare features (always run this first):
    python train_phase_scorer.py --prepare

Step 2 – Train & evaluate the model:
    python train_phase_scorer.py --train

Step 3 – Run both together:
    python train_phase_scorer.py --prepare --train
"""

import os
import argparse
import numpy as np
import pandas as pd
import pickle
import warnings
warnings.filterwarnings("ignore")

# ── Optional visual annotation dependency ─────────────────────────────────────
try:
    import cv2 as _cv2_test  # noqa: F401
    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False

# ── Paths ──────────────────────────────────────────────────────────────────────
ANNOTATION_PATH   = "pro_annotation_sample.csv"
DATA_DIR          = "data"
METRICS_DIR       = os.path.join(DATA_DIR, "metrics")
KEYFRAMES_DIR     = os.path.join(DATA_DIR, "keyframes")
FEATURES_OUT_PATH = os.path.join(DATA_DIR, "training_features.csv")
MODEL_OUT_PATH    = os.path.join("models", "phase_scorer.pkl")
POSES_DIR         = os.path.join(DATA_DIR, "extracted_poses")
VISUAL_OUTPUT_DIR = os.path.join("outputs", "visual_feedback")

# ── Feedback behavior knobs ───────────────────────────────────────────────────
FEEDBACK_SCORE_THRESHOLD = 85.0
FEEDBACK_MAX_ITEMS = 2
MIN_LOCAL_CONTRIB = 0.02
MIN_NORM_DEVIATION = 0.35

# ── Phase definitions ─────────────────────────────────────────────────────────
# Maps annotation CSV column  →  phase name used in the keyframe CSV
PHASE_LABEL_MAP = {
    "address_score":        "Address",
    "take_away_score":      "Takeaway",
    "mid_backswing_score":  "Mid-backswing",
    "top_score":            "Top",
    "impact_score":         "Impact",
    "mid_downswing_score":  "Mid-downswing",
    "follow_through_score": "Follow-through",
    "finish_score":         "Finish",
}

SCORE_COLS  = list(PHASE_LABEL_MAP.keys())   # label column names (8)
PHASE_NAMES = list(PHASE_LABEL_MAP.values()) # phase names as in CSVs (8)

# ── Biomechanical metric columns (from *_cleaned_metrics.csv) ─────────────────
METRIC_COLS = [
    "spine_angle", "spine_lateral_tilt",
    "shoulder_rotation", "hip_rotation", "hip_angle",
    "x_factor",
    "shoulder_rotation_3d", "hip_rotation_3d", "x_factor_3d",
    "lead_arm_angle", "trail_elbow_angle", "arm_extension",
    "wrist_hinge", "wrist_angle", "lag_angle",
    "lead_knee_flex", "trail_knee_flex",
    "stance_width", "shoulder_width", "stance_width_ratio",
    "head_lateral", "head_vertical", "head_displacement",
]

# ── Feature-to-feedback map: (phase_prefix, metric_name, 'high'|'low') → message
FEATURE_FEEDBACK_MAP = {
    # ADDRESS
    ("address", "spine_angle",            "high"): "Spine angle too upright at address — add more forward tilt",
    ("address", "spine_angle",            "low"):  "Too much forward bend at address — stand slightly more upright",
    ("address", "spine_lateral_tilt",     "high"): "Shoulders tilted excessively at address — level them out",
    ("address", "shoulder_rotation",      "low"):  "Shoulders closed at address — align square to target line",
    ("address", "shoulder_rotation",      "high"): "Shoulders open at address — square up to target line",
    ("address", "stance_width_ratio",     "low"):  "Stance too narrow at address — widen to shoulder width",
    ("address", "stance_width_ratio",     "high"): "Stance too wide at address — bring feet to shoulder width",
    ("address", "stance_width",           "low"):  "Stance too narrow at address — widen to shoulder width",
    ("address", "stance_width",           "high"): "Stance too wide at address — bring feet to shoulder width",
    ("address", "wrist_angle",            "high"): "Hands positioned too far from body at address",
    ("address", "wrist_angle",            "low"):  "Hands positioned too close to body at address",
    ("address", "lead_arm_angle",         "low"):  "Lead arm bent at address — extend naturally toward the ball",
    ("address", "lead_arm_angle",         "high"): "Lead arm too rigid at address — relax into natural extension",
    ("address", "trail_elbow_angle",      "low"):  "Trail arm too bent at address — extend naturally",
    ("address", "trail_elbow_angle",      "high"): "Trail arm too extended at address — relax arms",
    ("address", "arm_extension",          "low"):  "Arms too bent at address — extend toward the ball",
    ("address", "arm_extension",          "high"): "Arms too rigid at address — relax natural arm extension",
    ("address", "shoulder_width",         "low"):  "Narrow shoulder posture at address — open chest slightly",
    ("address", "shoulder_width",         "high"): "Wide shoulder posture detected at address — check setup",
    ("address", "lead_knee_flex",         "low"):  "Lead leg too straight at address — add knee flex",
    ("address", "lead_knee_flex",         "high"): "Excessive lead knee flex at address — stand more upright",
    ("address", "trail_knee_flex",        "low"):  "Trail leg too straight at address — add knee flex",
    ("address", "trail_knee_flex",        "high"): "Excessive trail knee flex at address — stand more upright",

    # TAKEAWAY
    ("takeaway", "x_factor",              "low"):  "Hip-shoulder separation low at takeaway position",
    ("takeaway", "x_factor",              "high"): "Hips over-rotating relative to shoulders at takeaway",
    ("takeaway", "shoulder_rotation",     "low"):  "Shoulder rotation low at takeaway checkpoint",
    ("takeaway", "shoulder_rotation",     "high"): "Shoulder over-rotation at takeaway — keep upper body controlled",
    ("takeaway", "wrist_angle",           "low"):  "Wrists relatively flat at takeaway position",
    ("takeaway", "wrist_angle",           "high"): "Wrists cupping at takeaway — keep them neutral",
    ("takeaway", "wrist_hinge",           "low"):  "Wrist hinge low at takeaway — wrists not yet loading",
    ("takeaway", "wrist_hinge",           "high"): "Wrists over-hinging early at takeaway",
    ("takeaway", "lead_arm_angle",        "low"):  "Lead arm bent at takeaway checkpoint",
    ("takeaway", "lead_arm_angle",        "high"): "Lead arm over-extended at takeaway — stay relaxed",
    ("takeaway", "trail_elbow_angle",     "low"):  "Trail elbow tucking too early at takeaway",
    ("takeaway", "trail_elbow_angle",     "high"): "Trail elbow flaring at takeaway — keep it soft",
    ("takeaway", "arm_extension",         "low"):  "Arms collapsing at takeaway — maintain extension",
    ("takeaway", "arm_extension",         "high"): "Arms overly extended at takeaway — stay relaxed",
    ("takeaway", "shoulder_width",        "low"):  "Narrow shoulder posture at takeaway",
    ("takeaway", "shoulder_width",        "high"): "Wide shoulder posture at takeaway",
    ("takeaway", "stance_width",          "low"):  "Narrow stance at takeaway — check foot position",
    ("takeaway", "stance_width",          "high"): "Stance too wide at takeaway",
    ("takeaway", "stance_width_ratio",    "low"):  "Stance width narrow relative to shoulders at takeaway",
    ("takeaway", "stance_width_ratio",    "high"): "Stance width wide relative to shoulders at takeaway",
    ("takeaway", "head_displacement",     "high"): "Head displaced from address position at takeaway",

    # MID-BACKSWING
    ("mid_backswing", "x_factor",         "low"):  "Low hip-shoulder separation at mid-backswing",
    ("mid_backswing", "x_factor",         "high"): "Hips over-rotating at mid-backswing — maintain separation",
    ("mid_backswing", "x_factor_3d",      "low"):  "3D hip-shoulder separation low at mid-backswing",
    ("mid_backswing", "shoulder_rotation", "low"): "Shoulder rotation low at mid-backswing checkpoint",
    ("mid_backswing", "shoulder_rotation", "high"): "Excessive shoulder rotation at mid-backswing",
    ("mid_backswing", "lead_arm_angle",   "low"):  "Lead arm bent at mid-backswing checkpoint",
    ("mid_backswing", "lead_arm_angle",   "high"): "Lead arm too rigid at mid-backswing — maintain natural flex",
    ("mid_backswing", "wrist_angle",      "high"): "Limited wrist hinge at mid-backswing",
    ("mid_backswing", "wrist_hinge",      "low"):  "Wrist hinge low at mid-backswing checkpoint",
    ("mid_backswing", "wrist_hinge",      "high"): "Over-hinging wrists at mid-backswing checkpoint",
    ("mid_backswing", "trail_elbow_angle", "low"): "Trail elbow tucking too early at mid-backswing",
    ("mid_backswing", "trail_elbow_angle", "high"): "Trail elbow flaring at mid-backswing — tuck it",
    ("mid_backswing", "arm_extension",    "low"):  "Loss of arm extension at mid-backswing — maintain width",
    ("mid_backswing", "arm_extension",    "high"): "Arms overly extended at mid-backswing",
    ("mid_backswing", "shoulder_width",   "low"):  "Narrow shoulder posture at mid-backswing",
    ("mid_backswing", "shoulder_width",   "high"): "Wide shoulder posture at mid-backswing",
    ("mid_backswing", "stance_width",     "low"):  "Stance too narrow at mid-backswing",
    ("mid_backswing", "stance_width",     "high"): "Stance too wide at mid-backswing",
    ("mid_backswing", "stance_width_ratio", "low"): "Stance width narrow relative to shoulders at mid-backswing",
    ("mid_backswing", "stance_width_ratio", "high"): "Stance width wide relative to shoulders at mid-backswing",
    ("mid_backswing", "trail_knee_flex",  "low"):  "Trail knee relatively straight at mid-backswing",
    ("mid_backswing", "trail_knee_flex",  "high"): "Trail knee over-flexing at mid-backswing — maintain stability",
    ("mid_backswing", "head_displacement", "high"): "Head displaced from address position at mid-backswing",

    # TOP
    ("top", "shoulder_rotation",          "low"):  "Shoulder rotation low at top of swing",
    ("top", "shoulder_rotation",          "high"): "Over-rotation at top — excessive shoulder turn detected",
    ("top", "shoulder_rotation_3d",       "low"):  "3D shoulder rotation low at top of swing",
    ("top", "x_factor",                   "low"):  "Low hip-shoulder separation at top of swing",
    ("top", "x_factor",                   "high"): "Hips over-rotating at top — maintain hip-shoulder separation",
    ("top", "head_displacement",          "high"): "Head displaced from address position at top of swing",
    ("top", "wrist_angle",                "high"): "Limited wrist hinge at top of swing",
    ("top", "wrist_hinge",                "low"):  "Wrist hinge low at top of swing",
    ("top", "wrist_hinge",                "high"): "Wrists over-hinged at top of swing",
    ("top", "spine_angle",                "high"): "Spine angle above ideal range at top of swing",
    ("top", "lead_arm_angle",             "low"):  "Lead arm bent at top of swing",
    ("top", "lead_arm_angle",             "high"): "Lead arm too rigid at top — allow natural flex",
    ("top", "trail_elbow_angle",          "low"):  "Trail elbow over-bent at top — let it point downward",
    ("top", "trail_elbow_angle",          "high"): "Trail elbow flaring at top of swing — tuck it",
    ("top", "arm_extension",              "low"):  "Lead arm bending too much at top — maintain extension",
    ("top", "arm_extension",              "high"): "Arms overly rigid at top of swing",
    ("top", "shoulder_width",             "low"):  "Narrow shoulder posture at top of swing",
    ("top", "shoulder_width",             "high"): "Wide shoulder posture at top of swing",
    ("top", "stance_width",               "low"):  "Stance narrowing at top — check balance",
    ("top", "stance_width",               "high"): "Stance too wide at top",
    ("top", "stance_width_ratio",         "low"):  "Stance width narrow relative to shoulders at top",
    ("top", "stance_width_ratio",         "high"): "Stance width wide relative to shoulders at top",
    ("top", "trail_knee_flex",            "low"):  "Trail knee straightening at top — maintain flex for power",
    ("top", "trail_knee_flex",            "high"): "Trail knee over-flexing at top of swing",

    # IMPACT
    ("impact", "hip_rotation",            "low"):  "Hip rotation low at impact position",
    ("impact", "hip_rotation_3d",         "low"):  "3D hip rotation low at impact position",
    ("impact", "shoulder_rotation",       "low"):  "Shoulders not clearing at impact — rotate through the ball",
    ("impact", "shoulder_rotation",       "high"): "Shoulder over-rotation at impact — control body rotation",
    ("impact", "lead_arm_angle",          "low"):  "Lead arm bent at impact position",
    ("impact", "lead_arm_angle",          "high"): "Lead arm too rigid at impact — allow natural release",
    ("impact", "x_factor",                "low"):  "Low hip-shoulder separation at impact",
    ("impact", "x_factor",                "high"): "Body rotation ahead of arm position at impact",
    ("impact", "lag_angle",               "low"):  "Releasing lag too early — maintain wrist angle into impact",
    ("impact", "lag_angle",               "high"): "Wrist lag angle low at impact position",
    ("impact", "wrist_angle",             "low"):  "Wrist angle low at impact — maintain lag through contact",
    ("impact", "wrist_angle",             "high"): "Wrist angle releasing early at impact",
    ("impact", "wrist_hinge",             "low"):  "Low wrist hinge at impact",
    ("impact", "wrist_hinge",             "high"): "Excessive wrist hinge at impact",
    ("impact", "trail_elbow_angle",       "low"):  "Trail elbow over-bent at impact — extend through the ball",
    ("impact", "trail_elbow_angle",       "high"): "Trail elbow not releasing at impact",
    ("impact", "arm_extension",           "low"):  "Arm extension low at impact — straighten through the ball",
    ("impact", "arm_extension",           "high"): "Arms overly extended at impact",
    ("impact", "shoulder_width",          "low"):  "Narrow shoulder posture at impact",
    ("impact", "shoulder_width",          "high"): "Wide shoulder posture at impact",
    ("impact", "stance_width",            "low"):  "Narrow stance at impact",
    ("impact", "stance_width",            "high"): "Stance too wide at impact",
    ("impact", "stance_width_ratio",      "low"):  "Stance width narrow relative to shoulders at impact",
    ("impact", "stance_width_ratio",      "high"): "Stance width wide relative to shoulders at impact",
    ("impact", "trail_knee_flex",         "low"):  "Trail leg straightening at impact — drive knee toward target",
    ("impact", "trail_knee_flex",         "high"): "Trail knee over-bent at impact",
    ("impact", "head_displacement",       "high"): "Head displaced from address position at impact",

    # MID-DOWNSWING
    ("mid_downswing", "lag_angle",        "low"):  "Releasing lag too early in downswing — hold angle longer",
    ("mid_downswing", "lag_angle",        "high"): "Wrist lag angle low at mid-downswing checkpoint",
    ("mid_downswing", "hip_rotation",     "low"):  "Hip rotation low at mid-downswing position",
    ("mid_downswing", "hip_rotation_3d",  "low"):  "3D hip rotation low at mid-downswing",
    ("mid_downswing", "shoulder_rotation", "low"): "Shoulder rotation low at mid-downswing — clear through",
    ("mid_downswing", "shoulder_rotation", "high"): "Shoulder rotation high relative to hips at mid-downswing",
    ("mid_downswing", "x_factor",         "low"):  "Low hip-shoulder separation at mid-downswing",
    ("mid_downswing", "x_factor",         "high"): "Hips over-rotating in downswing — maintain separation",
    ("mid_downswing", "lead_arm_angle",   "low"):  "Lead arm bent in downswing — maintain extension",
    ("mid_downswing", "lead_arm_angle",   "high"): "Lead arm too rigid in downswing — allow natural release",
    ("mid_downswing", "trail_elbow_angle", "low"): "Trail elbow over-bent in downswing — extend through ball",
    ("mid_downswing", "trail_elbow_angle", "high"): "Trail elbow flaring in downswing — tuck and drive",
    ("mid_downswing", "arm_extension",    "low"):  "Loss of arm extension in downswing — maintain swing width",
    ("mid_downswing", "arm_extension",    "high"): "Arms overly extended in downswing",
    ("mid_downswing", "wrist_angle",      "low"):  "Wrist angle low at mid-downswing — hold the lag",
    ("mid_downswing", "wrist_hinge",      "low"):  "Low wrist hinge at mid-downswing",
    ("mid_downswing", "wrist_hinge",      "high"): "Over-hinging at mid-downswing checkpoint",
    ("mid_downswing", "shoulder_width",   "low"):  "Narrow shoulder posture in downswing",
    ("mid_downswing", "shoulder_width",   "high"): "Wide shoulder posture in downswing",
    ("mid_downswing", "stance_width",     "low"):  "Narrow stance at mid-downswing",
    ("mid_downswing", "stance_width",     "high"): "Stance too wide at mid-downswing",
    ("mid_downswing", "stance_width_ratio", "low"): "Stance narrow relative to shoulders at mid-downswing",
    ("mid_downswing", "stance_width_ratio", "high"): "Stance wide relative to shoulders at mid-downswing",
    ("mid_downswing", "trail_knee_flex",  "low"):  "Trail leg straightening in downswing — drive knee through",
    ("mid_downswing", "trail_knee_flex",  "high"): "Trail knee over-bent in downswing",
    ("mid_downswing", "head_displacement", "high"): "Head displaced from address position at mid-downswing",

    # FOLLOW-THROUGH
    ("follow_through", "shoulder_rotation", "low"):  "Shoulder rotation low at follow-through checkpoint",
    ("follow_through", "shoulder_rotation", "high"): "Shoulder over-rotation at follow-through",
    ("follow_through", "x_factor",          "low"):  "Low hip-shoulder separation at follow-through",
    ("follow_through", "x_factor",          "high"): "Hips over-rotating at follow-through",
    ("follow_through", "lead_arm_angle",    "low"):  "Lead arm bent at follow-through checkpoint",
    ("follow_through", "lead_arm_angle",    "high"): "Lead arm too rigid at follow-through — allow natural fold",
    ("follow_through", "hip_rotation",      "low"):  "Hip rotation low at follow-through position",
    ("follow_through", "hip_rotation_3d",   "low"):  "3D hip rotation low at follow-through",
    ("follow_through", "trail_elbow_angle", "low"):  "Trail elbow over-bent at follow-through",
    ("follow_through", "trail_elbow_angle", "high"): "Trail elbow flaring at follow-through",
    ("follow_through", "arm_extension",     "low"):  "Arm extension low at follow-through",
    ("follow_through", "arm_extension",     "high"): "Arms overly extended at follow-through",
    ("follow_through", "wrist_angle",       "high"): "Limited wrist release at follow-through",
    ("follow_through", "wrist_hinge",       "high"): "Wrists over-hinged at follow-through",
    ("follow_through", "shoulder_width",    "low"):  "Narrow shoulder posture at follow-through",
    ("follow_through", "shoulder_width",    "high"): "Wide shoulder posture at follow-through",
    ("follow_through", "stance_width",      "low"):  "Narrow stance at follow-through",
    ("follow_through", "stance_width",      "high"): "Stance too wide at follow-through",
    ("follow_through", "stance_width_ratio", "low"):  "Stance narrow relative to shoulders at follow-through",
    ("follow_through", "stance_width_ratio", "high"): "Stance wide relative to shoulders at follow-through",
    ("follow_through", "trail_knee_flex",   "low"):  "Trail leg not releasing at follow-through — let it drive",
    ("follow_through", "trail_knee_flex",   "high"): "Trail knee over-bent at follow-through",
    ("follow_through", "spine_angle",       "high"): "Spine angle above ideal range at follow-through",

    # FINISH
    ("finish", "shoulder_rotation",       "low"):  "Shoulder rotation low at finish position",
    ("finish", "shoulder_rotation",       "high"): "Shoulder over-rotation at finish — check hip-shoulder sequencing",
    ("finish", "x_factor",                "low"):  "Low hip-shoulder separation at finish",
    ("finish", "x_factor",                "high"): "Hips over-rotating at finish",
    ("finish", "lead_knee_flex",          "high"): "Limited weight transfer at finish — lead knee over-flexed",
    ("finish", "hip_rotation",            "low"):  "Hip rotation low at finish position",
    ("finish", "hip_rotation_3d",         "low"):  "3D hip rotation low at finish position",
    ("finish", "wrist_angle",             "low"):  "Wrists flat at finish — check release pattern",
    ("finish", "wrist_angle",             "high"): "Wrists over-bent at finish position",
    ("finish", "wrist_hinge",             "low"):  "Low wrist hinge at finish",
    ("finish", "wrist_hinge",             "high"): "Wrists over-hinged at finish — check release",
    ("finish", "lag_angle",               "low"):  "Lag angle low at finish — release was early",
    ("finish", "lag_angle",               "high"): "Lag carried too long into finish position",
    ("finish", "lead_arm_angle",          "low"):  "Lead arm straight at finish — allow natural fold",
    ("finish", "lead_arm_angle",          "high"): "Lead arm over-bent at finish",
    ("finish", "trail_knee_flex",         "low"):  "Trail knee not flexing at finish — check weight transfer",
    ("finish", "trail_knee_flex",         "high"): "Trail knee over-bent at finish",
    ("finish", "trail_elbow_angle",       "low"):  "Trail elbow over-bent at finish",
    ("finish", "trail_elbow_angle",       "high"): "Trail elbow not folding at finish",
    ("finish", "arm_extension",           "low"):  "Arms folding too early at finish",
    ("finish", "arm_extension",           "high"): "Arms too extended at finish position",
    ("finish", "shoulder_width",          "low"):  "Narrow shoulder posture at finish — check balance",
    ("finish", "shoulder_width",          "high"): "Wide shoulder posture at finish",
    ("finish", "stance_width",            "low"):  "Narrow stance at finish",
    ("finish", "stance_width",            "high"): "Stance too wide at finish",
    ("finish", "stance_width_ratio",      "low"):  "Stance narrow relative to shoulders at finish",
    ("finish", "stance_width_ratio",      "high"): "Stance width too wide relative to shoulders at finish",
    ("finish", "head_displacement",       "high"): "Head significantly displaced at finish",
    ("finish", "spine_angle",             "high"): "Spine angle above ideal range at finish",
}


# ── Skeleton bone connections (MediaPipe landmark names from poses CSV) ────────
SKELETON_CONNECTIONS: list[tuple[str, str]] = [
    ("nose",           "left_eye"),
    ("nose",           "right_eye"),
    ("left_shoulder",  "right_shoulder"),
    ("left_shoulder",  "left_elbow"),
    ("left_elbow",     "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow",    "right_wrist"),
    ("left_shoulder",  "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip",       "right_hip"),
    ("left_hip",       "left_knee"),
    ("left_knee",      "left_ankle"),
    ("right_hip",      "right_knee"),
    ("right_knee",     "right_ankle"),
    ("left_ankle",     "left_heel"),
    ("left_heel",      "left_foot_index"),
    ("right_ankle",    "right_heel"),
    ("right_heel",     "right_foot_index"),
]

# ── Metric → joint segments to highlight red on deviation ─────────────────────
# Assumes right-handed golfer: lead = left side, trail = right side.
METRIC_TO_JOINTS: dict[str, list[tuple[str, str]]] = {
    "spine_angle":          [("left_shoulder",  "left_hip"),
                             ("right_shoulder", "right_hip")],
    "spine_lateral_tilt":   [("left_shoulder",  "left_hip"),
                             ("right_shoulder", "right_hip")],
    "shoulder_rotation":    [("left_shoulder",  "right_shoulder")],
    "hip_rotation":         [("left_hip",        "right_hip")],
    "hip_angle":            [("left_hip",        "left_knee"),
                             ("right_hip",       "right_knee")],
    "x_factor":             [("left_shoulder",  "right_shoulder"),
                             ("left_hip",        "right_hip")],
    "shoulder_rotation_3d": [("left_shoulder",  "right_shoulder")],
    "hip_rotation_3d":      [("left_hip",        "right_hip")],
    "x_factor_3d":          [("left_shoulder",  "right_shoulder"),
                             ("left_hip",        "right_hip")],
    "lead_arm_angle":       [("left_shoulder",   "left_elbow"),
                             ("left_elbow",      "left_wrist")],
    "trail_elbow_angle":    [("right_shoulder",  "right_elbow"),
                             ("right_elbow",     "right_wrist")],
    "arm_extension":        [("left_shoulder",   "left_elbow"),
                             ("left_elbow",      "left_wrist"),
                             ("right_shoulder",  "right_elbow"),
                             ("right_elbow",     "right_wrist")],
    "wrist_hinge":          [("left_elbow",      "left_wrist"),
                             ("right_elbow",     "right_wrist")],
    "wrist_angle":          [("left_elbow",      "left_wrist"),
                             ("right_elbow",     "right_wrist")],
    "lag_angle":            [("right_elbow",     "right_wrist"),
                             ("right_wrist",     "right_index")],
    "lead_knee_flex":       [("left_hip",        "left_knee"),
                             ("left_knee",       "left_ankle")],
    "trail_knee_flex":      [("right_hip",       "right_knee"),
                             ("right_knee",      "right_ankle")],
    "stance_width":         [("left_ankle",      "right_ankle")],
    "shoulder_width":       [("left_shoulder",   "right_shoulder")],
    "stance_width_ratio":   [("left_ankle",      "right_ankle"),
                             ("left_shoulder",   "right_shoulder")],
    "head_lateral":         [("nose",            "left_ear"),
                             ("nose",            "right_ear")],
    "head_vertical":        [("nose",            "left_eye"),
                             ("nose",            "right_eye")],
    "head_displacement":    [("nose",            "left_ear"),
                             ("nose",            "right_ear")],
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def _video_name(swing_id: str) -> str:
    """Strip _nn or _rb suffix to get the base video name."""
    return swing_id.replace("_nn", "").replace("_rb", "")


def _player_group(swing_id: str) -> str:
    """Return the player prefix (e.g. B12, B24) for grouped CV."""
    return swing_id.split("_")[0]


# ── Step 1: Feature extraction ─────────────────────────────────────────────────

def extract_features_for_swing(swing_id: str) -> dict | None:
    """
    For one swing, load its metrics CSV and phase CSV, then extract the
    biomechanical metrics at each phase's key frame.

    Returns a flat dict of 8*24 = 192 features, or None if data is missing.
    """
    video_name = _video_name(swing_id)

    metrics_path  = os.path.join(METRICS_DIR,  f"{video_name}_cleaned_metrics.csv")
    keyframe_dir  = os.path.join(KEYFRAMES_DIR, swing_id)
    keyframe_csv  = os.path.join(keyframe_dir,  f"{video_name}_cleaned_8phases.csv")

    if not os.path.exists(metrics_path):
        print(f"  [SKIP] {swing_id} — no metrics file: {metrics_path}")
        return None
    if not os.path.exists(keyframe_csv):
        print(f"  [SKIP] {swing_id} — no phase CSV: {keyframe_csv}")
        return None

    metrics_df = pd.read_csv(metrics_path)
    phases_df  = pd.read_csv(keyframe_csv)

    # Phase → key frame index  &  phase duration
    key_frame_map    = dict(zip(phases_df["Phase"], phases_df["Key_Frame"]))
    duration_map     = dict(zip(phases_df["Phase"], phases_df["Duration"]))

    features = {}
    for phase_name in PHASE_NAMES:
        prefix = phase_name.replace("-", "_").replace(" ", "_").lower()

        if phase_name not in key_frame_map:
            for col in METRIC_COLS:
                features[f"{prefix}_{col}"] = np.nan
            features[f"{prefix}_duration"] = np.nan
            continue

        key_frame = key_frame_map[phase_name]

        # Find the exact row (or nearest frame if not found)
        row_df = metrics_df[metrics_df["frame"] == key_frame]
        if row_df.empty:
            closest_idx = (metrics_df["frame"] - key_frame).abs().idxmin()
            row_df = metrics_df.iloc[[closest_idx]]

        for col in METRIC_COLS:
            features[f"{prefix}_{col}"] = (
                float(row_df[col].values[0]) if col in row_df.columns else np.nan
            )
        features[f"{prefix}_duration"] = float(duration_map.get(phase_name, 0))

    return features


def build_dataset() -> pd.DataFrame:
    """
    Join annotations + biomechanical features into one training-ready DataFrame.
    Saves result to FEATURES_OUT_PATH and returns it.
    """
    annotations = pd.read_csv(ANNOTATION_PATH)

    # Drop completely empty rows and rows without a skill label
    annotations = annotations.dropna(subset=["skill_level"])
    annotations = annotations[annotations["swing_id"].notna()]
    annotations = annotations[annotations["swing_id"].str.strip() != ""]

    rows = []
    skipped = []

    print(f"\n{'='*60}")
    print("BUILDING TRAINING DATASET")
    print(f"{'='*60}")
    print(f"Total annotated swings: {len(annotations)}")

    for _, ann in annotations.iterrows():
        swing_id = str(ann["swing_id"]).strip()

        # Check that at least one phase score exists
        scores = ann[SCORE_COLS]
        if scores.isna().all():
            skipped.append((swing_id, "all phase scores missing"))
            continue

        feats = extract_features_for_swing(swing_id)
        if feats is None:
            skipped.append((swing_id, "missing data files"))
            continue

        row = {
            "swing_id":    swing_id,
            "player_group": _player_group(swing_id),
            "camera_front": 1 if str(ann.get("camera_angle", "")).lower() == "front" else 0,
            "confidence":   float(ann.get("confidence", 3)),
            "skill_level":  str(ann["skill_level"]).lower().strip(),
        }
        row.update(feats)

        # Labels
        for col in SCORE_COLS:
            row[col] = ann[col] if not pd.isna(ann[col]) else np.nan

        rows.append(row)

    df = pd.DataFrame(rows)

    print(f"\n✓ Usable samples: {len(df)}")
    if skipped:
        print(f"✗ Skipped ({len(skipped)}):")
        for sid, reason in skipped:
            print(f"    {sid}: {reason}")

    # Summary of completeness
    print(f"\nPhase score completeness:")
    for col in SCORE_COLS:
        n_valid = df[col].notna().sum()
        print(f"  {col:<25} {n_valid}/{len(df)} filled")

    print(f"\nSkill level distribution:")
    print(df["skill_level"].value_counts().to_string())

    # Save
    os.makedirs(os.path.dirname(FEATURES_OUT_PATH), exist_ok=True)
    df.to_csv(FEATURES_OUT_PATH, index=False)
    print(f"\n✓ Features saved → {FEATURES_OUT_PATH}")

    return df


# ── Step 2: Training ───────────────────────────────────────────────────────────

def _get_feature_cols(df: pd.DataFrame) -> list[str]:
    """Return the input feature column names (not labels/metadata)."""
    exclude = {"swing_id", "player_group", "skill_level"} | set(SCORE_COLS)
    return [c for c in df.columns if c not in exclude]


def _build_phase_benchmarks(df: pd.DataFrame, feature_cols: list) -> dict:
    """
    For each phase, compute robust feature stats that represent 'what a pro
    looks like' — used as the target for feedback comparison.

    Priority:
      1. Pro swings only (skill_level == 'pro'), if >= 5 exist
      2. Fallback: all swings scoring >= 75 on that phase (mixed benchmark)
      3. Last resort: top half of available scores
    """
    PRO_MIN = 5  # minimum pro samples needed to use a pure-pro benchmark

    pro_df = df[df["skill_level"] == "pro"] if "skill_level" in df.columns else pd.DataFrame()
    use_pro_benchmark = len(pro_df) >= PRO_MIN
    if use_pro_benchmark:
        print(f"  [Benchmark] Using pro-only benchmark ({len(pro_df)} pro swings)")
    else:
        print(f"  [Benchmark] Not enough pros ({len(pro_df)}) — using score>=75 fallback")

    benchmarks = {}
    for score_col in SCORE_COLS:
        if score_col not in df.columns:
            continue

        if use_pro_benchmark:
            # Use pro swings; fall back to score>=75 per phase if a phase has few pro labels
            phase_pro = pro_df[pro_df[score_col].notna()]
            if len(phase_pro) >= PRO_MIN:
                good_df = phase_pro
            else:
                good_mask = df[score_col] >= 75
                good_df = df[good_mask] if good_mask.sum() >= 3 else df
        else:
            good_mask = df[score_col] >= 75
            if good_mask.sum() < 3:
                good_mask = df[score_col] >= df[score_col].quantile(0.5)
            good_df = df[good_mask]

        phase_bench = {}
        for col in feature_cols:
            vals = good_df[col].dropna()
            if len(vals) > 0:
                q1 = float(vals.quantile(0.25))
                med = float(vals.median())
                q3 = float(vals.quantile(0.75))
                iqr = max(q3 - q1, 1e-6)
                phase_bench[col] = {
                    "median": med,
                    "q1": q1,
                    "q3": q3,
                    "iqr": float(iqr),
                }
        benchmarks[score_col] = phase_bench
    return benchmarks


def _get_benchmark_stats(benchmark: dict, col: str) -> tuple[float, float] | None:
    """
    Return (median, iqr) for one feature from benchmark data.
    Supports both new dict stats format and legacy float mean format for
    backward compatibility with old model bundles.
    """
    raw = benchmark.get(col, None)
    if raw is None:
        return None

    if isinstance(raw, dict):
        med = raw.get("median", np.nan)
        iqr = raw.get("iqr", np.nan)
        if np.isnan(med) or np.isnan(iqr):
            return None
        return float(med), float(max(iqr, 1e-6))

    # Legacy model bundle: benchmark[col] was a single mean float
    if isinstance(raw, (int, float, np.floating)):
        med = float(raw)
        # Conservative fallback spread so legacy bundles still work
        iqr = max(abs(med) * 0.15, 1.0)
        return med, iqr

    return None


def _get_local_contribs_for_phase(model, X: np.ndarray, feature_cols: list[str]) -> dict[str, float]:
    """
    Compute per-feature local contribution magnitudes for one sample.
    Uses XGBoost pred_contribs when available; falls back to global
    feature_importances if pred_contribs is not available.
    """
    try:
        import xgboost as xgb  # noqa: PLC0415

        booster = model.get_booster()
        dmat = xgb.DMatrix(X, feature_names=feature_cols)
        contribs = booster.predict(dmat, pred_contribs=True)
        # Shape: (1, n_features + 1), last column is bias term
        vals = np.abs(contribs[0][:-1])
        return dict(zip(feature_cols, vals))
    except Exception:
        importances = dict(zip(feature_cols, model.feature_importances_))
        return {c: float(importances.get(c, 0.0)) for c in feature_cols}


def generate_phase_feedback(
    phase_name: str,
    score_col: str,
    features: dict,
    benchmark: dict,
    model,
    feature_cols: list,
    local_contribs: dict[str, float],
    score: float,
    max_items: int = FEEDBACK_MAX_ITEMS,
) -> list[str]:
    """
    Generate feedback for one phase by identifying which biomechanical features
    deviate most from the good-swing benchmark, weighted by XGBoost feature
    importance, then mapping to human-readable messages.
    """
    if score >= FEEDBACK_SCORE_THRESHOLD:
        return []

    prefix = phase_name.replace("-", "_").replace(" ", "_").lower()

    # When all local contribs are near-zero the model scored via its bias term;
    # bypass the contrib gate and rank purely by benchmark deviation instead.
    phase_locals = [
        float(local_contribs.get(col, 0.0))
        for col in feature_cols if col.startswith(prefix + "_")
    ]
    all_local_zero = bool(phase_locals) and all(abs(v) < 1e-6 for v in phase_locals)

    deviations = []
    fallback_candidates = []
    relaxed_candidates = []
    for col in feature_cols:
        if not col.startswith(prefix + "_"):
            continue
        metric = col[len(prefix) + 1:]  # strip phase prefix

        val       = features.get(col, np.nan)
        if np.isnan(val):
            continue

        stats = _get_benchmark_stats(benchmark, col)
        if stats is None:
            continue
        med, iqr = stats

        deviation = val - med
        norm_dev = abs(deviation) / max(iqr, 1e-6)
        local = float(local_contribs.get(col, 0.0))
        # Use pure deviation ranking when model relies on bias (all contribs zero)
        weighted = norm_dev if all_local_zero else local * norm_dev

        # Keep a relaxed ranking list so low-score phases still get at least
        # one concrete checkpoint when strict gates filter everything out.
        if weighted > 0:
            relaxed_candidates.append((weighted, metric))

        if norm_dev < MIN_NORM_DEVIATION:
            continue
        if not all_local_zero and local < MIN_LOCAL_CONTRIB:
            continue

        direction = "high" if deviation > 0 else "low"
        key = (prefix, metric, direction)
        if key in FEATURE_FEEDBACK_MAP:
            deviations.append((weighted, FEATURE_FEEDBACK_MAP[key]))
        else:
            fallback_candidates.append((weighted, metric))

    deviations.sort(key=lambda x: x[0], reverse=True)
    seen, feedback = set(), []
    for _, msg in deviations:
        if msg not in seen and len(feedback) < max_items:
            seen.add(msg)
            feedback.append(msg)

    # Fallback: if a phase is weak but no hand-authored mapping exists for the
    # strongest deviated metrics, still provide actionable generic guidance.
    if not feedback and fallback_candidates:
        fallback_candidates.sort(key=lambda x: x[0], reverse=True)
        for _, metric in fallback_candidates[:max_items]:
            metric_label = metric.replace("_", " ")
            feedback.append(
                f"{phase_name}: {metric_label} deviates from benchmark; prioritize correcting this checkpoint"
            )

    # Last fallback: if strict gates filtered everything for a weak phase,
    # use top relaxed deviations so detailed feedback is never empty.
    if not feedback and relaxed_candidates:
        relaxed_candidates.sort(key=lambda x: x[0], reverse=True)
        for _, metric in relaxed_candidates[:max_items]:
            metric_label = metric.replace("_", " ")
            feedback.append(
                f"{phase_name}: {metric_label} is the largest measurable deviation in this phase"
            )

    # Absolute safety net: do not leave weak phases without guidance text.
    if not feedback:
        feedback.append(
            f"{phase_name}: phase score is below target; review posture, rotation, and balance checkpoints"
        )

    return feedback


def _top_deviated_metric_names(
    phase_name: str,
    score_col: str,
    features: dict,
    benchmark: dict,
    model,
    feature_cols: list,
    local_contribs: dict[str, float],
    score: float,
    max_items: int = 3,
) -> list[str]:
    """
    Return the raw metric names (without phase prefix) that deviate most from
    the benchmark, weighted by XGBoost feature importance.
    Used to determine which joints to highlight in draw_annotated_keyframe().
    """
    if score >= FEEDBACK_SCORE_THRESHOLD:
        return []

    prefix = phase_name.replace("-", "_").replace(" ", "_").lower()

    # Mirror the same all_local_zero bypass used in generate_phase_feedback so
    # red joints always match the text feedback for bias-only phases (e.g. Finish).
    phase_locals = [
        float(local_contribs.get(col, 0.0))
        for col in feature_cols if col.startswith(prefix + "_")
    ]
    all_local_zero = bool(phase_locals) and all(abs(v) < 1e-6 for v in phase_locals)

    deviations = []
    relaxed_deviations = []
    for col in feature_cols:
        if not col.startswith(prefix + "_"):
            continue
        metric    = col[len(prefix) + 1:]
        val       = features.get(col, np.nan)
        if np.isnan(val):
            continue

        stats = _get_benchmark_stats(benchmark, col)
        if stats is None:
            continue
        med, iqr = stats

        deviation = val - med
        norm_dev = abs(deviation) / max(iqr, 1e-6)
        local = float(local_contribs.get(col, 0.0))
        weighted = norm_dev if all_local_zero else local * norm_dev

        if weighted > 0:
            relaxed_deviations.append((weighted, metric))

        passes = norm_dev >= MIN_NORM_DEVIATION and (
            all_local_zero or (local >= MIN_LOCAL_CONTRIB and weighted > 0)
        )
        if passes:
            deviations.append((weighted, metric))

    if deviations:
        deviations.sort(key=lambda x: x[0], reverse=True)
        return [m for _, m in deviations[:max_items]]

    # Fallback for weak phases when strict gates produce no metrics.
    relaxed_deviations.sort(key=lambda x: x[0], reverse=True)
    return [m for _, m in relaxed_deviations[:max_items]]


def draw_annotated_keyframe(
    swing_id: str,
    phase_name: str,
    deviated_metrics: list[str],
    score: float,
    output_dir: str = VISUAL_OUTPUT_DIR,
) -> str | None:
    """
    Draw the full skeleton on the saved key frame image for a phase, highlighting
    the joint segments associated with the top deviated metrics in red.

    Returns the path to the saved annotated image, or None if drawing failed.
    Requires opencv-python (cv2).
    """
    if not _CV2_AVAILABLE:
        print("  [WARN] opencv not available — skipping visual annotation (pip install opencv-python)")
        return None
    import cv2  # noqa: PLC0415

    video_name = _video_name(swing_id)

    # Load already-saved keyframe image
    img_path = os.path.join(KEYFRAMES_DIR, swing_id,
                            f"{video_name}_cleaned_{phase_name}.jpg")
    if not os.path.exists(img_path):
        return None
    img = cv2.imread(img_path)
    if img is None:
        return None
    h, w = img.shape[:2]

    # Load pose landmarks CSV and find the key frame row
    poses_path = os.path.join(POSES_DIR, f"{video_name}_cleaned_poses.csv")
    if not os.path.exists(poses_path):
        return None
    poses_df = pd.read_csv(poses_path)

    # Get key frame index from the phases CSV
    keyframe_csv = os.path.join(KEYFRAMES_DIR, swing_id,
                                f"{video_name}_cleaned_8phases.csv")
    if not os.path.exists(keyframe_csv):
        return None
    phases_df = pd.read_csv(keyframe_csv)
    phase_row = phases_df[phases_df["Phase"] == phase_name]
    if phase_row.empty:
        return None
    key_frame = int(phase_row["Key_Frame"].values[0])

    pose_row_df = poses_df[poses_df["frame"] == key_frame]
    if pose_row_df.empty:
        closest_idx = (poses_df["frame"] - key_frame).abs().idxmin()
        pose_row_df = poses_df.iloc[[closest_idx]]
    pose_row = pose_row_df.iloc[0]

    # Build landmark name → pixel coord mapping (coords are normalized 0-1)
    landmark_names = {c[:-2] for c in poses_df.columns if c.endswith("_x")}
    coords: dict[str, tuple[int, int]] = {}
    for lm in landmark_names:
        x_col, y_col = f"{lm}_x", f"{lm}_y"
        if x_col in pose_row.index and y_col in pose_row.index:
            px = int(float(pose_row[x_col]) * w)
            py = int(float(pose_row[y_col]) * h)
            coords[lm] = (px, py)

    # Collect all segments that should be highlighted red
    red_segs: set[tuple[str, str]] = set()
    for metric in deviated_metrics:
        for seg in METRIC_TO_JOINTS.get(metric, []):
            red_segs.add(seg)
    red_segs_lookup = red_segs | {(b, a) for a, b in red_segs}

    # Pass 1 — draw base skeleton in gray (skip segments that will be red)
    GRAY   = (120, 120, 120)
    GRAY_J = (170, 170, 170)
    for a, b in SKELETON_CONNECTIONS:
        if a not in coords or b not in coords:
            continue
        if (a, b) in red_segs_lookup:
            continue
        cv2.line(img, coords[a], coords[b], GRAY, 2, cv2.LINE_AA)
        cv2.circle(img, coords[a], 4, GRAY_J, -1)
        cv2.circle(img, coords[b], 4, GRAY_J, -1)

    # Pass 2 — draw deviated segments in red (thick, on top)
    RED = (0, 0, 255)
    red_joints: set[str] = set()
    for a, b in red_segs:
        if a not in coords or b not in coords:
            continue
        cv2.line(img, coords[a], coords[b], RED, 4, cv2.LINE_AA)
        red_joints.add(a)
        red_joints.add(b)
    for jt in red_joints:
        if jt in coords:
            cv2.circle(img, coords[jt], 7, RED, -1)

    # Top banner: phase name + score tag
    tag = "GOOD" if score >= 85 else ("OK" if score >= 70 else "NEEDS WORK")
    cv2.rectangle(img, (0, 0), (w, 44), (0, 0, 0), -1)
    cv2.putText(img, f"{phase_name}  |  {score:.0f}/100  [{tag}]",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                (255, 255, 255), 2, cv2.LINE_AA)

    # Save
    out_dir = os.path.join(output_dir, swing_id)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{phase_name}_annotated.jpg")
    cv2.imwrite(out_path, img)
    return out_path


def train_phase_scorer(df: pd.DataFrame | None = None):
    """
    Train one XGBoost regressor per phase on the prepared feature dataset.
    Saves models dict to MODEL_OUT_PATH.
    """
    try:
        from xgboost import XGBRegressor
    except ImportError:
        print("ERROR: xgboost not installed. Run:  pip install xgboost")
        return

    from sklearn.model_selection import GroupKFold, cross_val_score
    from sklearn.multioutput import MultiOutputRegressor
    from sklearn.metrics import mean_absolute_error
    from sklearn.impute import SimpleImputer

    if df is None:
        if not os.path.exists(FEATURES_OUT_PATH):
            print(f"ERROR: Features file not found. Run --prepare first.")
            return
        df = pd.read_csv(FEATURES_OUT_PATH)

    feature_cols = _get_feature_cols(df)

    print(f"\n{'='*60}")
    print("TRAINING PHASE SCORER")
    print(f"{'='*60}")
    print(f"Samples: {len(df)}  |  Features: {len(feature_cols)}")

    X_raw = df[feature_cols].values
    groups = df["player_group"].values

    # Impute missing features with column median
    imputer = SimpleImputer(strategy="median")
    X = imputer.fit_transform(X_raw)

    # ── Per-phase models ────────────────────────────────────────────────────
    models = {}
    cv_results = {}
    gkf = GroupKFold(n_splits=min(5, len(df["player_group"].unique())))

    for score_col, phase_name in PHASE_LABEL_MAP.items():
        y_series = df[score_col]
        valid_mask = y_series.notna()
        n_valid = valid_mask.sum()

        if n_valid < 5:
            print(f"  {phase_name:<20} SKIP (only {n_valid} labeled samples)")
            continue

        X_v = X[valid_mask]
        y_v = y_series[valid_mask].values
        g_v = groups[valid_mask]

        xgb = XGBRegressor(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            reg_alpha=0.5,
            reg_lambda=1.0,
            random_state=42,
            verbosity=0,
        )

        # Cross-validation
        n_splits = min(5, len(np.unique(g_v)))
        gkf_v = GroupKFold(n_splits=n_splits)
        cv_scores = cross_val_score(
            xgb, X_v, y_v,
            cv=gkf_v, groups=g_v,
            scoring="neg_mean_absolute_error",
        )
        mae = -cv_scores.mean()
        cv_results[phase_name] = mae

        # Fit on all data
        xgb.fit(X_v, y_v)
        models[score_col] = xgb

        print(f"  {phase_name:<20} MAE={mae:.1f}  (n={n_valid})")

    # ── Discrimination test ────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print("PRO vs BEGINNER DISCRIMINATION (predicted scores)")
    print(f"{'─'*60}")

    pro_mask      = df["skill_level"] == "pro"
    beginner_mask = df["skill_level"] == "beginner"

    if pro_mask.sum() > 0 and beginner_mask.sum() > 0:
        total_gaps = []
        for score_col, phase_name in PHASE_LABEL_MAP.items():
            if score_col not in models:
                continue
            xgb = models[score_col]
            preds = np.clip(xgb.predict(X), 0, 100)

            pro_mean = preds[pro_mask].mean()
            beg_mean = preds[beginner_mask].mean()
            gap = pro_mean - beg_mean
            total_gaps.append(gap)
            print(f"  {phase_name:<20} pro={pro_mean:.1f}  beg={beg_mean:.1f}  gap={gap:+.1f}")

        avg_gap = np.mean(total_gaps)
        print(f"\n  Average gap across phases: {avg_gap:+.1f} pts")
        target = 15
        status = "✓ TARGET MET" if avg_gap >= target else f"✗ target={target}"
        print(f"  {status}")

    # ── Phase benchmarks (good-swing averages for feedback) ──────────────
    benchmarks = _build_phase_benchmarks(df, feature_cols)

    # ── Save ───────────────────────────────────────────────────────────────
    bundle = {
        "models":          models,
        "imputer":         imputer,
        "feature_cols":    feature_cols,
        "score_cols":      SCORE_COLS,
        "phase_names":     PHASE_NAMES,
        "phase_label_map": PHASE_LABEL_MAP,
        "benchmarks":      benchmarks,
    }
    os.makedirs(os.path.dirname(MODEL_OUT_PATH), exist_ok=True)
    with open(MODEL_OUT_PATH, "wb") as f:
        pickle.dump(bundle, f)
    print(f"\n✓ Model saved → {MODEL_OUT_PATH}")

    return models


# ── Inference helper (used by pipeline) ───────────────────────────────────────

def predict_scores(swing_id: str, model_path: str = MODEL_OUT_PATH,
                   annotate: bool = False) -> dict | None:
    """
    Predict 8 phase scores + per-phase feedback for a single swing.

    Returns a dict with keys:
      - 'scores':   {phase_name: float}   — predicted 0-100 score per phase
      - 'feedback': {phase_name: [str]}   — list of feedback messages per phase
      - 'total':    float                 — average of all phase scores
      - 'images':   {phase_name: str}     — annotated keyframe paths (annotate=True only)
    """
    if not os.path.exists(model_path):
        return None

    with open(model_path, "rb") as f:
        bundle = pickle.load(f)

    feats = extract_features_for_swing(swing_id)
    if feats is None:
        return None

    models       = bundle["models"]
    imputer      = bundle["imputer"]
    feature_cols = bundle["feature_cols"]
    benchmarks   = bundle.get("benchmarks", {})

    # Backward compatibility:
    # older model bundles may contain legacy top-phase feature names prefixed
    # with "t__op_" due a historical typo. Map both ways at inference so
    # existing bundles keep working after the phase-name fix.
    row = {}
    for col in feature_cols:
        val = feats.get(col, np.nan)
        if pd.isna(val) and col.startswith("t__op_"):
            val = feats.get("top_" + col[len("t__op_"):], np.nan)
        elif pd.isna(val) and col.startswith("top_"):
            val = feats.get("t__op_" + col[len("top_"):], np.nan)
        row[col] = val
    X_raw = np.array([[row[c] for c in feature_cols]])
    X     = imputer.transform(X_raw)

    scores   = {}
    feedback = {}

    for score_col, phase_name in PHASE_LABEL_MAP.items():
        if score_col not in models:
            continue
        local_contribs = _get_local_contribs_for_phase(models[score_col], X, feature_cols)
        pred  = float(np.clip(models[score_col].predict(X)[0], 0, 100))
        score = round(pred, 1)
        scores[phase_name] = score

        phase_feedback = generate_phase_feedback(
            phase_name  = phase_name,
            score_col   = score_col,
            features    = feats,
            benchmark   = benchmarks.get(score_col, {}),
            model       = models[score_col],
            feature_cols= feature_cols,
            local_contribs= local_contribs,
            score       = score,
        )
        feedback[phase_name] = phase_feedback

    total = round(np.mean(list(scores.values())), 1) if scores else None

    # ── Visual annotation (optional) ──────────────────────────────────────────
    images: dict[str, str] = {}
    if annotate:
        for score_col, phase_name in PHASE_LABEL_MAP.items():
            if score_col not in models:
                continue
            score = scores.get(phase_name, 0.0)
            if score >= FEEDBACK_SCORE_THRESHOLD:
                continue  # no annotation needed for good phases
            local_contribs = _get_local_contribs_for_phase(models[score_col], X, feature_cols)
            deviated = _top_deviated_metric_names(
                phase_name   = phase_name,
                score_col    = score_col,
                features     = feats,
                benchmark    = benchmarks.get(score_col, {}),
                model        = models[score_col],
                feature_cols = feature_cols,
                local_contribs = local_contribs,
                score        = score,
            )
            if deviated:
                img_path = draw_annotated_keyframe(
                    swing_id         = swing_id,
                    phase_name       = phase_name,
                    deviated_metrics = deviated,
                    score            = score,
                )
                if img_path:
                    images[phase_name] = img_path

    return {"scores": scores, "feedback": feedback, "total": total, "images": images}


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train golf phase scorer from expert annotations.")
    parser.add_argument("--prepare", action="store_true", help="Build feature dataset from annotations + metrics")
    parser.add_argument("--train",   action="store_true", help="Train XGBoost models on the prepared features")
    parser.add_argument("--predict", type=str, default=None, metavar="SWING_ID",
                        help="Run inference on a single swing (e.g. B12_F1_I_nn)")
    parser.add_argument("--annotate", action="store_true",
                        help="Draw red-line skeleton feedback on key frame images for phases with issues (requires opencv-python)")
    args = parser.parse_args()

    if not any([args.prepare, args.train, args.predict]):
        parser.print_help()
    else:
        df = None
        if args.prepare:
            df = build_dataset()
        if args.train:
            train_phase_scorer(df)
        if args.predict:
            result = predict_scores(args.predict, annotate=args.annotate)
            if result:
                scores   = result["scores"]
                feedback = result["feedback"]
                total    = result["total"]
                images   = result.get("images", {})

                print(f"\n{'='*55}")
                print(f"SWING REPORT: {args.predict}")
                print(f"{'='*55}")
                print(f"  {'PHASE':<22} {'SCORE':>6}  FEEDBACK")
                print(f"  {'-'*22}  {'-'*6}  {'-'*25}")
                for phase_name, score in scores.items():
                    msgs = feedback.get(phase_name, [])
                    tag  = "GOOD" if score >= 85 else ("OK" if score >= 70 else "NEEDS WORK")
                    first = msgs[0] if msgs else ("No issues detected" if score >= 85 else "—")
                    print(f"  {phase_name:<22} {score:>5.1f}  [{tag}] {first}")
                    for msg in msgs[1:]:
                        print(f"  {'':22}         • {msg}")
                print(f"  {'─'*55}")
                print(f"  {'TOTAL SCORE':<22} {total:>5.1f}")
                print(f"{'='*55}")

                if images:
                    print(f"\n  Annotated keyframes saved:")
                    for phase_name, img_path in images.items():
                        print(f"    {phase_name:<22} → {img_path}")
            else:
                print(f"Could not predict for {args.predict} — run --prepare --train first.")
