"""
Video Processing Module
=======================
Video cleaning and preprocessing utilities.
"""

from .cleaner import detect_swing_bounds, crop_video, clean_videos

__all__ = ['detect_swing_bounds', 'crop_video', 'clean_videos']
