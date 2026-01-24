import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import numpy as np
import math
import os

# New API imports
BaseOptions = python.BaseOptions
PoseLandmarker = vision.PoseLandmarker
PoseLandmarkerOptions = vision.PoseLandmarkerOptions
VisionRunningMode = vision.RunningMode

# Pose connections for drawing skeleton
POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24), (23, 25), (24, 26), (25, 27),
    (26, 28), (27, 29), (28, 30), (29, 31), (30, 32), (21, 22)
]


class PoseDetector:
    def __init__(self, model_path='models/pose_landmarker_lite.task',
                 num_poses=1,
                 min_detection_confidence=0.5,
                 min_tracking_confidence=0.5):
        
        # Get project root (two levels up from src/pose/)
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        model_full_path = os.path.join(project_root, model_path)
        
        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_full_path),
            running_mode=VisionRunningMode.VIDEO,  # Good for video streams
            num_poses=num_poses,
            min_pose_detection_confidence=min_detection_confidence,
            min_pose_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
        self.landmarker = PoseLandmarker.create_from_options(options)
        self.results = None
        self.lmList = []  # Pixel coordinates [id, x, y]

    def findPose(self, img, draw=True):
        """Detect pose and optionally draw skeleton"""
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        self.results = self.landmarker.detect_for_video(mp_image, int(cv2.getTickCount()))  # Timestamp needed for VIDEO mode
        
        if draw and self.results.pose_landmarks:
            for landmarks in self.results.pose_landmarks:  # Usually only 1 pose
                self._draw_pose(img, landmarks)
        return img
    
    def _draw_pose(self, img, landmarks, circle_radius=3, circle_color=(0, 255, 0), line_color=(255, 0, 0)):
        """Custom pose drawing without mp_drawing dependency"""
        h, w, _ = img.shape
        
        # Draw skeleton lines
        for connection in POSE_CONNECTIONS:
            start_idx, end_idx = connection
            if start_idx < len(landmarks) and end_idx < len(landmarks):
                start = landmarks[start_idx]
                end = landmarks[end_idx]
                start_pos = (int(start.x * w), int(start.y * h))
                end_pos = (int(end.x * w), int(end.y * h))
                cv2.line(img, start_pos, end_pos, line_color, 2)
        
        # Draw landmark circles
        for landmark in landmarks:
            pos = (int(landmark.x * w), int(landmark.y * h))
            cv2.circle(img, pos, circle_radius, circle_color, -1)

    def findPosition(self, img, draw=True, drawColor=(255, 0, 0), circleSize=5):
        """Extract pixel coordinates"""
        self.lmList = []
        h, w, _ = img.shape
        
        if self.results.pose_landmarks:
            for idx, lm in enumerate(self.results.pose_landmarks[0]):  # Take first pose
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmList.append([idx, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), circleSize, drawColor, cv2.FILLED)
        return self.lmList

    def findAngle(self, img, p1, p2, p3, draw=True):
        if len(self.lmList) < max(p1, p2, p3) + 1:
            return 0
        
        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        x3, y3 = self.lmList[p3][1:]
        
        angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
        if angle < 0:
            angle += 360
        if angle > 180:
            angle = 360 - angle
        
        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 3)
            cv2.line(img, (x3, y3), (x2, y2), (255, 255, 255), 3)
            cv2.circle(img, (x1, y1), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x3, y3), 10, (0, 0, 255), cv2.FILLED)
            cv2.putText(img, f'{int(angle)}deg', (x2 - 50, y2 + 50),
                        cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)
        
        return angle

    def getLandmarks(self):
        """Raw normalized landmarks [id, x, y, z, visibility]"""
        landmarks = []
        if self.results.pose_landmarks:
            for idx, lm in enumerate(self.results.pose_landmarks[0]):
                landmarks.append([idx, lm.x, lm.y, lm.z, lm.visibility])
        return landmarks
    
    def getKeyLandmarks(self):
        """
        Get only golf-relevant landmarks
        
        Returns:
            Dictionary with key body points
        """
        if len(self.lmList) < 33:
            return {}
        
        key_landmarks = {
            'nose': self.lmList[0][1:],
            'left_shoulder': self.lmList[11][1:],
            'right_shoulder': self.lmList[12][1:],
            'left_elbow': self.lmList[13][1:],
            'right_elbow': self.lmList[14][1:],
            'left_wrist': self.lmList[15][1:],
            'right_wrist': self.lmList[16][1:],
            'left_hip': self.lmList[23][1:],
            'right_hip': self.lmList[24][1:],
            'left_knee': self.lmList[25][1:],
            'right_knee': self.lmList[26][1:],
            'left_ankle': self.lmList[27][1:],
            'right_ankle': self.lmList[28][1:],
        }
        
        return key_landmarks
    
    def getLmList(self):
        """
        Get current landmark list for external processing.
        
        Returns:
            List of [id, x, y] for each landmark
        """
        return self.lmList


# ============================================
# USAGE EXAMPLE (Similar to your original code)
# ============================================

def main():
    """
    Example usage - similar to your original code
    """
    # Initialize detector
    detector = PoseDetector()
    
    # Option 1: Process video file
    cap = cv2.VideoCapture('PoseVideos/1.mp4')
    
    # Option 2: Process single image
    # img = cv2.imread('AiTrainer/test.jpg')
    # img = detector.findPose(img)
    # lmList = detector.findPosition(img, draw=False)
    # if len(lmList) != 0:
    #     detector.findAngle(img, 12, 14, 16)
    # cv2.imshow("Image", img)
    # cv2.waitKey(0)
    
    while True:
        success, img = cap.read()
        if not success:
            break
        
        # Detect pose
        img = detector.findPose(img)
        
        # Get landmark positions
        lmList = detector.findPosition(img, draw=False)
        
        # Calculate angles if landmarks detected
        if len(lmList) != 0:
            # Example: Calculate right elbow angle (shoulder-elbow-wrist)
            angle = detector.findAngle(img, 12, 14, 16)
            print(f"Right elbow angle: {angle:.1f}°")
            
            # For comprehensive metrics, use:
            # from src.biomechanics import GolfBiomechanics
            # biomech = GolfBiomechanics()
            # metrics = biomech.calculate_all_metrics(lmList=lmList)
        
        # Display
        cv2.imshow("Golf Swing Analysis", img)
        
        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()