"""
SwingAI Coach - Full Pipeline Integration
Using your PoseDetector class style
"""

import cv2
import pandas as pd
import numpy as np
from .detector import PoseDetector
from ..biomechanics import GolfBiomechanics


class SwingAnalyzer:
    """
    Complete swing analysis pipeline
    Extracts poses and prepares data for ML models
    """
    
    def __init__(self):
        self.detector = PoseDetector(
            model_path='models/pose_landmarker_lite.task',
            num_poses=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.pose_data = []
        self.metrics_data = []
        self.biomechanics = GolfBiomechanics()
    
    def processVideo(self, video_path, output_csv=None, show_preview=True):
        """
        Process entire video and extract pose data
        
        Args:
            video_path: Path to golf swing video
            output_csv: Path to save CSV (optional)
            show_preview: Show live preview window
            
        Returns:
            DataFrame with pose data
        """
        cap = cv2.VideoCapture(video_path)
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Processing video: {video_path}")
        print(f"FPS: {fps}, Total Frames: {total_frames}")
        
        frame_num = 0
        self.pose_data = []
        self.metrics_data = []
        
        while True:
            success, img = cap.read()
            if not success:
                break
            
            # Detect pose
            img = self.detector.findPose(img, draw=True)
            lmList = self.detector.findPosition(img, draw=False)
            
            if len(lmList) != 0:
                # Get raw landmarks (for ML model)
                landmarks = self.detector.getLandmarks()
                
                # Create feature row: [frame, x0, y0, z0, vis0, x1, y1, z1, vis1, ...]
                row = [frame_num]
                for lm in landmarks:
                    row.extend([lm[1], lm[2], lm[3], lm[4]])  # x, y, z, visibility
                
                self.pose_data.append(row)
                
                # Calculate comprehensive golf biomechanics metrics
                metrics = self.biomechanics.calculate_all_metrics(lmList=self.detector.getLmList())
                metrics['frame'] = frame_num
                self.metrics_data.append(metrics)
                
                # Display frame info
                cv2.putText(img, f"Frame: {frame_num}/{total_frames}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(img, f"Frame: {frame_num} - NO POSE DETECTED", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Show preview
            if show_preview:
                cv2.imshow("SwingAI Analysis", img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            frame_num += 1
            
            # Progress
            if frame_num % 10 == 0:
                print(f"Progress: {(frame_num/total_frames)*100:.1f}%", end='\r')
        
        cap.release()
        cv2.destroyAllWindows()
        
        print(f"\nProcessing complete! Detected poses in {len(self.pose_data)} frames")
        
        # Convert to DataFrame
        df = self._createDataFrame()
        
        # Save to CSV if specified
        if output_csv:
            df.to_csv(output_csv, index=False)
            print(f"Saved to: {output_csv}")
        
        return df
    
    def _createDataFrame(self):
        """Convert pose data to DataFrame with proper column names"""
        
        # Create column names
        columns = ['frame']
        landmark_names = [
            'nose', 'left_eye_inner', 'left_eye', 'left_eye_outer',
            'right_eye_inner', 'right_eye', 'right_eye_outer',
            'left_ear', 'right_ear', 'mouth_left', 'mouth_right',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_pinky', 'right_pinky',
            'left_index', 'right_index', 'left_thumb', 'right_thumb',
            'left_hip', 'right_hip', 'left_knee', 'right_knee',
            'left_ankle', 'right_ankle', 'left_heel', 'right_heel',
            'left_foot_index', 'right_foot_index'
        ]
        
        for name in landmark_names:
            columns.extend([f'{name}_x', f'{name}_y', f'{name}_z', f'{name}_visibility'])
        
        df = pd.DataFrame(self.pose_data, columns=columns)
        return df
    
    def getMetricsDataFrame(self):
        """Get biomechanical metrics as DataFrame"""
        return pd.DataFrame(self.metrics_data)
    
    def visualizeMetrics(self):
        """Plot metrics over time"""
        import matplotlib.pyplot as plt
        
        metrics_df = self.getMetricsDataFrame()
        
        if metrics_df.empty:
            print("No metrics data available")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Shoulder rotation
        axes[0, 0].plot(metrics_df['frame'], metrics_df['shoulder_rotation'])
        axes[0, 0].set_title('Shoulder Rotation Over Time')
        axes[0, 0].set_xlabel('Frame')
        axes[0, 0].set_ylabel('Angle (degrees)')
        axes[0, 0].grid(True)
        
        # Hip rotation
        axes[0, 1].plot(metrics_df['frame'], metrics_df['hip_rotation'])
        axes[0, 1].set_title('Hip Rotation Over Time')
        axes[0, 1].set_xlabel('Frame')
        axes[0, 1].set_ylabel('Angle (degrees)')
        axes[0, 1].grid(True)
        
        # X-factor
        axes[1, 0].plot(metrics_df['frame'], metrics_df['x_factor'])
        axes[1, 0].set_title('X-Factor (Shoulder-Hip Separation)')
        axes[1, 0].set_xlabel('Frame')
        axes[1, 0].set_ylabel('Angle (degrees)')
        axes[1, 0].grid(True)
        
        # Arm angles (lead arm and trail elbow)
        axes[1, 1].plot(metrics_df['frame'], metrics_df['lead_arm_angle'], label='Lead Arm')
        axes[1, 1].plot(metrics_df['frame'], metrics_df['trail_elbow_angle'], label='Trail Elbow')
        axes[1, 1].set_title('Arm Angles')
        axes[1, 1].set_xlabel('Frame')
        axes[1, 1].set_ylabel('Angle (degrees)')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig('metrics_analysis.png')
        plt.show()
        
        print("Metrics visualization saved to: metrics_analysis.png")


# ============================================
# MAIN EXECUTION
# ============================================

def main():
    """
    Process golf swing videos and extract features
    """
    
    # Initialize analyzer
    analyzer = SwingAnalyzer()
    
    # Process video (similar to your original code style)
    video_path = 'PoseVideos/golf_swing_001.mp4'
    output_csv = 'data/pose_data_001.csv'
    
    # Extract pose data
    pose_df = analyzer.processVideo(
        video_path=video_path,
        output_csv=output_csv,
        show_preview=True
    )
    
    print("\n" + "="*60)
    print("POSE EXTRACTION COMPLETE")
    print("="*60)
    print(f"Shape: {pose_df.shape}")
    print(f"\nFirst 3 frames:")
    print(pose_df[['frame', 'nose_x', 'nose_y', 
                   'left_shoulder_x', 'right_shoulder_x']].head(3))
    
    # Get and save metrics
    metrics_df = analyzer.getMetricsDataFrame()
    metrics_df.to_csv('data/metrics_001.csv', index=False)
    print(f"\nMetrics saved to: data/metrics_001.csv")
    
    # Visualize metrics
    analyzer.visualizeMetrics()
    
    print("\n✅ Ready for Step 2: Club Detection")


if __name__ == "__main__":
    main()