"""
Unified Interface for Phase Detection
======================================
Supports multiple detection methods with factory pattern for easy switching:

1. rule-based: 8-phase detector using wrist trajectory (current default)
2. neural-network: LSTM-based detector (requires trained model)

This adapter allows the pipeline to easily switch between methods
when a trained neural network model becomes available.

Usage:
    # Rule-based (current)
    predictor = create_predictor('rule-based')
    results = predictor.process(csv_path, video_path, output_dir)
    
    # Neural network (future)
    predictor = create_predictor('neural-network', model_path='models/phase_model.pth')
    results = predictor.process(csv_path, video_path, output_dir)
"""

import os
import numpy as np
from .rule_based import EightPhaseDetector
from ..constants import PHASE_NAMES


class PhasePredictor:
    """
    Unified wrapper for different phase detection methods.
    
    Provides consistent interface regardless of underlying detection method,
    making it easy to swap between rule-based and neural network approaches.
    """
    
    PHASE_NAMES = PHASE_NAMES  # Use shared constant
    
    def __init__(self, model_type='rule-based', model_path=None):
        """
        Initialize phase predictor.
        
        Args:
            model_type: 'rule-based' or 'neural-network'
            model_path: Path to trained model (required for neural-network)
        """
        self.model_type = model_type
        self.model_path = model_path
        self.model = None
        
        if model_type == 'rule-based':
            print("✓ Initialized Rule-Based 8-Phase Detector (wrist trajectory)")
        elif model_type == 'neural-network':
            self._load_neural_network(model_path)
        else:
            raise ValueError(f"Unknown model type: {model_type}. Use 'rule-based' or 'neural-network'")
    
    def _load_neural_network(self, model_path):
        """Load neural network model for phase detection."""
        if model_path is None:
            raise ValueError("model_path is required for neural-network predictor")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        try:
            import torch
            from .neural_model import PoseSwingNet
            
            # Load model (9 classes = 8 phases + no-event)
            self.model = PoseSwingNet(input_size=132, hidden_size=128, num_layers=2, num_classes=9)
            self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
            self.model.eval()
            print(f"✓ Loaded Neural Network Model: {model_path}")
        except ImportError:
            raise ImportError("PyTorch is required for neural-network predictor")
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")
    
    def process(self, csv_path: str, video_path: str, output_dir: str = 'phase_frames') -> dict:
        """
        Process a single video and extract 8 key frames.
        
        Args:
            csv_path: Path to pose CSV file
            video_path: Path to video file  
            output_dir: Directory for output frames
            
        Returns:
            dict: Results containing phase_ranges, keyframes, and csv_path
        """
        if self.model_type == 'rule-based':
            return self._process_rule_based(csv_path, video_path, output_dir)
        elif self.model_type == 'neural-network':
            return self._process_neural_network(csv_path, video_path, output_dir)
    
    def _process_rule_based(self, csv_path, video_path, output_dir):
        """Process using rule-based wrist trajectory detector."""
        detector = EightPhaseDetector(csv_path, video_path, output_dir)
        
        # Detect phases
        phase_ranges = detector.detect_phases()
        
        # Extract key frames
        keyframes = detector.extract_8_frames()
        
        # Save CSV
        phases_csv = detector.save_phase_csv()
        
        return {
            'method': 'rule-based',
            'phase_ranges': phase_ranges,
            'keyframes': keyframes,
            'phases_csv': phases_csv
        }
    
    def _process_neural_network(self, csv_path, video_path, output_dir):
        """Process using neural network model."""
        import torch
        import pandas as pd
        import cv2
        from pathlib import Path
        
        # Load and prepare data
        df = pd.read_csv(csv_path)
        feature_cols = [col for col in df.columns if col != 'frame']
        poses = df[feature_cols].values
        frames = df['frame'].values
        
        # Normalize
        poses = (poses - poses.mean(axis=0)) / (poses.std(axis=0) + 1e-8)
        
        # Convert to tensor
        x = torch.FloatTensor(poses).unsqueeze(0)  # (1, seq_len, features)
        
        # Predict
        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=-1).squeeze().numpy()  # (seq_len, 9)
            predictions = np.argmax(probs, axis=-1)
        
        num_frames = len(frames)
        
        # Find phase boundaries using confidence-based peak detection
        phase_ranges = self._find_phase_peaks(probs, num_frames)
        
        # Extract key frames
        keyframes = self._extract_keyframes_nn(
            video_path, phase_ranges, output_dir, 
            Path(csv_path).stem.replace('_poses', '')
        )
        
        # Save CSV
        phases_csv = self._save_phases_csv_nn(
            phase_ranges, keyframes, output_dir,
            Path(csv_path).stem.replace('_poses', '')
        )
        
        return {
            'method': 'neural-network',
            'phase_ranges': phase_ranges,
            'keyframes': keyframes,
            'phases_csv': phases_csv,
            'predictions': predictions
        }
    
    def _find_phase_boundaries(self, predictions):
        """Find phase boundaries from frame-level predictions (legacy method)."""
        phase_ranges = {}
        
        for phase_id, phase_name in enumerate(self.PHASE_NAMES):
            phase_frames = np.where(predictions == phase_id)[0]
            
            if len(phase_frames) > 0:
                phase_ranges[phase_name] = (int(phase_frames[0]), int(phase_frames[-1]))
            else:
                phase_ranges[phase_name] = (0, 0)
        
        return phase_ranges
    
    def _find_phase_peaks(self, probs, num_frames):
        """
        Find phase key frames using confidence-based peak detection.
        Instead of looking for frames predicted as each class, find the 
        frame with highest confidence for each phase class.
        
        Args:
            probs: (num_frames, 9) softmax probabilities
            num_frames: total number of frames
            
        Returns:
            dict: phase_name -> (start, end) range around peak frame
        """
        phase_ranges = {}
        
        # Get confidence for each phase (excluding no-event class 8)
        for phase_id, phase_name in enumerate(self.PHASE_NAMES):
            phase_probs = probs[:, phase_id]  # Confidence for this phase
            
            # Find the frame with maximum confidence for this phase
            peak_frame = int(np.argmax(phase_probs))
            
            # Define a small window around the peak
            window = 3
            start = max(0, peak_frame - window)
            end = min(num_frames - 1, peak_frame + window)
            
            phase_ranges[phase_name] = (start, end)
        
        # Enforce temporal ordering: phases should occur in sequence
        phase_ranges = self._enforce_temporal_order(phase_ranges, num_frames)
        
        return phase_ranges
    
    def _enforce_temporal_order(self, phase_ranges, num_frames):
        """
        Ensure phases occur in temporal order (Address < Takeaway < ... < Finish).
        If phases are out of order, redistribute them.
        """
        # Get key frames (midpoints)
        key_frames = []
        for phase_name in self.PHASE_NAMES:
            start, end = phase_ranges[phase_name]
            key_frames.append((start + end) // 2)
        
        # Check if already in order
        is_ordered = all(key_frames[i] <= key_frames[i+1] for i in range(len(key_frames)-1))
        
        if is_ordered:
            return phase_ranges
        
        # Sort and redistribute
        sorted_frames = sorted(key_frames)
        
        # If still problematic, distribute evenly across video
        if sorted_frames[0] == sorted_frames[-1]:
            # All same frame - distribute evenly
            step = num_frames // 9
            sorted_frames = [i * step for i in range(8)]
        
        # Rebuild phase_ranges with sorted frames
        new_ranges = {}
        window = 3
        for i, phase_name in enumerate(self.PHASE_NAMES):
            frame = sorted_frames[i]
            start = max(0, frame - window)
            end = min(num_frames - 1, frame + window)
            new_ranges[phase_name] = (start, end)
        
        return new_ranges
    
    def _extract_keyframes_nn(self, video_path, phase_ranges, output_dir, video_name):
        """Extract key frames based on neural network predictions."""
        import cv2
        import os
        
        os.makedirs(output_dir, exist_ok=True)
        
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        keyframes = {}
        
        for phase_name in self.PHASE_NAMES:
            if phase_name not in phase_ranges:
                continue
            
            start, end = phase_ranges[phase_name]
            frame_idx = int((start + end) / 2)
            frame_idx = max(0, min(frame_idx, total_frames - 1))
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if ret:
                output_path = os.path.join(output_dir, f'{video_name}_{phase_name}.jpg')
                cv2.imwrite(output_path, frame)
                keyframes[phase_name] = {
                    'phase': phase_name,
                    'start_frame': start,
                    'end_frame': end,
                    'key_frame': frame_idx,
                    'image_path': output_path
                }
        
        cap.release()
        return keyframes
    
    def _save_phases_csv_nn(self, phase_ranges, keyframes, output_dir, video_name):
        """Save phase information to CSV."""
        import pandas as pd
        import os
        
        csv_path = os.path.join(output_dir, f'{video_name}_8phases.csv')
        
        rows = []
        for phase_name in self.PHASE_NAMES:
            if phase_name in keyframes:
                data = keyframes[phase_name]
                rows.append({
                    'Video': video_name,
                    'Phase': data['phase'],
                    'Start_Frame': data['start_frame'],
                    'End_Frame': data['end_frame'],
                    'Duration': data['end_frame'] - data['start_frame'] + 1,
                    'Key_Frame': data['key_frame'],
                    'Image_Path': data['image_path']
                })
        
        df = pd.DataFrame(rows)
        df.to_csv(csv_path, index=False)
        
        return csv_path


def create_predictor(model_type='rule-based', model_path=None):
    """
    Factory function to create appropriate phase predictor.
    
    Args:
        model_type: 'rule-based' or 'neural-network'
        model_path: Path to trained model (for neural-network only)
        
    Returns:
        PhasePredictor instance
        
    Usage:
        # Current: Rule-based detector (no model required)
        predictor = create_predictor('rule-based')
        results = predictor.process('poses.csv', 'video.mp4', 'output/')
        
        # Future: Neural network (requires trained model)
        predictor = create_predictor('neural-network', 'models/phase_model.pth')
        results = predictor.process('poses.csv', 'video.mp4', 'output/')
    """
    return PhasePredictor(model_type=model_type, model_path=model_path)
