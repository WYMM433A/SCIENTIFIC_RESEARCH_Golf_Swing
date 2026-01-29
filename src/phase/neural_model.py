"""
PoseSwingNet - Neural Network Model for Golf Swing Phase Detection
===================================================================
Bi-LSTM model that predicts golf swing phases from pose sequences.

This module contains:
- PoseSwingNet: The neural network model architecture
- Utility functions for loading and preprocessing pose data

For training, see: tools/train_with_golfdb.py
For inference, use: src/phase/adapter.py with create_predictor('neural-network', model_path)
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd

from ..constants import PHASE_NAMES, NUM_FEATURES


class PoseSwingNet(nn.Module):
    """
    Bidirectional LSTM model for golf swing phase detection.
    
    Architecture:
        Input (132 features) → Linear Projection → Bi-LSTM (2 layers) → Classifier → Output (9 classes)
    
    The model processes pose sequences and outputs per-frame phase probabilities.
    9 classes = 8 swing phases + 1 no-event class.
    """
    
    def __init__(self, input_size=132, hidden_size=128, num_layers=2, num_classes=9):
        """
        Initialize PoseSwingNet.
        
        Args:
            input_size: Number of input features per frame (default: 132 = 33 landmarks × 4)
            hidden_size: LSTM hidden dimension (default: 128)
            num_layers: Number of stacked LSTM layers (default: 2)
            num_classes: Number of output classes (default: 9 = 8 phases + no-event)
        """
        super(PoseSwingNet, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        
        # Input projection: expand features then compress
        self.input_projection = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, hidden_size),
            nn.ReLU()
        )
        
        # Bidirectional LSTM: processes sequence in both directions
        self.lstm = nn.LSTM(
            hidden_size, 
            hidden_size, 
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.2 if num_layers > 1 else 0
        )
        
        # Classifier: maps LSTM output to class probabilities
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),  # *2 for bidirectional
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, seq_len, input_size)
            
        Returns:
            logits: Output tensor of shape (batch, seq_len, num_classes)
        """
        # Project input features
        x = self.input_projection(x)
        
        # Process with Bi-LSTM
        lstm_out, _ = self.lstm(x)
        
        # Classify each frame
        logits = self.classifier(lstm_out)
        
        return logits


# ============================================
# DATA UTILITIES
# ============================================

def load_pose_csv(csv_path, normalize=True):
    """
    Load pose CSV file and prepare features for model input.
    
    Args:
        csv_path: Path to pose CSV file
        normalize: Whether to normalize features (default: True)
        
    Returns:
        poses: numpy array of shape (num_frames, 132)
        frames: array of frame numbers
    """
    df = pd.read_csv(csv_path)
    
    # Extract feature columns (all except 'frame')
    feature_cols = [col for col in df.columns if col != 'frame']
    poses = df[feature_cols].values
    frames = df['frame'].values
    
    if normalize:
        # Z-score normalization
        poses = (poses - poses.mean(axis=0)) / (poses.std(axis=0) + 1e-8)
    
    return poses, frames


def create_sequences(poses, sequence_length=64, stride=1):
    """
    Create overlapping sequences for training/inference.
    
    Args:
        poses: numpy array of shape (num_frames, features)
        sequence_length: Length of each sequence
        stride: Step size between sequences
        
    Returns:
        sequences: numpy array of shape (num_sequences, sequence_length, features)
        center_frames: Center frame index for each sequence
    """
    sequences = []
    center_frames = []
    
    for i in range(0, len(poses) - sequence_length + 1, stride):
        seq = poses[i:i + sequence_length]
        sequences.append(seq)
        center_frames.append(i + sequence_length // 2)
    
    return np.array(sequences), center_frames


def predict_phases(model, poses, device='cpu'):
    """
    Predict phase probabilities for a pose sequence.
    
    Args:
        model: Trained PoseSwingNet model
        poses: numpy array of shape (num_frames, 132)
        device: 'cpu' or 'cuda'
        
    Returns:
        probs: numpy array of shape (num_frames, num_classes) - softmax probabilities
        predictions: numpy array of shape (num_frames,) - predicted class per frame
    """
    model.eval()
    model.to(device)
    
    # Prepare input
    x = torch.FloatTensor(poses).unsqueeze(0).to(device)  # (1, seq_len, features)
    
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
        predictions = np.argmax(probs, axis=-1)
    
    return probs, predictions
