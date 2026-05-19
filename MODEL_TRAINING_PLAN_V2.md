# Golf Swing Neural Network Training Plan V2 (3 Weeks)

## Overview

Pivot from deterministic biomechanics scoring to **learned multi-output model** that:
1. Takes raw pose landmarks OR video frames
2. Outputs: 8 phase scores (0-100) + phase feedback text
3. Learns what actually discriminates pro/beginner from labeled expert data
4. Generalizes across camera angles (front/side/45°)

**Timeline:** 21 days  
**Dev effort:** ~40 hours  
**Expert effort:** ~20 hours (labeling)  
**Target:** ≥15pt discrimination between pro/beginner overall scores  
**Deliverable:** `models/swing_scorer.pth` (PyTorch) + `models/feedback_generator.pkl` (phase-level feedback)

---

## Why This Approach

**Dead ends with deterministic scoring:**
- Front-view MediaPipe can't measure 3D lag angle (critical for impact quality)
- Shoulder/hip rotation always ~175° (unreliable)
- Static positional angles (good address, top, finish) don't correlate with swing quality
- Result: Bad swings score 75-80 due to high static phase scores masking poor dynamics

**Neural network solution:**
- Learns non-linear combinations of landmarks that predict quality
- Can capture temporal patterns (velocity, acceleration, jerk)
- Expert labels provide ground truth for what "good" actually looks like
- Can output fine-grained feedback ("extend arm more", "shift weight earlier")

---

## Phase 1: Expert Labeling (Days 1–4)

### Step 1.1: Select Videos for Labeling

- **Total videos:** ~500 available
- **Label subset:** 150 swings (100 pro + 50 beginner/intermediate)
- **Pro sources:** GolfDB (~20), B80O + similar handicap 0-5 swings (~30), online samples (~50)
- **Beginner sources:** Your test videos (me, min2, golf_swing_001, etc.) + hand-collected videos (~50)

### Step 1.2: Design Expert Rating Form

Each expert rates **8 phase scores** (0-100) + root cause feedback:

```
Video: [swing_id] | Camera: [front/side/45°]
Expert: [name]     | Confidence: [1-5]

PHASE RATINGS (0-100 scale):
1. Address        [__]  | Issues if <70: _____________________________
2. Takeaway       [__]  | Issues if <70: _____________________________
3. Mid-backswing  [__]  | Issues if <70: _____________________________
4. Top            [__]  | Issues if <70: _____________________________
5. Mid-downswing  [__]  | Issues if <70: _____________________________
6. Impact         [__]  | Issues if <70: _____________________________
7. Follow-through [__]  | Issues if <70: _____________________________
8. Finish         [__]  | Issues if <70: _____________________________

OVERALL: [__]  (or auto-calculate as mean)

SWING QUALITY (select one):
[ ] Professional (handicap 0-5)
[ ] Advanced amateur (6-15)
[ ] Intermediate (16-25)
[ ] Beginner (26+)
```

**Process:**
1. Create video folders with swing IDs (B80O, min_indoor, me, etc.)
2. Send to golf pro/coach with form
3. Request completion in 1 week
4. Cross-check 20% overlap for inter-rater reliability (target: Pearson r > 0.85)

**Output:** `data/labels/expert_phase_ratings_150.csv`

---

## Phase 2: Prepare Training Dataset (Days 5–7)

### Step 2.1: Build Master CSV

```python
# prepare_training_data.py
import pandas as pd
import os

def build_training_dataset(expert_labels_csv, poses_dir, keyframes_dir, output_csv):
    """
    Combine expert labels + pose data + phase boundaries.
    One row per swing with 8 phase scores + metadata.
    """
    
    labels_df = pd.read_csv(expert_labels_csv)
    
    dataset = []
    for idx, row in labels_df.iterrows():
        swing_id = row['swing_id']
        
        # Find pose CSV
        pose_csv = os.path.join(poses_dir, f"{swing_id}_cleaned_poses.csv")
        if not os.path.exists(pose_csv):
            continue
        
        poses_df = pd.read_csv(pose_csv)
        
        # Find phase boundaries (from PoseSwingNet or manual annotation)
        phases_csv = os.path.join(keyframes_dir, f"{swing_id}_cleaned_8phases.csv")
        if not os.path.exists(phases_csv):
            continue
        
        phases_df = pd.read_csv(phases_csv)
        phases_dict = dict(zip(phases_df['Phase'], phases_df['Key_Frame']))
        
        # Build record
        record = {
            'swing_id': swing_id,
            'camera_angle': row.get('camera', 'front'),
            'skill_level': row.get('skill_level', 'unknown'),
            'num_frames': len(poses_df),
            'pose_csv': pose_csv,
            
            # 8 phase scores from expert (0-100)
            'address_score': row.get('address_score', None),
            'takeaway_score': row.get('takeaway_score', None),
            'mid_backswing_score': row.get('mid_backswing_score', None),
            'top_score': row.get('top_score', None),
            'mid_downswing_score': row.get('mid_downswing_score', None),
            'impact_score': row.get('impact_score', None),
            'follow_through_score': row.get('follow_through_score', None),
            'finish_score': row.get('finish_score', None),
            
            # Phase frame numbers
            'frame_address': phases_dict.get('Address', None),
            'frame_takeaway': phases_dict.get('Takeaway', None),
            'frame_mid_backswing': phases_dict.get('Mid-backswing', None),
            'frame_top': phases_dict.get('Top', None),
            'frame_mid_downswing': phases_dict.get('Mid-downswing', None),
            'frame_impact': phases_dict.get('Impact', None),
            'frame_follow_through': phases_dict.get('Follow-through', None),
            'frame_finish': phases_dict.get('Finish', None),
            
            # Feedback
            'feedback_text': row.get('feedback', ''),
            'expert_name': row.get('expert', ''),
            'confidence': row.get('confidence', 0.9),
        }
        dataset.append(record)
    
    df = pd.DataFrame(dataset)
    df.to_csv(output_csv, index=False)
    print(f"✓ Built training dataset: {len(df)} swings")
    print(f"  Skill distribution: {df['skill_level'].value_counts().to_dict()}")
    return df

if __name__ == '__main__':
    build_training_dataset(
        'data/labels/expert_phase_ratings_150.csv',
        'data/extracted_poses',
        'data/keyframes',
        'datasets/training_150_phases.csv'
    )
```

**Output:** `datasets/training_150_phases.csv`

---

## Phase 3: Extract Pose Features or Video Frames (Days 8–10)

### Option A: Pose-based model (faster, lower memory)

Extract per-frame landmarks as features:

```python
# extract_pose_features.py
import pandas as pd
import numpy as np

def extract_pose_sequences(training_csv, poses_dir, output_features_dir):
    """
    For each training video, extract landmark sequences aligned to phase keyframes.
    Output: npy files for model input.
    """
    
    df = pd.read_csv(training_csv)
    
    os.makedirs(output_features_dir, exist_ok=True)
    
    for idx, row in df.iterrows():
        swing_id = row['swing_id']
        poses_df = pd.read_csv(row['pose_csv'])
        
        # Extract 33 landmarks per frame (x, y, z, visibility)
        # Shape: (num_frames, 33, 4)
        landmarks = []
        for frame_idx, frame_row in poses_df.iterrows():
            frame_lms = []
            for i in range(33):  # 33 MediaPipe landmarks
                x = frame_row.get(f'landmark_{i}_x', 0)
                y = frame_row.get(f'landmark_{i}_y', 0)
                z = frame_row.get(f'landmark_{i}_z', 0)
                v = frame_row.get(f'landmark_{i}_visibility', 0)
                frame_lms.append([x, y, z, v])
            landmarks.append(frame_lms)
        
        landmarks = np.array(landmarks)
        
        # Save
        output_path = os.path.join(output_features_dir, f"{swing_id}_landmarks.npy")
        np.save(output_path, landmarks)
        
        if (idx + 1) % 20 == 0:
            print(f"Extracted {idx + 1}/{len(df)}")
    
    print(f"✓ Extracted pose sequences for {len(df)} swings")
```

### Option B: Video frame model (more powerful, higher memory)

Extract CNN features from keyframes:

```python
# extract_video_features.py
import torch
import torchvision.models as models
from torchvision import transforms
import os
import numpy as np
from PIL import Image

def extract_cnn_features(training_csv, keyframes_dir, output_dir):
    """
    Extract ResNet50 features for each of 8 keyframes per swing.
    Output: npy file with shape (8, 2048) for each swing.
    """
    
    model = models.resnet50(pretrained=True)
    model.eval()
    
    # Remove classification head, keep features
    features_layer = torch.nn.Sequential(*list(model.children())[:-1])
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    df = pd.read_csv(training_csv)
    
    for idx, row in df.iterrows():
        swing_id = row['swing_id']
        keyframe_folder = os.path.join(keyframes_dir, f"{swing_id}_nn")
        
        if not os.path.exists(keyframe_folder):
            continue
        
        features_list = []
        for phase_name in ['address', 'takeaway', 'mid_backswing', 'top', 
                          'mid_downswing', 'impact', 'follow_through', 'finish']:
            keyframe_path = os.path.join(keyframe_folder, f"keyframe_{phase_name}.jpg")
            
            if os.path.exists(keyframe_path):
                img = Image.open(keyframe_path)
                img_tensor = transform(img).unsqueeze(0)
                
                with torch.no_grad():
                    features = features_layer(img_tensor).squeeze().numpy()
                features_list.append(features)
            else:
                features_list.append(np.zeros(2048))
        
        # Stack: shape (8, 2048)
        features_array = np.array(features_list)
        
        output_path = os.path.join(output_dir, f"{swing_id}_cnn_features.npy")
        np.save(output_path, features_array)
        
        if (idx + 1) % 20 == 0:
            print(f"Extracted CNN features {idx + 1}/{len(df)}")
    
    print(f"✓ Extracted CNN features for {len(df)} swings")
```

**Recommendation:** Start with **Option A (Pose-based)** for faster iteration. Switch to Option B later if accuracy plateaus.

---

## Phase 4: Train Multi-Output Phase Scorer (Days 11–16)

### Step 4.1: Define Model Architecture

```python
# models/phase_scorer_nn.py
import torch
import torch.nn as nn

class PhaseScorer(nn.Module):
    """
    Multi-output regression model:
    - Input: Pose sequences (num_frames, 33, 4) or CNN features (8, 2048)
    - Output: 8 phase scores (0-100) + phase feedback
    """
    
    def __init__(self, input_type='pose', num_frames=150):
        super().__init__()
        
        self.input_type = input_type
        
        if input_type == 'pose':
            # LSTM to process frame sequences
            self.pose_encoder = nn.LSTM(
                input_size=33*4,  # 33 landmarks × 4 values (x,y,z,v)
                hidden_size=256,
                num_layers=2,
                batch_first=True,
                dropout=0.2
            )
            encoder_output_size = 256
            
        elif input_type == 'cnn':
            # CNN features already extracted, just use MLP
            self.cnn_encoder = nn.Sequential(
                nn.Linear(8 * 2048, 512),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(512, 256),
                nn.ReLU(),
            )
            encoder_output_size = 256
        
        # Shared layers for phase classification
        self.shared_layers = nn.Sequential(
            nn.Linear(encoder_output_size, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
        )
        
        # 8 phase score heads (regression, 0-100)
        self.phase_heads = nn.ModuleDict({
            'address': nn.Linear(256, 1),
            'takeaway': nn.Linear(256, 1),
            'mid_backswing': nn.Linear(256, 1),
            'top': nn.Linear(256, 1),
            'mid_downswing': nn.Linear(256, 1),
            'impact': nn.Linear(256, 1),
            'follow_through': nn.Linear(256, 1),
            'finish': nn.Linear(256, 1),
        })
        
        # 8 feedback generation heads (multi-class: 10 common issues per phase)
        self.feedback_heads = nn.ModuleDict({
            phase: nn.Linear(256, 10)  # 10 feedback classes per phase
            for phase in ['address', 'takeaway', 'mid_backswing', 'top',
                         'mid_downswing', 'impact', 'follow_through', 'finish']
        })
    
    def forward(self, x):
        """
        Args:
            x: Pose sequences (batch_size, num_frames, 33*4) or CNN features (batch_size, 8*2048)
        
        Returns:
            phase_scores: dict of phase -> (batch_size, 1) regression output
            feedback_logits: dict of phase -> (batch_size, 10) classification output
        """
        
        # Encode input
        if self.input_type == 'pose':
            x_flat = x.view(x.size(0), x.size(1), -1)  # (batch, frames, 132)
            lstm_out, (h_n, c_n) = self.pose_encoder(x_flat)
            # Take last hidden state
            encoded = h_n[-1]  # (batch, 256)
            
        elif self.input_type == 'cnn':
            encoded = self.cnn_encoder(x)  # (batch, 256)
        
        # Shared representation
        shared_rep = self.shared_layers(encoded)  # (batch, 256)
        
        # Predict 8 phase scores
        phase_scores = {}
        for phase_name in self.phase_heads.keys():
            raw_score = self.phase_heads[phase_name](shared_rep)  # (batch, 1)
            phase_scores[phase_name] = torch.clamp(raw_score, 0, 100)  # Clamp to [0, 100]
        
        # Predict 8 feedback categories
        feedback_logits = {}
        for phase_name in self.feedback_heads.keys():
            feedback_logits[phase_name] = self.feedback_heads[phase_name](shared_rep)  # (batch, 10)
        
        return phase_scores, feedback_logits
```

### Step 4.2: Training Loop

```python
# train_phase_scorer.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from models.phase_scorer_nn import PhaseScorer

class SwingDataset(Dataset):
    def __init__(self, training_csv, features_dir, input_type='pose'):
        self.df = pd.read_csv(training_csv)
        self.features_dir = features_dir
        self.input_type = input_type
        
        # Phase order for indexing
        self.phases = ['address', 'takeaway', 'mid_backswing', 'top', 
                      'mid_downswing', 'impact', 'follow_through', 'finish']
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        swing_id = row['swing_id']
        
        # Load features
        if self.input_type == 'pose':
            features_path = os.path.join(self.features_dir, f"{swing_id}_landmarks.npy")
            features = np.load(features_path)  # (num_frames, 33, 4)
            
        elif self.input_type == 'cnn':
            features_path = os.path.join(self.features_dir, f"{swing_id}_cnn_features.npy")
            features = np.load(features_path)  # (8, 2048)
            features = features.flatten()  # (8*2048,)
        
        # Load phase scores
        phase_scores = torch.tensor([
            row[f'{phase}_score'] for phase in self.phases
        ], dtype=torch.float32)
        
        # Load feedback (TODO: map text to feedback class indices)
        # For now, dummy feedback
        feedback_indices = torch.zeros(8, dtype=torch.long)
        
        return torch.tensor(features, dtype=torch.float32), phase_scores, feedback_indices

def train_model(training_csv, features_dir, output_model_path, epochs=50):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Dataset
    dataset = SwingDataset(training_csv, features_dir, input_type='pose')
    loader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    # Model
    model = PhaseScorer(input_type='pose').to(device)
    
    # Loss functions
    score_loss_fn = nn.MSELoss()  # Regression for phase scores
    feedback_loss_fn = nn.CrossEntropyLoss()  # Classification for feedback
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for features, phase_scores, feedback_indices in loader:
            features = features.to(device)
            phase_scores = phase_scores.to(device)
            feedback_indices = feedback_indices.to(device)
            
            # Forward pass
            pred_scores, pred_feedback = model(features)
            
            # Loss
            score_loss = sum(
                score_loss_fn(pred_scores[phase].squeeze(), phase_scores[:, i])
                for i, phase in enumerate(dataset.phases)
            ) / len(dataset.phases)
            
            # Feedback loss (optional, for now skip)
            total_loss = score_loss
            
            # Backward
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            total_loss += total_loss.item()
        
        scheduler.step()
        
        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")
        
        # Checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f"models/phase_scorer_epoch_{epoch+1}.pth")
    
    # Save final model
    torch.save(model.state_dict(), output_model_path)
    print(f"✓ Model saved: {output_model_path}")
    
    return model

if __name__ == '__main__':
    train_model(
        'datasets/training_150_phases.csv',
        'datasets/pose_features',
        'models/swing_scorer.pth'
    )
```

---

## Phase 5: Evaluate & Deploy (Days 17–21)

### Step 5.1: Cross-validation

```python
# evaluate_model.py
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

def evaluate_model(model, test_loader, phases):
    """Compute MAE per phase on test set."""
    
    model.eval()
    phase_mae = {phase: [] for phase in phases}
    
    with torch.no_grad():
        for features, phase_scores, _ in test_loader:
            pred_scores, _ = model(features)
            
            for i, phase in enumerate(phases):
                pred = pred_scores[phase].squeeze().cpu().numpy()
                true = phase_scores[:, i].cpu().numpy()
                mae = mean_absolute_error(true, pred)
                phase_mae[phase].append(mae)
    
    # Summary
    print("Phase-wise Mean Absolute Error:")
    for phase in phases:
        avg_mae = np.mean(phase_mae[phase])
        print(f"  {phase}: {avg_mae:.2f} points")
    
    overall_mae = np.mean([np.mean(phase_mae[p]) for p in phases])
    print(f"\nOverall MAE: {overall_mae:.2f} points")
    
    return phase_mae
```

### Step 5.2: Integration with Pipeline

Update `pipeline.py` to use neural network scorer:

```python
# In pipeline.py
from models.phase_scorer_nn import PhaseScorer

# ... after phase detection ...

# Load trained model
scorer_model = PhaseScorer(input_type='pose')
scorer_model.load_state_dict(torch.load('models/swing_scorer.pth'))
scorer_model.eval()

# Get pose features
pose_features = extract_pose_features(df)  # Shape: (num_frames, 33*4)

# Predict phase scores
with torch.no_grad():
    features_tensor = torch.tensor(pose_features, dtype=torch.float32).unsqueeze(0)
    phase_scores_dict, feedback_logits = scorer_model(features_tensor)

# Convert to output format
phase_scores = {
    phase: score.item() for phase, score in phase_scores_dict.items()
}

overall_score = np.mean(list(phase_scores.values()))

print(f"Phase scores: {phase_scores}")
print(f"Overall: {overall_score:.1f}")
```

### Step 5.3: Feedback Generation

Map feedback logits to text:

```python
FEEDBACK_TEMPLATES = {
    'address': {
        0: 'Good posture at address',
        1: 'Narrow stance - widen for stability',
        2: 'Bend knees more',
        # ... 10 categories
    },
    'impact': {
        0: 'Good lag at impact',
        1: 'Release lag early - delay wrist break',
        2: 'Extend arm more at impact',
        # ... 10 categories
    },
    # ... for each phase
}

def generate_feedback(feedback_logits, phases):
    """Convert logits to feedback text."""
    feedback = {}
    for phase in phases:
        pred_class = feedback_logits[phase].argmax(dim=1).item()
        feedback[phase] = FEEDBACK_TEMPLATES[phase].get(pred_class, "Keep improving")
    return feedback
```

---

## Success Criteria

- [ ] 150 labeled swings collected with >85% inter-rater agreement
- [ ] MAE per phase < 8 points (e.g., predict 72 when actual is 75)
- [ ] Pro vs Beginner overall scores differ by ≥15 points
- [ ] Model generalizes to unseen videos from same camera angles
- [ ] Feedback aligns with human expert feedback

---

## Notes

**Camera angle handling:** If videos are mixed (front/side/45°), add `camera_angle` as an input feature to model to condition phase scoring on viewpoint.

**Data augmentation:** Pose sequences can be augmented:
- Temporal jittering (add noise to frames)
- Scaling (different heights/positions)
- Temporal subsampling

**Future improvements:**
- Multi-angle model trained on side + front views together
- Club position tracking (if available)
- Real-time inference on video stream
