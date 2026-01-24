# Manual Annotation & Training Guide

## Step 1: Manual Annotation

### What You Need to Do

For each of your 11 cleaned videos, create a CSV file with frame-level labels.

### Format Required

**File name:** `golf_swing_001_labels.csv`

**Content:**
```csv
frame,phase_id
0,0
1,0
2,0
...
25,0
26,1
27,1
...
```

Where:
- `frame`: Frame number (0 to video_length)
- `phase_id`: Phase number (0-7)

**Phase ID Mapping:**
```
0 = Address
1 = Takeaway
2 = Mid-backswing
3 = Top
4 = Mid-downswing
5 = Impact
6 = Follow-through
7 = Finish
```

### Annotation Method

#### Option A: Manual Video Watching (Simplest)
1. Watch golf_swing_001_cleaned.mp4 frame by frame
2. Write down frame numbers where each phase starts/ends
3. Example:
   ```
   Address: frames 0-25
   Takeaway: frames 26-50
   Mid-backswing: frames 51-100
   Top: frames 101-120
   Mid-downswing: frames 121-150
   Impact: frames 151-165
   Follow-through: frames 166-190
   Finish: frames 191-297
   ```
4. Create CSV using these ranges

#### Option B: Use Your EightPhaseDetector Output
If you already ran EightPhaseDetector:
1. You have phase CSVs with start/end frames
2. Use `generate_training_labels.py` to convert automatically
3. (See Step 2 below)

#### Option C: Visual Annotation Tool
Use this Python script to help annotate:

```python
# simple_annotator.py - View video and click phase changes
import cv2
import sys

video_path = sys.argv[1]
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

boundaries = []

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        # User clicked - mark phase boundary at current frame
        current_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
        boundaries.append(int(current_frame))
        print(f"Boundary marked at frame {int(current_frame)}")

cv2.namedWindow('Video')
cv2.setMouseCallback('Video', mouse_callback)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.resize(frame, (800, 600))
    
    # Draw current frame number
    frame_num = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
    cv2.putText(frame, f'Frame: {frame_num}', (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, 'Click to mark phase boundary, Press Q to quit', 
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)
    
    cv2.imshow('Video', frame)
    
    key = cv2.waitKey(33) & 0xFF
    if key == ord('q'):
        break
    elif key == ord(' '):
        # Pause/play
        pass

cap.release()
cv2.destroyAllWindows()

print("\nBoundaries marked at frames:", boundaries)
# Manually assign phase IDs to each segment
```

---

## Step 2: Prepare Training Data

### If Using EightPhaseDetector Output:

```bash
# Generate labels from phase CSVs
python scripts/generate_training_labels.py
```

This will:
- Read your cleaned pose CSVs
- Read your EightPhaseDetector phase CSVs
- Create frame-level label CSVs
- Save to `data/training_data/`

### If Using Manual Annotations:

1. Save your label CSVs to `data/training_data/`:
   ```
   data/training_data/
   ├── golf_swing_001_poses.csv       (copy from extracted_poses)
   ├── golf_swing_001_labels.csv      (your manual annotations)
   ├── golf_swing_002_poses.csv
   ├── golf_swing_002_labels.csv
   ├── ...
   ```

2. Copy pose CSVs to training_data folder:
   ```bash
   cp data/extracted_poses/*_cleaned_poses.csv data/training_data/
   ```

3. Rename them:
   ```bash
   # On Windows PowerShell:
   Get-ChildItem data/training_data/*_cleaned_poses.csv | 
   Rename-Item -NewName { $_.Name -replace '_cleaned_poses', '_poses' }
   ```

---

## Step 3: Train the Model

### Run Training:

```bash
# Basic training (50 epochs, batch size 16)
python scripts/train_swingnet.py

# With custom parameters:
python scripts/train_swingnet.py --epochs 100 --batch_size 8 --lr 0.0005
```

### Command Line Options:

```
--epochs          Number of training epochs (default: 50)
--batch_size      Examples per batch (default: 16)
--lr              Learning rate (default: 0.001)
--hidden_size     LSTM hidden size (default: 128)
--num_layers      Number of LSTM layers (default: 2)
--seq_length      Sequence length (default: 30)
--patience        Early stopping patience (default: 10)
```

### What to Expect:

```
Epoch [1/50]   Train Loss: 2.1234 Acc: 0.1523 | Val Loss: 2.0834 Acc: 0.1687
Epoch [2/50]   Train Loss: 1.8456 Acc: 0.3241 | Val Loss: 1.7923 Acc: 0.3562
...
Epoch [50/50]  Train Loss: 0.0845 Acc: 0.9612 | Val Loss: 0.3245 Acc: 0.8723
  → Saved best model (val_acc: 0.8723)

✅ Training Complete!
   - models/best_phase_model.pth (saved automatically)
   - training_history.png (loss & accuracy curves)
```

### Training Time:

- **CPU:** 30-60 minutes for 50 epochs
- **GPU (CUDA):** 10-15 minutes for 50 epochs

If training on CPU seems slow:
```bash
# Skip some epochs and save partial models
python scripts/train_swingnet.py --epochs 20  # Quick test
```

---

## Step 4: Verify Results

### Check Training Curves:

Open `training_history.png` - should see:
- Loss decreasing over time
- Accuracy increasing over time
- Validation loss similar to train loss (not overfitting)

### Test on New Video:

```python
# inference_example.py
import torch
import pandas as pd
from src.ExtractKeyFrames import PoseSwingNet
from scripts.train_swingnet import create_sequences

# Load trained model
model = PoseSwingNet(input_size=132, hidden_size=128, num_layers=2, num_classes=8)
model.load_state_dict(torch.load('models/best_phase_model.pth'))
model.eval()

# Load test poses
test_poses = pd.read_csv('data/extracted_poses/golf_swing_010_cleaned_poses.csv')
features = [col for col in test_poses.columns if col != 'frame']
poses = test_poses[features].values

# Normalize using training stats (or fit new stats)
poses = (poses - poses.mean(axis=0)) / (poses.std(axis=0) + 1e-8)

# Create sequences
sequences, _ = create_sequences(poses)

# Predict
with torch.no_grad():
    predictions = model(torch.FloatTensor(sequences))
    phases = predictions.argmax(dim=-1)

print("Predicted phases:", phases)
```

---

## Step 5: Use Model in ExtractKeyFrames

Update `src/ExtractKeyFrames.py` to use trained model:

```python
# Line 407 in extract_key_phase_frames()
model_path = 'models/best_phase_model.pth'  # Add this line

model = PoseSwingNet(input_size=132, hidden_size=128, num_layers=2, num_classes=8)

if model_path and os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    print(f"✓ Loaded trained model from: {model_path}")
else:
    print("⚠ No model found")
```

Then run:
```bash
python src/ExtractKeyFrames.py
```

---

## Troubleshooting

### Problem: "CSV not found"
**Solution:** Make sure your label CSVs are in `data/training_data/` with format:
```
golf_swing_001_poses.csv
golf_swing_001_labels.csv
```

### Problem: "Only 10% frames labeled"
**Solution:** Your annotations are incomplete. Check:
- Are all frames 0 to num_frames labeled?
- No frames should have phase_id = -1
- Every row needs a label

### Problem: "Training stuck (loss not decreasing)"
**Solution:** Try:
- Reduce learning rate: `--lr 0.0001`
- Increase batch size: `--batch_size 32`
- Check annotation quality (bad labels = model can't learn)

### Problem: "High training loss, low validation loss"
**Solution:** Model is overfitting. Try:
- Reduce model size: `--hidden_size 64`
- Increase dropout in code
- Get more training data
- Reduce epochs: `--epochs 20`

### Problem: "Training too slow on CPU"
**Solution:**
- Use smaller subset: annotate only 5 videos first
- Reduce sequence length: `--seq_length 15`
- Skip validation: modify code to validate every 10 epochs instead of 1

---

## Summary

1. **Annotate:** Create label CSVs for each video (30 mins - 2 hours total)
2. **Organize:** Put CSVs in `data/training_data/`
3. **Train:** Run `python scripts/train_swingnet.py` (30-60 mins)
4. **Evaluate:** Check `training_history.png`
5. **Use:** Update ExtractKeyFrames.py to use trained model
6. **Test:** Run on new videos

---

## Recommended Annotation Order

Start with 5 videos for quick training:
1. golf_swing_001 - Easy, clear swing
2. golf_swing_005 - Different angle
3. golf_swing_007 - Faster swing
4. golf_swing_009 - Variable lighting
5. golf_swing_011 - Challenging view

Then add more videos (golf_swing_002, 003, 004, 006, 008, 010) for better accuracy.
