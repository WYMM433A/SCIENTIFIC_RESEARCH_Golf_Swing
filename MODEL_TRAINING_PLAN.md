# Golf Swing Model Training Plan (2 Weeks)

## Overview

This plan builds a working XGBoost classifier that separates beginner from pro golfers, identifies which biomechanics features actually discriminate, and generates phase-based coaching feedback.

**Timeline:** 14 days  
**Dev effort:** ~28 hours  
**Expert effort:** ~16 hours (labeling)  
**Target accuracy:** ≥80% beginner vs pro classification  
**Output:** Trained model + per-phase feedback system

---

## Phase 0: Prerequisites Checklist

Before starting:
- [ ] 500 videos extracted (raw_videos/)
- [ ] 500 pose CSVs extracted (extracted_poses/)
- [ ] PoseSwingNet phase boundaries available (phase start/end frames per video)
- [ ] Pro golfer or coach available for 16 hours of labeling
- [ ] Python environment with: pandas, xgboost, scikit-learn, shap, matplotlib

---

## Phase 1: Label Collection (Days 1–2)

### Goal
Create expert annotations for 100 swings (50 pro, 50 beginner).

### Step 1.1: Design Label Form

Create a Google Form or CSV template. Each swing takes ~10 minutes to label.

**Form structure:**
```
Swing ID: [auto-filled]
Video file: [link]
View: [side / front]

=== PHASE RATINGS (select one per phase) ===
Address:       [ ] Excellent  [ ] Good  [ ] Fair  [ ] Poor  | If Poor, issue: ________
Takeaway:      [ ] Excellent  [ ] Good  [ ] Fair  [ ] Poor  | If Poor, issue: ________
Mid-backswing: [ ] Excellent  [ ] Good  [ ] Fair  [ ] Poor  | If Poor, issue: ________
Top:           [ ] Excellent  [ ] Good  [ ] Fair  [ ] Poor  | If Poor, issue: ________
Mid-downswing: [ ] Excellent  [ ] Good  [ ] Fair  [ ] Poor  | If Poor, issue: ________
Impact:        [ ] Excellent  [ ] Good  [ ] Fair  [ ] Poor  | If Poor, issue: ________
Follow-through:[ ] Excellent  [ ] Good  [ ] Fair  [ ] Poor  | If Poor, issue: ________
Finish:        [ ] Excellent  [ ] Good  [ ] Fair  [ ] Poor  | If Poor, issue: ________

=== OVERALL ===
Overall skill level: [ ] Pro / Advanced  [ ] Intermediate  [ ] Beginner

Top 3 things to fix (if beginner/intermediate):
1. ________________
2. ________________
3. ________________

=== QUALITY CHECK ===
Confidence in this rating: [ ] Very confident  [ ] Somewhat  [ ] Unsure
```

### Step 1.2: Send to Pro Golfer

- Provide access to videos (organized by ID)
- Request completion within 3–5 days
- Aim for 100 swings (50 pro, 50 beginner)

**Pro tip:** If one person can't do it, split: 2 people × 50 swings each. Then cross-check ~10 swings for consistency.

### Step 1.3: Collect and Verify

```python
# verify_labels.py
import pandas as pd

labels = pd.read_csv('labels_raw.csv')

# Check completeness
print(f"Total rows: {len(labels)}")
print(f"Rows with all phases rated: {labels[['address', 'takeaway', 'mid_backswing', 'top', 'mid_downswing', 'impact', 'follow_through', 'finish']].notna().sum()}")

# Check balance
print(f"\nSkill distribution:")
print(labels['overall_skill_level'].value_counts())

# Check which videos are missing
print(f"\nSample of rows:")
print(labels.head())
```

**Success criteria:**
- 100 rows, all phases rated
- ~50 pro, ~50 beginner/intermediate
- No obvious inconsistencies

---

## Phase 2: Build Dataset Structure (Days 3–5)

### Goal
Create a unified CSV with one row per swing, combining expert labels + pose metadata.

### Step 2.1: Convert Label Scale

Map expert ratings to numeric scale:
```
Excellent = 3
Good      = 2
Fair      = 1
Poor      = 0
```

### Step 2.2: Create Master Dataset CSV

```python
# build_dataset.py
import pandas as pd
import os

def build_dataset_from_labels(expert_csv_path, poses_dir, output_path):
    """
    Combine expert labels + pose metadata into unified dataset.
    
    Args:
        expert_csv_path: Path to filled-out label form (CSV)
        poses_dir: Directory containing *_cleaned_poses.csv files
        output_path: Output path for master dataset CSV
    
    Returns:
        DataFrame with one row per swing
    """
    
    # Load expert labels
    expert_df = pd.read_csv(expert_csv_path)
    
    # Map string ratings to numeric
    rating_map = {'Excellent': 3, 'Good': 2, 'Fair': 1, 'Poor': 0}
    
    dataset = []
    for idx, row in expert_df.iterrows():
        swing_id = row['swing_id']
        pose_csv = os.path.join(poses_dir, f"{swing_id}_cleaned_poses.csv")
        
        # Verify pose file exists
        if not os.path.exists(pose_csv):
            print(f"⚠ Missing pose file: {pose_csv}")
            continue
        
        # Load pose data to get frame count
        poses = pd.read_csv(pose_csv)
        num_frames = len(poses)
        
        # Build record
        record = {
            'swing_id': swing_id,
            'video_id': row.get('video_id', swing_id),
            'view': row.get('view', 'side'),
            'handedness': row.get('handedness', 'right'),
            'skill_level': row['overall_skill_level'],  # 'Pro', 'Intermediate', 'Beginner'
            
            # Phase ratings (0-3 scale)
            'address_rating': rating_map.get(row['address'], -1),
            'takeaway_rating': rating_map.get(row['takeaway'], -1),
            'mid_backswing_rating': rating_map.get(row['mid_backswing'], -1),
            'top_rating': rating_map.get(row['top'], -1),
            'mid_downswing_rating': rating_map.get(row['mid_downswing'], -1),
            'impact_rating': rating_map.get(row['impact'], -1),
            'follow_through_rating': rating_map.get(row['follow_through'], -1),
            'finish_rating': rating_map.get(row['finish'], -1),
            
            # Metadata
            'overall_score': None,  # Will be computed later
            'issues_tags': row.get('top_3_issues', ''),
            'num_frames': num_frames,
            'pose_path': pose_csv,
            'label_source': 'expert',
            'confidence': row.get('confidence_score', 0.9)
        }
        dataset.append(record)
    
    # Create DataFrame
    df = pd.DataFrame(dataset)
    
    # Compute overall_score as mean of phase ratings
    phase_cols = ['address_rating', 'takeaway_rating', 'mid_backswing_rating', 'top_rating',
                  'mid_downswing_rating', 'impact_rating', 'follow_through_rating', 'finish_rating']
    df['overall_score'] = df[phase_cols].mean(axis=1) * 25  # Scale 0-3 to 0-75
    
    # Save
    df.to_csv(output_path, index=False)
    print(f"✓ Built dataset: {len(df)} swings")
    print(f"  Pro: {(df['skill_level'] == 'Pro').sum()}")
    print(f"  Beginner: {(df['skill_level'] == 'Beginner').sum()}")
    
    return df

if __name__ == '__main__':
    df = build_dataset_from_labels(
        'labels_raw.csv',
        'data/extracted_poses',
        'datasets/swing_labels_100.csv'
    )
    print(df.head())
```

**Output:** `datasets/swing_labels_100.csv` with columns:
- swing_id, video_id, view, handedness, skill_level
- address_rating, takeaway_rating, ..., finish_rating (0–3 scale)
- overall_score, issues_tags, num_frames, pose_path, label_source, confidence

---

## Phase 3: Extract Engineered Features (Days 6–8)

### Goal
Compute biomechanics features for all 500 swings.

### Step 3.1: Feature Extraction Script

```python
# extract_features_batch.py
import os
import pandas as pd
import numpy as np
from src.biomechanics.angles import AngleCalculator

def extract_features_batch(poses_dir, phase_boundaries_path, output_csv):
    """
    For every swing CSV, compute biomechanics features per frame and aggregate.
    
    Args:
        poses_dir: Directory with *_cleaned_poses.csv files
        phase_boundaries_path: JSON or CSV with phase start/end frames per swing
        output_csv: Output path for features table
    """
    
    calc = AngleCalculator()
    
    # Load phase boundaries (from PoseSwingNet)
    phase_bounds_df = pd.read_csv(phase_boundaries_path)  
    # Expected cols: swing_id, address_start, address_end, takeaway_start, ..., finish_end
    
    results = []
    pose_files = sorted([f for f in os.listdir(poses_dir) if f.endswith('_cleaned_poses.csv')])
    
    for idx, pose_file in enumerate(pose_files):
        swing_id = pose_file.replace('_cleaned_poses.csv', '')
        pose_path = os.path.join(poses_dir, pose_file)
        
        if (idx + 1) % 50 == 0:
            print(f"Processing {idx + 1}/{len(pose_files)}...")
        
        try:
            df = pd.read_csv(pose_path)
            
            # Get phase boundaries for this swing
            phase_row = phase_bounds_df[phase_bounds_df['swing_id'] == swing_id]
            if phase_row.empty:
                print(f"⚠ No phase boundaries for {swing_id}, skipping")
                continue
            
            phase_row = phase_row.iloc[0]
            
            # Compute features per frame (this is computationally intensive)
            # For efficiency, compute only once per frame and cache
            frame_features = {}
            
            for frame_idx, row in df.iterrows():
                try:
                    lmList = extract_landmarks_from_row(row)
                    
                    frame_features[frame_idx] = {
                        'spine_angle': calc.get_spine_angle(lmList, frame_idx),
                        'spine_lateral_tilt': calc.get_spine_lateral_tilt(lmList, frame_idx),
                        'lead_arm_angle': calc.get_lead_arm_angle(lmList, frame_idx),
                        'trail_elbow_angle': calc.get_trail_elbow_angle(lmList, frame_idx),
                        'lead_knee_flex': calc.get_lead_knee_flex(lmList, frame_idx),
                        'trail_knee_flex': calc.get_trail_knee_flex(lmList, frame_idx),
                        'stance_width_ratio': calc.get_stance_width_ratio(lmList, frame_idx),
                        'head_displacement': calc.get_head_movement(lmList, frame_idx)[0],
                    }
                except Exception as e:
                    print(f"  Error processing frame {frame_idx}: {e}")
            
            # Aggregate per phase
            phases = {
                'address': (phase_row['address_start'], phase_row['address_end']),
                'takeaway': (phase_row['takeaway_start'], phase_row['takeaway_end']),
                'mid_backswing': (phase_row['mid_backswing_start'], phase_row['mid_backswing_end']),
                'top': (phase_row['top_start'], phase_row['top_end']),
                'mid_downswing': (phase_row['mid_downswing_start'], phase_row['mid_downswing_end']),
                'impact': (phase_row['impact_start'], phase_row['impact_end']),
                'follow_through': (phase_row['follow_through_start'], phase_row['follow_through_end']),
                'finish': (phase_row['finish_start'], phase_row['finish_end']),
            }
            
            record = {'swing_id': swing_id}
            
            for phase_name, (start_frame, end_frame) in phases.items():
                phase_frames = {k: v for k, v in frame_features.items() if start_frame <= k <= end_frame}
                
                if not phase_frames:
                    continue
                
                # Extract per-metric statistics
                for metric_name in ['spine_angle', 'spine_lateral_tilt', 'lead_arm_angle', 'trail_elbow_angle',
                                   'lead_knee_flex', 'trail_knee_flex', 'stance_width_ratio', 'head_displacement']:
                    values = [v[metric_name] for v in phase_frames.values() if metric_name in v]
                    
                    if values:
                        record[f'{phase_name}_{metric_name}_mean'] = np.mean(values)
                        record[f'{phase_name}_{metric_name}_std'] = np.std(values)
                        record[f'{phase_name}_{metric_name}_min'] = np.min(values)
                        record[f'{phase_name}_{metric_name}_max'] = np.max(values)
            
            results.append(record)
            
        except Exception as e:
            print(f"✗ Error processing {swing_id}: {e}")
    
    # Create DataFrame
    feature_df = pd.DataFrame(results)
    feature_df.to_csv(output_csv, index=False)
    print(f"✓ Extracted features for {len(feature_df)} swings")
    print(f"  Feature columns: {len(feature_df.columns)}")
    
    return feature_df

def extract_landmarks_from_row(row):
    """Convert CSV row to lmList format for angle calculations."""
    # Assuming CSV has columns: left_shoulder_x, left_shoulder_y, ..., etc.
    lmList = []
    # Build list of (x, y, z, visibility) for each landmark
    # This depends on your CSV structure
    return lmList

if __name__ == '__main__':
    extract_features_batch(
        'data/extracted_poses',
        'data/phase_boundaries.csv',  # Or load from PoseSwingNet output
        'datasets/features_500.csv'
    )
```

**Output:** `datasets/features_500.csv` with ~200+ columns:
- swing_id
- per-phase statistics for each metric (mean, std, min, max)
- Example: address_spine_angle_mean, address_spine_angle_std, takeaway_lead_arm_angle_mean, etc.

**Tip:** This runs in parallel if you have many videos. Consider using `joblib.Parallel` or `multiprocessing` to speed it up.

---

## Phase 4: Train XGBoost Classifier (Days 9–11)

### Goal
Train a model that separates beginner from pro with high accuracy. Extract feature importance.

### Step 4.1: Prepare Training Data

```python
# train_xgboost.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import shap

def train_xgboost_classifier(labels_csv, features_csv, output_model_path):
    """
    Train XGBoost classifier on labeled swings.
    """
    
    # Load data
    labels_df = pd.read_csv(labels_csv)
    features_df = pd.read_csv(features_csv)
    
    print(f"Loaded {len(labels_df)} labeled swings")
    print(f"Loaded {len(features_df)} feature swings")
    
    # Join on swing_id (only keep labeled swings)
    merged = labels_df.merge(features_df, on='swing_id', how='inner')
    print(f"Merged: {len(merged)} swings with both labels and features")
    
    # Prepare features and labels
    X = merged.drop(columns=['swing_id', 'video_id', 'view', 'handedness', 'skill_level', 
                             'address_rating', 'takeaway_rating', 'mid_backswing_rating', 'top_rating',
                             'mid_downswing_rating', 'impact_rating', 'follow_through_rating', 'finish_rating',
                             'overall_score', 'issues_tags', 'num_frames', 'pose_path', 'label_source', 'confidence'])
    
    # Encode skill_level: Pro=1, Beginner=0, Intermediate=0.5 (treat as separate class for now)
    skill_map = {'Pro': 1, 'Beginner': 0, 'Intermediate': 0}
    y = merged['skill_level'].map(skill_map).astype(int)
    
    # Handle missing values
    X = X.fillna(X.mean())
    
    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Label distribution:\n{y.value_counts()}")
    
    # Train/test split (stratified to keep class balance)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train model
    print("\n--- Training XGBoost ---")
    model = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='logloss',
        verbose=1
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        early_stopping_rounds=20,
        verbose=False
    )
    
    # Evaluate
    print("\n--- Evaluation ---")
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    print(f"\nAccuracy: {(y_pred == y_test).mean():.3f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Beginner', 'Pro']))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # Cross-validation
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    print(f"\n5-Fold CV Scores: {cv_scores}")
    print(f"Mean CV Accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")
    
    # Feature importance
    print("\n--- Feature Importance (Top 30) ---")
    importance_df = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(importance_df.head(30))
    
    # Save top features
    importance_df.to_csv('datasets/feature_importance.csv', index=False)
    
    # SHAP analysis (optional but informative)
    print("\n--- SHAP Analysis ---")
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
        
        # Plot top features
        shap.summary_plot(shap_values, X_test, plot_type="bar", max_display=15)
        plt.savefig('outputs/shap_summary.png', dpi=150, bbox_inches='tight')
        print("✓ Saved SHAP summary to outputs/shap_summary.png")
        
    except Exception as e:
        print(f"⚠ SHAP analysis failed: {e}")
    
    # Save model
    model.save_model(output_model_path)
    print(f"\n✓ Saved model to {output_model_path}")
    
    # Save training metadata
    metadata = {
        'test_accuracy': (y_pred == y_test).mean(),
        'cv_mean_accuracy': cv_scores.mean(),
        'cv_std_accuracy': cv_scores.std(),
        'n_features': X.shape[1],
        'n_samples_train': len(X_train),
        'n_samples_test': len(X_test),
        'feature_columns': list(X.columns),
        'top_features': list(importance_df.head(20)['feature'].values)
    }
    
    import json
    with open('models/xgboost_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return model, importance_df, X_test, y_test, y_pred_proba

if __name__ == '__main__':
    model, importance_df, X_test, y_test, y_pred_proba = train_xgboost_classifier(
        'datasets/swing_labels_100.csv',
        'datasets/features_500.csv',
        'models/xgboost_beginner_pro.model'
    )
```

**Expected outcome:**
- ≥80% accuracy on test set if metrics are good
- 60–70% if metrics are mediocre
- <60% if metrics don't discriminate

**If accuracy < 75%:** Your current engineered features may not separate skill levels well. Revisit metric definitions before moving forward.

---

## Phase 5: Generate Phase-Based Feedback (Days 12–14)

### Goal
Create a feedback system that maps top discriminative features to coaching advice per phase.

### Step 5.1: Feedback Template Builder

```python
# generate_feedback.py
import pandas as pd
import json

FEEDBACK_TEMPLATES = {
    'address': {
        'spine_angle_high': {
            'message': "📌 Address: Reduce forward spine tilt. Stand more upright.",
            'drill': "Practice: Mirror drill - check posture 10 reps",
            'impact': 8
        },
        'stance_width_narrow': {
            'message': "📌 Address: Widen your stance to shoulder width.",
            'drill': "Practice: Stance calibration drill 5 reps",
            'impact': 6
        }
    },
    'top': {
        'lead_arm_angle_low': {
            'message': "🔝 Top: Extend your lead arm more. Avoid bent-arm backswing.",
            'drill': "Practice: Lead arm extension 10 reps",
            'impact': 10
        },
        'x_factor_low': {
            'message': "🔝 Top: Increase shoulder turn. Rotate more fully.",
            'drill': "Practice: Full shoulder turn 8 reps",
            'impact': 12
        }
    },
    'mid_downswing': {
        'hip_rotates_after_shoulder': {
            'message': "⬇️ Mid-downswing: Initiate with hips first, then shoulders. Poor sequencing.",
            'drill': "Practice: Hip-first transition drill 8 reps",
            'impact': 15
        },
        'trail_elbow_angle_high': {
            'message': "⬇️ Mid-downswing: Drop your elbow. Too much arm lift.",
            'drill': "Practice: Elbow drop drill 10 reps",
            'impact': 11
        }
    },
    'impact': {
        'head_movement_high': {
            'message': "💥 Impact: Keep your head steady. Reduce head sway.",
            'drill': "Practice: Head-steady impact drill 10 reps",
            'impact': 9
        },
        'spine_angle_changed': {
            'message': "💥 Impact: Maintain your spine angle through impact.",
            'drill': "Practice: Posture maintenance drill 8 reps",
            'impact': 10
        }
    }
}

def generate_feedback_from_model(swing_id, labels_row, features_row, model, importance_df):
    """
    Generate phase-specific feedback based on model predictions and feature importance.
    
    Args:
        swing_id: Swing identifier
        labels_row: Expert label row (phase ratings)
        features_row: Extracted features row
        model: Trained XGBoost model
        importance_df: Feature importance DataFrame
    
    Returns:
        Feedback structure with phase-by-phase coaching advice
    """
    
    feedback_items = []
    
    # Get top 20 important features
    top_features = set(importance_df.head(20)['feature'].values)
    
    phases = ['address', 'takeaway', 'mid_backswing', 'top', 'mid_downswing', 'impact', 'follow_through', 'finish']
    
    for phase in phases:
        phase_rating = labels_row.get(f'{phase}_rating', 2)  # Default to "Good"
        
        if phase_rating == 3:
            # Excellent - no feedback needed
            feedback_items.append({
                'phase': phase,
                'status': 'good',
                'message': f"✓ {phase.title()}: Well executed",
                'priority': 0
            })
        elif phase_rating <= 1:
            # Poor or Fair - find discriminative features for this phase
            phase_features = [f for f in features_row.index if f.startswith(phase) and f in top_features]
            
            issues_for_phase = []
            for feat in phase_features:
                feature_name = feat.replace(f'{phase}_', '')
                value = features_row[feat]
                
                # Simple heuristic: if value is outlier, flag it
                if pd.notna(value):
                    issues_for_phase.append({
                        'feature': feature_name,
                        'value': value,
                        'template_key': feature_name
                    })
            
            # Use template if available
            if phase in FEEDBACK_TEMPLATES and issues_for_phase:
                issue = issues_for_phase[0]  # Top issue
                template_key = issue['template_key']
                
                if template_key in FEEDBACK_TEMPLATES[phase]:
                    template = FEEDBACK_TEMPLATES[phase][template_key]
                    feedback_items.append({
                        'phase': phase,
                        'status': 'needs_work',
                        'message': template['message'],
                        'drill': template['drill'],
                        'impact': template['impact'],
                        'priority': 1
                    })
                else:
                    # Generic fallback
                    feedback_items.append({
                        'phase': phase,
                        'status': 'needs_work',
                        'message': f"⚠️ {phase.title()}: Check {template_key}. Look at video for improvement areas.",
                        'priority': 1
                    })
            else:
                feedback_items.append({
                    'phase': phase,
                    'status': 'needs_work',
                    'message': f"⚠️ {phase.title()}: Room for improvement",
                    'priority': 1
                })
    
    # Sort by priority and take top 3 issues
    high_priority = [f for f in feedback_items if f.get('priority', 0) > 0]
    high_priority = sorted(high_priority, key=lambda x: x.get('impact', 0), reverse=True)[:3]
    
    result = {
        'swing_id': swing_id,
        'skill_level': labels_row.get('skill_level', 'Unknown'),
        'all_phases': feedback_items,
        'top_issues': high_priority,
        'message': f"Focus on these 3 areas: {', '.join([item['message'][:30] + '...' for item in high_priority[:3]])}"
    }
    
    return result

if __name__ == '__main__':
    # Example usage
    labels_df = pd.read_csv('datasets/swing_labels_100.csv')
    features_df = pd.read_csv('datasets/features_500.csv')
    importance_df = pd.read_csv('datasets/feature_importance.csv')
    
    merged = labels_df.merge(features_df, on='swing_id')
    
    all_feedback = []
    for idx, row in merged.head(10).iterrows():
        swing_id = row['swing_id']
        labels_row = labels_df[labels_df['swing_id'] == swing_id].iloc[0]
        features_row = features_df[features_df['swing_id'] == swing_id].iloc[0]
        
        feedback = generate_feedback_from_model(swing_id, labels_row, features_row, None, importance_df)
        all_feedback.append(feedback)
    
    # Save feedback
    with open('outputs/feedback_samples.json', 'w') as f:
        json.dump(all_feedback, f, indent=2)
    
    print("✓ Generated feedback for 10 swings")
    print(json.dumps(all_feedback[0], indent=2))
```

**Output:** Per-swing feedback JSON with structure:
```json
{
  "swing_id": "golf_swing_001",
  "skill_level": "Beginner",
  "all_phases": [
    {
      "phase": "address",
      "status": "good",
      "message": "✓ Address: Well executed",
      "priority": 0
    },
    {
      "phase": "mid_downswing",
      "status": "needs_work",
      "message": "⬇️ Mid-downswing: Initiate with hips first, then shoulders",
      "drill": "Hip-first transition drill 8 reps",
      "impact": 15,
      "priority": 1
    }
  ],
  "top_issues": [
    {
      "phase": "mid_downswing",
      "message": "⬇️ Hip sequencing issue...",
      "impact": 15
    }
  ]
}
```

---

## Phase 6: End-to-End Validation (Day 14)

### Goal
Test the full pipeline on unseen videos and validate beginner/pro separation.

```python
# validate_pipeline.py
import pandas as pd
from xgboost import XGBClassifier

def validate_full_pipeline():
    """
    Load trained model and test on remaining unlabeled swings.
    Use swing-level labels to validate separation.
    """
    
    # Load model
    model = XGBClassifier()
    model.load_model('models/xgboost_beginner_pro.model')
    
    # Load features for all 500 swings
    all_features = pd.read_csv('datasets/features_500.csv')
    
    # Predict on all
    X = all_features.drop('swing_id', axis=1).fillna(all_features.mean())
    y_pred_proba = model.predict_proba(X)
    y_pred = model.predict(X)
    
    all_features['predicted_prob_pro'] = y_pred_proba[:, 1]
    all_features['predicted_label'] = y_pred
    
    # Load ground truth labels
    labels_df = pd.read_csv('datasets/swing_labels_100.csv')
    
    # Merge
    validation = all_features.merge(labels_df[['swing_id', 'skill_level']], on='swing_id', how='left')
    
    # Evaluate on labeled set
    labeled = validation[validation['skill_level'].notna()]
    
    skill_map = {'Pro': 1, 'Beginner': 0}
    y_true = labeled['skill_level'].map(skill_map)
    y_pred_val = labeled['predicted_label']
    
    accuracy = (y_true == y_pred_val).mean()
    print(f"Validation Accuracy (labeled): {accuracy:.3f}")
    
    # Check separation on unlabeled
    unlabeled = validation[validation['skill_level'].isna()]
    print(f"\nPredictions on 400 unlabeled swings:")
    print(f"  Predicted Pro: {(unlabeled['predicted_label'] == 1).sum()}")
    print(f"  Predicted Beginner: {(unlabeled['predicted_label'] == 0).sum()}")
    
    # Save results
    validation.to_csv('outputs/validation_results_all_500.csv', index=False)
    print("✓ Saved validation results")

if __name__ == '__main__':
    validate_full_pipeline()
```

---

## Success Criteria

| Metric | Target | Pass/Fail |
|--------|--------|-----------|
| XGBoost test accuracy | ≥80% | ??? |
| Confusion matrix (true negatives) | ≥75% | ??? |
| Top 20 features identified | ✓ | ??? |
| Per-phase feedback generated | ✓ | ??? |
| Feedback on 100 labeled swings | ✓ | ??? |
| All 500 swings processed | ✓ | ??? |

---

## Output Deliverables

After 2 weeks, you will have:

1. **datasets/swing_labels_100.csv** — Master dataset with expert labels
2. **datasets/features_500.csv** — Engineered features for all 500 swings
3. **datasets/feature_importance.csv** — Top 50 discriminative features
4. **models/xgboost_beginner_pro.model** — Trained XGBoost classifier
5. **models/xgboost_metadata.json** — Model metadata and top features
6. **outputs/feedback_samples.json** — Phase-based feedback for 100 labeled swings
7. **outputs/validation_results_all_500.csv** — Predictions on full dataset

---

## What's Next?

**If accuracy ≥80%:**
- ✅ Features work! Use top features to improve rule-based thresholds
- ✅ Or: bootstrap training data for an LSTM model
- ✅ Deploy XGBoost model now, train LSTM in parallel

**If accuracy 70–79%:**
- 🔄 Features partially work. Analyze which phases separate well vs. poorly
- 🔄 Retrain with per-phase calibration
- 🔄 Add more features (trajectory smoothness, velocity analysis)

**If accuracy <70%:**
- ❌ Features don't discriminate. Revisit metric definitions
- ❌ Check data quality (pose confidence, phase boundaries)
- ❌ Consider that skill separation may need temporal modeling (LSTM), not static features

---

## Troubleshooting

### Issue: Missing pose files
**Solution:** Run pose extraction for all 500 videos first. Check `data/extracted_poses/` is complete.

### Issue: Phase boundaries missing
**Solution:** PoseSwingNet must have run on all videos. Check phase output structure.

### Issue: Feature extraction times out
**Solution:** Process videos in parallel using `joblib.Parallel`. Typical: 500 videos = 2–4 hours single-thread.

### Issue: Accuracy below 70%
**Solution:**
1. Check feature distributions (use `.describe()`)
2. Verify labels are consistent (cross-check 10 swings)
3. Add trajectory-based features (velocity, smoothness)
4. Try different threshold for what counts as "Pro" vs. "Beginner"

---

## Timeline Summary

| Week | Days | Tasks |
|------|------|-------|
| 1 | 1–2 | Design labels, brief expert |
| 1 | 3–5 | Build dataset CSV structure |
| 1 | 6–8 | Extract features (batch automation) |
| 2 | 9–11 | Train XGBoost, feature importance |
| 2 | 12–14 | Feedback generation, validation |

**Total: 14 days, ~28 dev hours, ~16 expert hours**

📋 ACTIONABLE RECOMMENDATIONS:
Before friend starts:

 Confirm 100 swings to be labeled (which videos?)
 Verify phase boundaries are exportable for all 500
 Run data quality check on pose CSVs (remove corrupted ones)
For friend's Phase 1:

Start with 50 swings max (reduces labeling load)
Use stratified sample (different video angles/quality)
Document labeling criteria (consistency)
For friend's Phase 3:

Reuse GolfBiomechanics class instead of rebuilding
Run on subset (100 videos) first to validate pipeline
Then scale to 500
For friend's Phase 4:

If accuracy < 70%, don't proceed - revisit metrics
Add sanity checks: "Does model say pros are better than beginners?"
🎯 VERDICT:
Plan is SOLID but needs these updates:

✅ Use deterministic scoring we built (complement, not replace XGBoost)
⚠️ Validate data quality upfront (remove bad videos)
⚠️ Reduce labeled swings from 100 → 50-75 (speed up Phase 1)
✅ Reuse existing biomechanics code (don't reimplement)
Want me to create a revised, tighter version of the plan that accounts for what we've learned? Or send current version as-is to your friend?

