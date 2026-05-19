"""
Validate expert labels and organize videos for training dataset.
"""
import pandas as pd
import os
import shutil
import numpy as np

def validate_expert_labels(csv_path):
    """Check expert labels for completeness and consistency."""
    
    df = pd.read_csv(csv_path)
    
    print("\n" + "="*70)
    print("EXPERT LABELS VALIDATION")
    print("="*70)
    
    # 1. Check completeness
    print(f"\nTotal rows: {len(df)}")
    
    phase_cols = ['address_score', 'takeaway_score', 'mid_backswing_score', 'top_score',
                  'mid_downswing_score', 'impact_score', 'follow_through_score', 'finish_score']
    
    missing_scores = df[phase_cols].isna().sum()
    print(f"\nMissing phase scores:")
    for col in phase_cols:
        if missing_scores[col] > 0:
            print(f"  {col}: {missing_scores[col]} rows")
    
    complete_rows = df[phase_cols].notna().all(axis=1).sum()
    print(f"\n✓ Complete rows (all 8 phases rated): {complete_rows}/{len(df)}")
    
    # 2. Check skill level distribution
    print(f"\nSkill level distribution:")
    print(df['skill_level'].value_counts())
    
    # 3. Check score ranges
    print(f"\nPhase score ranges:")
    for col in phase_cols:
        phase_name = col.replace('_score', '')
        scores = df[col].dropna()
        print(f"  {phase_name}: min={scores.min():.0f}, max={scores.max():.0f}, mean={scores.mean():.1f}")
    
    # 4. Check consistency (do pro swings generally score higher?)
    print(f"\nAverage scores by skill level:")
    for skill in df['skill_level'].unique():
        subset = df[df['skill_level'] == skill][phase_cols]
        avg = subset.mean().mean()
        print(f"  {skill}: {avg:.1f}/100")
    
    # 5. Check for outliers (any row with huge variance across phases?)
    df['phase_score_std'] = df[phase_cols].std(axis=1)
    high_variance = df[df['phase_score_std'] > 30]
    if len(high_variance) > 0:
        print(f"\n⚠ High variance in phase scores (std > 30) - may indicate inconsistent rating:")
        for idx, row in high_variance.iterrows():
            print(f"  {row['swing_id']}: {row['phase_score_std']:.1f} std")
    
    # 6. Verify video files exist
    print(f"\nVerifying video files...")
    missing_videos = []
    for idx, row in df.iterrows():
        swing_id = row['swing_id']
        poses_csv = f"data/extracted_poses/{swing_id}_cleaned_poses.csv"
        phases_csv = f"data/keyframes/{swing_id}_nn/{swing_id}_cleaned_8phases.csv"
        
        if not os.path.exists(poses_csv):
            missing_videos.append((swing_id, "poses CSV"))
        if not os.path.exists(phases_csv):
            missing_videos.append((swing_id, "phases CSV"))
    
    if missing_videos:
        print(f"⚠ Missing files:")
        for swing_id, file_type in missing_videos[:10]:
            print(f"  {swing_id}: missing {file_type}")
        if len(missing_videos) > 10:
            print(f"  ... and {len(missing_videos) - 10} more")
    else:
        print(f"✓ All {len(df)} swings have pose + phase data")
    
    # Summary
    print(f"\n" + "="*70)
    if complete_rows >= 140:  # Want 150, accept 140+
        print("✓ READY TO USE for training!")
    else:
        print(f"⚠ Need {150 - complete_rows} more complete ratings")
    print("="*70 + "\n")
    
    return df

def organize_videos_for_labeling(label_csv, output_dir="data/videos_for_labeling"):
    """
    Create organized folder structure for expert to access videos.
    Group by skill level (pro vs beginner) for reference.
    """
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/PRO_REFERENCE", exist_ok=True)
    os.makedirs(f"{output_dir}/BEGINNER_REFERENCE", exist_ok=True)
    os.makedirs(f"{output_dir}/TO_LABEL", exist_ok=True)
    
    df = pd.read_csv(label_csv)
    
    print(f"\nOrganizing videos...")
    
    # Organize by skill level
    for idx, row in df.iterrows():
        swing_id = row['swing_id']
        skill = row.get('skill_level', 'unknown')
        
        # Create video info file
        info_path = f"{output_dir}/TO_LABEL/{swing_id}_info.txt"
        with open(info_path, 'w') as f:
            f.write(f"Swing ID: {swing_id}\n")
            f.write(f"Camera angle: {row.get('camera_angle', 'unknown')}\n")
            f.write(f"Rating this as: [PRO / ADVANCED / INTERMEDIATE / BEGINNER]\n")
            f.write(f"\n--- Rate each phase 0-100 ---\n")
            f.write(f"Address: ___/100\n")
            f.write(f"Takeaway: ___/100\n")
            f.write(f"Mid-backswing: ___/100\n")
            f.write(f"Top: ___/100\n")
            f.write(f"Mid-downswing: ___/100\n")
            f.write(f"Impact: ___/100\n")
            f.write(f"Follow-through: ___/100\n")
            f.write(f"Finish: ___/100\n")
            f.write(f"\nTop issues if beginner/intermediate:\n1. ___________\n2. ___________\n3. ___________\n")
        
        # Create symlink to poses CSV for analysis
        poses_src = f"data/extracted_poses/{swing_id}_cleaned_poses.csv"
        if os.path.exists(poses_src):
            try:
                os.symlink(os.path.abspath(poses_src), f"{output_dir}/TO_LABEL/{swing_id}_poses.csv")
            except:
                pass  # Windows may not support symlinks
    
    print(f"✓ Created labeling folder structure at: {output_dir}")
    print(f"  - TO_LABEL/ : {len(df)} swings to rate")
    print(f"  - Reference videos available for calibration")
    
    return output_dir

def create_training_csv(expert_labels_csv, output_path="datasets/training_150_labeled.csv"):
    """
    Convert expert labels to training dataset format.
    Verify all required files exist.
    """
    
    df = pd.read_csv(expert_labels_csv)
    
    # Filter to only complete rows
    phase_cols = ['address_score', 'takeaway_score', 'mid_backswing_score', 'top_score',
                  'mid_downswing_score', 'impact_score', 'follow_through_score', 'finish_score']
    df = df[df[phase_cols].notna().all(axis=1)]
    
    # Verify files exist
    valid_rows = []
    for idx, row in df.iterrows():
        swing_id = row['swing_id']
        poses_csv = f"data/extracted_poses/{swing_id}_cleaned_poses.csv"
        phases_csv = f"data/keyframes/{swing_id}_nn/{swing_id}_cleaned_8phases.csv"
        
        if os.path.exists(poses_csv) and os.path.exists(phases_csv):
            valid_rows.append(row)
    
    if len(valid_rows) == 0:
        print("⚠ No valid swings found!")
        return None
    
    training_df = pd.DataFrame(valid_rows)
    
    # Compute overall score as mean of phases
    training_df['overall_score'] = training_df[phase_cols].mean(axis=1)
    
    # Save
    os.makedirs('datasets', exist_ok=True)
    training_df.to_csv(output_path, index=False)
    
    print(f"\n✓ Created training dataset: {output_path}")
    print(f"  Swings: {len(training_df)}")
    print(f"  Skill distribution:")
    print(training_df['skill_level'].value_counts().to_string())
    print(f"\n  Score distribution (by skill):")
    for skill in training_df['skill_level'].unique():
        subset = training_df[training_df['skill_level'] == skill]['overall_score']
        print(f"    {skill}: mean={subset.mean():.1f}, std={subset.std():.1f}")
    
    return training_df

if __name__ == '__main__':
    # Step 1: Validate labels once expert fills them out
    print("\n1. VALIDATING EXPERT LABELS...")
    df = validate_expert_labels('data/labels/EXPERT_RATING_FORM.csv')
    
    # Step 2: Organize videos for easy access
    print("\n2. ORGANIZING VIDEOS FOR LABELING...")
    organize_videos_for_labeling('data/labels/EXPERT_RATING_FORM.csv')
    
    # Step 3: Create training dataset
    print("\n3. CREATING TRAINING DATASET...")
    create_training_csv('data/labels/EXPERT_RATING_FORM.csv')
    
    print("\n✓ All done! Ready for neural network training.\n")
