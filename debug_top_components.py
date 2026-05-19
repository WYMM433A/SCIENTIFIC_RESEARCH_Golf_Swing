import pandas as pd
from src.biomechanics.angles import GolfBiomechanics
from src.biomechanics.phase_scorer import PhaseScorer

# Load pose data
pose_csv = 'data/extracted_poses/B80O_cleaned_poses.csv'
df = pd.read_csv(pose_csv)

# Set your top frame number (from your pipeline output)
top_frame = 94 # Change if your top frame is different

biomech = GolfBiomechanics(df)
metrics = biomech.calculate_all_metrics(frame=top_frame)

scorer = PhaseScorer()
score, details = scorer.score_top(metrics)

print(f"Top phase score: {score:.2f}/100\n")
print("Component breakdown:")
for k, v in details.get('components', {}).items():
    print(f"  {k:15}: {v:.2f}")
