import pandas as pd
from src.biomechanics.angles import GolfBiomechanics

# Load pose data
pose_csv = 'data/extracted_poses/B80O_cleaned_poses.csv'
df = pd.read_csv(pose_csv)

# Set your top frame number (from your pipeline output)
top_frame = 37  # Change if your top frame is different

biomech = GolfBiomechanics(df)

# Print all metrics at top frame
metrics = biomech.calculate_all_metrics(frame=top_frame)
print(f"Top frame: {top_frame}")
for k, v in metrics.items():
    print(f"{k:20}: {v}")
