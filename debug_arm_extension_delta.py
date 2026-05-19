import pandas as pd
from src.biomechanics.angles import GolfBiomechanics

# Load pose data
pose_csv = 'data/extracted_poses/B80O_cleaned_poses.csv'
df = pd.read_csv(pose_csv)

# Set your top and impact frame numbers (from your pipeline output)
top_frame = 37  # Change if your top frame is different
impact_frame = 51

biomech = GolfBiomechanics(df)
top_angle = biomech.get_lead_arm_angle(frame=top_frame)
impact_angle = biomech.get_lead_arm_angle(frame=impact_frame)
delta = impact_angle - top_angle
print(f"Top frame: {top_frame}, Lead arm angle: {top_angle:.2f}°")
print(f"Impact frame: {impact_frame}, Lead arm angle: {impact_angle:.2f}°")
print(f"Delta (impact - top): {delta:.2f}°")
print(f"Ideal delta: 15°")
print(f"Score (before weighting): {max(0, 100 - abs(delta-15)*5):.2f}/100")
