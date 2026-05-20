"""Check if x_factor exists in B90 metrics"""
import pandas as pd
import numpy as np

# Load metrics
metrics_df = pd.read_csv('data/metrics/B90_cleaned_metrics.csv')

print("Checking x_factor in B90 metrics:")
print(f"  Columns with 'factor': {[c for c in metrics_df.columns if 'factor' in c.lower()]}")
print()

# Check impact frame (50-65% of total)
total = len(metrics_df)
impact_start = int(total * 0.5)
impact_end = int(total * 0.65)

print(f"Impact frames: {impact_start}-{impact_end}")
print()

if 'x_factor' in metrics_df.columns:
    print("x_factor values in impact phase:")
    x_factors = metrics_df.iloc[impact_start:impact_end]['x_factor']
    print(f"  Mean: {x_factors.mean():.2f}")
    print(f"  Min: {x_factors.min():.2f}")
    print(f"  Max: {x_factors.max():.2f}")
    print(f"  Any NaN?: {x_factors.isna().any()}")
else:
    print("ERROR: x_factor column not found!")
    print(f"Available metric columns: {metrics_df.columns.tolist()}")
