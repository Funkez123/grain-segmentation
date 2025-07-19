import pandas as pd
from pathlib import Path

# Input/output paths
csv_path = Path(r"C:\Users\danie\Documents\muesli\features.csv")
output_csv_path = csv_path.parent / "aggregated_features_per_image.csv"

# Load CSV
df = pd.read_csv(csv_path)

# Group by image and calculate mean and std for relevant columns
agg_df = df.groupby("image").agg({
    "major": ["mean", "std"],
    "minor": ["mean", "std"],
    "aspect_ratio" : ["mean", "std"],
    "contour_area" : ["mean", "std"],
    "roughness" : ["mean", "std"],
    "solidity" : ["mean", "std"],
    "mean_red": ["mean"],
    "mean_green": ["mean"],
    "mean_blue": ["mean"]
})

# Flatten MultiIndex column names
agg_df.columns = ['_'.join(col) for col in agg_df.columns]
agg_df.reset_index(inplace=True)

# Optional: Save to CSV
agg_df.to_csv(output_csv_path, index=False)
print(f"Aggregated features saved to: {output_csv_path}")