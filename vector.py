import pandas as pd
from pathlib import Path

# Input/output paths
csv_path = Path(r"/Users/danielanders/projekte/grain-segmentation/features.csv")
output_csv_path = csv_path.parent / "aggregated_features_per_image.csv"

# Load mapping from variety mapping CSV
mapping_df = pd.read_csv("samplelist_alias.csv", skiprows=1, names=["VarietyID", "ImageNumber"])

# Convert ImageNumber like '001' to filename format like 'r001.JPG'
mapping_df["image"] = mapping_df["ImageNumber"].apply(lambda x: f"r{int(x):03}.JPG")
image_to_variety = dict(zip(mapping_df["image"], mapping_df["VarietyID"]))

# Load extracted features CSV
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

agg_df["VarietyID"] = agg_df.index.map(image_to_variety).astype("Int32")

# Flatten MultiIndex column names
agg_df.columns = ['_'.join(col) for col in agg_df.columns]
agg_df.reset_index(inplace=True)

# Optional: Save to CSV
agg_df.to_csv(output_csv_path, index=False)
print(f"Aggregated features saved to: {output_csv_path}")