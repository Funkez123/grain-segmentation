import pandas as pd
import matplotlib.pyplot as plt

# Load CSV
file_path = 'aggregated_features_per_image.csv'  # Adjust if needed
df = pd.read_csv(file_path)

# Create histograms
plt.figure(figsize=(12, 5))

# Histogram for contour_area_mean
plt.subplot(1, 2, 1)
plt.hist(df['contour_area_mean'], bins=20, color='lightgreen', edgecolor='black')
plt.title('Histogram of Contour Area Mean', fontsize=14)
plt.xlabel('Contour Area (20 bins)')
plt.ylabel('Frequency')
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Histogram for aspect_ratio_mean
plt.subplot(1, 2, 2)
plt.hist(df['aspect_ratio_mean'], bins=20, color='plum', edgecolor='black')
plt.title('Histogram of Aspect Ratio Mean', fontsize=14)
plt.xlabel('Aspect Ratio (20 bins)')
plt.ylabel('Frequency')
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig("aspectRatioAndContourArea.png")
