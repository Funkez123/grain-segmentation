import pandas as pd
import matplotlib.pyplot as plt

# Load CSV
file_path = 'aggregated_features_per_image.csv'  # Adjust if needed
df = pd.read_csv(file_path)

# Create histograms
plt.figure(figsize=(12, 5))

# Histogram for solidity_mean
plt.subplot(1, 2, 1)
plt.hist(df['solidity_mean'], bins=20, color='skyblue', edgecolor='black')
plt.title('Histogram gemittelter Flächenübereinstimmung (Solidity)', fontsize=14)
plt.xlabel('Solidity (20 bins)')
plt.ylabel('Häufigkeit')
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Histogram for roughness_mean
plt.subplot(1, 2, 2)
plt.hist(df['roughness_mean'], bins=20, color='salmon', edgecolor='black')
plt.title('Histogram gemittelter Rauheit (Roughness)', fontsize=14)
plt.xlabel('Roughness (20 bins)')
plt.ylabel('Häufigkeit')
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig("solidity_roughness_histogram.png")
