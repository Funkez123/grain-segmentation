import pandas as pd
import matplotlib.pyplot as plt

file_path = 'aggregated_features_per_image.csv'
df = pd.read_csv(file_path)

rgb_df = df[['mean_red_mean', 'mean_green_mean', 'mean_blue_mean']]

plt.figure(figsize=(8, 6))
plt.boxplot(
    [rgb_df['mean_red_mean'], rgb_df['mean_green_mean'], rgb_df['mean_blue_mean']],
    labels=['Red', 'Green', 'Blue'],
    patch_artist=True,
    boxprops=dict(facecolor='lightgray', color='black'),
    medianprops=dict(color='red', linewidth=2)
)

plt.title('Distribution of average RGB values', fontsize=14)
plt.ylabel('average channel value', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig("avg_color_values.png")