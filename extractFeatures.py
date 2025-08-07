import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import time

# Paths
input_dir = Path(r"muesli_projekt")
output_dir = Path(r"fitted_ellipses")
csv_path = "features.csv"
output_dir.mkdir(parents=True, exist_ok=True)

# Resize factor
scale = 0.5

# Get image files
image_files = list(input_dir.glob("*.JPG")) + list(input_dir.glob("*.jpg"))

# Liste für Merkmal-Daten
ellipse_data = []

global_start_time = time.time()

def process_image(image_path):
    start_time = time.time()
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Failed to read {image_path.name}")
        return

    # Resize
    image = cv2.resize(image, (0, 0), fx=scale, fy=scale)
    height, width = image.shape[:2]
    min_dim = min(height, width)
    x_start = (width - min_dim) // 2
    y_start = (height - min_dim) // 2
    image = image[y_start:y_start + min_dim, x_start:x_start + min_dim]

    # Convert to grayscale and threshold
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 110, 255, cv2.THRESH_BINARY)

    # Morphological operations (noise reduction)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    opening = cv2.erode(opening, kernel, iterations=5)

    # Distance transform and markers
    sure_bg = cv2.dilate(opening, kernel, iterations=4)

    dist = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist, 0.25 * dist.max(), 255, 0)

    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    # Watershed
    markers = cv2.watershed(image, markers)
    segmentation = np.uint8(markers > 1) * 255

    # image copy for color analysis
    orig_image = image.copy()

    # iterating through each marker, starting at 2 because 0 = unknown and 1 = background
    for marker_id in range(2, np.max(markers) + 1):
        mask = np.uint8(markers == marker_id)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0 or len(contours[0]) < 5:
            continue

        ellipse = cv2.fitEllipse(contours[0])
        (x, y), (major, minor), angle = ellipse

        # size and location filter
        if major > 80 or minor > 220 or major < 30 or minor < 25 or x < 100 or x > 1000 or y < 100 or y > 1000:
            continue
        
        aspect_ratio =  major / minor

        ellipse_area = np.pi * (major / 2) * (minor / 2)
        contour_area = cv2.contourArea(contours[0])
        solidity = contour_area / ellipse_area

        # R G and B color channel seperation
        masked_pixels = orig_image[mask == 1]  # Shape: (N, 3), N = Anzahl Pixel im Segment
        gray_masked_pixels = gray[mask == 1]

        if masked_pixels.size == 0:
            continue 

        mean_r, mean_g, mean_b = np.mean(masked_pixels, axis=0)
        #std_r, std_g, std_b = np.std(masked_pixels, axis=0)

        roughness = np.std(gray_masked_pixels, axis=0)

        # try drawing the ellipses
        try:
            cv2.ellipse(image, ellipse, (0, 255, 0), 4)
        except cv2.error:
            continue

        # buffer data for csv
        ellipse_data.append({
            "image": image_path.name,
            "cx": x,
            "cy": y,
            "angle": angle,
            "major": max(major, minor) / 2,  # Halbachse
            "minor": min(major, minor) / 2,  # Halbachse
            "contour_area" : contour_area,
            "aspect_ratio": aspect_ratio,
            "solidity" : solidity,
            "mean_red": mean_r,
            "mean_green": mean_g,
            "mean_blue": mean_b,
            "roughness" : roughness
        })

    # Save output
    output_path = output_dir / image_path.name
    cv2.imwrite(str(output_path), image)
    print(f"{image_path.name} processed in {time.time() - start_time:.2f}s")

# Run multithreaded processing
with ThreadPoolExecutor() as executor:
    executor.map(process_image, image_files)

global_end_time = time.time()
computation_time = global_end_time-global_start_time
print(f"Processing Time for 476 images: {computation_time:.2f}s")
print("All images processed.")

#save data as pandas dataframe
df = pd.DataFrame(ellipse_data)
df.to_csv(csv_path, index=False)
print(f"\nAll ellipse data saved under: {csv_path}")

# # statistics and plots
# if not df.empty:
#     stats = df[["major", "minor"]].agg(["mean", "std"])
#     print("\nStatistiken:")
#     print(stats)

#     # Plot
#     plt.figure(figsize=(8, 5))
#     for col in ["major", "minor"]:
#         plt.errorbar(
#             col, stats.loc["mean", col],
#             yerr=stats.loc["std", col],
#             fmt='o', capsize=5, label=f"{col.capitalize()}-Achse"
#         )
#     plt.title("Mittelwerte und Standardabweichungen der Ellipsenhalbachsen")
#     plt.ylabel("Pixel")
#     plt.legend()
#     plt.grid(True)
#     plt.tight_layout()
#     plt.savefig(output_dir / "ellipse_statistics.png")
#     plt.show()

print("All images have been analyzed")