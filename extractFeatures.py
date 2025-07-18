import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import time

# Paths
input_dir = Path(r"C:\Users\danie\Documents\muesli\muesli_project")
output_dir = Path(r"C:\Users\danie\Documents\muesli\fitted_ellipses")
csv_path = output_dir / "ellipsen_merkmale.csv"
output_dir.mkdir(parents=True, exist_ok=True)

# Resize factor
scale = 0.5

# Get image files
image_files = list(input_dir.glob("*.JPG")) + list(input_dir.glob("*.jpg"))

# Liste für Merkmal-Daten
ellipse_data = []


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
    
    # Morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    opening = cv2.erode(opening, kernel, iterations=7)

    # Distance transform and markers
    sure_bg = cv2.dilate(opening, kernel, iterations=4)
    dist = cv2.distanceTransform(opening, cv2.DIST_L2, 5)

    _, sure_fg = cv2.threshold(dist, 0.30 * dist.max(), 255, 0)

    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    # Watershed
    markers = cv2.watershed(image, markers)
    segmentation = np.uint8(markers > 1) * 255

    # Kopie des Originalbilds zur Farbanalyse
    orig_image = image.copy()

    # Für jeden Marker (ab 2, da 0 = unbekannt, 1 = Hintergrund)
    for marker_id in range(2, np.max(markers) + 1):
        # Maske erstellen für den aktuellen Marker
        mask = np.uint8(markers == marker_id)

        # Konturen des Bereichs finden (optional für Ellipse)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0 or len(contours[0]) < 5:
            continue

        # Ellipse fitten
        ellipse = cv2.fitEllipse(contours[0])
        (x, y), (major, minor), angle = ellipse

        # Farbe im maskierten Bereich berechnen (Mittelwert in RGB)
        mean_color = cv2.mean(orig_image, mask=mask)
        mean_r, mean_g, mean_b = mean_color[:3]

        # Optional: Maske anzeigen/debuggen
        # cv2.imshow("Mask", mask * 255)
        # cv2.waitKey(0)

        ellipse_data.append({
            "bild": image_path.name,
            "cx": x,
            "cy": y,
            "major": max(major, minor) / 2,
            "minor": min(major, minor) / 2,
            "angle": angle,
            "mean_r": mean_r,
            "mean_g": mean_g,
            "mean_b": mean_b
        })

    # Fit ellipses
    contours, _ = cv2.findContours(segmentation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if len(contour) >= 5:
            ellipse = cv2.fitEllipse(contour)
            (x, y), (major, minor), angle = ellipse
            if major > 20 and minor > 20 and minor < 220 and major < 80: #only plausible ellipses
                try:
                    cv2.ellipse(image, ellipse, (0, 255, 0), 2)
                    ellipse_data.append({
                        "bild": image_path.name,
                        "cx": x,
                        "cy": y,
                        "major": max(major, minor) / 2,  # Halbachse
                        "minor": min(major, minor) / 2,  # Halbachse
                        "angle": angle
                    })
                except cv2.error:
                    continue

    # Save output
    output_path = output_dir / image_path.name
    cv2.imwrite(str(output_path), image)
    print(f"{image_path.name} processed in {time.time() - start_time:.2f}s")

# Run multithreaded processing
with ThreadPoolExecutor() as executor:
    executor.map(process_image, image_files)

print("All images processed.")

# Daten als DataFrame speichern
df = pd.DataFrame(ellipse_data)
df.to_csv(csv_path, index=False)
print(f"\nMerkmalsdaten gespeichert unter: {csv_path}")

# Statistik und Plot
if not df.empty:
    stats = df[["major", "minor"]].agg(["mean", "std"])
    print("\nStatistiken:")
    print(stats)

    # Plot
    plt.figure(figsize=(8, 5))
    for col in ["major", "minor"]:
        plt.errorbar(
            col, stats.loc["mean", col],
            yerr=stats.loc["std", col],
            fmt='o', capsize=5, label=f"{col.capitalize()}-Achse"
        )
    plt.title("Mittelwerte und Standardabweichungen der Ellipsenhalbachsen")
    plt.ylabel("Pixel")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / "ellipse_statistik.png")
    plt.show()

print("Alle Bilder verarbeitet und Merkmale statistisch ausgewertet.")