import cv2
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import time

# Paths
input_dir = Path(r"C:\Users\danie\Documents\muesli\muesli_project")
output_dir = Path(r"C:\Users\danie\Documents\muesli\fitted_ellipses")
output_dir.mkdir(parents=True, exist_ok=True)

# Resize factor
scale = 0.5

# Get image files
image_files = list(input_dir.glob("*.JPG")) + list(input_dir.glob("*.jpg"))

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
    kernel_erroded = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel_background = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    kernel_morph = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_morph, iterations=2)

    opening = cv2.erode(opening, kernel_erroded, iterations=8)

    # Distance transform and markers
    sure_bg = cv2.dilate(opening, kernel_erroded, iterations=4)
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

    # Fit ellipses
    contours, _ = cv2.findContours(segmentation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if len(contour) >= 5:
            ellipse = cv2.fitEllipse(contour)
            (x, y), (major, minor), angle = ellipse
            if major > 0 and minor > 0:
                try:
                    cv2.ellipse(image, ellipse, (0, 255, 0), 2)
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