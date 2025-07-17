import cv2
import numpy as np
import os
from pathlib import Path

# Paths
input_dir = Path("/home/daniel/Downloads/muesli/muesli_project")
output_dir = Path("/home/daniel/Downloads/muesli/fitted_ellipses")
output_dir.mkdir(parents=True, exist_ok=True)

# List all JPG/jpg files
image_files = list(input_dir.glob("*.JPG")) + list(input_dir.glob("*.jpg"))

# Resize factor
scale = 0.5

for image_path in image_files:
    print(f"Processing: {image_path.name}")
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Failed to read {image_path}")
        continue

    # Resize:
    image = cv2.resize(image, (0, 0), fx=scale, fy=scale)
    # Get image dimensions
    height, width = image.shape[:2]

    # Determine the size of the square (min of height and width)
    min_dim = min(height, width)

    # Compute the top-left corner of the square crop
    x_start = (width - min_dim) // 2
    y_start = (height - min_dim) // 2

    # Crop the center square
    image = image[y_start:y_start + min_dim, x_start:x_start + min_dim]

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Blur
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    # Threshold
    _, thresh = cv2.threshold(gray, 140, 255, cv2.THRESH_BINARY)

    # Morphological closing to close gaps / noise removal
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1))
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

    # Sure background
    sure_bg = cv2.dilate(opening, kernel, iterations=10)

    # Distance transform segmentation
    dist = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist, 0.3 * dist.max(), 255, 0)

    # Unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Marker labeling
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1 #sure background is 1 not 0
    markers[unknown == 255] = 0

    # Watershed
    #markers = cv2.watershed(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), markers)
    markers = cv2.watershed(image, markers)

    segmentation = np.uint8(markers > 1) * 255

    # Find contours
    contours, _ = cv2.findContours(segmentation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if len(contour) >= 5:
            ellipse = cv2.fitEllipse(contour)
            (x, y), (major, minor), angle = ellipse

            if major <= 0 or minor <= 0:
                continue

            try:
                cv2.ellipse(image, ellipse, (0, 255, 0), 2)
            except cv2.error as e:
                print(f"Skipped bad ellipse: {e}")
                continue

    # Save output image
    output_path = output_dir / image_path.name
    cv2.imwrite(str(output_path), image)

print("All images processed and saved.")
