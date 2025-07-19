# Müsli Test Merkmalsextraktion aus Bilddatensatz

Dieses Repo enthält zwei Python scripts: extractFeatures.py extrahiert die Merkmale und Hüllkurve der Getreidekörner aus einem Bilddatensatz mittels OpenCV und demWatershed-Algorithmus. vector.py macht daraus einen aggregierten Eingangsvektor. 

Aggregierter Eingangsvektor für jedes Bild befindet sich in `aggregated_features_per_image.csv`

---

Input:
In beiden Skripts muss man den Pfad der Eingangsbilddatensatzen und des output-ordners angeben.

## Verarbeitungspipeline

Jedes Bild durchläuft
1. **Preprocessing**  
   - Skalierung & quadratischer Crop  
   - grayscale threshholding  
   - Morphological cleaning

2. **Segmentierung**  
   - Watershed Algorithmus

3. **Feature Extraction**  
   - Ellipse fitting (`cv2.fitEllipse`)
   - Shape features (Fläche, aspect ratio, solidity, etc.)
   - Color statistics (mean and standard deviation of R, G, B)
   - Surface roughness (Graustufen Standardabweichung)

> Achtung: mean R G und B beziehen sich auf den Mittelwert der RGB Farbwerte innerhalb eines Getreidekorns. Die Standardabweichung der RGB Werte beschreibt die Abweichung zwischen den Getreidekörnern, nicht innerhalb eines Getreidekorns!

4. **Aggregierten Vektor erstellen**  
    - wird aus ellipsen_merkmale.csv erstellt

---

## Aggregierte Merkmale je Bild

| Feature Group | Feature Name                     | Description                                |
|---------------|----------------------------------|--------------------------------------------|
| Shape         | `major_mean`, `major_std`        | Length of major axis                       |
|               | `minor_mean`, `minor_std`        | Length of minor axis                       |
|               | `aspect_ratio_mean`, `aspect_ratio_std` | Elliptical elongation                 |
|               | `contour_area_mean`, `contour_area_std` | Area of the fitted grain contour     |
| Texture       | `roughness_mean`, `roughness_std`| Grayscale std within grain mask (surface roughness) |
| Solidity      | `solidity_mean`, `solidity_std`  | Ratio of contour area to convex hull area  |
| Color         | `mean_red_mean`, `mean_green_mean`, `mean_blue_mean` | Average RGB values across grains |

Die Features werden in den CSV dateien abgespeichert.  
`aggregated_features_per_image.csv`
`ellipsen_merkmale.csv`
---
