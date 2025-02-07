import cv2
import numpy as np

def adaptive_closing(image_path):
    # Load grayscale image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    original = img.copy()

    # Preprocessing: Apply Gaussian Blur to smooth noise
    img_blurred = cv2.GaussianBlur(img, (5, 5), 0)

    # Edge Detection to highlight gaps
    edges = cv2.Canny(img_blurred, 5, 250)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a mask for large gaps
    gap_mask = np.zeros_like(img)

    for cnt in contours:
        if cv2.arcLength(cnt, closed=False) > 50:  # Adjust threshold for detecting gaps
            cv2.drawContours(gap_mask, [cnt], -1, 255, thickness=cv2.FILLED)

    # Use a larger kernel **only on the gaps**
    kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
    closed_gaps = cv2.morphologyEx(gap_mask, cv2.MORPH_CLOSE, kernel_large, iterations=2)

    # Merge closed gaps with the original image
    result = cv2.bitwise_or(img, closed_gaps)

    return original, result

# Load and process the image
original, processed = adaptive_closing(r"D:\SleepApneaPtData\set_1_masked\0000.bmp")

scale_factor = 0.5  # Adjust scale factor as needed
original_resized = cv2.resize(original, (0, 0), fx=scale_factor, fy=scale_factor)
processed_resized = cv2.resize(processed, (0, 0), fx=scale_factor, fy=scale_factor)

# Show the resized images
cv2.imshow("Original (Resized)", original_resized)
cv2.imshow("Processed with Contour Interpolation (Resized)", processed_resized)
cv2.waitKey(0)
cv2.destroyAllWindows()