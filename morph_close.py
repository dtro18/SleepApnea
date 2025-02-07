import cv2
import numpy as np
import os 
def process_oct_image(image_path, kernel_size=100, iterations=5):
    """
    Process OCT image using morphological operations to close gaps
    and calculate area.
    
    Parameters:
    image_path: str, path to the image file
    kernel_size: int, size of the kernel for morphological operations
    iterations: int, number of times to apply the closing operation
    
    Returns:
    tuple: (processed image, area in pixels, original image)
    """

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    original = img.copy()
    img = cv2.GaussianBlur(img, (5, 5), 0)
    # Create kernel for morphological operations
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    
    # Apply morphological closing
    # dilated = cv2.dilate(img, kernel, iterations=1)
    closed = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=iterations)
    
    # Find contours in the closed image
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw contours and fill them
    result = np.zeros_like(img)
    area = 0
    if contours:
        # Get the largest contour (assuming it's the ring)
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        cv2.drawContours(result, [largest_contour], -1, (255, 255, 255), -1)
    
    return result, area, original

def display_results(original, processed, area, scale_factor=0.25):
    """
    Display the original and processed images side by side
    with the calculated area.
    """
    # Create a side-by-side comparison
    comparison = np.hstack((original, processed))
    height, width = comparison.shape
    new_width = int(width*scale_factor)
    new_height = int(height*scale_factor)

    resized_comparison = cv2.resize(comparison, (new_width, new_height))
    # Display images
    cv2.imshow('Original vs Processed (Area: {} pixels)'.format(int(area)), resized_comparison)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


processed_img, area, original = process_oct_image(r"D:\SleepApneaPtData\set_1_masked\0004.bmp")

# Display results
display_results(original, processed_img, area)