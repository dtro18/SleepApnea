import numpy as np
import random
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import os

'''
Program implements a processing pipeline for two folders, an oct image folder and a mask folder.
Assumes that the oct image folder and the mask folder are matched up 
i.e. the first image in oct folder corresponds to the first in the mask folder.

1) Remove the outer ring of oct image
2) Center crop the image to include the relevant anatomy
3) Apply the center crop to the corresponding mask image

Results are output into a specified folder for both oct and mask images.

'''
def preprocess_image(image_path_oct, image_path_mask, oct_output, msk_output, intensity_threshold=60, center_radius=60, mask_size=(250, 250)):
    oct_image = cv2.imread(image_path_oct, cv2.IMREAD_GRAYSCALE)  # Convert to grayscale
    msk_image = cv2.imread(image_path_mask, cv2.IMREAD_GRAYSCALE)

    # width, height = oct_image.size
    image_np = np.array(oct_image)
    
    # Apply intensity threshold to retain only bright structures (ring)
    image_np[image_np < intensity_threshold] = 0
    
    # Mask out outer ring
    masked_ring = remove_outer_ring(image_np)

    # Crop
    # Crop the image
    crop_params = auto_crop_image(masked_ring) 
   
    cropped_oct = masked_ring[crop_params[0]:crop_params[1], crop_params[2]:crop_params[3]]
    cropped_mask = msk_image[crop_params[0]:crop_params[1], crop_params[2]:crop_params[3]]

    # Create image and save it
    processed_oct = Image.fromarray(cropped_oct)
    processed_mask = Image.fromarray(cropped_mask)

    # Save the image in BMP format
    processed_oct.save(oct_output, format="PNG")
    processed_mask.save(msk_output, format="PNG")

    print(f"Processed image saved as {oct_output}")



# Takes in grayscale image as input. Must be square
def remove_outer_ring(imageArr):
    
    # Get image dimensions
    height, width = imageArr.shape
    center = (width // 2, height // 2)  # Assume center is the middle of the image

    radius = int(width * 0.40)
    # Create a circular mask
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.circle(mask, center, radius, 255, thickness=-1)  # White-filled circle

    # Apply the mask
    masked_image = cv2.bitwise_and(imageArr, imageArr, mask=mask)

    return masked_image

# Square crop that returns params of crop
def auto_crop_image(image, min_contour_size=50, buffer=50):
    h, w = image.shape
    # Apply threshold to get binary mask of important features
    _, binary = cv2.threshold(image, 50, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours by size
    valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_size]

    if not valid_contours:
        print("No large contours found. Skipping crop.")
        return image

    # Get bounding box around all valid contours
    x_min = min([cv2.boundingRect(c)[0] for c in valid_contours])
    y_min = min([cv2.boundingRect(c)[1] for c in valid_contours])
    x_max = max([cv2.boundingRect(c)[0] + cv2.boundingRect(c)[2] for c in valid_contours])
    y_max = max([cv2.boundingRect(c)[1] + cv2.boundingRect(c)[3] for c in valid_contours])

    # Apply buffer
    x_min, x_max = max(0, x_min - buffer), min(w, x_max + buffer)
    y_min, y_max = max(0, y_min - buffer), min(h, y_max + buffer)

    # Compute width and height
    width, height = x_max - x_min, y_max - y_min

    # Determine max side length for square crop
    max_side = max(width, height)

    # Center the crop area
    x_center = (x_min + x_max) // 2
    y_center = (y_min + y_max) // 2

    # Compute new square crop boundaries
    x_min = max(0, x_center - max_side // 2)
    x_max = min(w, x_min + max_side)

    y_min = max(0, y_center - max_side // 2)
    y_max = min(h, y_min + max_side)

    
    return [y_min, y_max, x_min, x_max]

oct_directory = r"C:\SleepApnea\test_images"
mask_directory = r"C:\SleepApnea\mask_images"
oct_output_dir = r"C:\SleepApnea\oct_processed_output"
msk_output_dir = r"C:\SleepApnea\mask_processed_output"


oct_image_paths = []
msk_image_paths = []
for dirpath, dirnames, filenames in os.walk(oct_directory):
    for file in filenames:
        filepath = oct_directory + "\\" + file
        oct_image_paths.append(filepath)

for dirpath, dirnames, filenames in os.walk(mask_directory):
    for file in filenames:
        filepath = mask_directory + "\\" + file
        msk_image_paths.append(filepath)
    
for i in range(len(oct_image_paths)):
    oct_output_path = oct_output_dir + "\\" + str(i).zfill(4) + ".png"
    msk_output_path = msk_output_dir + "\\" + str(i).zfill(4) + ".png"
    preprocess_image(oct_image_paths[i], msk_image_paths[i], oct_output_path, msk_output_path)