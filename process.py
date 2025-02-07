import numpy as np
import random
from PIL import Image
import matplotlib.pyplot as plt
import os

def preprocess_image(image_path, output_folder="/set_1_masked/", intensity_threshold=100, center_radius=60, mask_size=(250, 250)):
    image = Image.open(image_path).convert("L")  # Convert to grayscale
    width, height = image.size
    image_np = np.array(image)
    
    # Apply intensity threshold to retain only bright structures (ring)
    image_np[image_np < intensity_threshold] = 0
    
    # Remove central circular region (probe area)
    cx, cy = width // 2, height // 2
    y, x = np.ogrid[:height, :width]
    mask = (x - cx) ** 2 + (y - cy) ** 2 <= center_radius ** 2
    image_np[mask] = 0
    
    # Define middle 40% region
    mid_x_start, mid_x_end = int(0.3 * width), int(0.7 * width)
    mid_y_start, mid_y_end = int(0.3 * height), int(0.7 * height)
    
    # Select a random mask position inside the middle 40% of the image
    mask_x = random.randint(mid_x_start, mid_x_end - mask_size[0])
    mask_y = random.randint(mid_y_start, mid_y_end - mask_size[1])
    
    # Apply the mask (black out region)
    image_np[mask_y:mask_y + mask_size[1], mask_x:mask_x + mask_size[0]] = 0  
    
    processed_image = Image.fromarray(image_np)

    # Save the image in BMP format
    processed_image.save(output_path, format="BMP")
    print(f"Processed image saved as {output_path}")

    # Display the processed image
    # plt.imshow(processed_image, cmap="gray")
    # plt.axis("off")  # Hide axes for better visualization
    # plt.show()

    return processed_image

# preprocess_image(r"D:\SleepApneaPtData\set_1_complete\Frame0000.bmp")


working_directory = r"C:\SleepApnea\rotated_images_raw"
output_directory = r"C:\SleepApnea\oct_processed_incomplete"
# Assumes directory only contains images

num = 0
for dirpath, dirnames, filenames in os.walk(working_directory):
    for file in filenames:
        filepath = working_directory + "\\" + file
        output_path = output_directory + "\\" + str(num).zfill(4) + ".bmp"
        preprocess_image(filepath, output_path)
        num += 1