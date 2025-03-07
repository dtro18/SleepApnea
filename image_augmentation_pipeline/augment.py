'''
Script to augment a folder of images by applying transformtions.
Input: Image directory to walk through
Output: New image directory with augmented images (including the original)

'''
import os

IMAGE_DIR = r"C:\SleepApnea\trial1_421_images\msk_processed_imgs"
OUTPUT_DIR = r"C:\SleepApnea\trial1_421_images\msk_augmented_imgs"
global_naming_count = 0

os.makedirs(OUTPUT_DIR, exist_ok=True)

from PIL import Image
def augment_image(image_path, outputdir):
    global global_naming_count
    # Open the image
    img = Image.open(image_path)
    # Save original copy
    
    transformations = [
        img,  # Original
        img.transpose(Image.FLIP_LEFT_RIGHT),  # Flip X
        img.transpose(Image.FLIP_TOP_BOTTOM),  # Flip Y
        img.transpose(Image.ROTATE_90),  # Rotate 90°
        img.transpose(Image.ROTATE_180),  # Rotate 180°
        img.transpose(Image.ROTATE_270)  # Rotate 270°
    ]
    
    # index = start_idx  # Start numbering from the given index

    for transformed_img in transformations:
        filename = os.path.join(outputdir, f"{global_naming_count:04d}.png")  # Zero-padded name
        transformed_img.save(filename, format="PNG")
        global_naming_count += 1


for dirpath, dirnames, filenames in os.walk(IMAGE_DIR):
    for file in filenames:
        filepath = IMAGE_DIR + "\\" + file
        augment_image(filepath, OUTPUT_DIR)