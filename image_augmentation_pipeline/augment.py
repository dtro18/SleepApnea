# horiz flip, vertical flip, rotation 90, rotation 180, rotation 270 

from PIL import Image

# Open the image
img = Image.open(r"C:\SleepApnea\0000.png")

# Reflect along the x-axis (horizontal flip)
img_x_reflect = img.transpose(Image.FLIP_LEFT_RIGHT)

# Reflect along the y-axis (vertical flip)
img_y_reflect = img.transpose(Image.FLIP_TOP_BOTTOM)

img_rotate_90 = img.transpose(Image.ROTATE_90)  # Rotates the image by 90 degrees
img_rotate_180 = img.transpose(Image.ROTATE_180)  # Rotates the image by 180 degrees
img_rotate_270 = img.transpose(Image.ROTATE_270)  # Rotates the image by 270 degrees

# Show the reflected images
# img_x_reflect.show()
# img_y_reflect.show()
img.show()
img_rotate_90.show()
img_rotate_180.show()
img_rotate_270.show()
