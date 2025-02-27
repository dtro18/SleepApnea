gitimport os
import numpy as np
import tensorflow as tf
import cv2
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.callbacks import ModelCheckpoint

# Define the paths to your folders
complete_folder = 'C:\SleepApnea\oct_processed_complete'
incomplete_folder = 'C:\SleepApnea\oct_processed_incomplete'

# Image dimensions (you can change it depending on your dataset)
img_height = 256
img_width = 256
channels = 1  # Use 1 for grayscale, 3 for RGB

# Function to load images and preprocess them
def load_and_preprocess_image(image_path, target_size=(img_height, img_width)):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Read in grayscale mode
    img = cv2.resize(img, target_size)  # Resize to target size
    img = img.astype('float32') / 255.0  # Normalize pixel values
    img = np.expand_dims(img, axis=-1)  # Add channel dimension (H, W, 1)
    return img


# Load all image pairs (complete and incomplete)
def load_dataset(complete_folder, incomplete_folder):
    complete_images = []
    incomplete_images = []
    for filename in os.listdir(complete_folder):
        if filename.endswith(".bmp") or filename.endswith(".jpg"):
            complete_img_path = os.path.join(complete_folder, filename)
            incomplete_img_path = os.path.join(incomplete_folder, filename)
            
            complete_img = load_and_preprocess_image(complete_img_path)
            incomplete_img = load_and_preprocess_image(incomplete_img_path)
            
            complete_images.append(complete_img)
            incomplete_images.append(incomplete_img)

    return np.array(incomplete_images), np.array(complete_images)

# Load the dataset
X_train, y_train = load_dataset(complete_folder, incomplete_folder)

# Reshape arrays to match TensorFlow's expected input format
X_train = np.expand_dims(X_train, axis=-1)  # (num_samples, 256, 256, 1)
y_train = np.expand_dims(y_train, axis=-1)  # (num_samples, 256, 256, 1)

# U-Net Model Definition
def unet(input_size=(256, 256, 1)):  # Change channels from 3 to 1
    inputs = layers.Input(input_size)
    
    # Contracting path (downsampling)
    c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)
    
    c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)
    
    c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
    c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c3)
    p3 = layers.MaxPooling2D((2, 2))(c3)
    
    c4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(p3)
    c4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(c4)
    p4 = layers.MaxPooling2D((2, 2))(c4)
    
    c5 = layers.Conv2D(1024, (3, 3), activation='relu', padding='same')(p4)
    c5 = layers.Conv2D(1024, (3, 3), activation='relu', padding='same')(c5)
    
    # Expansive path (upsampling)
    u6 = layers.Conv2DTranspose(512, (3, 3), strides=(2, 2), padding='same')(c5)
    u6 = layers.concatenate([u6, c4])
    c6 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(u6)
    c6 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(c6)
    
    u7 = layers.Conv2DTranspose(256, (3, 3), strides=(2, 2), padding='same')(c6)
    u7 = layers.concatenate([u7, c3])
    c7 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(u7)
    c7 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c7)
    
    u8 = layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same')(c7)
    u8 = layers.concatenate([u8, c2])
    c8 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(u8)
    c8 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c8)
    
    u9 = layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(c8)
    u9 = layers.concatenate([u9, c1])
    c9 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u9)
    c9 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c9)
    
    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)  # Change to 1 channel
    
    model = models.Model(inputs, outputs)
    return model

# Create and compile the model
model = unet()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Training the model
checkpoint = ModelCheckpoint('unet_model.keras', monitor='val_loss', save_best_only=True)

history = model.fit(X_train, y_train, batch_size=16, epochs=50, validation_split=0.1, callbacks=[checkpoint])

# Optionally, evaluate the model or use it for inference