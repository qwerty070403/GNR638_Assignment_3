import os
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Dropout
from tensorflow.keras.layers import Flatten, Dense, Activation
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from google.colab import drive

# Check GPU availability
print("Num GPUs Available:", len(tf.config.experimental.list_physical_devices('GPU')))


# Mount Google Drive
drive.mount('/content/drive')

# Dataset path
dataset_path = "/content/drive/MyDrive/Images"

# Image parameters
IMG_SIZE = (128, 128)  # Resize images
BATCH_SIZE = 32
NUM_CLASSES = 21  # UC Merced has 21 classes

# Get image paths and labels
valid_extensions = ('.tif', '.tiff', '.png', '.jpg', '.jpeg')
image_paths, labels = [], []

for root, dirs, files in os.walk(dataset_path):
    for file in files:
        if file.lower().endswith(valid_extensions):
            image_paths.append(os.path.join(root, file))
            labels.append(os.path.basename(root))  # Folder name as label

# Convert labels to numeric encoding
label_to_index = {label: i for i, label in enumerate(sorted(set(labels)))}
labels = np.array([label_to_index[label] for label in labels])

# Split dataset
train_paths, temp_paths, train_labels, temp_labels = train_test_split(image_paths, labels, test_size=0.3, random_state=42)
val_paths, test_paths, val_labels, test_labels = train_test_split(temp_paths, temp_labels, test_size=2/3, random_state=42)

print(f"Train: {len(train_paths)}, Val: {len(val_paths)}, Test: {len(test_paths)}")

# Data generator function
def image_generator(paths, labels, batch_size, shuffle=True):
    while True:
        indices = np.arange(len(paths))
        if shuffle:
            np.random.shuffle(indices)

        for start in range(0, len(paths), batch_size):
            end = min(start + batch_size, len(paths))
            batch_paths = [paths[i] for i in indices[start:end]]
            batch_labels = [labels[i] for i in indices[start:end]]

            batch_images = np.array([keras.preprocessing.image.load_img(p, target_size=IMG_SIZE) for p in batch_paths])
            batch_images = np.array([keras.preprocessing.image.img_to_array(img) / 255.0 for img in batch_images])

            yield batch_images, np.array(batch_labels)

# Create generators
train_gen = image_generator(train_paths, train_labels, BATCH_SIZE)
val_gen = image_generator(val_paths, val_labels, BATCH_SIZE)
test_gen = image_generator(test_paths, test_labels, BATCH_SIZE, shuffle=False)
