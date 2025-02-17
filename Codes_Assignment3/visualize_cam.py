import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Conv2D, Input, BatchNormalization, Activation, MaxPooling2D, GlobalAveragePooling2D, Dense, Dropout, LeakyRelU
import cv2

# Choose the best model below
# *** Rebuild the Model Exactly As Trained ***
def build_model(input_shape=(128, 128, 3), num_classes=21):
    inputs = Input(shape=input_shape)  # EXPLICIT INPUT LAYER

    # Conv Block 1
    x = Conv2D(32, (1, 1), padding="same")(inputs)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.01)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Conv Block 2
    x = Conv2D(64, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.01)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Conv Block 3
    x = Conv2D(128, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.01)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Flatten and Fully Connected Layers
    x = GlobalAveragePooling2D()(x)
    x = Dense(128)(x)
    x = LeakyReLU(alpha=0.01)(x)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation="softmax")(x)  # Output layer

    # Create model
    model = Model(inputs=inputs, outputs=outputs)

    # Compile model
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

# *** Build the Model and Load Weights ***
model = build_model()
model.load_weights("best_model_7.h5")  # Load saved weights
model.summary()  # PRINT SUMMARY TO CONFIRM LAYER NAMES

# *** Force Initialization with Dummy Input ***
dummy_input = np.zeros((1, 128, 128, 3), dtype=np.float32)
model.predict(dummy_input)  # Make a dummy prediction to initialize inputs

# *** Select a Test Image ***
test_img_path = test_paths[1]  # Change index for a different image
test_img = tf.keras.preprocessing.image.load_img(test_img_path, target_size=(128, 128))
test_img_array = tf.keras.preprocessing.image.img_to_array(test_img)
test_img_array = np.expand_dims(test_img_array, axis=0) / 255.0  # Normalize

# *** Get the Last Conv2D Layer Properly ***
# Get all layers that are instances of Conv2D
conv_layers = [layer for layer in model.layers if isinstance(layer, Conv2D)]
if not conv_layers:
    raise ValueError("No Conv2D layers found in the model.")
last_conv_layer = conv_layers[-1]  # Get the last Conv2D layer
print(f"Last Conv2D Layer Name: {last_conv_layer.name}")  # PRINT LAYER NAME FOR CONFIRMATION

# *** Create CAM Model ***
# Model to map input image → conv layer output + final prediction
cam_model = Model(inputs=model.input, outputs=[last_conv_layer.output, model.output])

# *** Get Feature Maps & Predictions ***
conv_output, preds = cam_model.predict(test_img_array)
pred_class = np.argmax(preds[0])  # Predicted class

# *** Compute Gradients of Predicted Class w.r.t Last Conv Layer Output ***
with tf.GradientTape() as tape:
    conv_output, preds = cam_model(test_img_array)
    loss = preds[:, pred_class]  # Class score
grads = tape.gradient(loss, conv_output)

# *** Compute the Mean Intensity of Gradients ***
weights = tf.reduce_mean(grads, axis=(0, 1, 2))

# *** Compute Weighted Activation Map ***
activation_map = np.dot(conv_output[0], weights.numpy().reshape(-1, 1))
activation_map = np.squeeze(activation_map)

# *** Normalize & Resize to Match Input Size ***
activation_map = cv2.resize(activation_map, (128, 128))
activation_map = np.maximum(activation_map, 0)  # ReLU-like
activation_map /= np.max(activation_map)  # Normalize

# *** Overlay Heatmap on the Original Image ***
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.imshow(test_img)
plt.axis("off")
plt.title("Original Image")

plt.subplot(1, 2, 2)
plt.imshow(test_img)
plt.imshow(activation_map, cmap="jet", alpha=0.5)  # Blend heatmap
plt.axis("off")
plt.title(f"Grad-CAM (Class: {pred_class})")

plt.show()
