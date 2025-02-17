from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Dense, Dropout, GlobalAveragePooling2D, LeakyReLU
import tensorflow as tf

# Default choice best Test Accuracy Model is Given below (Model 7 of LeakyReLU as mentioned in README.md)
def build_model(input_shape=(128, 128, 3), num_classes=21):
    model = keras.Sequential([
        Conv2D(32, (1,1), padding="same", input_shape=input_shape), #make changes to kernel size as per wish, choose between(1x1,3x3,5x5)
        BatchNormalization(),
        LeakyReLU(alpha=0.01),
        MaxPooling2D(pool_size=(2,2)),

        Conv2D(64, (3,3), padding="same"), #make changes to kernel size as per wish, choose between(1x1,3x3,5x5)
        BatchNormalization(),
        LeakyReLU(alpha=0.01),  #For Activation to be ReLU Replace this Line with Activation("relu"),
        MaxPooling2D(pool_size=(2,2)),

        Conv2D(128, (3,3), padding="same"),  #make changes to kernel size as per wish, choose between(1x1,3x3,5x5)
        BatchNormalization(),
        LeakyReLU(alpha=0.01),  #For Activation to be ReLU Replace this Line with Activation("relu"),
        MaxPooling2D(pool_size=(2,2)),

        GlobalAveragePooling2D(),
        Dense(128),
        LeakyReLU(alpha=0.01), #For Activation to be ReLU Replace this Line with Activation("relu"),
        Dropout(0.3),  #Choose an optimal Dropout Ration
        Dense(num_classes, activation="softmax")
    ])

    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


# Build model
model = build_model()
model.summary()
