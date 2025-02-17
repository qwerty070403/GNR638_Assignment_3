from tensorflow.keras.callbacks import ModelCheckpoint
from Load_Preprocess_Dataset import train_paths, train_gen, val_gen, val_paths, test_paths, test_gen
from build_model import model
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

EPOCHS = 25

# Save the best model based on validation accuracy
checkpoint = ModelCheckpoint("best_model.h5", monitor="val_accuracy", save_best_only=True, mode="max")

#Training part given below
history = model.fit(
    train_gen,
    steps_per_epoch=len(train_paths) // BATCH_SIZE,
    validation_data=val_gen,
    validation_steps=len(val_paths) // BATCH_SIZE,
    epochs=EPOCHS,
    callbacks=[checkpoint]  # Save best model
)


# Extract accuracy and validation accuracy
train_acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
epochs = range(1, len(train_acc) + 1)

# Plot accuracy vs. epoch
plt.figure(figsize=(8, 5))
plt.plot(epochs, train_acc, label="Training Accuracy", marker='o')
plt.plot(epochs, val_acc, label="Validation Accuracy", marker='o')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Training and Validation Accuracy")
plt.legend()
plt.grid(True)
plt.show()

#Testing Part given below (Test for model with best Val accuracy)
# Load the best saved model
best_model = load_model("best_model.h5")

# Evaluate on test set
test_loss, test_accuracy = best_model.evaluate(test_gen, steps=len(test_paths) // BATCH_SIZE)

print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

