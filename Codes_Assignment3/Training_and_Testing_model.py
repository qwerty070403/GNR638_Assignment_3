from tensorflow.keras.callbacks import ModelCheckpoint
from Load_Preprocess_Dataset import train_paths, train_gen, val_gen, val_paths, test_paths, test_gen
from build_model import model
from tensorflow.keras.models import load_model

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


#Testing Part given below (Test for model with best Val accuracy)
# Load the best saved model
best_model = load_model("best_model.h5")

# Evaluate on test set
test_loss, test_accuracy = best_model.evaluate(test_gen, steps=len(test_paths) // BATCH_SIZE)

print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

