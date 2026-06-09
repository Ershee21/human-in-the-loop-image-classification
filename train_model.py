# Used for building, training, evaluating, and saving
# the Convolutional Neural Network (CNN) model
import tensorflow as tf 

# Keras Layers and Models Module
# layers - used to create CNN layers such as Conv2D,
# MaxPooling, Flatten, Dense, and Dropout
# models - used to create the Sequential CNN architecture
from tensorflow.keras import layers, models

# NumPy Library
# Used for numerical computations, array manipulation,
# image preprocessing, and handling prediction outputs
import numpy as np


#initializing the dataset training path
dataset_path = "dataset/train"

img_size = (128, 128)
batch_size = 32

train_data = tf.keras.utils.image_dataset_from_directory(
    dataset_path,
    image_size=img_size,
    batch_size=batch_size,
    validation_split=0.2,
    subset="training",
    seed=42
)

val_data = tf.keras.utils.image_dataset_from_directory(
    dataset_path,
    image_size=img_size,
    batch_size=batch_size,
    validation_split=0.2,
    subset="validation",
    seed=42
)

class_names = train_data.class_names
print("Classes:", class_names)

#list of layers in the model
model = models.Sequential(
    [
    #input layer / normalizatoin layer
    #Converts pixel values from 0-255 to 0-1 for faster and more stable training
    layers.Rescaling(1./255, input_shape=(128, 128, 3)),

    #Convolutional Layer 1 + ReLU Activation
    layers.Conv2D(
        filters=32,
        kernel_size=3,
        activation='relu'
    ),

    #Pooling Layer 1
    layers.MaxPooling2D(),

    #Convolutional Layer 2 + ReLU Activation
    layers.Conv2D(
        filters=63,
        kernel_size=3,
        activation='relu'
    ),

    #Pooling Layer 2
    layers.MaxPooling2D(),

    #Convolutional Layer 3 + ReLU Activation
    layers.Conv2D(
        filters=128,
        kernel_size=3,
        activation='relu'
    ),

    #Pooling Layer 3
    layers.MaxPooling2D(),


    layers.Flatten(), #flatten layer / Converts the 2D feature maps into a 1D vector
    layers.Dense(128, activation='relu'), #fully connected layer with 128 neurons and ReLU activation

    # Dropout Layer
    layers.Dropout(0.5),

    #output layer with softmax activation for multi-class classification
    layers.Dense(len(class_names), activation='softmax') 
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Load Test Dataset
test_data = tf.keras.utils.image_dataset_from_directory(
    "dataset/test",
    image_size=img_size,
    batch_size=batch_size,
    shuffle=False
)

"""
#calls the model to train on the training data and validate on the validation data for 10 epochs
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=10
)

print("\n" + "="*50)
print("TRAINING RESULTS")
print("="*50)

print(f"Final Training Accuracy: {history.history['accuracy'][-1] * 100:.2f}%")
print(f"Final Validation Accuracy: {history.history['val_accuracy'][-1] * 100:.2f}%")
"""

# Load previously trained model
model = tf.keras.models.load_model("plant_disease_model.h5")

# Testing the model on the test dataset to evaluate its performance on unseen data
test_loss, test_acc = model.evaluate(test_data)

# Printing sample predictions for the first batch of test data
print("\n\nSAMPLE PREDICTIONS")

# Get one batch from test dataset
for images, labels in test_data.take(1):

    predictions = model.predict(images)

    predicted_classes = np.argmax(predictions, axis=1)

    for i in range(min(20, len(images))):

        actual_class = class_names[labels[i].numpy()]
        predicted_class = class_names[predicted_classes[i]]

        print(f"Image {i+1}")
        print(f"Actual:    {actual_class}")
        print(f"Predicted: {predicted_class}")
        print("-" * 40)

print("\n\nTEST RESULTS")
print(f"Test Accuracy: {test_acc * 100:.2f}%")
print(f"Test Loss: {test_loss:.4f}")

model.save("plant_disease_model.h5")

# Save class names
with open("class_names.txt", "w") as f:
    for name in class_names:
        f.write(name + "\n")