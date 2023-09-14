import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

# Function to load and preprocess image data
def load_and_preprocess_data(data_path, label, image_size=(128, 128)):
    data = []
    labels = []
    
    # Loop through image files in the given path
    for img_file in os.listdir(data_path):
        img = Image.open(os.path.join(data_path, img_file))
        img = img.resize(image_size)
        img = img.convert('RGB')
        # Normalize pixel values
        img = np.array(img) / 255.0  
        data.append(img)
        labels.append(label)
    
    return data, labels

# Load and preprocess the data for images with masks
with_mask_data, with_mask_labels = load_and_preprocess_data('data/with_mask', 1)

# Load and preprocess the data for images without masks
without_mask_data, without_mask_labels = load_and_preprocess_data('data/without_mask', 0)

# Combine data and labels
data = with_mask_data + without_mask_data
labels = with_mask_labels + without_mask_labels

# Convert data and labels to NumPy arrays
X = np.array(data)
Y = np.array(labels)

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

# Create an image data generator for data augmentation
datagen = ImageDataGenerator(
    # Rotate images randomly
    rotation_range=20, 
    # Shift images horizontally 
    width_shift_range=0.2,  
    # Shift images vertically
    height_shift_range=0.2,  
    # Flip images horizontally
    horizontal_flip=True,  
    # Zoom in on images randomly
    zoom_range=0.2  
)

# Define the model architecture
num_of_classes = 2

model = keras.Sequential([
    keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(128, 128, 3)),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),
    keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(num_of_classes, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['acc'])

# Train the model using data augmentation
model.fit(datagen.flow(X_train, Y_train, batch_size=32), validation_data=(X_test, Y_test), epochs=5)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, Y_test)
print('Test Accuracy =', accuracy)

# Save the trained model to the root folder
model.save('mask-detection-model.h5')