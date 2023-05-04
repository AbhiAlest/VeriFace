import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.utils.class_weight import compute_class_weight

# Define actual faces and deepfaked faces directories
actual_faces_dir = '/path/to/actual_faces'
deepfaked_faces_dir = '/path/to/deepfaked_faces'

# Load actual faces
X_actual = []
y_actual = []
for filename in os.listdir(actual_faces_dir):
    image_path = os.path.join(actual_faces_dir, filename)
    image = load_img(image_path, target_size=(224, 224))
    image = img_to_array(image)
    X_actual.append(image)
    y_actual.append(0)  # label 0 for actual faces

# Load deepfaked faces
X_deepfake = []
y_deepfake = []
for filename in os.listdir(deepfaked_faces_dir):
    image_path = os.path.join(deepfaked_faces_dir, filename)
    image = load_img(image_path, target_size=(224, 224))
    image = img_to_array(image)
    X_deepfake.append(image)
    y_deepfake.append(1)  # label 1 for deepfaked faces

# Combine data
X = np.concatenate([X_actual, X_deepfake], axis=0)
y = np.concatenate([y_actual, y_deepfake], axis=0)

# Convert to numpy arrays
X = np.array(X)
y = np.array(y)

# Split data into training, validation, and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, stratify=y_train, random_state=42)

# Calculate class weights to handle imbalanced classes
class_weights = compute_class_weight('balanced', np.unique(y_train), y_train)

# Define the model
def create_model(learning_rate=0.001, dropout_rate=0.5, filters=64, kernel_size=3, units=128):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, activation='relu', input_shape=(224, 224, 3)),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(filters=filters//2, kernel_size=kernel_size, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units=units, activation='relu'),
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Dense(units=1, activation='sigmoid')
    ])
    optimizer = tf.keras.optimizers.Adam(lr=learning_rate)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

# Hyperparameter tuning with grid search
param_grid = {
    'learning_rate': [0.001, 0.01, 0.1],
    'dropout_rate': [0.2, 0.3, 0.4],
    'filters': [32, 64, 128],
    'kernel_size': [3, 5],
    'units': [64, 128, 256]
}
model = tf.keras.wrappers.scikit_learn.K
