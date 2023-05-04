import os
import random
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Reshape, Conv2D, MaxPooling2D, Flatten, Dense, concatenate, Dropout, LSTM, RNN
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from sklearn.utils import class_weight
from sklearn.model_selection import GridSearchCV

# Set paths
actual_faces_path = '/path/to/actual_faces/'
deepfaked_faces_path = '/path/to/deepfaked_faces/'

# Load image data and labels
X = []
y = []
for img_path in os.listdir(actual_faces_path):
    img = load_img(os.path.join(actual_faces_path, img_path), target_size=(128, 128))
    X.append(img_to_array(img))
    y.append(0)  # 0 for actual faces
for img_path in os.listdir(deepfaked_faces_path):
    img = load_img(os.path.join(deepfaked_faces_path, img_path), target_size=(128, 128))
    X.append(img_to_array(img))
    y.append(1)  # 1 for deepfaked faces
X = np.array(X)
y = np.array(y)

# Split data to training, validation, and test sets
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42, stratify=y_train_val)

# Calculate class weights for handling imbalanced classes
class_weights = class_weight.compute_class_weight('balanced', np.unique(y_train), y_train)

# Hyperparameter tuning using grid search
param_grid = {
    'lr': [0.001, 0.01, 0.1],
    'batch_size': [32, 64, 128],
    'dropout_rate': [0.25, 0.5],
    'lstm_units': [32, 64],
    'rnn_units': [32, 64]
}
model = Sequential()
model.add(Reshape((128, 128, 3), input_shape=(128, 128, 3)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(concatenate([model.layers[-1].output, model.layers[-5].output]))
model.add(Dropout(0.5))
model.add(LSTM(64, return_sequences=True))
model.add(RNN(64))
model.add(LSTM(64))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))
optimizer = Adam(lr=0.001)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
grid_search = GridSearchCV(model, param_grid, cv=3, scoring='roc_auc', verbose=1, n_jobs=-1)
grid_search.fit(train_generator, steps_per_epoch=len(train_generator), validation_data=valid_generator, validation_steps=len(valid_generator), class_weight=class_weights)
                
# Train the model with the best hyperparameters
model.fit(train_generator, epochs=20, validation_data=valid_generator, callbacks=[early_stopping, checkpoint])

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(test_generator)
print("Test loss:", test_loss)
print("Test accuracy:", test_accuracy)

# Make predictions on the test set
y_pred = model.predict(test_generator)

# Calculate AUC
auc = roc_auc_score(test_generator.classes, y_pred)
print("AUC:", auc)
                
