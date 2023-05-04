import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

actual_faces_path = '/path/to/actual_faces/'
deepfaked_faces_path = '/path/to/deepfaked_faces/'

img_size = 128

def preprocess_image(img):
    img = cv2.resize(img, (img_size, img_size))
    img = img.astype('float32') / 255.0
    img -= np.mean(img, axis=(0, 1))
    return img

actual_faces = []
for filename in os.listdir(actual_faces_path):
    img = cv2.imread(os.path.join(actual_faces_path, filename))
    img = preprocess_image(img)
    actual_faces.append(img)
actual_faces = np.array(actual_faces)

deepfaked_faces = []
for filename in os.listdir(deepfaked_faces_path):
    img = cv2.imread(os.path.join(deepfaked_faces_path, filename))
    img = preprocess_image(img)
    deepfaked_faces.append(img)
deepfaked_faces = np.array(deepfaked_faces)

X_train, X_val, y_train, y_val = train_test_split(
    np.concatenate([actual_faces, deepfaked_faces]), 
    np.concatenate([np.ones(len(actual_faces)), np.zeros(len(deepfaked_faces))]),
    test_size=0.3,
    random_state=42
)
