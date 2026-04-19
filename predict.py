import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load model
model = load_model("deepfake_model.h5")

# Image path
image_path = "test.jpg"

# Check if file exists
if not os.path.exists(image_path):
    print("❌ Error: test.jpg not found in folder")
    exit()

# Read image
image = cv2.imread(image_path)

# Check if image loaded properly
if image is None:
    print("❌ Error: Unable to read image (wrong format or corrupted)")
    exit()

# Resize and preprocess
image = cv2.resize(image, (64, 64))
image = image / 255.0
image = np.reshape(image, (1, 64, 64, 3))

# Predict
prediction = model.predict(image)

# Output result
if prediction[0][0] > prediction[0][1]:
    print("✅ REAL IMAGE")
else:
    print("❌ FAKE IMAGE")