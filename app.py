from flask import Flask, render_template, request
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os

app = Flask(__name__)
model = load_model("deepfake_model.h5")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    path = os.path.join("uploads", file.filename)
    file.save(path)

    image = cv2.imread(path)

    if image is None:
        return "Error reading image"

    image = cv2.resize(image, (64, 64))
    image = image / 255.0
    image = np.reshape(image, (1, 64, 64, 3))

    prediction = model.predict(image)

    if prediction[0][0] > prediction[0][1]:
        result = "REAL IMAGE ✅"
    else:
        result = "FAKE IMAGE ❌"

    return render_template('index.html', prediction=result)

if __name__ == "__main__":
    app.run(debug=True)