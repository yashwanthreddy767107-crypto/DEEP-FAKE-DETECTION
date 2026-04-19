from flask import Flask, render_template, request, jsonify
import os
import cv2
import numpy as np
import base64
import tensorflow as tf
from tensorflow.keras.models import load_model
import librosa
import joblib
from werkzeug.utils import secure_filename

app = Flask(__name__)

# -------- FOLDERS --------
UPLOAD_FOLDER = "static/uploads"
OUTPUT_FOLDER = "static/outputs"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# -------- MODELS --------
image_model = load_model("deepfake_model.h5")
voice_model = joblib.load("voice_model.pkl")

# -------- PREPROCESS --------
def preprocess(img):
    img = cv2.resize(img, (224,224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# -------- RESULT --------
def get_result(pred):
    real = pred[0][0]
    fake = pred[0][1]
    return f"REAL ✅ ({real*100:.2f}%)" if real > fake else f"FAKE ❌ ({fake*100:.2f}%)"

# -------- HEATMAP --------
def generate_heatmap(img_array):
    last_conv_layer = "Conv_1"

    grad_model = tf.keras.models.Model(
        [image_model.inputs],
        [image_model.get_layer(last_conv_layer).output, image_model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, tf.argmax(predictions[0])]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.reduce_max(heatmap)
    return heatmap.numpy()

# -------- HOME --------
@app.route('/')
def home():
    return render_template("index.html")

# -------- IMAGE --------
@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    filename = secure_filename(file.filename)

    path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(path)

    img = cv2.imread(path)
    processed = preprocess(img)

    pred = image_model.predict(processed)
    result = get_result(pred)

    # heatmap
    heatmap = generate_heatmap(processed)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    final = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

    heatmap_name = "heatmap_" + filename
    cv2.imwrite(os.path.join(UPLOAD_FOLDER, heatmap_name), final)

    return render_template("index.html",
                           prediction=result,
                           filename=filename,
                           heatmap="uploads/" + heatmap_name)

# -------- VIDEO --------
@app.route('/predict_video', methods=['POST'])
def predict_video():
    file = request.files['video']
    filename = secure_filename(file.filename)

    path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(path)

    cap = cv2.VideoCapture(path)

    real, fake = 0, 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = preprocess(frame)
        pred = image_model.predict(frame)

        if pred[0][0] > pred[0][1]:
            real += 1
        else:
            fake += 1

    cap.release()

    result = "REAL VIDEO ✅" if real > fake else "FAKE VIDEO ❌"

    return render_template("index.html", prediction=result)

# -------- VOICE --------
@app.route('/predict_voice', methods=['POST'])
def predict_voice():
    file = request.files['audio']
    filename = secure_filename(file.filename)

    path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(path)

    audio, sr = librosa.load(path, sr=None)
    mfcc = np.mean(librosa.feature.mfcc(y=audio, sr=sr), axis=1)
    mfcc = mfcc.reshape(1,-1)

    pred = voice_model.predict(mfcc)

    result = "REAL VOICE ✅" if pred[0] == 0 else "FAKE VOICE ❌"

    return render_template("index.html", prediction=result)

# -------- WEBCAM --------
@app.route('/webcam')
def webcam():
    return render_template("webcam.html")

@app.route('/webcam_predict', methods=['POST'])
def webcam_predict():
    data = request.json['image']

    image_data = data.split(",")[1]
    image_bytes = base64.b64decode(image_data)

    np_arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    processed = preprocess(img)
    pred = image_model.predict(processed)

    return jsonify({"result": get_result(pred)})

# -------- HEATMAP WEBCAM --------
@app.route('/webcam_heatmap')
def webcam_heatmap():
    return render_template("webcam_heatmap.html")

@app.route('/webcam_heatmap_predict', methods=['POST'])
def webcam_heatmap_predict():
    data = request.json['image']

    image_data = data.split(",")[1]
    image_bytes = base64.b64decode(image_data)

    np_arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    processed = preprocess(img)
    pred = image_model.predict(processed)
    result = get_result(pred)

    heatmap = generate_heatmap(processed)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    final = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

    _, buffer = cv2.imencode('.jpg', final)
    img_base64 = base64.b64encode(buffer).decode()

    return jsonify({
        "result": result,
        "image": "data:image/jpeg;base64," + img_base64
    })

# -------- RUN --------
if __name__ == "__main__":
    app.run(debug=True)