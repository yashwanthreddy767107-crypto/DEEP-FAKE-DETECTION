import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os

# Load model
model = load_model("deepfake_model.h5")

# Video path (safe)
video_path = 0

# Check file exists
print("📂 Checking video file...")
if not os.path.exists(video_path):
    print("❌ Error: Video file not found")
    print("👉 Check file name and location")
    exit()
else:
    print("✅ Video found")

# Open video
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("❌ Error: Cannot open video")
    exit()

print("🎬 Processing video... Press 'q' to quit")

# Face detector (better accuracy)
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]

        # Preprocess
        face = cv2.resize(face, (64, 64))
        face = face / 255.0
        face = np.reshape(face, (1, 64, 64, 3))

        # Predict
        pred = model.predict(face, verbose=0)

        real_score = float(pred[0][0]) * 100
        fake_score = float(pred[0][1]) * 100

        if real_score > fake_score:
            label = f"REAL ({real_score:.1f}%)"
            color = (0, 255, 0)
        else:
            label = f"FAKE ({fake_score:.1f}%)"
            color = (0, 0, 255)

        # Draw box + label
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, label, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # Show video
    cv2.imshow("Deepfake Video Detection", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print("✅ Done")