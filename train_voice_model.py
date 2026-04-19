import os
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

data = []
labels = []

real_path = "voice_dataset/real"
fake_path = "voice_dataset/fake"

def extract_features(file):
    audio, sr = librosa.load(file, sr=None)
    mfccs = np.mean(librosa.feature.mfcc(y=audio, sr=sr), axis=1)
    return mfccs

# REAL = 0
for file in os.listdir(real_path):
    path = os.path.join(real_path, file)
    try:
        features = extract_features(path)
        data.append(features)
        labels.append(0)
    except:
        print("Error:", file)

# FAKE = 1
for file in os.listdir(fake_path):
    path = os.path.join(fake_path, file)
    try:
        features = extract_features(path)
        data.append(features)
        labels.append(1)
    except:
        print("Error:", file)

X = np.array(data)
y = np.array(labels)

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Accuracy
accuracy = model.score(X_test, y_test)
print("Accuracy:", accuracy)

# Save model
joblib.dump(model, "voice_model.pkl")

print("✅ Voice model trained successfully")