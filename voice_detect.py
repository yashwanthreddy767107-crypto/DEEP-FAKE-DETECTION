import librosa
import numpy as np
import os

file_path = r"C:\Users\yashw\Desktop\deepfake-detector\voice.mp3"

print("📂 Checking file...")

if not os.path.exists(file_path):
    print("❌ File not found")
    exit()
else:
    print("✅ File found")

try:
    audio, sr = librosa.load(file_path, sr=None)
except Exception as e:
    print("❌ Error loading audio:", e)
    exit()

mfccs = np.mean(librosa.feature.mfcc(y=audio, sr=sr), axis=1)

print("🎵 Features extracted")

if np.mean(mfccs) > 0:
    print("REAL VOICE ✅")
else:
    print("FAKE VOICE ❌")