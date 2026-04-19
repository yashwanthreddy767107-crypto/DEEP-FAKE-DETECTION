import librosa
import numpy as np
import joblib

model = joblib.load("voice_model.pkl")

file = "voice.mp3"

audio, sr = librosa.load(file, sr=None)
mfccs = np.mean(librosa.feature.mfcc(y=audio, sr=sr), axis=1)
mfccs = mfccs.reshape(1, -1)

prediction = model.predict(mfccs)

if prediction[0] == 0:
    print("REAL VOICE ✅")
else:
    print("FAKE VOICE ❌")