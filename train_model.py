import os
import cv2
import numpy as np
import json
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

IMG_SIZE = 224

real_path = "dataset/real"
fake_path = "dataset/fake"

data = []
labels = []

# -------- LOAD IMAGES --------
def load_images(folder, label):
    for img in os.listdir(folder):
        path = os.path.join(folder, img)
        image = cv2.imread(path)

        if image is None:
            continue

        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
        data.append(image)
        labels.append(label)

# ✅ FIXED LABELS
load_images(real_path, 0)   # real = 0
load_images(fake_path, 1)   # fake = 1

X = np.array(data) / 255.0
y = to_categorical(labels, 2)

# -------- SPLIT --------
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------- AUGMENTATION --------
datagen = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

datagen.fit(X_train)

# -------- MODEL --------
base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)

for layer in base_model.layers:
    layer.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(2, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# -------- TRAIN --------
model.fit(
    datagen.flow(X_train, y_train, batch_size=32),
    validation_data=(X_val, y_val),
    epochs=10
)

# -------- SAVE --------
model.save("deepfake_model.h5")

# -------- SAVE LABELS --------
labels_dict = {"real": 0, "fake": 1}

with open("labels.json", "w") as f:
    json.dump(labels_dict, f)

print("✅ Model trained successfully")
print("Labels:", labels_dict)