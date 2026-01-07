import os
import json
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten,
    Dense, Dropout, BatchNormalization
)
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# ===============================
# CONFIG
# ===============================
IMG_SIZE = 48
BATCH_SIZE = 32
EPOCHS = 25

TRAIN_DIR = "data/train"
VAL_DIR = "data/val"
MODEL_DIR = "model"

os.makedirs(MODEL_DIR, exist_ok=True)

# ===============================
# DATA GENERATORS
# ===============================
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1
)

val_datagen = ImageDataGenerator(
    rescale=1./255
)

train_gen = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=True
)

val_gen = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

NUM_CLASSES = train_gen.num_classes
print("Number of classes:", NUM_CLASSES)

# ===============================
# SAVE CLASS INDEX MAPPING
# ===============================
class_indices = train_gen.class_indices
with open(os.path.join(MODEL_DIR, "class_indices.json"), "w") as f:
    json.dump(class_indices, f, indent=4)

print("Class indices saved")

# ===============================
# CNN MODEL
# ===============================
model = Sequential([
    Conv2D(32, (3,3), activation="relu", input_shape=(IMG_SIZE, IMG_SIZE, 1)),
    BatchNormalization(),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation="relu"),
    BatchNormalization(),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation="relu"),
    BatchNormalization(),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(256, activation="relu"),
    Dropout(0.5),
    Dense(NUM_CLASSES, activation="softmax")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ===============================
# CALLBACKS
# ===============================
checkpoint = ModelCheckpoint(
    filepath=os.path.join(MODEL_DIR, "asl_cnn_best.h5"),
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

# ===============================
# TRAIN
# ===============================
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=[checkpoint, early_stop]
)

# ===============================
# SAVE FINAL MODEL
# ===============================
model.save(os.path.join(MODEL_DIR, "asl_cnn_final.h5"))
print(" Training completed and model saved")
