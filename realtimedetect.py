import cv2
import numpy as np
import json
import tensorflow as tf
from collections import deque

# ===============================
# CONFIG
# ===============================
IMG_SIZE = 48
MODEL_PATH = "model/asl_cnn_best.h5"
CLASS_INDEX_PATH = "model/class_indices.json"

CONF_THRESHOLD = 0.75      # strong confidence gate
SMOOTHING_FRAMES = 12      # temporal smoothing

# ===============================
# LOAD MODEL
# ===============================
model = tf.keras.models.load_model(MODEL_PATH)

# ===============================
# LOAD CLASS INDICES
# ===============================
with open(CLASS_INDEX_PATH, "r") as f:
    class_indices = json.load(f)

index_to_class = {v: k for k, v in class_indices.items()}
print("✅ Model & class indices loaded")

# ===============================
# PREDICTION BUFFERS
# ===============================
pred_buffer = deque(maxlen=SMOOTHING_FRAMES)
conf_buffer = deque(maxlen=SMOOTHING_FRAMES)

# ===============================
# HAND PRESENCE CHECK
# ===============================
def hand_present(gray_img):
    _, thresh = cv2.threshold(gray_img, 60, 255, cv2.THRESH_BINARY)
    return cv2.countNonZero(thresh) > 800

# ===============================
# WEBCAM
# ===============================
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]

    # ROI (center)
    x1, y1 = w // 2 - 150, h // 2 - 150
    x2, y2 = w // 2 + 150, h // 2 + 150

    roi = frame[y1:y2, x1:x2]
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # ===============================
    # PREPROCESS
    # ===============================
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))

    # ===============================
    # HAND PRESENCE CHECK
    # ===============================
    if not hand_present(resized):
        pred_buffer.clear()
        conf_buffer.clear()
        text = "No sign"
        color = (0, 0, 255)
    else:
        normalized = resized / 255.0
        input_img = normalized.reshape(1, IMG_SIZE, IMG_SIZE, 1)

        preds = model.predict(input_img, verbose=0)[0]
        pred_index = np.argmax(preds)
        confidence = np.max(preds)

        pred_buffer.append(pred_index)
        conf_buffer.append(confidence)

        # Majority voting
        final_pred = max(set(pred_buffer), key=pred_buffer.count)
        avg_conf = sum(conf_buffer) / len(conf_buffer)

        if avg_conf >= CONF_THRESHOLD:
            label = index_to_class[final_pred]
            text = f"{label} ({avg_conf*100:.1f}%)"
            color = (0, 255, 0)
        else:
            text = "No sign"
            color = (0, 0, 255)

    # ===============================
    # DISPLAY
    # ===============================
    cv2.putText(
        frame,
        text,
        (x1, y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        color,
        2
    )

    cv2.imshow("ASL Real-Time Detection (Stable)", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
