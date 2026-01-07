import cv2
import os
import numpy as np

# ===============================
# CONFIG
# ===============================
BASE_DIR = "data/raw"   # RAW data only
IMG_SIZE = 48
ROI_SIZE = 300

# ===============================
# CREATE FOLDERS
# ===============================
if not os.path.exists(BASE_DIR):
    os.makedirs(BASE_DIR)

for i in range(65, 91):  # A-Z
    letter = chr(i)
    os.makedirs(os.path.join(BASE_DIR, letter), exist_ok=True)

os.makedirs(os.path.join(BASE_DIR, "blank"), exist_ok=True)

# ===============================
# QUALITY CHECK FUNCTIONS
# ===============================
def is_blurry(img, threshold=100):
    return cv2.Laplacian(img, cv2.CV_64F).var() < threshold

def is_too_dark_or_bright(img, low=40, high=215):
    mean = img.mean()
    return mean < low or mean > high

def hand_present(img):
    _, thresh = cv2.threshold(img, 60, 255, cv2.THRESH_BINARY)
    return cv2.countNonZero(thresh) > 500

# ===============================
# CAPTURE
# ===============================
cap = cv2.VideoCapture(0)

print("Press A-Z to capture that sign")
print("Press . for blank")
print("Press ESC to exit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]

    # Center ROI
    x1, y1 = w//2 - ROI_SIZE//2, h//2 - ROI_SIZE//2
    x2, y2 = w//2 + ROI_SIZE//2, h//2 + ROI_SIZE//2

    roi = frame[y1:y2, x1:x2]
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)

    cv2.imshow("Webcam", frame)

    # Preprocess ROI
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))

    cv2.imshow("ROI", resized)

    key = cv2.waitKey(10) & 0xFF

    if key == 27:  # ESC
        break

    # Detect key A-Z or blank
    label = None
    if ord('a') <= key <= ord('z'):
        label = chr(key).upper()
    elif key == ord('.'):
        label = "blank"

    if label:
        # QUALITY FILTERS
        if is_blurry(resized):
            print("❌ Blurry image discarded")
            continue

        if is_too_dark_or_bright(resized):
            print("❌ Bad lighting discarded")
            continue

        if not hand_present(resized):
            print("❌ No hand detected")
            continue

        save_dir = os.path.join(BASE_DIR, label)
        count = len(os.listdir(save_dir))
        save_path = os.path.join(save_dir, f"{count}.jpg")

        cv2.imwrite(save_path, resized)
        print(f"✅ Saved {label}/{count}.jpg")

cap.release()
cv2.destroyAllWindows()
