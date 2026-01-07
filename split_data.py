import os
import shutil
import random

RAW_DIR = "data/raw"
TRAIN_DIR = "data/train"
VAL_DIR = "data/val"

SPLIT_RATIO = 0.8  # 80% train, 20% val

# Create train & val directories
os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(VAL_DIR, exist_ok=True)

classes = os.listdir(RAW_DIR)

for cls in classes:
    cls_raw_path = os.path.join(RAW_DIR, cls)

    if not os.path.isdir(cls_raw_path):
        continue

    images = os.listdir(cls_raw_path)
    random.shuffle(images)

    split_index = int(len(images) * SPLIT_RATIO)

    train_images = images[:split_index]
    val_images = images[split_index:]

    os.makedirs(os.path.join(TRAIN_DIR, cls), exist_ok=True)
    os.makedirs(os.path.join(VAL_DIR, cls), exist_ok=True)

    for img in train_images:
        src = os.path.join(cls_raw_path, img)
        dst = os.path.join(TRAIN_DIR, cls, img)
        shutil.copy(src, dst)

    for img in val_images:
        src = os.path.join(cls_raw_path, img)
        dst = os.path.join(VAL_DIR, cls, img)
        shutil.copy(src, dst)

    print(f"{cls}: {len(train_images)} train | {len(val_images)} val")

print(" Data split completed")
