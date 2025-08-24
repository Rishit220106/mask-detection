import os
import shutil
import random

# Paths
base_dir = "./data"   # put your original dataset here (with_mask/ without_mask inside)
TRAIN_DIR = "data/train"
VAL_DIR   = "data/val"

# Make dirs if not exist
for split in ["train", "val"]:
    for category in ["with_mask", "without_mask"]:
        os.makedirs(os.path.join("data", split, category), exist_ok=True)

# Split ratio
val_ratio = 0.2

for category in ["with_mask", "without_mask"]:
    img_dir = os.path.join(base_dir, category)
    images = os.listdir(img_dir)
    random.shuffle(images)

    val_count = int(len(images) * val_ratio)
    val_images = images[:val_count]
    train_images = images[val_count:]

    # Copy files
    for img in train_images:
        shutil.copy(os.path.join(img_dir, img), os.path.join(TRAIN_DIR, category, img))
    for img in val_images:
        shutil.copy(os.path.join(img_dir, img), os.path.join(VAL_DIR, category, img))

print("✅ Dataset split into training and validation sets!")
