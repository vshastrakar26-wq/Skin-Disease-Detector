import os
import cv2
import random
from tqdm import tqdm

# ---------------- CONFIG ----------------
TRAIN_DIR = "dataset/train_set"
TEST_DIR = "dataset/test_set"

TARGET_SIZE = (128, 128)
MAX_TRAIN = 500   # keep this many per class
MAX_TEST = 150    # keep this many per class
VALID_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")

# ---------------- FUNCTION ----------------
def resize_and_limit(folder, max_count):
    """Resize all images and limit to max_count per class."""
    for class_name in sorted(os.listdir(folder)):
        class_path = os.path.join(folder, class_name)
        if not os.path.isdir(class_path):
            continue

        files = [f for f in os.listdir(class_path) if f.lower().endswith(VALID_EXTS)]
        random.shuffle(files)

        if len(files) > max_count:
            to_delete = files[max_count:]
            for f in to_delete:
                try:
                    os.remove(os.path.join(class_path, f))
                except Exception:
                    pass
            files = files[:max_count]

        for f in tqdm(files, desc=f"Resizing {class_name}"):
            path = os.path.join(class_path, f)
            try:
                img = cv2.imread(path)
                if img is None:
                    os.remove(path)
                    continue
                img = cv2.resize(img, TARGET_SIZE, interpolation=cv2.INTER_AREA)
                cv2.imwrite(path, img, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            except Exception as e:
                print(f"⚠️ Skipping {path}: {e}")

# ---------------- RUN ----------------
print("🧹 Cleaning up training set...")
resize_and_limit(TRAIN_DIR, MAX_TRAIN)

print("\n🧹 Cleaning up test set...")
resize_and_limit(TEST_DIR, MAX_TEST)

print("\n✅ Dataset cleaned, resized, and limited successfully!")
