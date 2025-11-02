import os
import numpy as np
import cv2
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import tensorflow as tf

# ---------------- CONFIG ----------------
TEST_DIR = "dataset/test_set"
IMG_SIZE = (128, 128)
ORB_FEATURES = 100
MAX_IMAGES_PER_CLASS = 20  # limit to 20 per class for quick evaluation

# ---------------- LOAD MODELS ----------------
mobilenet = MobileNetV2(weights="imagenet", include_top=False, pooling="avg")
clf = joblib.load("hybrid_svm_model.pkl")
pca = joblib.load("pca_transform.pkl")
scaler = joblib.load("scaler.pkl")

# ---------------- ORB SETUP ----------------
orb = cv2.ORB_create(nfeatures=ORB_FEATURES)

def extract_orb_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    keypoints, descriptors = orb.detectAndCompute(gray, None)
    if descriptors is None:
        descriptors = np.zeros((1, 32))
    descriptors = descriptors.flatten()
    target_len = ORB_FEATURES * 32
    if len(descriptors) < target_len:
        descriptors = np.pad(descriptors, (0, target_len - len(descriptors)))
    else:
        descriptors = descriptors[:target_len]
    return descriptors

def extract_mobilenet_features(image_path):
    img = load_img(image_path, target_size=IMG_SIZE)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    with tf.device('/CPU:0'):
        feat = mobilenet.predict(img_array, verbose=0).flatten()
    return feat

# ---------------- LOAD TEST DATA ----------------
class_names = sorted([d for d in os.listdir(TEST_DIR) if os.path.isdir(os.path.join(TEST_DIR, d))])
y_true, y_pred = [], []

print("🔍 Evaluating model on test dataset...\n")

for label_idx, class_name in enumerate(class_names):
    class_dir = os.path.join(TEST_DIR, class_name)
    image_files = os.listdir(class_dir)[:MAX_IMAGES_PER_CLASS]
    print(f"🧩 Processing {class_name} ({len(image_files)} images)...")

    for img_file in image_files:
        img_path = os.path.join(class_dir, img_file)
        image = cv2.imread(img_path)
        if image is None:
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        orb_feat = extract_orb_features(image).reshape(1, -1)
        orb_feat = scaler.transform(orb_feat)
        orb_feat = pca.transform(orb_feat)

        cnn_feat = extract_mobilenet_features(img_path).reshape(1, -1)

        combined = np.concatenate((cnn_feat, orb_feat), axis=1)
        pred = clf.predict(combined)[0]
        y_true.append(label_idx)
        y_pred.append(pred)

# ---------------- RESULTS ----------------
acc = accuracy_score(y_true, y_pred) * 100
print(f"\n✅ Test Accuracy: {acc:.2f}%")
print("\n📊 Classification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))
print("\n🧩 Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))
