import os, cv2, numpy as np, joblib, time
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# ---------------- PATHS ----------------
TRAIN_DIR = "dataset/train_set"
TEST_DIR = "dataset/test_set"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# ---------------- PARAMETERS ----------------
ORB_FEATURES = 80           # reduced → faster
PCA_COMPONENTS = 20         # smaller → lighter PCA
IMG_SIZE = (128, 128)       # smaller images
mobilenet = MobileNetV2(weights="imagenet", include_top=False, pooling="avg", input_shape=(128,128,3))
orb = cv2.ORB_create(nfeatures=ORB_FEATURES)

# ---------------- CLASSES ----------------
class_names = sorted(os.listdir(TRAIN_DIR))
print("Detected classes:", class_names)

# ---------------- FEATURE EXTRACTORS ----------------
def extract_orb_features(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    kp, des = orb.detectAndCompute(gray, None)
    if des is None: des = np.zeros((1, 32))
    f = des.flatten()
    f = np.pad(f, (0, ORB_FEATURES*32 - len(f)))[:ORB_FEATURES*32]
    return f

def extract_mobilenet_features(path):
    img = load_img(path, target_size=IMG_SIZE)
    arr = preprocess_input(np.expand_dims(img_to_array(img), 0))
    return mobilenet.predict(arr, verbose=0).flatten()

def load_dataset(folder, limit_per_class=500):
    Xo, Xc, y = [], [], []
    for label, cls in enumerate(class_names):
        path = os.path.join(folder, cls)
        files = os.listdir(path)[:limit_per_class]  # limit count per class
        for f in tqdm(files, desc=f"Processing {cls}"):
            img_path = os.path.join(path, f)
            img = cv2.imread(img_path)
            if img is None: continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            Xo.append(extract_orb_features(img))
            Xc.append(extract_mobilenet_features(img_path))
            y.append(label)
    return np.array(Xo), np.array(Xc), np.array(y)

# ---------------- LOAD ----------------
print("📦 Loading dataset (limited for speed)...")
Xo_tr, Xc_tr, y_tr = load_dataset(TRAIN_DIR)
Xo_te, Xc_te, y_te = load_dataset(TEST_DIR, limit_per_class=200)

# ---------------- NORMALIZE + PCA ----------------
print("🔧 Scaling & PCA...")
sc_orb, sc_cnn = StandardScaler(), StandardScaler()
Xo_tr, Xo_te = sc_orb.fit_transform(Xo_tr), sc_orb.transform(Xo_te)
Xc_tr, Xc_te = sc_cnn.fit_transform(Xc_tr), sc_cnn.transform(Xc_te)
pca = PCA(n_components=min(PCA_COMPONENTS, Xo_tr.shape[1]-1))
Xo_tr, Xo_te = pca.fit_transform(Xo_tr), pca.transform(Xo_te)

X_train, X_test = np.hstack((Xc_tr, Xo_tr)), np.hstack((Xc_te, Xo_te))

# ---------------- TRAIN ----------------
print("⚙️ Training fast Linear SVM ...")
clf = LinearSVC(max_iter=3000, class_weight='balanced', dual=False)
t0 = time.time(); clf.fit(X_train, y_tr)
print("✅ Done in", round((time.time()-t0)/60,2), "min")

acc = accuracy_score(y_te, clf.predict(X_test))
print("🎯 Accuracy:", round(acc*100,2), "%")

# ---------------- SAVE ----------------
joblib.dump(clf, f"{MODEL_DIR}/hybrid_svm_model.pkl")
joblib.dump(pca, f"{MODEL_DIR}/pca_transform.pkl")
joblib.dump(sc_orb, f"{MODEL_DIR}/scaler_orb.pkl")
joblib.dump(sc_cnn, f"{MODEL_DIR}/scaler_cnn.pkl")
print("💾 All models saved in /models/")
