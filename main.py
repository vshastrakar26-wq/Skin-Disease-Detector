from flask import Flask, render_template, request
import os
import cv2
import numpy as np
import joblib
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

app = Flask(__name__)

# ---------------- PATHS ----------------
MODEL_PATH = "models/hybrid_svm_model.pkl"
PCA_PATH = "models/pca_transform.pkl"
SCALER_ORB_PATH = "models/scaler_orb.pkl"
SCALER_CNN_PATH = "models/scaler_cnn.pkl"

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ---------------- LOAD MODELS ----------------
mobilenet = MobileNetV2(weights="imagenet", include_top=False, pooling="avg")

# ⚙️ Match ORB settings to training configuration
orb = cv2.ORB_create(nfeatures=80)

# Load trained models and scalers
svm_model = joblib.load(MODEL_PATH)
pca_orb = joblib.load(PCA_PATH)
scaler_orb = joblib.load(SCALER_ORB_PATH)
scaler_cnn = joblib.load(SCALER_CNN_PATH)

class_names = [
    "Eczema", "Melanoma", "Psoriasis", "Ringworm",
    "Seborrheic dermatitis", "Vitiligo", "Warts"
]

# ---------------- ROUTES ----------------
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/detect')
def detect():
    return render_template('detect.html')

# ---------------- PREDICTION ROUTE ----------------
@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return render_template('detect.html', prediction="❌ No file uploaded.")
        
        file = request.files['file']
        if file.filename == '':
            return render_template('detect.html', prediction="⚠️ Please select an image.")

        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)

        # ---- Extract ORB + CNN features ----
        img = cv2.imread(file_path)
        if img is None:
            return render_template('detect.html', prediction="⚠️ Invalid image file.")

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        def extract_orb_features(image):
            keypoints, descriptors = orb.detectAndCompute(image, None)
            if descriptors is None:
                descriptors = np.zeros((1, 32))
            descriptors = descriptors.flatten()
            target_len = 80 * 32
            if len(descriptors) < target_len:
                descriptors = np.pad(descriptors, (0, target_len - len(descriptors)))
            else:
                descriptors = descriptors[:target_len]
            return descriptors

        orb_feat = extract_orb_features(img_rgb)
        cnn_feat = mobilenet.predict(
            np.expand_dims(preprocess_input(img_to_array(load_img(file_path, target_size=(160, 160)))), axis=0),
            verbose=0
        ).flatten()

        # ---- Normalize + PCA + Merge ----
        orb_feat = scaler_orb.transform([orb_feat])
        cnn_feat = scaler_cnn.transform([cnn_feat])
        orb_feat = pca_orb.transform(orb_feat)
        features = np.hstack((cnn_feat, orb_feat))

        # ---- Predict ----
        pred = svm_model.predict(features)[0]
        label = class_names[pred]
        info_link = f"/info/{label}"

        return render_template('detect.html', prediction=f"🩺 Prediction: {label}", info_link=info_link)

    except Exception as e:
        print("❌ Error during prediction:", e)
        return render_template('detect.html', prediction=f"⚠️ Error: {str(e)}")

# ---------------- INFO ROUTE ----------------
@app.route('/info/<disease>')
def info(disease):
    info_dict = {
        "Eczema": "Eczema causes red, itchy, and inflamed skin. Regular moisturizing and avoiding triggers help manage it.",
        "Melanoma": "Melanoma is a serious skin cancer. Early detection is crucial—watch for changes in moles.",
        "Psoriasis": "Psoriasis creates scaly patches on the skin due to rapid cell growth. Manageable with creams and light therapy.",
        "Ringworm": "Ringworm is a contagious fungal infection that forms circular rashes. Treatable with antifungal medication.",
        "Seborrheic dermatitis": "Causes flaky, itchy skin, often on the scalp or face. Medicated shampoos help.",
        "Vitiligo": "Vitiligo leads to white patches as pigment cells lose function. It’s non-contagious.",
        "Warts": "Warts are small skin growths from HPV infection. They can be treated or removed easily."
    }

    description = info_dict.get(disease, "No information available for this condition.")
    return render_template('info.html', disease=disease, description=description)

# ---------------- RUN APP ----------------
if __name__ == '__main__':
    app.run(debug=True)
