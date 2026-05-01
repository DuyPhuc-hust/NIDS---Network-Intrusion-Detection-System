import os
import joblib
from src.utils.config import MODEL_DIR

def save_artifacts(model, scaler, features):
    os.makedirs(MODEL_DIR, exist_ok=True)

    joblib.dump(model, f"{MODEL_DIR}/model.pkl")
    joblib.dump(scaler, f"{MODEL_DIR}/scaler.pkl")
    joblib.dump(features, f"{MODEL_DIR}/features.pkl")

    print("[+] Model saved!")