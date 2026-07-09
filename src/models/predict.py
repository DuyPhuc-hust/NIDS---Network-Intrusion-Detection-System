import joblib
import numpy as np
from src.utils.config import MODEL_DIR


_artifacts = None


def load_artifacts():
    global _artifacts
    if _artifacts is None:
        _artifacts = {
            "model": joblib.load(f"{MODEL_DIR}/model.pkl"),
            "scaler": joblib.load(f"{MODEL_DIR}/scaler.pkl"),
            "features": joblib.load(f"{MODEL_DIR}/features.pkl"),
        }
    return _artifacts


def align_features(df):
    features = load_artifacts()["features"]
    df = df.copy()

    for col in features:
        if col not in df.columns:
            df[col] = 0

    return df[features]


def predict(df):
    artifacts = load_artifacts()
    df = align_features(df)
    X = artifacts["scaler"].transform(df)
    predictions = artifacts["model"].predict(X)
    predictions = np.asarray(predictions)
    if predictions.ndim > 1:
        return predictions.argmax(axis=1)
    return predictions
