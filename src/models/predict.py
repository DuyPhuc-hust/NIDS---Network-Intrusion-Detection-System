import joblib
import pandas as pd
from src.utils.config import MODEL_DIR

model = joblib.load(f"{MODEL_DIR}/model.pkl")
scaler = joblib.load(f"{MODEL_DIR}/scaler.pkl")
features = joblib.load(f"{MODEL_DIR}/features.pkl")


def align_features(df):
    for col in features:
        if col not in df.columns:
            df[col] = 0

    return df[features]


def predict(df):
    df = align_features(df)
    X = scaler.transform(df)
    return model.predict(X)