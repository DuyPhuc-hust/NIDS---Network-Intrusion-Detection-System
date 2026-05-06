from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import os

from src.data.loader import load_cicids
from src.data.preprocess import clean_data, split_xy, scale_features, encode_labels
from src.models.train import train_model


def run_train_pipeline(data_path, model_name="xgb", sample_size=None):
    print("[+] Starting pipeline...")

    # LOAD DATA
    df = load_cicids(data_path, sample_size)

    # SPLIT
    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df["Label"]
    )

    print(f"[+] Train shape: {train_df.shape}")
    print(f"[+] Test shape: {test_df.shape}")

    # CLEAN
    train_df = clean_data(train_df)
    test_df = clean_data(test_df)

    # SPLIT X, y
    X_train, y_train = split_xy(train_df)
    X_test, y_test = split_xy(test_df)

    # ENCODE LABEL
    y_train, y_test, label_encoder = encode_labels(y_train, y_test)

    # SCALE
    X_train, X_test, scaler = scale_features(X_train, X_test)

    # TRAIN
    model = train_model(X_train, y_train, model_name)

    # EVALUATE
    y_pred = model.predict(X_test)

    print("\n[+] Evaluation:")
    print(classification_report(y_test, y_pred))

    # SAVE
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, f"models/{model_name}.pkl")

    print(f"[+] Model saved: models/{model_name}.pkl")

    return model, scaler, label_encoder