from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import os
import pandas as pd # Thêm import này

# Import thêm hàm handle_imbalance từ preprocess
from src.data.loader import load_cicids
from src.data.preprocess import clean_data, split_xy, scale_features, encode_labels, handle_imbalance
from src.models.train import train_model

def run_train_pipeline(data_path, model_name="xgb", sample_size=None):
    print("[+] Starting pipeline...")

    # 1. LOAD DATA
    df = load_cicids(data_path, sample_size)

    # 2. SPLIT TRAIN/TEST
    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df["Label"]
    )

    # 3. CLEAN DATA
    train_df = clean_data(train_df)
    test_df = clean_data(test_df)

    # 4. SPLIT X, y
    X_train, y_train = split_xy(train_df)
    X_test, y_test = split_xy(test_df)
    
    # --- SỬA NHỎ 1: Lưu tên cột ---
    feature_names = X_train.columns.tolist()

    # 5. HANDLE IMBALANCE
    X_train, y_train = handle_imbalance(X_train, y_train)
    # --- SỬA NHỎ 2: Khôi phục DataFrame sau SMOTE ---
    X_train = pd.DataFrame(X_train, columns=feature_names)

    # 6. ENCODE LABEL
    y_train, y_test, label_encoder = encode_labels(y_train, y_test)

    # 7. SCALE FEATURES
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    # --- SỬA NHỎ 3: Khôi phục DataFrame sau Scaling ---
    X_train_final = pd.DataFrame(X_train_scaled, columns=feature_names)
    X_test_final = pd.DataFrame(X_test_scaled, columns=feature_names)

    # 8. TRAIN
    # Truyền X_train_final (DataFrame) thay vì mảng numpy
    model = train_model(X_train_final, y_train, model_name)

    # 9. EVALUATE
    y_pred = model.predict(X_test_final)

    print("\n[+] Evaluation on Unbalanced Test Set:")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

    # 10. SAVE
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, f"models/{model_name}.pkl")
    joblib.dump(scaler, "models/scaler.pkl")
    joblib.dump(label_encoder, "models/label_encoder.pkl")

    print(f"[+] Model saved: models/{model_name}.pkl")

    return model, scaler, label_encoder