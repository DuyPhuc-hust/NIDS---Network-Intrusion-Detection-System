from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import os

# Import thêm hàm handle_imbalance từ preprocess
from src.data.loader import load_cicids
from src.data.preprocess import clean_data, split_xy, scale_features, encode_labels, handle_imbalance
from src.models.train import train_model


def run_train_pipeline(data_path, model_name="xgb", sample_size=None):
    print("[+] Starting pipeline...")

    # 1. LOAD DATA
    df = load_cicids(data_path, sample_size)

    # 2. SPLIT TRAIN/TEST (Tách df thô trước để tránh leakage)
    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df["Label"] # Đảm bảo tỉ lệ các lớp ở 2 tập là như nhau
    )

    # 3. CLEAN DATA
    train_df = clean_data(train_df)
    test_df = clean_data(test_df)

    # 4. SPLIT X, y
    X_train, y_train = split_xy(train_df)
    X_test, y_test = split_xy(test_df)

    # 5. HANDLE IMBALANCE (Chỉ thực hiện trên tập Train)
    # Bước này giúp model học được đặc trưng của các lớp thiểu số tốt hơn
    X_train, y_train = handle_imbalance(X_train, y_train)

    # 6. ENCODE LABEL
    y_train, y_test, label_encoder = encode_labels(y_train, y_test)

    # 7. SCALE FEATURES
    # Scaler fit trên tập Train đã balance và transform cho cả 2
    X_train, X_test, scaler = scale_features(X_train, X_test)

    # 8. TRAIN
    model = train_model(X_train, y_train, model_name)

    # 9. EVALUATE
    y_pred = model.predict(X_test)

    print("\n[+] Evaluation on Unbalanced Test Set:")
    # Chú ý: Classification report sẽ hiển thị theo index của LabelEncoder
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

    # 10. SAVE
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, f"models/{model_name}.pkl")
    # Nên save cả scaler và encoder để dùng cho inference/realtime sau này
    joblib.dump(scaler, "models/scaler.pkl")
    joblib.dump(label_encoder, "models/label_encoder.pkl")

    print(f"[+] Model saved: models/{model_name}.pkl")

    return model, scaler, label_encoder