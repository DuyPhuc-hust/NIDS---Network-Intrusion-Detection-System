from src.data.loader import load_cicids
from src.data.preprocess import (
    clean_data,
    split_xy,
    encode_features_train_test,
    scale_features
)

from src.models.train import train_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report


def run_train_pipeline(data_path):
    print("[+] Starting pipeline...")

    df = load_cicids(data_path)

    # đảm bảo có Label
    df.columns = df.columns.str.strip()

    if "Label" not in df.columns:
        raise Exception("❌ Không tìm thấy cột Label")

    # split trước
    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df["Label"]
    )

    print(f"[+] Train shape: {train_df.shape}")
    print(f"[+] Test shape: {test_df.shape}")

    # clean riêng
    train_df = clean_data(train_df)
    test_df = clean_data(test_df)

    # split X y
    X_train, y_train = split_xy(train_df)
    X_test, y_test = split_xy(test_df)

    # encode label (multi-class)
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train)
    y_test = label_encoder.transform(y_test)

    print("\n[+] Classes:")
    print(label_encoder.classes_)

    # encode feature
    X_train, X_test, encoder = encode_features_train_test(X_train, X_test)

    # scale
    X_train, X_test, scaler = scale_features(X_train, X_test)

    # train
    model = train_model(X_train, y_train)

    # predict
    y_pred = model.predict(X_test)

    print("\n[+] Evaluation:")
    print(classification_report(y_test, y_pred))

    return model, scaler, encoder, label_encoder