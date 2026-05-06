import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

def clean_data(df):
    """
    Làm sạch dữ liệu, xử lý giá trị vô hạn và map nhãn về các nhóm chính.
    """
    # Xóa khoảng trắng thừa trong tên cột
    df.columns = df.columns.str.strip()

    # Fix label column nếu có space
    if "Label" not in df.columns and "Label" in [c.strip() for c in df.columns]:
        df.rename(columns={col: col.strip() for col in df.columns}, inplace=True)

    # Thay thế giá trị vô hạn (inf) bằng NaN và xóa hàng có NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    # 🔥 MAP LABEL: Gom nhóm các cuộc tấn công nhỏ lẻ vào nhóm lớn
    label_mapping = {
        "BENIGN": "Normal",
        "DoS Hulk": "DoS",
        "DoS GoldenEye": "DoS",
        "DoS Slowloris": "DoS",
        "DoS slowloris": "DoS",
        "DoS Slowhttptest": "DoS",
        "DDoS": "DDoS",
        "PortScan": "PortScan",
        "FTP-Patator": "BruteForce",
        "SSH-Patator": "BruteForce",
        "Bot": "Bot",
        "Web Attack  Brute Force": "Web",
        "Web Attack  XSS": "Web",
        "Web Attack  Sql Injection": "Web",
        "Infiltration": "Infiltration",
        "Heartbleed": "Misc"
    }

    df["Label"] = df["Label"].map(label_mapping)

    # Drop các dòng không map được nhãn (unknown)
    df = df.dropna(subset=["Label"])

    print("\n[+] Thống kê các lớp sau khi gom nhóm:")
    print(df["Label"].value_counts())

    return df

def split_xy(df):
    """
    Tách đặc trưng (X) và nhãn (y).
    """
    X = df.drop("Label", axis=1)
    y = df["Label"]
    return X, y

def handle_imbalance(X_train, y_train):
    """
    Sử dụng chiến lược Hybrid: 
    1. Under-sampling lớp đa số (Normal) để giảm tải.
    2. Over-sampling (SMOTE) các lớp thiểu số để tăng khả năng nhận diện.
    """
    print("\n[+] Đang xử lý imbalance bằng chiến lược Hybrid (SMOTE + Under-sampling)...")
    print(f"Trước xử lý: {dict(pd.Series(y_train).value_counts())}")

    # Bước 1: Under-sampling lớp Normal xuống mức 500,000 mẫu (hoặc tùy bạn chọn)
    # Nếu lớp Normal đang ít hơn mức này thì nó sẽ giữ nguyên.
    under_strategy = {
        'Normal': 500000 
    }
    # Chỉ thực hiện under-sample nếu lớp Normal thực sự lớn hơn ngưỡng
    current_normal = (y_train == 'Normal').sum()
    if current_normal > 500000:
        rus = RandomUnderSampler(sampling_strategy=under_strategy, random_state=42)
        X_train, y_train = rus.fit_resample(X_train, y_train)

    # Bước 2: SMOTE các lớp còn lại lên bằng mức của lớp Normal sau khi đã cắt bớt
    # Lúc này các lớp Bot, Infiltration, DoS... sẽ được nâng lên mức 500,000
    smote = SMOTE(sampling_strategy='not majority', random_state=42, k_neighbors=1)
    X_res, y_res = smote.fit_resample(X_train, y_train)

    print(f"Sau xử lý: {dict(pd.Series(y_res).value_counts())}")
    return X_res, y_res

def encode_labels(y_train, y_test):
    """
    Chuyển đổi nhãn dạng chữ sang số.
    """
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)

    print("\n[+] Danh sách các lớp (Encoded):")
    for index, label in enumerate(le.classes_):
        print(f"{index}: {label}")

    return y_train_enc, y_test_enc, le

def scale_features(X_train, X_test):
    """
    Chuẩn hóa dữ liệu về cùng một thang đo (Z-score normalization).
    """
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, scaler