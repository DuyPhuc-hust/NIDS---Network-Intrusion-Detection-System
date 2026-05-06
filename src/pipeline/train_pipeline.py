from src.data.loader import load_cicids
from src.data.preprocess import (
    clean_data,
    split_xy,
    encode_features_train_test,
    scale_features
)
from src.models.train import train_all_models
from sklearn.metrics import classification_report


def run_train_pipeline(data_path):
    print("[+] Starting pipeline...")

    df = load_cicids(data_path)

    # ===== SPLIT =====
    from sklearn.model_selection import train_test_split
    train_df, test_df = train_test_split(
        df, test_size=0.2, stratify=df["Label"], random_state=42
    )

    print(f"[+] Train shape: {train_df.shape}")
    print(f"[+] Test shape: {test_df.shape}")

    # ===== CLEAN =====
    train_df = clean_data(train_df)
    test_df = clean_data(test_df)

    # ===== SPLIT XY =====
    X_train, y_train = split_xy(train_df, "Label")
    X_test, y_test = split_xy(test_df, "Label")

    # ===== ENCODE =====
    X_train, X_test, encoder = encode_features_train_test(X_train, X_test)

    # ===== SCALE =====
    X_train, X_test, scaler = scale_features(X_train, X_test)

    # ===== TRAIN MULTI MODELS =====
    results = train_all_models(X_train, y_train)

    print("\n================ MODEL COMPARISON ================")

    best_model = None
    best_score = 0

    for r in results:
        print(f"""
Model: {type(r['model']).__name__}
CV Score: {r['cv_score']:.4f}
Time: {r['time']:.2f}s
Memory: {r['memory']:.2f} MB
CPU max: {r['cpu_max']}%
CPU avg: {r['cpu_avg']:.2f}%
        """)

        if r["cv_score"] > best_score:
            best_score = r["cv_score"]
            best_model = r["model"]

    # ===== FINAL EVAL =====
    print("\n[+] Best model evaluation:")
    y_pred = best_model.predict(X_test)

    print(classification_report(y_test, y_pred))

    return best_model, scaler, encoder