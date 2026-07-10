from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, f1_score, confusion_matrix
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.base import clone
import joblib
import os
import numpy as np
import pandas as pd

from src.data.loader import load_cicids
from src.data.preprocess import clean_data, split_xy, scale_features, encode_labels, handle_imbalance
from src.models.train import train_model
from src.utils.config import BASE_DIR, MODEL_DIR, RANDOM_STATE


def _stratify_if_possible(labels):
    counts = labels.value_counts()
    return labels if len(counts) > 1 and counts.min() >= 2 else None


def _xgb_sample_weight(y_train_enc):
    weights = compute_sample_weight(class_weight="balanced", y=y_train_enc)
    weights = pd.Series(weights).clip(lower=0.75, upper=2.0).to_numpy()
    print(f"[+] Conservative XGBoost sample weights enabled: min={weights.min():.2f}, max={weights.max():.2f}")
    return weights


def _print_report(title, y_true, y_pred, label_encoder):
    print(f"\n[+] Evaluation on {title}:")
    print(classification_report(
        y_true,
        y_pred,
        labels=range(len(label_encoder.classes_)),
        target_names=label_encoder.classes_,
        zero_division=0
    ))
    print(f"[+] {title} Macro-F1: {f1_score(y_true, y_pred, average='macro', zero_division=0):.4f}")
    print(f"[+] {title} Weighted-F1: {f1_score(y_true, y_pred, average='weighted', zero_division=0):.4f}")


def _predict_classes(model, X):
    predictions = model.predict(X)
    predictions = np.asarray(predictions)
    if predictions.ndim > 1:
        return predictions.argmax(axis=1)
    return predictions


def _save_xgb_diagnostics(model, feature_names, label_encoder, y_test_enc, y_pred, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    if hasattr(model, "feature_importances_"):
        importance_df = pd.DataFrame({
            "feature": feature_names,
            "importance": model.feature_importances_
        }).sort_values("importance", ascending=False)
        importance_path = os.path.join(output_dir, "xgb_feature_importance.csv")
        importance_df.to_csv(importance_path, index=False)
        print(f"[+] Feature importance saved: {importance_path}")

    cm = confusion_matrix(
        y_test_enc,
        y_pred,
        labels=range(len(label_encoder.classes_))
    )
    cm_df = pd.DataFrame(
        cm,
        index=label_encoder.classes_,
        columns=label_encoder.classes_
    )
    cm_path = os.path.join(output_dir, "xgb_confusion_matrix.csv")
    cm_df.to_csv(cm_path)
    print(f"[+] Confusion matrix saved: {cm_path}")


def run_train_pipeline(data_path, model_name="xgb", sample_size=None, model_dir=MODEL_DIR, use_imbalance=True):
    print("[+] Starting pipeline...")
    model_dir = model_dir or MODEL_DIR
    output_dir = os.path.join(BASE_DIR, "outputs")

    # 1. LOAD & CLEAN DATA
    df = load_cicids(data_path, sample_size)
    df = clean_data(df)

    label_col = "Label" if "Label" in df.columns else "label"
    if label_col not in df.columns:
        raise ValueError("Dataset must contain a Label column after cleaning.")

    # 2. SPLIT TRAIN/TEST/VALIDATION on cleaned labels
    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=_stratify_if_possible(df[label_col])
    )

    train_df, val_df = train_test_split(
        train_df,
        test_size=0.125,
        random_state=RANDOM_STATE,
        stratify=_stratify_if_possible(train_df[label_col])
    )

    # 5. SPLIT X, y
    X_train, y_train = split_xy(train_df)
    X_val, y_val = split_xy(val_df)
    X_test, y_test = split_xy(test_df)
    
    feature_names = X_train.columns.tolist()

    # 6. HANDLE IMBALANCE
    if use_imbalance:
        X_train, y_train = handle_imbalance(X_train, y_train)
        X_train = pd.DataFrame(X_train, columns=feature_names)
    else:
        print("\n[!] Skipping imbalance handling; training on original class distribution.")

    # 7. ENCODE LABELS
    y_train_enc, y_test_enc, y_val_enc, label_encoder = encode_labels(y_train, y_test, y_val)

    # 8. SCALE FEATURES
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    X_val_scaled = scaler.transform(X_val)

    X_train_final = pd.DataFrame(X_train_scaled, columns=feature_names)
    X_val_final = pd.DataFrame(X_val_scaled, columns=feature_names)
    X_test_final = pd.DataFrame(X_test_scaled, columns=feature_names)

    # 9. TRAIN
    eval_set = [(X_val_final, y_val_enc)] if model_name == "xgb" else None
    sample_weight = _xgb_sample_weight(y_train_enc) if model_name == "xgb" and use_imbalance else None
    model = train_model(
        X_train_final,
        y_train_enc,
        model_name,
        eval_set=eval_set,
        early_stopping_rounds=50,
        verbose=False,
        sample_weight=sample_weight
    )

    # 10. VALIDATION EVALUATION
    y_val_pred = _predict_classes(model, X_val_final)
    _print_report("Validation Set", y_val_enc, y_val_pred, label_encoder)

    # 11. TEST EVALUATION
    y_pred = _predict_classes(model, X_test_final)
    test_accuracy = (y_pred == y_test_enc).mean()

    _print_report("Test Set", y_test_enc, y_pred, label_encoder)
    if model_name == "xgb":
        _save_xgb_diagnostics(
            model,
            feature_names,
            label_encoder,
            y_test_enc,
            y_pred,
            output_dir
        )

    # 12. K-FOLD CROSS-VALIDATION (sampled for large datasets)
    print("\n[+] Running 5-Fold Cross-Validation on a stratified training sample:")
    cv_train_size = 20000 if model_name == "knn" else 200000
    if len(X_train_final) > cv_train_size:
        print(f"[+] Large training set detected; using a {cv_train_size:,} stratified sample for faster CV.")
        X_cv, _, y_cv, _ = train_test_split(
            X_train_final,
            y_train_enc,
            train_size=cv_train_size,
            stratify=y_train_enc,
            random_state=RANDOM_STATE
        )
    else:
        X_cv, y_cv = X_train_final, y_train_enc

    min_cv_class = pd.Series(y_cv).value_counts().min()
    if min_cv_class >= 5:
        cv_model = clone(model)
        if model_name == "xgb":
            cv_model.set_params(early_stopping_rounds=None)
        cv_scores = cross_val_score(cv_model, X_cv, y_cv, cv=5, scoring='f1_weighted', n_jobs=1)
        print(f"CV Scores: {cv_scores}")
        print(f"CV Weighted-F1 Mean: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        print(f"[!] Accuracy vs CV Weighted-F1 Gap = {test_accuracy - cv_scores.mean():.4f}")
        print("    (Large gap can indicate class imbalance or overfitting.)\n")
    else:
        print(f"[!] Skipping 5-Fold CV because the smallest class has only {min_cv_class} samples.\n")

    # 11. SAVE
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(model, os.path.join(model_dir, f"{model_name}.pkl"))
    joblib.dump(model, os.path.join(model_dir, "model.pkl"))
    joblib.dump(scaler, os.path.join(model_dir, "scaler.pkl"))
    joblib.dump(label_encoder, os.path.join(model_dir, "label_encoder.pkl"))
    joblib.dump(feature_names, os.path.join(model_dir, "features.pkl"))

    print(f"[+] Model saved: {os.path.join(model_dir, model_name + '.pkl')}")

    return model, scaler, label_encoder
