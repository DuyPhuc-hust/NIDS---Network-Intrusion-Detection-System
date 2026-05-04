from xgboost import XGBClassifier


def train_model(X, y):
    print("[DEBUG] Using XGBoost multi-class")

    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        eval_metric="mlogloss",
        n_jobs=-1
    )

    model.fit(X, y)

    return model