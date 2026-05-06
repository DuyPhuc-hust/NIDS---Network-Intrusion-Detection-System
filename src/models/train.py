from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

def train_model(X_train, y_train, model_name="xgb"):

    if model_name == "rf":
        print("[DEBUG] Using Random Forest")
        model = RandomForestClassifier(
            n_estimators=100,
            n_jobs=-1,
            random_state=42
        )

    elif model_name == "xgb":
        print("[DEBUG] Using XGBoost")
        model = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="mlogloss",
            tree_method="hist"
        )

    elif model_name == "lr":
        print("[DEBUG] Using Logistic Regression")
        model = LogisticRegression(
            max_iter=1000,
            n_jobs=-1
        )

    elif model_name == "knn":
        print("[DEBUG] Using KNN")
        model = KNeighborsClassifier(
            n_neighbors=5,
            n_jobs=-1
        )

    else:
        raise ValueError("Unknown model")

    model.fit(X_train, y_train)
    return model