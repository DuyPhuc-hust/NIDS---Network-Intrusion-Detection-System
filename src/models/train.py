from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
import inspect
import numpy as np
from xgboost.callback import EarlyStopping


def train_model(X_train, y_train, model_name="xgb", eval_set=None, early_stopping_rounds=None, verbose=False, sample_weight=None):
    num_classes = len(np.unique(y_train))

    if model_name == "rf":
        print("[DEBUG] Using Random Forest")
        model = RandomForestClassifier(
            n_estimators=100,
            n_jobs=-1,
            random_state=42
        )

    elif model_name == "xgb":
        print("[DEBUG] Using anti-overfit XGBoost for NIDS")
        model = XGBClassifier(
            n_estimators=600,
            max_depth=4,
            learning_rate=0.04,
            subsample=0.8,
            colsample_bytree=0.75,
            reg_alpha=2.0,
            reg_lambda=8.0,
            gamma=2.0,
            min_child_weight=10,
            max_delta_step=1,
            objective="multi:softmax",
            num_class=num_classes,
            eval_metric="merror",
            tree_method="hist",
            verbosity=0,
            random_state=42,
            n_jobs=-1
        )

    elif model_name == "lr":
        print("[DEBUG] Using Logistic Regression")
        model = LogisticRegression(
            solver="saga",
            max_iter=300,
            tol=1e-3,
            n_jobs=-1,
            random_state=42
        )

    elif model_name == "knn":
        print("[DEBUG] Using KNN")
        model = KNeighborsClassifier(
            n_neighbors=5,
            n_jobs=-1
        )

    else:
        raise ValueError("Unknown model")

    fit_kwargs = {}
    if sample_weight is not None:
        fit_kwargs['sample_weight'] = sample_weight

    if model_name == "xgb" and eval_set is not None:
        sig = inspect.signature(model.fit)
        if early_stopping_rounds is not None and 'early_stopping_rounds' not in sig.parameters and 'callbacks' not in sig.parameters:
            model.set_params(early_stopping_rounds=early_stopping_rounds)

        # Handle different xgboost versions: some accept early_stopping_rounds, others require callbacks
        try:
            if 'early_stopping_rounds' in sig.parameters:
                model.fit(
                    X_train,
                    y_train,
                    eval_set=eval_set,
                    early_stopping_rounds=early_stopping_rounds,
                    verbose=verbose,
                    **fit_kwargs
                )
            elif 'callbacks' in sig.parameters and early_stopping_rounds is not None:
                model.fit(
                    X_train,
                    y_train,
                    eval_set=eval_set,
                    callbacks=[EarlyStopping(rounds=early_stopping_rounds)],
                    verbose=verbose,
                    **fit_kwargs
                )
            else:
                model.fit(X_train, y_train, eval_set=eval_set, verbose=verbose, **fit_kwargs)
        except Exception:
            # Fallback to basic fit if something unexpected happens
            model.fit(X_train, y_train, **fit_kwargs)
    else:
        model.fit(X_train, y_train, **fit_kwargs)

    return model
