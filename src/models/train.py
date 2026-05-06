import time
import threading
import psutil

from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier

from memory_profiler import memory_usage


# ================= CPU MONITOR =================
def monitor_cpu(stop_flag, cpu_usage):
    while not stop_flag.is_set():
        cpu_usage.append(psutil.cpu_percent(interval=0.1))


# ================= GENERIC TRAIN =================
def measure_model(model, X_train, y_train, name="Model"):
    print(f"\n[DEBUG] Training {name}...")

    cpu_usage = []
    stop_flag = threading.Event()

    def train():
        model.fit(X_train, y_train)

    # start cpu monitor
    t = threading.Thread(target=monitor_cpu, args=(stop_flag, cpu_usage))
    t.start()

    start = time.time()
    mem = max(memory_usage((train,)))
    train_time = time.time() - start

    stop_flag.set()
    t.join()

    cv_score = cross_val_score(model, X_train, y_train, cv=3).mean()

    result = {
        "model": model,
        "cv_score": cv_score,
        "memory": mem,
        "time": train_time,
        "cpu_max": max(cpu_usage) if cpu_usage else 0,
        "cpu_avg": sum(cpu_usage) / len(cpu_usage) if cpu_usage else 0
    }

    return result


# ================= ALL MODELS =================
def train_all_models(X_train, y_train):
    results = []

    # 1. Random Forest
    rf = RandomForestClassifier(n_estimators=100, n_jobs=-1)
    results.append(measure_model(rf, X_train, y_train, "Random Forest"))

    # 2. XGBoost
    xgb = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        eval_metric="mlogloss",
        tree_method="hist"
    )
    results.append(measure_model(xgb, X_train, y_train, "XGBoost"))

    # 3. Neural Network
    nn = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        max_iter=20,
        early_stopping=True
    )
    results.append(measure_model(nn, X_train, y_train, "Neural Network"))

    return results