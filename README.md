# Network Intrusion Detection System (NIDS)

A Machine Learning-based Network Intrusion Detection System (NIDS) built using the **CICIDS2017** dataset.  
This project focuses on detecting and classifying malicious network traffic using multiple machine learning models and preprocessing techniques.

---

# 📌 Features

- Multi-class intrusion detection
- Binary classification support (Normal vs Attack)
- Data preprocessing and feature scaling
- Class imbalance handling using oversampling / SMOTE
- Multiple ML models:
  - XGBoost
  - Random Forest
  - Logistic Regression
  - K-Nearest Neighbors (KNN)
- Performance evaluation:
  - Accuracy
  - Precision
  - Recall
  - F1-score
  - Classification report
- Real-world dataset: CICIDS2017
- Modular training pipeline
- Model saving and reuse

---

# 🧠 Dataset

This project uses the CICIDS2017 dataset.

The dataset contains modern attack traffic including:

- DDoS
- DoS
- PortScan
- Brute Force
- Botnet
- Web Attacks
- Infiltration

---

# 📂 Project Structure

```bash
NIDS---Network-Intrusion-Detection-System/
│
├── data/
│   └── raw/
│       └── CICIDS2017/
│
├── models/
│   ├── xgb.pkl
│   ├── rf.pkl
│   ├── lr.pkl
│   └── knn.pkl
│
├── outputs/
│
├── src/
│   ├── data/
│   │   ├── loader.py
│   │   └── preprocess.py
│   │
│   ├── pipeline/
│   │   └── train_pipeline.py
│   │
│   └── utils/
│
├── main.py
├── requirements.txt
└── README.md
```

---

# ⚙️ Installation

## 1. Clone the repository

```bash
git clone <your-repo-link>
cd NIDS---Network-Intrusion-Detection-System
```

## 2. Install dependencies

```bash
pip install -r requirements.txt
```

---

# 📦 Required Libraries

Main libraries used:

```bash
pandas
numpy
scikit-learn
xgboost
imbalanced-learn
joblib
```

---

# 🚀 Training Models

## Train XGBoost

```bash
python3 main.py --train data/raw/CICIDS2017 --model xgb
```

## Train Random Forest

```bash
python3 main.py --train data/raw/CICIDS2017 --model rf
```

## Train Logistic Regression

```bash
python3 main.py --train data/raw/CICIDS2017 --model lr
```

## Train KNN

```bash
python3 main.py --train data/raw/CICIDS2017 --model knn
```

---

# 🛠 Preprocessing Pipeline

The preprocessing stage includes:

- Data cleaning
- Handling missing and infinite values
- Feature scaling using StandardScaler
- Label encoding
- Class balancing for minority attack classes

---

# ⚖️ Class Imbalance Handling

The CICIDS2017 dataset is highly imbalanced.

To improve detection performance on minority classes such as:

- Bot
- Web Attacks
- Infiltration

the project applies:

- Oversampling
- SMOTE (optional)
- Balanced training strategy

---

# 🤖 Implemented Models

| Model | Description |
|---|---|
| XGBoost | Gradient boosting ensemble model |
| Random Forest | Bagging-based tree ensemble |
| Logistic Regression | Linear baseline classifier |
| KNN | Distance-based classifier |

---

# 📈 Evaluation Metrics

The system evaluates models using:

- Accuracy
- Precision
- Recall
- F1-score
- Macro Average
- Weighted Average
- Confusion Matrix

---

# 🧪 Current Results Summary

| Model | Performance |
|---|---|
| XGBoost | Best overall performance |
| Random Forest | Strong and stable |
| KNN | Good accuracy but slower inference |
| Logistic Regression | Fast baseline model |

---

# 🔒 Security Use Cases

This NIDS can be used for:

- Network traffic monitoring
- Intrusion detection
- Malware traffic identification
- SOC analysis
- Cybersecurity research
- Educational purposes

---

# 📌 Future Improvements

- Deep Learning models (MLP / LSTM)
- Real-time packet analysis
- Live traffic capture with CICFlowMeter
- Feature selection optimization
- Hyperparameter tuning
- Deployment with Flask/FastAPI

---

# 👨‍💻 Author

Developed for research and educational purposes in:

- Network Security
- Intrusion Detection Systems
- Machine Learning for Cybersecurity