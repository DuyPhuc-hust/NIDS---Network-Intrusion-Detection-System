# NIDS - Network Intrusion Detection System

This project is a Machine Learning-based Network Intrusion Detection System using the CICIDS2017 flow dataset. It supports multi-class intrusion detection, dataset preprocessing, class imbalance handling, model comparison, and offline PCAP testing.

## Objectives

- Classify network traffic into `Normal`, `DoS`, `DDoS`, `PortScan`, `BruteForce`, `Bot`, and `Infiltration`.
- Compare Logistic Regression, KNN, Random Forest, and XGBoost.
- Evaluate the effect of class imbalance handling before and after resampling.
- Test trained models on offline PCAP files through flow extraction.

## Project Structure

```text
NIDS---Network-Intrusion-Detection-System/
├── data/
│   ├── raw/CICIDS2017/        # CICIDS2017 CSV flow files
│   └── pcap/                  # offline PCAP files for inference testing
├── models/final/              # default model artifacts used for PCAP inference
├── experiments/models/        # archived model artifacts from comparison runs
├── outputs/                   # generated prediction outputs and diagnostics
├── reports/training_runs/     # saved training logs and metric summaries
├── src/
│   ├── data/                  # dataset loading, cleaning, label mapping, imbalance handling
│   ├── models/                # model definitions and prediction helpers
│   ├── pipeline/              # end-to-end training pipeline
│   └── utils/                 # shared configuration
├── live_analyze.py            # PCAP-to-flow extraction and model inference
├── main.py                    # command-line entry point
├── requirements.txt
└── README.md
```

The dataset and generated model files can be large, so the README describes them at the folder level instead of listing every CSV, PCAP, or `.pkl` file.

`models/final/` is intentionally small and is the default inference location. Older comparison artifacts are kept under `experiments/models/` so the main project tree stays readable.

## Dataset

The main dataset is CICIDS2017 in CSV flow format.

Dataset location:

```text
data/raw/CICIDS2017/
```

Total loaded data:

```text
2,830,743 rows
79 columns including Label
```

Original class distribution after label grouping:

| Class | Samples |
|---|---:|
| Normal | 2,273,097 |
| DoS | 252,661 |
| PortScan | 158,930 |
| DDoS | 128,027 |
| BruteForce | 13,835 |
| Bot | 1,966 |
| Infiltration | 47 |

The dataset is highly imbalanced. `Normal` traffic dominates the dataset, while `Infiltration` has only 47 samples.

## Label Grouping

The original CICIDS2017 labels are grouped in `src/data/preprocess.py`.

| Original Label | Grouped Label |
|---|---|
| `BENIGN`, `Benign`, `Normal` | `Normal` |
| `DoS Hulk`, `DoS GoldenEye`, `DoS Slowloris`, `DoS Slowhttptest` | `DoS` |
| `DDoS` | `DDoS` |
| `PortScan` | `PortScan` |
| `FTP-Patator`, `SSH-Patator` | `BruteForce` |
| `Bot` | `Bot` |
| `Infiltration`, `Heartbleed` | `Infiltration` |
| `Web Attack ...` | `Web` |

In the current full-dataset experiments, the final observed classes are `Normal`, `DoS`, `PortScan`, `DDoS`, `BruteForce`, `Bot`, and `Infiltration`.

## Preprocessing Pipeline

The preprocessing logic is implemented in `src/data/preprocess.py`.

### 1. Column Name Normalization

Column names are stripped to remove leading/trailing whitespace.

```python
df_clean.columns = df_clean.columns.str.strip()
```

### 2. Metadata Removal

Identifier-like metadata columns are removed before training:

```text
Flow ID
Source IP
Source Port
Destination IP
Destination Port
Protocol
Timestamp
```

The pipeline also removes equivalent columns produced by flow extractors:

```text
flow_id
src_ip
src_port
dst_ip
dst_port
protocol
timestamp
```

These columns are removed because they may cause the model to learn dataset-specific identifiers instead of general traffic behavior.

### 3. Missing and Infinite Values

- `inf` and `-inf` are replaced with `NaN`.
- Missing values are filled with `0`.
- Non-numeric feature columns are converted to numeric values.

```python
df_clean.replace([np.inf, -np.inf], np.nan, inplace=True)
df_clean.fillna(0, inplace=True)
```

### 4. Train/Validation/Test Split

The cleaned dataset is split into:

| Split | Ratio |
|---|---:|
| Train | 70% |
| Validation | 10% |
| Test | 20% |

Stratified splitting is used when each class has enough samples.

### 5. Label Encoding

String labels are encoded using `LabelEncoder`.

### 6. Feature Scaling

Features are scaled using `StandardScaler`.

Important details:

- The scaler is fitted only on the training set.
- Validation and test sets only use `transform`.
- In inference mode, the saved `scaler.pkl` is reused.

## Class Imbalance Handling

The class imbalance strategy is implemented in `handle_imbalance()`.

The current approach is conservative and consists of:

1. Under-sampling the `Normal` class.
2. Applying light SMOTE to selected minority classes.
3. Avoiding aggressive synthetic oversampling for extremely rare classes.

### 1. Normal Under-Sampling

The `Normal` class is reduced to at most 300,000 training samples:

```python
target_normal = min(300000, normal_count)
```

This prevents the model from being dominated by normal traffic and producing misleadingly high accuracy or weighted-F1.

### 2. Conservative SMOTE

SMOTE is applied only when a class has enough samples. Very rare classes are not aggressively oversampled because synthetic samples from tiny classes may be unreliable.

| Condition | Strategy |
|---|---|
| Class has fewer than 100 samples | Do not increase |
| `Bot` | Increase up to 1.5x, capped at 2,500 |
| Class has fewer than 2,000 samples | Increase up to 1.5x, capped at 4,000 |
| Class has fewer than 10,000 samples | Increase up to 1.25x, capped at 12,000 |
| Large classes | Keep unchanged |

### Training Distribution After Imbalance Handling

For full-dataset experiments, the training distribution after imbalance handling is:

| Class | Training Samples After Handling |
|---|---:|
| Normal | 300,000 |
| DoS | 176,863 |
| PortScan | 111,251 |
| DDoS | 89,618 |
| BruteForce | 12,000 |
| Bot | 2,064 |
| Infiltration | 33 |

`Infiltration` remains extremely small, so its metrics can fluctuate heavily and should be interpreted carefully.

## Models

| Model | Role |
|---|---|
| Logistic Regression | Linear baseline |
| KNN | Distance-based baseline |
| Random Forest | Stable tree ensemble |
| XGBoost | Boosting model with regularization and feature importance |

### XGBoost Configuration

XGBoost is configured with anti-overfitting settings:

```text
n_estimators = 600
max_depth = 4
learning_rate = 0.04
subsample = 0.8
colsample_bytree = 0.75
reg_alpha = 2.0
reg_lambda = 8.0
gamma = 2.0
min_child_weight = 10
tree_method = hist
```

When training XGBoost with imbalance handling, the pipeline also uses clipped balanced sample weights:

```text
min weight = 0.75
max weight = 2.00
```

## Installation

```bash
pip install -r requirements.txt
```

For PCAP analysis, make sure the Scapy/CICFlowMeter-related dependencies are installed correctly.

## Training

By default, training uses imbalance handling:

```bash
python3 main.py --train data/raw/CICIDS2017 --model xgb
```

Train a specific model:

```bash
python3 main.py --train data/raw/CICIDS2017 --model rf
python3 main.py --train data/raw/CICIDS2017 --model lr
python3 main.py --train data/raw/CICIDS2017 --model knn
```

Train without imbalance handling:

```bash
python3 main.py --train data/raw/CICIDS2017 --model xgb --no-imbalance
```

Train on a smaller sample for quick experiments:

```bash
python3 main.py --train data/raw/CICIDS2017 --model xgb --sample 100000
```

Save model artifacts to a custom directory:

```bash
python3 main.py --train data/raw/CICIDS2017 --model xgb --model-dir models/final
```

Saved artifacts:

```text
model.pkl
<model_name>.pkl
scaler.pkl
label_encoder.pkl
features.pkl
```

## Testing

### 1. Internal Validation/Test Split

Each training run automatically reports:

- Precision
- Recall
- F1-score
- Macro-F1
- Weighted-F1
- Cross-validation when applicable

These are the main results used for model evaluation.

### 2. PCAP Testing

Run offline PCAP analysis:

```bash
python3 main.py --pcap data/pcap/test1.pcap
```

Examples:

```bash
python3 main.py --pcap data/pcap/task1.dos_attacker.pcap
python3 main.py --pcap data/pcap/task3.dos_victim.pcap
```

Important: PCAP mode loads artifacts from `models/final/` by default:

```text
models/final/model.pkl
models/final/scaler.pkl
models/final/features.pkl
models/final/label_encoder.pkl
```

To test a different trained model, pass its artifact directory with `--model-dir`.

Example for testing the XGBoost model trained with imbalance handling:

```bash
python3 main.py --pcap data/pcap/test1.pcap --model-dir experiments/models/report_full_xgb_with_imbalance
```

## Full Dataset Results

The following results are based on training logs stored in:

```text
reports/training_runs/
```

Detailed summary:

```text
reports/training_runs/model_results_summary.md
```

### Overall Results

| Model | Imbalance Handling | Validation Macro-F1 | Validation Weighted-F1 | Test Macro-F1 | Test Weighted-F1 | CV Weighted-F1 |
|---|---|---:|---:|---:|---:|---:|
| Logistic Regression | No | 0.4788 | 0.9224 | 0.4791 | 0.9224 | Skipped |
| Logistic Regression | Yes | 0.5030 | 0.8940 | 0.4554 | 0.8938 | 0.8821 +/- 0.0018 |
| KNN | No | 0.8818 | 0.9981 | 0.8769 | 0.9982 | 0.9834 +/- 0.0011 |
| KNN | Yes | 0.8252 | 0.9916 | 0.8378 | 0.9914 | Skipped |
| XGBoost | No | 0.8682 | 0.9987 | 0.8243 | 0.9988 | Skipped |
| XGBoost | Yes | 0.9459 | 0.9984 | 0.9215 | 0.9985 | 0.9981 +/- 0.0002 |
| Random Forest | No | 0.9536 | 0.9988 | 0.9378 | 0.9987 | Skipped |
| Random Forest | Yes | 0.9497 | 0.9983 | 0.9280 | 0.9985 | 0.9979 +/- 0.0002 |

### Per-Class Test F1

| Model | Imbalance | Bot | BruteForce | DDoS | DoS | Infiltration | Normal | PortScan |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression | No | 0.00 | 0.00 | 0.75 | 0.78 | 0.00 | 0.96 | 0.87 |
| Logistic Regression | Yes | 0.00 | 0.00 | 0.69 | 0.76 | 0.00 | 0.93 | 0.80 |
| KNN | No | 0.70 | 0.99 | 1.00 | 1.00 | 0.46 | 1.00 | 1.00 |
| KNN | Yes | 0.55 | 0.91 | 1.00 | 0.99 | 0.46 | 0.99 | 0.96 |
| XGBoost | No | 0.78 | 1.00 | 1.00 | 1.00 | 0.00 | 1.00 | 1.00 |
| XGBoost | Yes | 0.75 | 1.00 | 1.00 | 1.00 | 0.71 | 1.00 | 1.00 |
| Random Forest | No | 0.77 | 1.00 | 1.00 | 1.00 | 0.80 | 1.00 | 1.00 |
| Random Forest | Yes | 0.79 | 1.00 | 1.00 | 1.00 | 0.71 | 1.00 | 1.00 |

## Result Interpretation

### 1. Weighted-F1 Is Not Enough

Weighted-F1 and accuracy are very high for most models because the dataset is dominated by `Normal` traffic.

For this project, the more important metrics are:

- Macro-F1
- Minority-class recall/F1
- Confusion matrix

### 2. Random Forest Achieved the Best Overall Result

Random Forest without imbalance handling achieved:

```text
Test Macro-F1 = 0.9378
Test Weighted-F1 = 0.9987
```

This is the best full-dataset result among the trained models.

### 3. XGBoost Benefited the Most From Imbalance Handling

XGBoost before imbalance handling:

```text
Test Macro-F1 = 0.8243
Infiltration F1 = 0.00
```

XGBoost after imbalance handling:

```text
Test Macro-F1 = 0.9215
Infiltration F1 = 0.71
```

This shows that the conservative imbalance strategy helped XGBoost detect rare attack classes more effectively.

### 4. KNN Is Expensive for Large-Scale Inference

KNN completed on the full dataset, but prediction was slow because it needs to compute distances against a large training set.

KNN without imbalance handling performed better than KNN with imbalance handling:

```text
KNN no imbalance Test Macro-F1 = 0.8769
KNN with imbalance Test Macro-F1 = 0.8378
```

This is expected because KNN is sensitive to sample density. Under-sampling `Normal` changes the feature-space structure and can reduce performance.

### 5. Logistic Regression Is Only a Baseline

Full-dataset Logistic Regression produced convergence warnings:

```text
ConvergenceWarning: The max_iter was reached which means the coef_ did not converge
```

It failed to detect `Bot`, `BruteForce`, and `Infiltration` in the full-dataset test set. It should be treated as a linear baseline, not as the main model.

## Recommended Model

If the goal is the best overall metric:

```text
Random Forest without imbalance handling
```

If the goal is a stronger imbalance-handling story and better explainability:

```text
XGBoost with conservative imbalance handling
```

Suggested report wording:

```text
Random Forest achieved the best overall Macro-F1 on the full dataset, while XGBoost with conservative imbalance handling showed the clearest improvement on minority attack classes, especially Infiltration.
```

## Training Logs

All training logs are stored in:

```text
reports/training_runs/
```

Logs include:

```text
full_xgb_no_imbalance.log
full_xgb_with_imbalance.log
full_rf_no_imbalance.log
full_rf_with_imbalance.log
full_lr_no_imbalance.log
full_lr_with_imbalance.log
full_knn_no_imbalance.log
full_knn_with_imbalance.log
sample100k_lr_no_imbalance.log
sample100k_lr_with_imbalance.log
sample100k_knn_no_imbalance.log
sample100k_knn_with_imbalance.log
```

## Limitations

- CICIDS2017 has extremely rare classes, especially `Infiltration` with only 47 samples.
- Metrics for rare classes can fluctuate because test support is very small.
- PCAP testing depends heavily on flow extraction quality.
- If extracted PCAP features differ from CICIDS2017 feature distributions, predictions may be unreliable.
- KNN has high inference cost on large datasets.
- Logistic Regression does not converge well with the current full-dataset setup.

## Future Work

- Add a dedicated CSV-flow prediction command that can compute metrics when labels are available.
- Try LightGBM or CatBoost.
- Separate binary detection from multi-class classification.
- Add SHAP or richer feature-importance analysis.
- Improve PCAP flow extraction to better match CICIDS2017 feature definitions.
