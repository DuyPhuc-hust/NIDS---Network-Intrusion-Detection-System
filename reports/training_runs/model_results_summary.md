# NIDS Model Training Results

## Dataset

- Dataset: CICIDS2017 CSV flow dataset
- Path: `data/raw/CICIDS2017`
- Total rows loaded: 2,830,743
- Number of features before label split: 78

### Original Class Distribution

| Class | Samples |
|---|---:|
| Normal | 2,273,097 |
| DoS | 252,661 |
| PortScan | 158,930 |
| DDoS | 128,027 |
| BruteForce | 13,835 |
| Bot | 1,966 |
| Infiltration | 47 |

### Conservative Imbalance Handling

For full-dataset experiments, imbalance handling produced this training distribution:

| Class | Train Samples After Handling |
|---|---:|
| Normal | 300,000 |
| DoS | 176,863 |
| PortScan | 111,251 |
| DDoS | 89,618 |
| BruteForce | 12,000 |
| Bot | 2,064 |
| Infiltration | 33 |

## Full Dataset Results

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

## Full Dataset Per-Class Test F1

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

## Sample 100k Results

KNN and Logistic Regression were also run on the full dataset. KNN completed but was much slower during prediction than tree-based models. Logistic Regression produced convergence warnings and weak minority-class performance.

Sample class distribution:

| Class | Samples |
|---|---:|
| Normal | 80,263 |
| DoS | 9,013 |
| PortScan | 5,632 |
| DDoS | 4,456 |
| BruteForce | 493 |
| Bot | 69 |
| Infiltration | 2 |

| Model | Imbalance Handling | Validation Macro-F1 | Validation Weighted-F1 | Test Macro-F1 | Test Weighted-F1 | CV Weighted-F1 |
|---|---|---:|---:|---:|---:|---:|
| Logistic Regression | No | 0.5816 | 0.9399 | 0.5760 | 0.9361 | Skipped |
| Logistic Regression | Yes | 0.5897 | 0.9425 | 0.5826 | 0.9380 | Skipped |
| KNN | No | 0.9260 | 0.9938 | 0.8832 | 0.9932 | 0.9831 +/- 0.0012 |
| KNN | Yes | 0.9191 | 0.9932 | 0.8823 | 0.9930 | 0.9830 +/- 0.0018 |

## Interpretation

- Weighted-F1 is very high for most models because the dataset is dominated by Normal traffic.
- Macro-F1 is more useful for judging imbalance because it treats minority classes more equally.
- XGBoost improved strongly after imbalance handling: Test Macro-F1 increased from 0.8243 to 0.9215, and Infiltration F1 increased from 0.00 to 0.71.
- Random Forest achieved the best full-dataset Test Macro-F1 at 0.9378 without imbalance handling.
- Random Forest with imbalance handling improved Bot recall but slightly reduced overall Macro-F1.
- Logistic Regression performed poorly on both the full dataset and the 100k sample; on the full-dataset test set it did not detect Bot, BruteForce, or Infiltration.
- KNN achieved moderate full-dataset Macro-F1, but it is not ideal for full-scale deployment because prediction cost grows with the training set.

## Saved Training Summary

This folder intentionally keeps only `model_results_summary.md`.

Raw per-run `.log` files are local runtime artifacts and are not required for the GitHub submission. The final report and README use this summary file together with the diagnostic CSV files in `reports/final/`.
