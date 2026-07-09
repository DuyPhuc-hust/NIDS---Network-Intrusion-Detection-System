# NIDS - Network Intrusion Detection System

Du an xay dung he thong phat hien xam nhap mang dua tren Machine Learning, su dung dataset CICIDS2017. Pipeline ho tro train nhieu model, xu ly mat can bang du lieu, danh gia bang validation/test set, test voi PCAP va phan tich file log.

## Muc Tieu

- Phan loai luu luong mang thanh cac lop: `Normal`, `DoS`, `DDoS`, `PortScan`, `BruteForce`, `Bot`, `Infiltration`.
- So sanh cac model: Logistic Regression, KNN, Random Forest, XGBoost.
- Danh gia anh huong cua dataset imbalance truoc va sau xu ly.
- Ho tro test offline bang PCAP thong qua flow extraction.
- Co module phu `log_analyze.py` de phan tich log dang text/CSV theo rule; module nay khong duoc dung trong pipeline train/test model ML.

## Cau Truc Thu Muc

```text
NIDS---Network-Intrusion-Detection-System/
├── data/
│   ├── raw/
│   │   └── CICIDS2017/
│   │       ├── Monday-WorkingHours.pcap_ISCX.csv
│   │       ├── Tuesday-WorkingHours.pcap_ISCX.csv
│   │       ├── Wednesday-workingHours.pcap_ISCX.csv
│   │       ├── Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
│   │       ├── Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
│   │       ├── Friday-WorkingHours-Morning.pcap_ISCX.csv
│   │       ├── Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
│   │       └── Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
│   ├── pcap/
│   │   ├── test1.pcap
│   │   ├── task1.dos_attacker.pcap
│   │   ├── task1.dos_victim.pcap
│   │   ├── task3.dos_attacker.pcap
│   │   └── task3.dos_victim.pcap
│   └── logs/
│       └── *.log / *.csv
├── models/
│   ├── model.pkl
│   ├── scaler.pkl
│   ├── label_encoder.pkl
│   ├── features.pkl
│   ├── report_full_xgb_with_imbalance/
│   ├── report_full_rf_no_imbalance/
│   ├── report_full_lr_no_imbalance/
│   └── report_full_knn_no_imbalance/
├── outputs/
│   ├── log_analysis_findings.csv
│   ├── xgb_feature_importance.csv
│   └── xgb_confusion_matrix.csv
├── reports/
│   └── training_runs/
│       ├── model_results_summary.md
│       └── *.log
├── src/
│   ├── data/
│   │   ├── loader.py
│   │   └── preprocess.py
│   ├── models/
│   │   ├── train.py
│   │   └── predict.py
│   ├── pipeline/
│   │   └── train_pipeline.py
│   └── utils/
├── live_analyze.py
├── log_analyze.py              # optional rule-based log analyzer, khong phai ML model
├── main.py
├── requirements.txt
└── README.md
```

## Dataset

Dataset chinh duoc dung la CICIDS2017 flow CSV.

Vi tri dataset:

```text
data/raw/CICIDS2017/
```

Tong so flow da load:

```text
2,830,743 rows
79 columns, bao gom Label
```

Phan bo lop ban dau sau khi gom nhom:

| Lop | So mau |
|---|---:|
| Normal | 2,273,097 |
| DoS | 252,661 |
| PortScan | 158,930 |
| DDoS | 128,027 |
| BruteForce | 13,835 |
| Bot | 1,966 |
| Infiltration | 47 |

Nhan xet: dataset bi mat can bang rat manh. `Normal` chiem phan lon, trong khi `Infiltration` chi co 47 mau tren toan bo dataset.

## Gom Nhom Label

Trong `src/data/preprocess.py`, cac label goc cua CICIDS2017 duoc gom thanh cac lop chinh:

| Label goc | Label sau gom nhom |
|---|---|
| `BENIGN`, `Benign`, `Normal` | `Normal` |
| `DoS Hulk`, `DoS GoldenEye`, `DoS Slowloris`, `DoS Slowhttptest` | `DoS` |
| `DDoS` | `DDoS` |
| `PortScan` | `PortScan` |
| `FTP-Patator`, `SSH-Patator` | `BruteForce` |
| `Bot` | `Bot` |
| `Infiltration`, `Heartbleed` | `Infiltration` |
| `Web Attack ...` | `Web` |

Luu y: trong cac lan train hien tai, sau khi clean va map label, cac lop duoc ghi nhan trong full dataset la `Normal`, `DoS`, `PortScan`, `DDoS`, `BruteForce`, `Bot`, `Infiltration`.

## Tien Xu Ly Du Lieu

Pipeline tien xu ly nam trong `src/data/preprocess.py`.

### 1. Chuan hoa ten cot

Tat ca ten cot duoc strip khoang trang:

```python
df_clean.columns = df_clean.columns.str.strip()
```

### 2. Loai bo cot metadata

Nhung cot dinh danh khong dung de train duoc loai bo:

```text
Flow ID
Source IP
Source Port
Destination IP
Destination Port
Protocol
Timestamp
```

Va cac bien the tu flow extractor:

```text
flow_id
src_ip
src_port
dst_ip
dst_port
protocol
timestamp
```

Ly do: cac cot nay co the lam model hoc "dau vet dataset" thay vi hoc hanh vi traffic that.

### 3. Xu ly gia tri loi

- Thay `inf`, `-inf` thanh `NaN`.
- Dien gia tri thieu bang `0`.
- Ep cac cot feature ve numeric.

```python
df_clean.replace([np.inf, -np.inf], np.nan, inplace=True)
df_clean.fillna(0, inplace=True)
```

### 4. Chia du lieu

Pipeline chia dataset thanh:

| Tap | Ty le |
|---|---:|
| Train | 70% |
| Validation | 10% |
| Test | 20% |

Viec split co su dung stratify neu so mau moi lop du dieu kien.

### 5. Encode label

Label dang chuoi duoc ma hoa bang `LabelEncoder`.

### 6. Scale feature

Feature duoc scale bang `StandardScaler`.

Quan trong:

- Scaler chi `fit` tren train set.
- Validation/test chi dung `transform`.
- Khi inference, model dung lai `scaler.pkl` da save.

## Xu Ly Dataset Imbalance

Chien luoc xu ly imbalance nam trong ham `handle_imbalance()`.

Phuong phap hien tai la conservative imbalance handling, gom:

1. Under-sampling lop `Normal`.
2. SMOTE nhe cho mot so lop thieu so.
3. Khong ep SMOTE manh voi lop qua hiem.

### 1. Under-sampling Normal

Lop `Normal` duoc giam toi da ve 300,000 mau trong train set:

```python
target_normal = min(300000, normal_count)
```

Ly do: `Normal` qua lon co the lam accuracy/weighted-F1 rat cao nhung model bo qua attack hiem.

### 2. Conservative SMOTE

SMOTE chi duoc ap dung khi lop co du mau. Cac lop qua nho se khong bi tao mau tong hop qua nhieu.

Quy tac:

| Dieu kien | Cach xu ly |
|---|---|
| Lop co duoi 100 mau | Khong tang |
| `Bot` | Tang toi da 1.5 lan, gioi han 2,500 |
| Lop duoi 2,000 mau | Tang toi da 1.5 lan, gioi han 4,000 |
| Lop duoi 10,000 mau | Tang toi da 1.25 lan, gioi han 12,000 |
| Lop lon | Giu nguyen |

Ly do: neu lop qua it, SMOTE co the tao mau nhan tao kem tin cay va lam ket qua trong dep hon thuc te.

### Phan bo train sau xu ly imbalance

Voi full dataset, train set sau xu ly co phan bo:

| Lop | So mau train sau xu ly |
|---|---:|
| Normal | 300,000 |
| DoS | 176,863 |
| PortScan | 111,251 |
| DDoS | 89,618 |
| BruteForce | 12,000 |
| Bot | 2,064 |
| Infiltration | 33 |

Nhan xet: `Infiltration` van rat it, nen ket qua cua lop nay dao dong manh va khong nen dien giai qua muc.

## Model Da Su Dung

| Model | Vai tro |
|---|---|
| Logistic Regression | Baseline tuyen tinh |
| KNN | Baseline dua tren khoang cach |
| Random Forest | Ensemble tree-based, manh va on dinh |
| XGBoost | Boosting, co regularization va feature importance |

### Cau hinh XGBoost

XGBoost duoc cau hinh theo huong giam overfitting:

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

Khi train XGBoost voi imbalance handling, pipeline dung them sample weight co gioi han:

```text
min weight = 0.75
max weight = 2.00
```

## Cach Cai Dat

```bash
pip install -r requirements.txt
```

Neu dung PCAP, can dam bao cac thu vien lien quan den Scapy/CICFlowMeter da cai thanh cong.

## Cach Train Model

Train mac dinh co xu ly imbalance:

```bash
python3 main.py --train data/raw/CICIDS2017 --model xgb
```

Chon model:

```bash
python3 main.py --train data/raw/CICIDS2017 --model rf
python3 main.py --train data/raw/CICIDS2017 --model lr
python3 main.py --train data/raw/CICIDS2017 --model knn
```

Train khong xu ly imbalance:

```bash
python3 main.py --train data/raw/CICIDS2017 --model xgb --no-imbalance
```

Train voi sample nho de thu nhanh:

```bash
python3 main.py --train data/raw/CICIDS2017 --model xgb --sample 100000
```

Luu model vao thu muc rieng:

```bash
python3 main.py --train data/raw/CICIDS2017 --model xgb --model-dir models/report_full_xgb_with_imbalance
```

Artifact sau khi train:

```text
model.pkl
<model_name>.pkl
scaler.pkl
label_encoder.pkl
features.pkl
```

## Cach Test Model

### 1. Test noi bo bang validation/test split

Moi lan train, pipeline tu dong in:

- Precision
- Recall
- F1-score
- Macro-F1
- Weighted-F1
- Cross-validation neu du dieu kien

Day la ket qua dung cho bao cao model.

### 2. Test bang PCAP

Dung PCAP trong `data/pcap/`:

```bash
python3 main.py --pcap data/pcap/test1.pcap
```

Vi du:

```bash
python3 main.py --pcap data/pcap/task1.dos_attacker.pcap
python3 main.py --pcap data/pcap/task3.dos_victim.pcap
```

Luu y: PCAP mode dang load artifact mac dinh tu:

```text
models/model.pkl
models/scaler.pkl
models/features.pkl
models/label_encoder.pkl
```

Neu muon test mot model cu the, can dua artifact cua model do ve thu muc `models/`.

Vi du muon test XGBoost da xu ly imbalance:

```bash
cp models/report_full_xgb_with_imbalance/model.pkl models/model.pkl
cp models/report_full_xgb_with_imbalance/scaler.pkl models/scaler.pkl
cp models/report_full_xgb_with_imbalance/features.pkl models/features.pkl
cp models/report_full_xgb_with_imbalance/label_encoder.pkl models/label_encoder.pkl
python3 main.py --pcap data/pcap/test1.pcap
```

### 3. Phan tich log optional

`log_analyze.py` la module phu, khong dung XGBoost/RF/KNN/LR va khong tham gia vao ket qua train model. Module nay chi scan log text/CSV bang rule de tim cac dau hieu nhu SQL injection, XSS, path traversal, scanner, sensitive path hoac brute force.

Phan tich file log:

```bash
python3 main.py --log data/logs/sample_access.log
```

Ket qua mac dinh luu vao:

```text
outputs/log_analysis_findings.csv
```

Co the doi threshold brute force:

```bash
python3 main.py --log data/logs/vpn_auth.log --brute-force-threshold 10
```

## Ket Qua Train Full Dataset

Ket qua duoi day duoc lay tu cac lan chay trong:

```text
reports/training_runs/
```

Tong hop chi tiet nam tai:

```text
reports/training_runs/model_results_summary.md
```

### Bang Tong Hop

| Model | Xu ly imbalance | Validation Macro-F1 | Validation Weighted-F1 | Test Macro-F1 | Test Weighted-F1 | CV Weighted-F1 |
|---|---|---:|---:|---:|---:|---:|
| Logistic Regression | No | 0.4788 | 0.9224 | 0.4791 | 0.9224 | Skipped |
| Logistic Regression | Yes | 0.5030 | 0.8940 | 0.4554 | 0.8938 | 0.8821 +/- 0.0018 |
| KNN | No | 0.8818 | 0.9981 | 0.8769 | 0.9982 | 0.9834 +/- 0.0011 |
| KNN | Yes | 0.8252 | 0.9916 | 0.8378 | 0.9914 | Skipped |
| XGBoost | No | 0.8682 | 0.9987 | 0.8243 | 0.9988 | Skipped |
| XGBoost | Yes | 0.9459 | 0.9984 | 0.9215 | 0.9985 | 0.9981 +/- 0.0002 |
| Random Forest | No | 0.9536 | 0.9988 | 0.9378 | 0.9987 | Skipped |
| Random Forest | Yes | 0.9497 | 0.9983 | 0.9280 | 0.9985 | 0.9979 +/- 0.0002 |

### Per-class Test F1

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

## Nhan Xet Ket Qua

### 1. Weighted-F1 rat cao nhung khong du de ket luan model tot

Dataset co qua nhieu `Normal`, nen Weighted-F1 va accuracy de bi cao. Vi vay can xem them:

- Macro-F1
- Recall/F1 cua lop nho
- Confusion matrix

### 2. Random Forest dang co ket qua tong the tot nhat

Random Forest khong xu ly imbalance dat:

```text
Test Macro-F1 = 0.9378
Test Weighted-F1 = 0.9987
```

Day la ket qua cao nhat tren full dataset.

### 3. XGBoost the hien ro tac dung cua xu ly imbalance

XGBoost truoc imbalance:

```text
Test Macro-F1 = 0.8243
Infiltration F1 = 0.00
```

XGBoost sau imbalance:

```text
Test Macro-F1 = 0.9215
Infiltration F1 = 0.71
```

Dieu nay cho thay imbalance handling giup XGBoost khong bo qua lop rat hiem.

### 4. KNN khong hop de deploy voi dataset lon

KNN full dataset chay duoc, nhung rat cham o buoc prediction vi phai tinh khoang cach voi tap train lon.

KNN khong xu ly imbalance tot hon KNN da xu ly:

```text
KNN no imbalance Test Macro-F1 = 0.8769
KNN with imbalance Test Macro-F1 = 0.8378
```

Ly do: KNN phu thuoc vao mat do diem du lieu. Under-sampling `Normal` lam thay doi cau truc khong gian feature, co the lam ket qua giam.

### 5. Logistic Regression chi phu hop lam baseline

LR full dataset co warning:

```text
ConvergenceWarning: The max_iter was reached which means the coef_ did not converge
```

LR khong detect duoc `Bot`, `BruteForce`, `Infiltration` tren test set. Do do LR nen duoc trinh bay la baseline tuyen tinh, khong phai model chinh.

## Model Nen Chon

Neu uu tien metric tong the:

```text
Random Forest without imbalance handling
```

Neu uu tien cau chuyen xu ly imbalance va kha nang giai thich:

```text
XGBoost with conservative imbalance handling
```

De bao cao, co the viet:

```text
Random Forest achieved the best overall Macro-F1 on the full dataset, while XGBoost with conservative imbalance handling showed the clearest improvement on minority attack classes, especially Infiltration.
```

## File Log Ket Qua

Tat ca log train da luu tai:

```text
reports/training_runs/
```

Gom:

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

## Han Che

- CICIDS2017 co mot so lop qua it, dac biet `Infiltration` chi co 47 mau.
- Ket qua voi lop hiem co the dao dong manh vi test support rat nho.
- PCAP test phu thuoc chat luong flow extraction; neu flow feature khac CICIDS2017, ket qua co the lech.
- KNN co chi phi inference cao voi dataset lon.
- Logistic Regression hoi tu kem voi full dataset hien tai.

## Huong Phat Trien

- Them script chon `model-dir` khi test PCAP thay vi phai copy artifact ve `models/`.
- Them file test CSV flow rieng de predict va tinh metric neu co label.
- Thu them LightGBM/CatBoost.
- Tach binary detection va multi-class detection.
- Them SHAP/feature importance cho giai thich model.
- Cai thien flow extraction tu PCAP de gan hon feature cua CICIDS2017.
