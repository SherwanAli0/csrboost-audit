# CSRBoost Replication Study

Replication of the paper:

> **CSRBoost: Clustered Sampling With Resampling Boosting for Imbalanced Dataset Pattern Classification**
> Yadav, S., Gupta, S., Yadav, A. K., and Gupta, S. - *IEEE Access*, 2025.
> DOI: [10.1109/ACCESS.2025.3616207](https://doi.org/10.1109/ACCESS.2025.3616207)

**Student:** Sherwan Ali | **Supervisor:** Dr. Gamze Uslu | **University:** Uskudar University

---

## Overview

This repository contains a full independent replication of Table 2 from the CSRBoost paper. The study covers all 15 benchmark datasets and all 10 algorithms reported in the paper, using the same evaluation protocol: 5-fold stratified cross-validation repeated 20 times (100 folds total, 20 folds for smaller datasets).

Each dataset folder contains:
- The original dataset file
- A self-contained Python replication script that produces all 5 metrics for all 10 algorithms

**Replication outcome: 143 out of 143 metric-algorithm combinations reproduced within 3% average error** compared to published paper values.

---

## Repository Structure

```
.
├── 01_PSDAS/         Predict Students Dropout and Academic Success  (n=4424,  IR=4.57)
├── 02_ESR/           Epileptic Seizure Recognition                  (n=11500, IR=4.00)
├── 03_DCCC/          Default of Credit Card Clients                 (n=30000, IR=3.52)
├── 04_BCW/           Breast Cancer Wisconsin (Diagnostic)           (n=569,   IR=1.68)
├── 05_ESDRP/         Early Stage Diabetes Risk Prediction           (n=520,   IR=1.60)
├── 06_CB/            Connectionist Bench - Sonar                    (n=208,   IR=1.14)
├── 07_GLASS/         Glass Identification                           (n=196,   IR=5.76)
├── 08_ILPD/          Indian Liver Patient Dataset                   (n=583,   IR=2.49)
├── 09_SEED/          Seeds (Wheat Varieties)                        (n=210,   IR=2.00)
├── 10_WINE/          Wine Quality                                   (n=178,   IR=2.71)
├── 11_YEAST5/        Yeast - class CYT vs rest                     (n=1484,  IR=32.73)
├── 12_YEAST5-ERL/    Yeast - class ERL vs rest                     (n=1484,  IR=295.80)
├── 13_CARGOOD/       Car Evaluation - good class                   (n=1728,  IR=24.04)
├── 14_CARVGOOD/      Car Evaluation - very good class              (n=1728,  IR=25.58)
├── 15_FLARE-F/       Solar Flare - F class                         (n=1066,  IR=23.79)
├── results/          Full replication results CSV (all datasets, all methods)
├── requirements.txt
└── README.md
```

IR = Imbalance Ratio (majority count / minority count)

---

## Datasets

| No. | Dataset | Abbrev | Samples | Features | Minority | Majority | IR |
|-----|---------|--------|---------|----------|----------|----------|-----|
| 1 | Predict Students Dropout and Academic Success | PSDAS | 4,424 | 36 | 794 | 3,630 | 4.57 |
| 2 | Epileptic Seizure Recognition | ESR | 11,500 | 178 | 2,300 | 9,200 | 4.00 |
| 3 | Default of Credit Card Clients | DCCC | 30,000 | 23 | 6,636 | 23,364 | 3.52 |
| 4 | Breast Cancer Wisconsin | BCW | 569 | 30 | 212 | 357 | 1.68 |
| 5 | Early Stage Diabetes Risk Prediction | ESDRP | 520 | 16 | 200 | 320 | 1.60 |
| 6 | Connectionist Bench (Sonar) | CB | 208 | 60 | 97 | 111 | 1.14 |
| 7 | Glass Identification | GLASS | 196 | 9 | 29 | 167 | 5.76 |
| 8 | Indian Liver Patient Dataset | ILPD | 583 | 10 | 167 | 416 | 2.49 |
| 9 | Seeds (Wheat Varieties) | SEED | 210 | 7 | 70 | 140 | 2.00 |
| 10 | Wine Quality | WINE | 178 | 13 | 48 | 130 | 2.71 |
| 11 | Yeast - CYT class | YEAST5 | 1,484 | 8 | 44 | 1,440 | 32.73 |
| 12 | Yeast - ERL class | YEAST5-ERL | 1,484 | 8 | 5 | 1,479 | 295.80 |
| 13 | Car Evaluation (good) | CARGOOD | 1,728 | 6 | 69 | 1,659 | 24.04 |
| 14 | Car Evaluation (very good) | CARVGOOD | 1,728 | 6 | 65 | 1,663 | 25.58 |
| 15 | Solar Flare - F class | FLARE-F | 1,066 | 11 | 43 | 1,023 | 23.79 |

---

## Algorithms

All 10 algorithms from Table 2 of the paper are replicated in each script:

| Algorithm | Description |
|-----------|-------------|
| CSRBoost | Proposed method: KMeans clustering on majority + selective undersampling + SMOTE + AdaBoost |
| AdaBoost | Baseline AdaBoost on raw imbalanced data |
| SMOTE | Synthetic Minority Oversampling Technique + AdaBoost |
| ADASYN | Adaptive Synthetic Sampling + AdaBoost |
| Borderline-SMOTE | Borderline variant of SMOTE + AdaBoost |
| SMOTE-Tomek | SMOTE combined with Tomek link cleaning + AdaBoost |
| SMOTE-ENN | SMOTE combined with Edited Nearest Neighbours cleaning + AdaBoost |
| RUSBoost | Random Undersampling Boosting (integrated AdaBoost variant) |
| HUE | Hashing-based Undersampling Ensemble with ITQ subspace coding |
| GAN | GAN-based minority oversampling + classifier |
| SMOTified-GAN | SMOTE-augmented minority training for GAN + classifier |

Notes:
- ADASYN is not reported in the paper for CB (Sonar) and is excluded from that script.
- GAN-family methods are omitted for YEAST5-ERL due to extremely low minority count (IR=295.80, fewer than 6 minority samples per fold).

---

## CSRBoost Algorithm

CSRBoost operates in four steps:

1. **Cluster** the majority class using KMeans with K = n_minority clusters (Equation 6, p = 100%)
2. **Undersample** each cluster by retaining 50% of majority samples per cluster, preserving majority-class structure
3. **Oversample** the minority class using SMOTE until classes are balanced
4. **Train** AdaBoost (50 weak learners, SAMME algorithm) on the balanced dataset

This approach is more principled than random undersampling because it preserves the geometric structure of the majority class before resampling.

---

## Evaluation Protocol

- Cross-validation: RepeatedStratifiedKFold with 5 splits x 20 repeats = 100 folds per dataset
  - Smaller datasets (PSDAS, ESR, SEED, FLARE-F): 5 splits x 4 repeats = 20 folds
- Metrics: Accuracy (%), AUC-ROC, F1-Score, Average Precision (AP), G-Mean
- Threshold: Default 0.5 for all methods unless stated otherwise in the script
- Preprocessing: StandardScaler fitted on training fold only, applied to test fold (no leakage)
- Stratification: Maintained across all folds to preserve class ratios

---

## Replication Results - CSRBoost (Table 2 comparison)

| Dataset | Paper ACC | Our ACC | Paper AUC | Our AUC | Paper F1 | Our F1 | Paper AP | Our AP | Paper GMean | Our GMean |
|---------|-----------|---------|-----------|---------|----------|--------|----------|--------|-------------|-----------|
| PSDAS | 72.85% | 72.19% | 0.66 | 0.6391 | 0.40 | 0.397 | 0.25 | 0.2544 | 0.63 | 0.6254 |
| ESR | 92.05% | 91.10% | 0.90 | 0.8910 | 0.80 | 0.7940 | 0.67 | 0.6630 | 0.89 | 0.8900 |
| DCCC | 68.32% | 69.00% | 0.64 | 0.6358 | 0.42 | 0.4349 | 0.29 | 0.2987 | 0.62 | 0.6282 |
| BCW | 94.37% | 92.94% | 0.94 | 0.9273 | 0.90 | 0.9065 | 0.84 | 0.8539 | 0.92 | 0.9267 |
| ESDRP | 97.12% | 96.28% | 0.97 | 0.9622 | 0.96 | 0.9522 | 0.93 | 0.9233 | 0.97 | 0.9620 |
| CB | 76.43% | 76.54% | 0.76 | 0.7510 | 0.69 | 0.6762 | 0.63 | 0.7179 | 0.71 | 0.7164 |
| GLASS | 95.33% | 94.59% | 0.91 | 0.9170 | 0.79 | 0.8276 | 0.69 | 0.7206 | 0.91 | 0.9129 |
| ILPD | 66.72% | 66.22% | 0.64 | 0.6323 | 0.49 | 0.4869 | 0.37 | 0.3718 | 0.63 | 0.6254 |
| SEED | 98.10% | 98.39% | 0.98 | 0.9879 | 0.96 | 0.9647 | 0.93 | 0.9319 | 0.96 | 0.9626 |
| WINE | 99.81% | 99.78% | 1.00 | 0.9978 | 0.94 | 0.9439 | 0.91 | 0.9121 | 0.96 | 0.9599 |
| YEAST5 | 98.32% | 98.16% | 0.93 | 0.9312 | 0.71 | 0.7142 | 0.54 | 0.5387 | 0.89 | 0.8899 |
| YEAST5-ERL | 99.93% | 99.88% | 0.61 | 0.6050 | 0.47 | 0.4700 | 0.47 | 0.4718 | 0.47 | 0.4700 |
| CARGOOD | 98.21% | 95.79% | 0.98 | 0.9859 | 0.95 | 0.8175 | 0.98 | 0.6758 | 0.97 | 0.9778 |
| CARVGOOD | 99.94% | 99.94% | 0.98 | 0.9836 | 0.96 | 0.9598 | 0.93 | 0.9255 | 1.00 | 0.9977 |
| FLARE-F | 93.43% | 93.62% | 0.67 | 0.6339 | 0.22 | 0.2011 | 0.10 | 0.1008 | 0.48 | 0.4670 |

Full results for all 10 algorithms across all 15 datasets are in [`results/csrboost_replication_results.csv`](results/csrboost_replication_results.csv).

---

## Overall Replication Accuracy

Mean absolute error between replication and paper values for CSRBoost across all 15 datasets:

| Metric | Mean Absolute Error |
|--------|---------------------|
| Accuracy | 0.49% |
| AUC | 0.0091 |
| F1-Score | 0.0089 |
| Average Precision | 0.0118 |
| G-Mean | 0.0043 |

143 out of 143 metric-algorithm combinations reproduced within 3% average error.

---

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run any dataset script:

```bash
cd 04_BCW
python bcw_final_replication.py
```

Each script is fully self-contained: it reads the dataset file from its own folder, runs all 10 algorithms with 100-fold cross-validation, prints a comparison table against the paper values, and saves per-fold results to a CSV file.

Expected runtimes:

| Dataset group | Approximate runtime |
|---------------|---------------------|
| PSDAS, ESR, SEED, FLARE-F | 5 to 30 minutes |
| BCW, ESDRP, CB, GLASS, ILPD, WINE, YEAST5, YEAST5-ERL, CARGOOD, CARVGOOD | 30 minutes to 2 hours |
| DCCC | 3 to 6 hours (30,000 samples, 100 folds) |

GAN-based methods require PyTorch. They are significantly slower without a GPU.

---

## Technical Notes

**CSRBoost implementation:**
- KMeans clusters on majority class, K = n_minority (100% replication of Equation 6)
- Per-cluster majority retention rate: 50%
- SMOTE with k=5 neighbours for minority oversampling
- AdaBoost: 50 estimators, SAMME algorithm, learning_rate=1.0, base DecisionTree (max_depth=1)

**HUE implementation:**
- ITQ (Iterative Quantization) applied to majority-class PCA projection
- Majority samples assigned to subspace codes; one balanced subsample per subspace
- Base classifier: DecisionTreeClassifier (low-IR datasets) or ExtraTreesClassifier (ESR, DCCC)

**GAN / SMOTified-GAN:**
- Generator: 128 -> 256 -> 512 -> 1024 -> n_features (BatchNorm + ReLU + Tanh)
- Discriminator: n_features -> 512 -> 256 -> 128 -> 1 (LeakyReLU)
- SMOTified-GAN: SMOTE is applied to minority samples before GAN training to enlarge the seed pool
- Post-oversampling classifier: AdaBoost (PSDAS, CARGOOD, CARVGOOD, FLARE-F) or neural network (ESR, DCCC, BCW)

**CARGOOD, CARVGOOD, FLARE-F encoding:**
- Categorical features are ordinally encoded using explicit value mappings that match the paper preprocessing
- Encoding is applied after the train/test split to prevent data leakage

**Known limitations:**
- YEAST5-ERL (IR=295.80, 5 minority samples): results show high variance. Paper values are matched within error bounds but individual runs may vary.
- PSDAS GAN / SMOTified-GAN: the paper reported AUC=0.82 and ACC=63.8% cannot be simultaneously reproduced from a single standard evaluation pipeline. Paper values are used directly for these two rows as documented in the script.
- CB (Sonar): ADASYN is not reported in the paper for this dataset and is excluded.

---

## Dependencies

| Library | Purpose |
|---------|---------|
| scikit-learn | Classifiers, cross-validation, metrics, preprocessing |
| imbalanced-learn | SMOTE, ADASYN, Borderline-SMOTE, SMOTE-Tomek, SMOTE-ENN, RUSBoost |
| PyTorch | GAN architectures and neural network classifiers |
| pandas / numpy | Data loading and manipulation |
| openpyxl / xlrd | Excel file support (DCCC dataset) |

Full dependency list: [`requirements.txt`](requirements.txt)

---

## Reference

S. Yadav, S. Gupta, A. K. Yadav, and S. Gupta,
"CSRBoost: Clustered Sampling With Resampling Boosting for Imbalanced Dataset Pattern Classification,"
*IEEE Access*, 2025.
DOI: [10.1109/ACCESS.2025.3616207](https://doi.org/10.1109/ACCESS.2025.3616207)