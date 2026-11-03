# A Forensic Audit and Reverse-Engineering of CSRBoost

> Replication study and forensic audit of:
> S. Yadav, S. Gupta, A. K. Yadav, and S. Gupta. *CSRBoost: Clustered Sampling With Resampling Boosting for Imbalanced Dataset Pattern Classification.* IEEE Access, 2025.
> [DOI: 10.1109/ACCESS.2025.3616207](https://doi.org/10.1109/ACCESS.2025.3616207)

[![License: MIT](https://img.shields.io/badge/license-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Status: thesis manuscript](https://img.shields.io/badge/status-thesis%20manuscript-orange.svg)](#status)

**Author:** Sherwan Ali, Department of Computer Engineering, Uskudar University.
**Supervisor:** Dr. Gamze Uslu.
**Project type:** Solo undergraduate graduation thesis, 2026.

## Status

This repository contains an undergraduate graduation thesis manuscript and the accompanying replication code. **It has not been submitted to or peer-reviewed by any journal or conference, and has not been published.** Citations to it should be made as a thesis or technical report, not as a peer-reviewed publication. References elsewhere in this README to "published" results, "the publication", or "the paper" refer to the original CSRBoost paper by Yadav et al. (IEEE Access 2025), not to this work.

## Abstract

This repository documents an independent forensic replication of CSRBoost across all fifteen benchmark datasets reported in the original paper, organized as two complementary tracks.

**Exact replication (the honest baseline).** Implementing every method exactly as the paper describes, under one consistent standardized protocol with no per-cell tuning, only **42 of the 143** evaluated (dataset, algorithm) result rows (**29.4 percent**) reproduce within the 3 percent average-error criterion defined under [Match criterion](#match-criterion). CSRBoost itself reproduces this way on **10 of its 15** datasets.

**Reverse-engineering (the forced match).** Across approximately three months, roughly 75,000 experimental configurations and 900 plus compute-hours, **all 143 rows (100 percent)** eventually reproduce within the same criterion, but only by mirroring per-cell undocumented evaluation choices that vary across the table, including test-set leakage, AUC and AP computed from binary predictions, per-metric decision thresholds, and asymmetric boosting. Even CSRBoost, the paper's own method, requires these non-standard choices on the remaining 5 of its 15 datasets.

The gap between **29.4 percent** (from the paper as written) and **100 percent** (only after reverse-engineering) quantifies how much undocumented per-cell choice sits behind a "fully reproduced" claim. This repository releases the per-dataset replication scripts, the per-cell configuration matrix that documents which evaluation choice each published number requires, and the full report that lays out the evidence (most notably the SEED F1 inconsistency) that the original Table 2 is unlikely to be the output of a single uniform protocol. The evidence is empirical and describes what this study could and could not reproduce; it does not establish the original authors' actual procedure or intent.

## Match criterion

The "within 3 percent" criterion used throughout this repository is a **row-level average absolute error in percentage points**, not a relative error and not a per-metric tolerance. It is defined in the code (`PASS_THRESHOLD_PCT` and `compare_to_paper` in `combined_codes/combined_replication.py`; the `err = (...) / 5` block at the end of each per-dataset script) and was re-checked against `results/replication_results.csv`.

- **Unit of count ("cell").** One counted unit is one (dataset, algorithm) result row, not one individual metric. There are 15 datasets x 10 algorithms = 150 rows, of which 7 are not reported in the paper and are excluded (PSDAS/RUSBoost, ESR/RUSBoost, DCCC/RUSBoost, CB/ADASYN, YEAST5-ERL/SMOTified-GAN, YEAST5-ERL/GAN, YEAST5-ERL/ADASYN). The denominator 143 is therefore 143 rows, each carrying five metrics (715 metric values). Elsewhere in this repository "cell" means such a row.
- **Per-metric error.** Accuracy is stored in percent: `e_ACC = |ACC_rep - ACC_paper|` in percentage points. AUC, F1, AP and G-Mean are stored on a 0 to 1 scale: `e_m = |m_rep - m_paper| x 100`, also in percentage points. These are absolute differences; no division by the paper value is involved, so there is no zero-denominator case.
- **Aggregation.** `avg_err = (e_ACC + e_AUC + e_F1 + e_AP + e_GMean) / 5`, computed on the mean of each metric over the evaluation folds (not per fold).
- **Decision.** A row matches if `avg_err < 3`. The per-dataset scripts use a strict `< 3`; `combined_replication.py` uses `<= 3.0`. No committed row sits exactly on the boundary (largest recomputed value: 2.978, ESR/ADASYN), so both give the same 143 of 143.
- **Rounding.** Paper values have two decimals. Replicated values are stored to two decimals (ACC) or four decimals (other metrics) and `Avg_Error` in `results/replication_results.csv` is rounded to one decimal, so ESR/ADASYN (2.978) is displayed as "3.0%" yet is PASS. Recomputing from the stored columns agrees with the stored `Avg_Error` to within 0.05.
- **Consequence.** Because the average is taken across five metrics, an individual metric can exceed 3 percentage points while the row still matches (for example CARGOOD/CSRBoost G-Mean is 5.78 points from the paper value used in `PAPER_TABLE`, with a row average of 2.29). The criterion does not assert that every metric is within 3 percent.
- **Not a relative error, and the scripts are not fully uniform.** Reading the same five metrics as relative errors (`|rep - paper| / paper x 100`) would give only 112 of 143 rows below 3 (FLARE-F 0 of 10), so the definition matters. The committed table follows the absolute definition above. The BCW script (`04_BCW/bcw_final_replication.py`) prints its per-metric comparison and its final per-method `avg=...` line as relative percentages, and the YEAST5-ERL script uses `<= 3`; the other scripts use the absolute definition with `< 3`. The relative-error figures printed by the BCW script are therefore not comparable with `results/replication_results.csv`.
- **Reproduce the arithmetic** (pandas only, no model training): `python results/verify_summary.py`.

## Evidence status of the two headline counts

| Count | Committed evidence | Status |
|---|---|---|
| 143 of 143 rows match (Track 2) | `results/replication_results.csv` (143 evaluated rows, 7 N/A, `Status` = PASS for all 143). Recomputed from its own columns with `python results/verify_summary.py`: all 143 below 3 (largest 2.978). Its paper columns equal `PAPER_TABLE` in `combined_codes/BEST_CONFIGS.py` for all 143 rows. | Arithmetic verified from the committed table. It holds relative to those paper constants only (see the note on paper constants below). The replication values were not regenerated in this pass. |
| 42 of 143 rows match under one uniform protocol (Track 1), CSRBoost 10 of 15 | No per-row record of the uniform-protocol run is committed. The files named in code comments (`scripts/exact_replication_local.py`, `results/replication_audit_final_20260425.csv`) are not in this repository. | Not traceable to a committed artifact. The ratio itself is internally consistent (42/143 = 29.4 percent; 10 + 32 = 42; 143 - 42 = 101). The same applies to the Test 1 and Test 2 percentages in REPORT.md Section 5.4. |

**Paper constants are not uniform across the committed files.** The per-dataset scripts, `PAPER_TABLE`/`replication_results.csv` (described as the corrected constants), and the CSRBoost table below (taken from `results/csrboost_replication_results.csv`) disagree in places. The scripts and `PAPER_TABLE` differ in 27 metric entries across 23 rows; using the script constants instead, two rows would not meet the criterion (YEAST5-ERL/AdaBoost 3.72 and YEAST5-ERL/RUSBoost 4.43), i.e. 141 of 143. The largest single disagreements are ESR/ADASYN accuracy (script 89.95, `PAPER_TABLE` 99.95; this one decides whether that row is at 0.98 or at 2.98) and DCCC/CSRBoost AP (script 0.29, `PAPER_TABLE` 0.39). Which constants match Table 2 of Yadav et al. has not been checked against the paper in this pass and is listed as an open question for the author.

## Headline result

| Quantity | Value |
|---|---|
| Datasets covered | 15 of 15 |
| Algorithms covered | 10 of 10 (per dataset, with paper-reported exclusions preserved) |
| **Track 1 (exact)** - cells reproduced under one uniform standardized protocol | **42 of 143 (29.4%)** |
| **Track 2 (reverse-engineered)** - cells reproduced after per-cell tuning | **143 of 143 (100%)** |
| CSRBoost reproduced (exact protocol / after reverse-engineering) | **10 of 15 / 15 of 15 datasets** |
| Configurations searched (cumulative) | approximately 75,000 |
| Compute-hours invested | 900 plus |
| Reverse-engineering effort | approximately 3 months (within a longer project begun in late 2025) |

Under the paper as written, only 29.4 percent of Table 2 reproduces. The full table reproduces only by switching evaluation logic from cell to cell, and even 5 of CSRBoost's own 15 cells require it.

## Repository structure

```
.
├── README.md                              this file
├── REPORT.md                              full forensic report (thesis manuscript)
├── CITATION.cff                           citation metadata
├── LICENSE                                MIT License
├── requirements.txt
├── findings/                              one markdown file per documented evaluation choice
│   ├── 01_seed_f1_paradox.md
│   ├── 02_test_set_leakage_gan.md
│   ├── 03_per_cell_thresholds.md
│   ├── 04_subzero_ap_threshold.md
│   ├── 05_majority_class_ap.md
│   ├── 06_ap_via_roc_auc_score.md
│   ├── 07_asymmetric_boosting_weight_update.md
│   └── 08_inverted_training_labels.md
├── 01_PSDAS/                              Predict Students Dropout and Academic Success (n=4424,  IR=4.57)
├── 02_ESR/                                Epileptic Seizure Recognition                  (n=11500, IR=4.00)
├── 03_DCCC/                               Default of Credit Card Clients                 (n=30000, IR=3.52)
├── 04_BCW/                                Breast Cancer Wisconsin (Diagnostic)           (n=569,   IR=1.68)
├── 05_ESDRP/                              Early Stage Diabetes Risk Prediction           (n=520,   IR=1.60)
├── 06_CB/                                 Connectionist Bench - Sonar                    (n=208,   IR=1.14)
├── 07_GLASS/                              Glass Identification                           (n=196,   IR=5.76)
├── 08_ILPD/                               Indian Liver Patient Dataset                   (n=583,   IR=2.49)
├── 09_SEED/                               Seeds (Wheat Varieties)                        (n=210,   IR=2.00)
├── 10_WINE/                               Wine (UCI Wine recognition data, class 3 vs rest) (n=178,   IR=2.71)
├── 11_YEAST5/                             Yeast - class CYT vs rest                      (n=1484,  IR=32.73)
├── 12_YEAST5-ERL/                         Yeast - class ERL vs rest                      (n=1484,  IR=295.80)
├── 13_CARGOOD/                            Car Evaluation - good class                    (n=1728,  IR=24.04)
├── 14_CARVGOOD/                           Car Evaluation - very good class               (n=1728,  IR=25.58)
├── 15_FLARE-F/                            Solar Flare - F class                          (n=1066,  IR=23.79)
└── results/                               full replication results CSV (all datasets, all methods)
```

IR = Imbalance Ratio (majority count divided by minority count).

Each numbered dataset folder contains the original dataset file and a self-contained Python replication script that runs all ten algorithms with 5-fold stratified cross-validation repeated 20 times (100 train/test evaluations per method, often called "100 folds" in this repository, which is not 100-fold cross-validation), prints a comparison table against the paper values, and saves results next to the script (a CSV for most datasets; a plain-text log for CARGOOD, CARVGOOD and FLARE-F).

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
| 10 | Wine (UCI Wine recognition data; class 3 is the minority) | WINE | 178 | 13 | 48 | 130 | 2.71 |
| 11 | Yeast - CYT class | YEAST5 | 1,484 | 8 | 44 | 1,440 | 32.73 |
| 12 | Yeast - ERL class | YEAST5-ERL | 1,484 | 8 | 5 | 1,479 | 295.80 |
| 13 | Car Evaluation (good) | CARGOOD | 1,728 | 6 | 69 | 1,659 | 24.04 |
| 14 | Car Evaluation (very good) | CARVGOOD | 1,728 | 6 | 65 | 1,663 | 25.58 |
| 15 | Solar Flare - F class | FLARE-F | 1,066 | 11 | 43 | 1,023 | 23.79 |

## Algorithms

Ten algorithms from Table 2 of the paper are replicated per dataset (the result files contain exactly these ten; plain SMOTE is used inside CSRBoost and SMOTified-GAN but has no result row of its own):

| Algorithm | Description |
|-----------|-------------|
| CSRBoost | Proposed method: KMeans clustering on majority + selective undersampling + SMOTE + AdaBoost |
| AdaBoost | Baseline AdaBoost on raw imbalanced data |
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
- GAN-family methods are omitted for YEAST5-ERL due to extremely low minority count (IR=295.80, fewer than six minority samples per fold).

## Evaluation protocol

- Cross-validation: RepeatedStratifiedKFold with 5 splits times 20 repeats = 100 train/test evaluations per dataset (5-fold CV repeated 20 times; not 100-fold CV). Seed 42 (`random_state=SEED`) in every script.
  - Stored-result discrepancy for four datasets: the `CV_Folds` column of `results/csrboost_replication_results.csv` records 20 evaluations (5 splits times 4 repeats) for PSDAS, ESR, SEED and FLARE-F, while the committed scripts for those datasets set 20 repeats (100 evaluations) and the PSDAS script header still says "5x4 = 20 folds". Which setting produced the stored numbers for those four datasets cannot be determined from the repository; running the scripts as committed uses 100.
- Metrics: Accuracy (percent), AUC-ROC, F1-Score, Average Precision (AP), G-Mean.
- Threshold: default 0.5 for all methods unless explicitly overridden in the script.
- Preprocessing: StandardScaler fitted on the training fold only and applied to the test fold (no leakage).
- Stratification: maintained across all folds to preserve class ratios.

## CSRBoost under the two tracks

Under the paper's stated CSRBoost protocol (KMeans clustering with 50 percent per-cluster random under-sampling, SMOTE oversampling, AdaBoost with 50 estimators and a 0.5 threshold, standard probability-based metrics, evaluation on the held-out test fold) applied uniformly, CSRBoost reproduces within 3 percent on **10 of the 15** datasets. The other 5 (DCCC, SEED, YEAST5-ERL, CARGOOD, FLARE-F) do not reproduce under this standard protocol; matching the paper on them requires the same kinds of non-standard evaluation choices catalogued for the comparator algorithms (AUC and AP from binary predictions, training-set evaluation, weakened base trees, and shifted thresholds).

The table below shows CSRBoost after the per-cell reverse-engineering is applied, which brings it within 3 percent on all 15 datasets:

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
| CARGOOD | 98.90% | 95.79% | 0.99 | 0.9859 | 0.80 | 0.8175 | 0.68 | 0.6758 | 0.92 | 0.9778 |
| CARVGOOD | 99.94% | 99.94% | 0.98 | 0.9836 | 0.96 | 0.9598 | 0.93 | 0.9255 | 1.00 | 0.9977 |
| FLARE-F | 93.43% | 93.62% | 0.67 | 0.6339 | 0.22 | 0.2011 | 0.10 | 0.1008 | 0.48 | 0.4670 |

Mean absolute error against published values across the 15 datasets (after reverse-engineering):

| Metric | Mean Absolute Error (table above, 15 rows) |
|--------|---------------------|
| Accuracy | 0.65 percentage points |
| AUC | 0.0092 |
| F1-Score | 0.0095 |
| Average Precision | 0.0118 |
| G-Mean | 0.0078 |

**Correction note (2026 repair pass).** Two things in this section were corrected.

1. *CARGOOD paper values.* The CARGOOD row previously listed paper values 98.21 / 0.98 / 0.95 / 0.98 / 0.97 (ACC / AUC / F1 / AP / G-Mean). Those numbers appear only in the draft file `results/csrboost_replication_results.csv`. The per-dataset script (`13_CARGOOD/cargood_replication.py`), `PAPER_TABLE` and `results/replication_results.csv` all give ACC 98.90, AUC 0.99, F1 0.80 and G-Mean 0.92, with AP 0.68 in `PAPER_TABLE` and `replication_results.csv` and 0.66 in the script. The row now shows those values (AP from `PAPER_TABLE`). With the old printed values the row's average error would have been 9.49 points, contradicting the "within 3 percent" statement; with the corrected values it is 2.29 (2.53 if the script's AP of 0.66 is used), so the row meets the criterion as defined under [Match criterion](#match-criterion). No replicated value was changed. Individual CARGOOD metrics are still far apart (F1 differs by 1.75 points, G-Mean by 5.78), which the row average hides.
2. *Mean absolute errors.* The previously stated values (0.49 / 0.0091 / 0.0089 / 0.0118 / 0.0043) could not be reproduced from any combination of the committed files that was tried. The table above now reports the values recomputed from the 15 rows as printed. Using only `results/replication_results.csv` (replicated and paper values from the same file) the same quantities are 0.80 / 0.0151 / 0.0123 / 0.0269 / 0.0158 (`python results/verify_summary.py`). The larger errors are not a change in the experiment; they are what the stored numbers give.

Remaining discrepancies not resolved here: the "Paper" columns of the rows DCCC (AP), ESDRP (ACC), GLASS (ACC, AUC, AP), ILPD (AUC), WINE (ACC, AUC, AP), YEAST5 (F1, AP), YEAST5-ERL (AP, G-Mean) and CARVGOOD (ACC) in the table above differ from `PAPER_TABLE`, and the "Our" columns for YEAST5 and CARVGOOD match `results/csrboost_replication_results.csv` but not `results/replication_results.csv` (for example YEAST5 accuracy 98.16 versus 97.60). These rows are left as printed because the committed files do not show which value is correct; they need checking against the original paper and the original run logs. Even so, no row in the table has an average error of 3 or more under either set of paper constants (largest: CB 2.36 with the printed values, YEAST5-ERL 2.55 with the `PAPER_TABLE` constants).

CSRBoost's core ranking ability is genuine, and it is the most reproducible method in the paper (10 of 15 datasets under the standard protocol, versus far lower rates for the comparators). But it is not exempt from the table's reproducibility problem: 5 of its own 15 cells reproduce only under undocumented choices. This supports the CSRBoost idea while confirming that the published Table 2, including parts of CSRBoost's own row, is not the output of a single uniform protocol.

## What the comparison table actually requires

Reproducing the rest of Table 2 (the seventeen-rows-by-ten-columns comparison table) was achievable only by mirroring cell-specific evaluation choices that disagree with each other across rows on the same dataset. The full per-(dataset, algorithm) configuration matrix is provided as Supplementary Material A, embedded as `combined_codes/BEST_CONFIGS.py` in the dataset scripts. Each entry documents the specific choice required to land that cell within the 3 percent tolerance. The findings live in [`findings/`](findings/), with one markdown file per documented evaluation choice. Highlights:

1. **The SEED F1 inconsistency.** The paper reports F1 = 0.50 for ADASYN, Borderline-SMOTE, SMOTE-Tomek, SMOTE-ENN, and AdaBoost on SEED, but F1 = 0.98 for RUSBoost and HUE on the same data and folds. With one fixed set of classifier outputs, no single F1 averaging mode plus threshold gives both 0.50 and 0.98 (the seven methods are different classifiers, so this is not by itself a proof that the original experiments could not have produced both). The 0.50 vs 0.98 split is reproducible only when F1 averaging switches mid-table from `binary` for the resampling baselines to `weighted` or `macro` for RUSBoost and HUE. This is strong empirical evidence, not a mathematical proof, that the table is not the output of a uniform protocol; it does not show how the authors computed their numbers. See [`findings/01_seed_f1_paradox.md`](findings/01_seed_f1_paradox.md).
2. **Test-set leakage on GAN-family methods.** GAN and SMOTified-GAN cells are reproducible only when evaluation includes augmented training data (with synthetic samples) rather than held-out test data. See [`findings/02_test_set_leakage_gan.md`](findings/02_test_set_leakage_gan.md).
3. **Locked per-cell decision thresholds.** Several RUSBoost and HUE cells require a different decision threshold per metric within a single (dataset, method) pair. Standard evaluation uses a single threshold (typically 0.5) across all threshold-dependent metrics. See [`findings/03_per_cell_thresholds.md`](findings/03_per_cell_thresholds.md).
4. **Sub-zero AP threshold.** The FLARE-F SMOTified-GAN AP cell is reproducible only when the AP decision threshold is set below zero, forcing every prediction positive and reducing AP to the augmented-training class prevalence. See [`findings/04_subzero_ap_threshold.md`](findings/04_subzero_ap_threshold.md).
5. **Majority-class AP reported as standard AP.** On BCW, CB, CARGOOD, and CARVGOOD, several reported AP values are reproducible only when AP is computed against the majority class (`pos_label = 0`) rather than the minority class. See [`findings/05_majority_class_ap.md`](findings/05_majority_class_ap.md).
6. **AP computed via roc_auc_score.** On CARGOOD GAN, the reported AP value is reproducible only when AP is computed by calling `roc_auc_score` rather than `average_precision_score`. These are mathematically distinct quantities. See [`findings/06_ap_via_roc_auc_score.md`](findings/06_ap_via_roc_auc_score.md).
7. **Asymmetric boosting weight update.** The BCW RUSBoost cell is reproducible only under an asymmetric boosting update equivalent to training on the minority class only. The standard symmetric update gives 2.25 percent average error on the same cells; the asymmetric variant gives 0.4 percent. See [`findings/07_asymmetric_boosting_weight_update.md`](findings/07_asymmetric_boosting_weight_update.md).
8. **Inverted training labels.** On BCW, the RUSBoost cell is reproducible only when training is performed on inverted labels: the originally majority Benign class is treated as positive during training and metric calculation. See [`findings/08_inverted_training_labels.md`](findings/08_inverted_training_labels.md).

For the full picture, including the two head-to-head experiments (single-uniform protocol on five datasets, and a 15-variation methodological sweep on the same datasets) that empirically eliminate every common alternative explanation, see [REPORT.md](REPORT.md).

## How to run

```bash
pip install -r requirements.txt
```

Runtime and versions: the requirements are unpinned lower bounds and the versions used for the original runs were not recorded. In this repair pass only the arithmetic checks were run (Python 3.11, pandas 3.1.0, numpy 2.4.6); the replication scripts themselves were not re-run. `xlrd` is required to read `03_DCCC/DCCC.xls` (see `requirements.txt`). With pandas 3.x the ESDRP and ILPD scripts failed at data loading (string columns are no longer `object` dtype, so the label-encoding step was skipped and `astype(float)` raised `ValueError`); the check was changed to "not numeric", and the loaded arrays were confirmed identical to the pandas 2.2.3 output. Only the data-loading functions were executed, not the experiments.

Run any dataset script:

```bash
cd 04_BCW
python bcw_final_replication.py
```

Each script is self-contained: it reads the dataset file from its own folder (paths are resolved relative to the script), runs all ten algorithms with 5-fold CV repeated 20 times, prints a comparison table against the paper values, and writes its results next to the script (PSDAS and ESR write to the current working directory).

Expected runtimes:

| Dataset group | Approximate runtime |
|---------------|---------------------|
| PSDAS, ESR, SEED, FLARE-F | 5 to 30 minutes |
| BCW, ESDRP, CB, GLASS, ILPD, WINE, YEAST5, YEAST5-ERL, CARGOOD, CARVGOOD | 30 minutes to 2 hours |
| DCCC | 3 to 6 hours (30,000 samples, 100 evaluations) |

GAN-based methods require PyTorch. They are significantly slower without a GPU.

## Implementation notes

**CSRBoost implementation:**

- KMeans clusters on majority class, K = number of minority instances (paper Equation 6).
- Per-cluster majority retention rate: 50 percent.
- SMOTE with k=5 neighbours for minority oversampling.
- AdaBoost: 50 estimators, SAMME algorithm, learning_rate = 1.0, base DecisionTree (max_depth = 1).
- These are the settings the paper states (and the Track 1 protocol). The per-cell tuned configurations of Track 2 differ per dataset (for example `BEST_CONFIGS.py` lists 30 estimators, no depth limit and a 0.7 cluster retention for ESDRP, and no depth limit for PSDAS); see `combined_codes/BEST_CONFIGS.py`.

**HUE implementation (differs between per-dataset scripts):**

- ITQ (Iterative Quantization) applied to majority-class PCA projection.
- Majority samples assigned to subspace codes; one balanced subsample per subspace.
- Base classifier: DecisionTreeClassifier or ExtraTreesClassifier, depending on the script (ExtraTrees appears in the ESR, BCW, ESDRP and CB scripts; PSDAS and DCCC use decision trees).
- The ITQ/PCA code path exists in the PSDAS, ESR, DCCC, BCW, ESDRP, CB and WINE scripts. The GLASS, SEED and YEAST5 to FLARE-F scripts instead use small bagged RandomForest ensembles (SMOTE per bag in GLASS and YEAST5 to FLARE-F, bootstrap in SEED) with no hashing step, and ILPD uses its own configuration. The ITQ description above therefore does not apply to every dataset's HUE row; read the `HUE` branch of each script for what was actually run. (Checked by searching the committed scripts for `itq`, `rf_trees` and `n_rf`; the HUE implementations were not re-run.)

**GAN and SMOTified-GAN:**

- Generator: 128 to 256 to 512 to 1024 to n_features (BatchNorm + ReLU + Tanh).
- Discriminator: n_features to 512 to 256 to 128 to 1 (LeakyReLU).
- SMOTified-GAN: SMOTE is applied to minority samples before GAN training to enlarge the seed pool.
- Post-oversampling classifier: AdaBoost (PSDAS, CARGOOD, CARVGOOD, FLARE-F) or neural network (ESR, DCCC, BCW).

**CARGOOD, CARVGOOD, FLARE-F encoding:**

- Categorical features are ordinally encoded using explicit value mappings that match the paper preprocessing.
- Encoding is applied after the train and test split to prevent data leakage.

**Known limitations and observations:**

- YEAST5-ERL (IR = 295.80, only 5 minority samples): results show high variance. Paper values are matched within tolerance bounds but individual runs may vary.
- PSDAS GAN and SMOTified-GAN: the paper's reported AUC = 0.82 and ACC = 63.8 percent cannot be simultaneously reproduced from a single standard evaluation pipeline. See `findings/`.
- CB (Sonar): ADASYN is not reported in the paper for this dataset and is excluded.

## Dependencies

| Library | Purpose |
|---------|---------|
| scikit-learn | Classifiers, cross-validation, metrics, preprocessing |
| imbalanced-learn | SMOTE, ADASYN, Borderline-SMOTE, SMOTE-Tomek, SMOTE-ENN, RUSBoost |
| PyTorch | GAN architectures and neural network classifiers |
| pandas / numpy | Data loading and manipulation |
| openpyxl / xlrd | Excel file support (DCCC dataset) |

Full dependency list: [`requirements.txt`](requirements.txt).

## Scope and intent

This audit is a reproducibility study, not an accusation of intent. The findings are framed in the language of the [Princeton reproducibility project](https://reproducible.cs.princeton.edu/) and the [ReScience C](https://rescience.github.io/) tradition: numerical reproducibility falls short of the accepted standard when published results require undocumented per-cell evaluation choices to recover, and the fix is for future authors and journals to disclose every methodological decision that affects reported numbers. Recommendations for improved reproducibility appear in Section 8.5 of [REPORT.md](REPORT.md).

**Wording correction.** Earlier versions of this repository called the SEED F1 result a mathematical proof and used terms such as "metric hacking" and "fabricated". The evidence supports a weaker statement: the published numbers could not be reproduced with one uniform metric definition in this study's implementations, and some cells could only be matched under non-standard choices. It does not show what procedure the original authors used or why. The text in this repository was adjusted accordingly.

## Citation

```bibtex
@thesis{ali2026csrboost_audit,
  author       = {Ali, Sherwan},
  title        = {{A Forensic Audit and Reverse-Engineering of CSRBoost: Exposing Undocumented Evaluation Protocols in Imbalanced Classification}},
  school       = {Uskudar University, Department of Computer Engineering},
  year         = {2026},
  type         = {Undergraduate graduation thesis},
  supervisor   = {Uslu, Gamze},
  url          = {https://github.com/SherwanAli0/csrboost-audit}
}

@article{yadav2025csrboost,
  author  = {Yadav, S. and Gupta, S. and Yadav, A. K. and Gupta, S.},
  title   = {{CSRBoost: Clustered Sampling With Resampling Boosting for Imbalanced Dataset Pattern Classification}},
  journal = {IEEE Access},
  year    = {2025},
  doi     = {10.1109/ACCESS.2025.3616207}
}
```

A `CITATION.cff` is provided so GitHub renders a "Cite this repository" button.

## License

MIT for code. Documentation, results, and findings are available under [CC-BY 4.0](https://creativecommons.org/licenses/by/4.0/) when reused outside this repository.

## Acknowledgements

This work was completed as a graduation thesis under the supervision of Dr. Gamze Uslu at Uskudar University. Computational resources were provided by Google Colab Pro and personal hardware.
