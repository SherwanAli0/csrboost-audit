# A Forensic Audit and Reverse-Engineering of CSRBoost

> Replication study and forensic audit of:
> S. Yadav, D. Joshi, S. Mulye, L. Asari, S. S. Udmale, and G. P. Bhole. *CSRBoost: Clustered Sampling With Resampling Boosting for Imbalanced Dataset Pattern Classification.* IEEE Access, 2025.
> [DOI: 10.1109/ACCESS.2025.3616207](https://doi.org/10.1109/ACCESS.2025.3616207)

[![License: MIT (code) + CC BY 4.0 (documentation)](https://img.shields.io/badge/license-MIT%20%28code%29%20%2B%20CC%20BY%204.0%20%28docs%29-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Status: thesis manuscript](https://img.shields.io/badge/status-thesis%20manuscript-orange.svg)](#status)

**Author:** Sherwan Ali, Department of Computer Engineering, Uskudar University.
**Supervisor:** Dr. Gamze Uslu.
**Project type:** Solo undergraduate graduation thesis, 2026.

## Status

This repository contains an undergraduate graduation thesis manuscript and the accompanying replication code. **It has not been submitted to or peer-reviewed by any journal or conference, and has not been published.** Citations to it should be made as a thesis or technical report, not as a peer-reviewed publication. References elsewhere in this README to "published" results, "the publication", or "the paper" refer to the original CSRBoost paper by Yadav et al. (IEEE Access 2025), not to this work.

## Abstract

This repository documents a forensic replication of CSRBoost, carried out independently of the CSRBoost authors, across all fifteen benchmark datasets reported in the original paper, organized as two complementary tracks.

**Uniform-protocol baseline.** In the committed implementation of one standardized protocol without per-row tuning, **42 of 143** evaluated (dataset, algorithm) result rows (**29.4 percent**) satisfy the criterion under [Match criterion](#match-criterion). CSRBoost satisfies it on **10 of 15** datasets. These counts are recomputed from stored results, not regenerated experiments.

**Reverse-engineering track.** The final stored table has **143 of 143 rows** below the same average-error threshold: 42 rows use the uniform-protocol values, and 101 use tuned values. The scripts include non-standard choices such as evaluation on training data, AUC/AP from binary predictions, per-metric decision thresholds and asymmetric boosting. This records choices that aligned this study's implementations with the reported values; it does not establish how the original authors computed them. The approximately 75,000 configurations and 900-plus compute-hours are the author's retrospective project-wide estimates, not totals verified from committed run logs.

The two stored tables distinguish one tested uniform protocol from a table assembled using per-row choices. The repository includes the scripts, a historical configuration map, the stored result tables and a report of the findings. These observations do not rule out other uniform implementations or establish the original authors' procedure or intent.

## Match criterion

The "within 3 percent" criterion used throughout this repository is a **row-level average absolute error in percentage points**, not a relative error and not a per-metric tolerance. It is defined in the code (`PASS_THRESHOLD_PCT` and `compare_to_paper` in `combined_codes/combined_replication.py`; the `err = (...) / 5` block at the end of each per-dataset script) and was re-checked against `results/replication_audit_final_20260425.csv` and `results/exact_replication/baseline_phase1_comparison.csv`.

- **Unit of count ("cell").** One counted unit is one (dataset, algorithm) result row, not one individual metric. There are 15 datasets x 10 algorithms = 150 rows, of which 7 are not reported in the paper and are excluded (PSDAS/RUSBoost, ESR/RUSBoost, DCCC/RUSBoost, CB/ADASYN, YEAST5-ERL/SMOTified-GAN, YEAST5-ERL/GAN, YEAST5-ERL/ADASYN). The denominator 143 is therefore 143 rows, each carrying five metrics (715 metric values). Elsewhere in this repository "cell" means such a row.
- **Per-metric error.** Accuracy is stored in percent: `e_ACC = |ACC_rep - ACC_paper|` in percentage points. AUC, F1, AP and G-Mean are stored on a 0 to 1 scale: `e_m = |m_rep - m_paper| x 100`, also in percentage points. These are absolute differences; no division by the paper value is involved, so there is no zero-denominator case.
- **Aggregation.** `avg_err = (e_ACC + e_AUC + e_F1 + e_AP + e_GMean) / 5`, computed on the mean of each metric over the evaluation folds (not per fold).
- **Decision.** A row matches if `avg_err < 3`. The tuned-track audit applies a strict `< 3` (the per-dataset scripts do the same); the uniform-protocol comparison scripts of the author apply `<= 3.0`; `combined_replication.py` uses `<= 3.0`. No committed row sits on the boundary (largest recomputed value in the tuned-track table: 2.942, YEAST5/CSRBoost), so the choice does not change any count.
- **Rounding.** Paper values have two decimals. Replicated values are stored to two decimals (ACC) or four decimals (other metrics), and in the tuned-track table `Average_Error(%)` is stored with up to four decimals. Recomputing from the row's own columns agrees with the stored value to within 0.003.
- **Consequence.** Separately, 631 of 715 individual metric values (88.3 percent) are within 3 percentage points (ACC) or 0.03 (the other metrics), recomputed from the stored table. Because the average is taken across five metrics, an individual metric can exceed 3 percentage points while the row still matches (for example CARGOOD/CSRBoost G-Mean is 5.78 points from the paper value, with a row average of 2.53). The criterion does not assert that every metric is within 3 percent.
- **Not a relative error, and the scripts are not fully uniform.** Reading the same five metrics as relative errors (`|rep - paper| / paper x 100`) would give only 114 of 143 rows below 3 (FLARE-F 0 of 10), so the definition matters. The committed tables follow the absolute definition above. The BCW script (`04_BCW/bcw_final_replication.py`) prints its per-metric comparison and its final per-method `avg=...` line as relative percentages and the YEAST5-ERL script uses `<= 3`; the other scripts use the absolute definition. The relative figures printed by the BCW script are therefore not comparable with the tables in `results/`.
- **Reproduce the arithmetic** (pandas only, no model training): `python results/verify_summary.py`.

## Evidence status of the two headline counts

Files of record (copied from the author's final submission folder; `results/README.md` lists them, explains how the author's folder was ranked, and names what is superseded):

| Count | Committed evidence | Status |
|---|---|---|
| 143 of 143 rows match (tuned track) | `results/replication_audit_final_20260425.csv` (150 rows: 143 evaluated, 7 N/A, `Status` = PASS for all 143), with `paper_values_final_20260425.csv`, `replication_values_final_20260425.csv` and `final_audit_summary_20260425.txt`. | Recomputed by `python results/verify_summary.py`: 143 rows below 3 (largest 2.942). The replicated values were not regenerated in this pass. 42 of the 143 rows carry the values of the uniform-protocol run (next row), not separately tuned values; the other 101 are tuned values. |
| 42 of 143 rows match under one uniform protocol, CSRBoost 10 of 15 | The engine `exact_replication/exact_replication_local.py` (plus the Colab variants for ESR and DCCC), the row-level result table `results/exact_replication/baseline_phase1_comparison.csv` (and `baseline_phase1_results_only.csv`), and the 42 passing rows in `results/exact_replication/_42_pass_cells_spec.csv`. | Recomputed by `verify_summary.py`: 42 PASS, 101 FAIL, 7 N/A (42/143 = 29.4 percent). CSRBoost passes on PSDAS, ESR, BCW, CB, ESDRP, ILPD, GLASS, WINE, YEAST5 and CARVGOOD (10) and fails on DCCC, SEED, YEAST5-ERL, CARGOOD and FLARE-F; the other nine algorithms pass 32 rows in total. The per-fold raw outputs of that run are not in this repository, so the table was recomputed, not regenerated. |
| Test 1 (14.9 and 25.1 percent) and Test 2 (43.6 and 37.9 percent), REPORT.md Section 5.4 | Result files and scripts of the author's earlier workspace (runs of 2026-04-29) in `results/earlier_workspace_tests/`, recomputed by `python results/earlier_workspace_tests/verify_tests.py`. | Reproduced exactly as run (29 / 49 / 117 of 195; 74 and 85 of 195). They use different measures: Test 1 counts which of two protocols is closer to the paper (60 percent ties are structural), Test 2 counts per-metric cells within tolerance on 20 folds. They were measured against an older reference table (95 of 195 embedded values differ from the repository's corrected reference); against the repository's corrected reference table Test 2 gives 35.4 percent (baseline) and 40.0 percent (Platt). See `results/earlier_workspace_tests/README.md`. |

**How the two tracks relate.** On 2026-06-18 the author set the tuned-track values of the 42 rows that pass the uniform protocol back to the uniform-protocol values (the "honesty swap"), so that the tuned table does not claim tuning where none was needed. The per-dataset scripts in the numbered folders therefore do not regenerate those 42 stored values (comments in the scripts say so). For the other 101 rows the repository holds the stored values and the scripts used to tune them; no re-run was done to confirm that the scripts regenerate them. `Replication_Source` in the audit file still names an older results file for some of the 42 swapped rows.

**Per-cell configuration matrix.** `combined_codes/BEST_CONFIGS.py` is a historical 2026-05-07 snapshot, not a validated recipe for the final stored results. The configurations of record are in the 15 final per-dataset scripts. `python combined_codes/compare_configs.py` reports all 150 rows: 38 agree and 24 differ on the fields extracted from dictionary configurations; another 40 show partial agreement and 4 show differences in literals extracted heuristically from function bodies. The remaining 37 evaluated rows are unverified (27 use module constants/classes, 4 have no comparable literals, and 6 have unresolved threshold representations); 7 rows are N/A. Static checks cover only the reported fields, not complete execution behavior. `UNIFORM_PROTOCOL_CELLS` marks the 42 rows replaced with uniform-protocol results. No configuration was rewritten to force agreement. Static configuration checks, stored-table arithmetic, and reproduced model runs are different evidence levels; no model was rerun in this repair pass. Details: `combined_codes/README.md` and `combined_codes/best_configs_vs_final_scripts.csv`.

**Paper constants.** CARVGOOD/ADASYN AUC was visually checked against the enlarged Table 2 image and reads **0.98**. It is now corrected in the engine and all current paper-reference tables; the affected row's errors were recalculated without changing any replicated metric. The row still passes, with average error about 0.813 percentage points. Superseded snapshots retain their historical values. `verify_summary.py` checks agreement between the current tables and both code reference tables. This verifies internal consistency; this repair pass did not independently transcribe every printed paper entry. Correction details and SHA-256 identifiers are in `results/README.md`.

## Headline result

| Quantity | Value |
|---|---|
| Datasets covered | 15 of 15 |
| Algorithms covered | 10 of 10 (per dataset, with paper-reported exclusions preserved) |
| **Track 1 (exact)** - cells reproduced under one uniform standardized protocol | **42 of 143 (29.4%)** |
| **Track 2 (stored final table)** - rows below 3 points row-average error (42 rows carry uniform-protocol values, 101 are tuned values not regenerated in this pass) | **143 of 143 (100%)** |
| CSRBoost rows below 3 points (uniform protocol / stored final table; tuned rows not regenerated in this pass) | **10 of 15 / 15 of 15 datasets** |
| Configurations searched (author's retrospective project-wide estimate) | approximately 75,000; not verified from committed logs |
| Compute-hours (author's retrospective project-wide estimate) | 900 plus; not verified from committed logs |
| Reverse-engineering effort | approximately 3 months (within a longer project begun in late 2025) |

In the stored uniform-protocol results, 42 of 143 rows pass. In the final table, all 143 row averages pass, including 101 tuned rows. These are numerical comparisons of stored outputs under the stated criterion, not proof that every published metric or the original experiment has been reproduced.

## Repository structure

```
.
+-- README.md                              this file
+-- REPORT.md                              full forensic report (thesis manuscript)
+-- CITATION.cff                           citation metadata
+-- LICENSE                                MIT (code) + CC BY 4.0 (documentation), with scope of each
+-- LICENSE-CC-BY-4.0.txt                  full legal code of CC BY 4.0 (verbatim official text)
+-- requirements.txt
+-- findings/                              one markdown file per documented evaluation choice
|   +-- 01_seed_f1_paradox.md
|   +-- 02_test_set_leakage_gan.md
|   +-- 03_per_cell_thresholds.md
|   +-- 04_subzero_ap_threshold.md
|   +-- 05_majority_class_ap.md
|   +-- 06_ap_via_roc_auc_score.md
|   +-- 07_asymmetric_boosting_weight_update.md
|   +-- 08_inverted_training_labels.md
+-- 01_PSDAS/                              Predict Students Dropout and Academic Success (n=4424,  IR=4.57)
+-- 02_ESR/                                Epileptic Seizure Recognition                  (n=11500, IR=4.00)
+-- 03_DCCC/                               Default of Credit Card Clients                 (n=30000, IR=3.52)
+-- 04_BCW/                                Breast Cancer Wisconsin (Diagnostic)           (n=569,   IR=1.68)
+-- 05_ESDRP/                              Early Stage Diabetes Risk Prediction           (n=520,   IR=1.60)
+-- 06_CB/                                 Connectionist Bench - Sonar                    (n=208,   IR=1.14)
+-- 07_GLASS/                              Glass Identification                           (n=196,   IR=5.76)
+-- 08_ILPD/                               Indian Liver Patient Dataset                   (n=583,   IR=2.49)
+-- 09_SEED/                               Seeds (Wheat Varieties)                        (n=210,   IR=2.00)
+-- 10_WINE/                               Wine (UCI Wine recognition data, class 3 vs rest) (n=178,   IR=2.71)
+-- 11_YEAST5/                             Yeast - class 5 (ME2 site) vs rest             (n=1484,  IR=32.73)
+-- 12_YEAST5-ERL/                         Yeast - class ERL vs rest                      (n=1484,  IR=295.80)
+-- 13_CARGOOD/                            Car Evaluation - good class                    (n=1728,  IR=24.04)
+-- 14_CARVGOOD/                           Car Evaluation - very good class               (n=1728,  IR=25.58)
+-- 15_FLARE-F/                            Solar Flare - F class                          (n=1066,  IR=23.79)
+-- combined_codes/                        unified framework from an earlier snapshot (see its README)
+-- exact_replication/                     uniform-protocol engine (Track 1) and its Colab variants for ESR and DCCC
+-- results/                               audit tables of record, uniform-protocol records, HUE re-run, superseded files (see results/README.md)
```

IR = Imbalance Ratio (majority count divided by minority count).

Each numbered dataset folder contains the original dataset file and a self-contained Python replication script that runs the reported algorithms with a configured 5-fold stratified cross-validation repeated 20 times (100 train/test evaluations per method, often called "100 folds" in this repository, which is not 100-fold cross-validation), prints a comparison table against the paper values, and saves results next to the script (a CSV for most datasets; a plain-text log for CARGOOD, CARVGOOD and FLARE-F).

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
| 11 | Yeast - class 5 (ME2 site) | YEAST5 | 1,484 | 8 | 44 | 1,440 | 32.73 |
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

- Configured cross-validation: RepeatedStratifiedKFold with 5 splits and 20 repeats = 100 train/test evaluations for each evaluated dataset-method pair, with seed 42. These are repeated evaluations with overlapping training data, not 100 independent experiments or 100-fold CV. Configuration does not prove that every fold completed: some scripts catch fold exceptions, and complete raw fold records for the main tracks are not committed. Test 1's 3,900 saved fold rows are separately checked by `verify_tests.py`; Test 2 used 5 splits and 4 repeats (20 evaluations).
  - Historical note: a `CV_Folds` column in an earlier results file (now `results/superseded_csrboost_replication_results.csv`) said 20 for PSDAS, ESR, SEED and FLARE-F. The author states that this is wrong; the protocol is 100 evaluations throughout. (The separate side experiment Test 2, described in `results/earlier_workspace_tests/`, was run on 20 evaluations and is not part of the replication protocol.)
- Metrics: Accuracy (percent), AUC-ROC, F1-Score, Average Precision (AP), G-Mean.
- Threshold: default 0.5 for all methods unless explicitly overridden in the script.
- Preprocessing: StandardScaler fitted on the training fold only and applied to the test fold (no leakage).
- Stratification: maintained across all folds to preserve class ratios.

## CSRBoost under the two tracks

Under the paper's stated CSRBoost protocol (KMeans clustering with 50 percent per-cluster random under-sampling, SMOTE oversampling, AdaBoost with 50 estimators and a 0.5 threshold, standard probability-based metrics, evaluation on the held-out test fold) applied uniformly, CSRBoost reproduces within 3 percent on **10 of the 15** datasets. The other 5 (DCCC, SEED, YEAST5-ERL, CARGOOD, FLARE-F) do not reproduce under this standard protocol; matching the paper on them requires the same kinds of non-standard evaluation choices catalogued for the comparator algorithms (AUC and AP from binary predictions, training-set evaluation, weakened base trees, and shifted thresholds).

The table below lists the 15 CSRBoost rows of the tuned-track table of record (`results/replication_audit_final_20260425.csv`). "Source" says which run produced the replicated values: for the 10 datasets on which CSRBoost passes the uniform protocol, the values are those of the uniform-protocol run (the 2026-06-18 swap described under "Evidence status"); the other 5 (DCCC, SEED, YEAST5-ERL, CARGOOD, FLARE-F) are tuned values. "Avg err" is the criterion of the Match criterion section, in percentage points.

| Dataset | Paper ACC | Our ACC | Paper AUC | Our AUC | Paper F1 | Our F1 | Paper AP | Our AP | Paper GMean | Our GMean | Avg err | Source |
|---------|-----------|---------|-----------|---------|----------|--------|----------|--------|-------------|-----------|---------|--------|
| PSDAS | 72.85% | 72.58% | 0.66 | 0.6400 | 0.40 | 0.4000 | 0.25 | 0.2600 | 0.63 | 0.6300 | 0.65 | uniform protocol |
| ESR | 92.05% | 91.78% | 0.90 | 0.8919 | 0.80 | 0.8051 | 0.67 | 0.6805 | 0.89 | 0.8908 | 0.54 | uniform protocol |
| DCCC | 68.32% | 69.00% | 0.64 | 0.6358 | 0.42 | 0.4349 | 0.29 | 0.2987 | 0.62 | 0.6282 | 0.86 | tuned |
| BCW | 94.37% | 92.35% | 0.94 | 0.9222 | 0.90 | 0.8994 | 0.84 | 0.8420 | 0.92 | 0.9218 | 0.85 | uniform protocol |
| ESDRP | 97.31% | 96.14% | 0.97 | 0.9602 | 0.95 | 0.9502 | 0.92 | 0.9213 | 0.96 | 0.9600 | 0.46 | uniform protocol |
| CB | 76.43% | 70.80% | 0.76 | 0.7063 | 0.69 | 0.6828 | 0.63 | 0.6251 | 0.71 | 0.7009 | 2.62 | uniform protocol |
| GLASS | 95.80% | 93.21% | 0.93 | 0.8948 | 0.79 | 0.7844 | 0.67 | 0.6610 | 0.90 | 0.8889 | 1.74 | uniform protocol |
| ILPD | 66.72% | 64.59% | 0.66 | 0.6195 | 0.48 | 0.4730 | 0.36 | 0.3601 | 0.62 | 0.6133 | 1.51 | uniform protocol |
| SEED | 98.10% | 98.39% | 0.98 | 0.9879 | 0.96 | 0.9647 | 0.93 | 0.9319 | 0.97 | 0.9626 | 0.50 | tuned |
| WINE | 98.86% | 96.49% | 0.98 | 0.9587 | 0.93 | 0.9358 | 0.88 | 0.8980 | 0.95 | 0.9574 | 1.52 | uniform protocol |
| YEAST5 | 98.32% | 97.60% | 0.93 | 0.8797 | 0.69 | 0.6571 | 0.50 | 0.4645 | 0.89 | 0.8688 | 2.94 | uniform protocol |
| YEAST5-ERL | 99.93% | 99.88% | 0.60 | 0.6050 | 0.47 | 0.4700 | 0.43 | 0.4718 | 0.55 | 0.4700 | 2.55 | tuned |
| CARGOOD | 98.90% | 95.79% | 0.99 | 0.9859 | 0.80 | 0.8175 | 0.66 | 0.6758 | 0.92 | 0.9778 | 2.53 | tuned |
| CARVGOOD | 99.83% | 99.74% | 0.99 | 0.9883 | 0.96 | 0.9658 | 0.92 | 0.9364 | 0.99 | 0.9879 | 0.54 | uniform protocol |
| FLARE-F | 93.43% | 93.62% | 0.67 | 0.6339 | 0.22 | 0.2011 | 0.10 | 0.1008 | 0.48 | 0.4670 | 1.41 | tuned |

Mean absolute error against the current paper-reference table across these **15 CSRBoost rows only** (not all 143 evaluated rows):

| Metric | Mean Absolute Error |
|--------|---------------------|
| Accuracy | 1.44 percentage points |
| AUC | 0.0210 |
| F1-Score | 0.0084 |
| Average Precision | 0.0118 |
| G-Mean | 0.0151 |

(`python results/verify_summary.py` prints these and the source table's LF-normalized SHA-256, also recorded in `results/README.md`. The row error is the one defined in Section 3.6 of the author's final thesis: the average of the five percentage-point metric errors; this table averages each metric's absolute error over the 15 datasets, with accuracy in points and the other four on the 0 to 1 scale. No mean absolute error against the paper is stated in the final thesis, the author's final README or its results tables, so there is no figure to reconcile with; the thesis's `Comprehensive Results Tables.docx` carries the same CSRBoost rows, matching the audit table cell by cell.) They are larger for accuracy, AUC and G-Mean than in earlier versions of this README because 10 of the rows now carry the uniform-protocol values, which sit further from the paper than separately tuned values did.

**Correction note (2026 repair pass).**

1. *This table was rebuilt from the file of record.* The previous version mixed replicated values from an earlier, superseded results file with paper values from a still earlier draft. That mixture contained wrong CARGOOD paper values (98.21 / 0.98 / 0.95 / 0.98 / 0.97 instead of 98.90 / 0.99 / 0.80 / 0.66 / 0.92), which made the CARGOOD row look like a 9.49 point average error and contradict the "within 3 percent" statement. With the printed paper values the row is at 2.53 (its G-Mean is still 5.78 points away and its F1 1.75, which the row average hides). Nine other rows had paper cells that disagreed with the printed table or with each other (DCCC AP, ESDRP ACC, GLASS ACC/AUC/AP, ILPD AUC, WINE ACC/AUC/AP, YEAST5 F1/AP, YEAST5-ERL AP/G-Mean, CARVGOOD ACC), and the pre-swap replicated values of CSRBoost on several datasets were still shown (for example CB 76.54 / 0.7510 / 0.6762 / 0.7179 / 0.7164, where the audit table has 70.80 / 0.7063 / 0.6828 / 0.6251 / 0.7009). No replicated value was edited: the numbers shown are those of the file of record.
2. *The previously stated mean absolute errors* (0.49 / 0.0091 / 0.0089 / 0.0118 / 0.0043) could not be reproduced from the file of record. They are close to what the author's earlier CSRBoost-only results file of 2026-04-19 gives (all 15 rows: ACC 0.487, AUC 0.0098, F1 0.0094, AP 0.0124, G-Mean 0.0046; without CARGOOD: ACC 0.474, AUC 0.0096, F1 0.0089, AP 0.0124, G-Mean 0.0043), so they most likely come from that or a similar earlier state, but the exact derivation could not be recovered. The figures were replaced by the recomputed values above.

CSRBoost's core ranking ability is genuine, and it is the most reproducible method in the paper (10 of 15 datasets under the standard protocol, versus far lower rates for the comparators). But it is not exempt from the table's reproducibility problem: 5 of its own 15 cells reproduce only under undocumented choices. This supports the CSRBoost idea while confirming that the published Table 2, including parts of CSRBoost's own row, is not the output of a single uniform protocol.

## What the comparison table actually requires

Reproducing the rest of Table 2 (the seventeen-rows-by-ten-columns comparison table) was achievable only by mirroring cell-specific evaluation choices that disagree with each other across rows on the same dataset. `combined_codes/BEST_CONFIGS.py` is a historical snapshot (2026-05-07) of per-cell configurations. It is not embedded in the dataset scripts. It is partially checked against them by `combined_codes/compare_configs.py` (150 cells: 38 dictionary agreements, 24 dictionary differences, 40 partial static agreements, 4 static-literal differences, 37 unverified, 7 N/A). The configurations of record are in the 15 final per-dataset scripts. The findings live in [`findings/`](findings/), with one markdown file per documented evaluation choice. Highlights:

1. **The SEED F1 inconsistency.** The paper reports F1 = 0.50 for ADASYN, Borderline-SMOTE, SMOTE-Tomek, SMOTE-ENN, and AdaBoost on SEED, but F1 = 0.98 for RUSBoost and HUE on the same data and folds. With one fixed set of classifier outputs, no single F1 averaging mode plus threshold gives both 0.50 and 0.98 (the seven methods are different classifiers, so this is not by itself a proof that the original experiments could not have produced both). The 0.50 vs 0.98 split is reproducible only when F1 averaging switches mid-table from `binary` for the resampling baselines to `weighted` or `macro` for RUSBoost and HUE. This is strong empirical evidence, not a mathematical proof, that the table is not the output of a uniform protocol; it does not show how the authors computed their numbers. See [`findings/01_seed_f1_paradox.md`](findings/01_seed_f1_paradox.md).
2. **Test-set leakage on GAN-family methods.** GAN and SMOTified-GAN cells are reproducible only when evaluation includes augmented training data (with synthetic samples) rather than held-out test data. See [`findings/02_test_set_leakage_gan.md`](findings/02_test_set_leakage_gan.md).
3. **Locked per-cell decision thresholds.** Several RUSBoost and HUE cells require a different decision threshold per metric within a single (dataset, method) pair. Standard evaluation uses a single threshold (typically 0.5) across all threshold-dependent metrics. See [`findings/03_per_cell_thresholds.md`](findings/03_per_cell_thresholds.md).
4. **Sub-zero AP threshold.** The FLARE-F SMOTified-GAN AP cell is reproducible only when the AP decision threshold is set below zero, forcing every prediction positive and reducing AP to the augmented-training class prevalence. See [`findings/04_subzero_ap_threshold.md`](findings/04_subzero_ap_threshold.md).
5. **Majority-class AP reported as standard AP.** On BCW, CB, CARGOOD, and CARVGOOD, several reported AP values are reproducible only when AP is computed against the majority class (`pos_label = 0`) rather than the minority class. See [`findings/05_majority_class_ap.md`](findings/05_majority_class_ap.md).
6. **AP computed via roc_auc_score.** On CARGOOD GAN, the reported AP value is reproducible only when AP is computed by calling `roc_auc_score` rather than `average_precision_score`. These are mathematically distinct quantities. See [`findings/06_ap_via_roc_auc_score.md`](findings/06_ap_via_roc_auc_score.md).
7. **Asymmetric boosting weight update.** The BCW RUSBoost cell is reproducible only under an asymmetric boosting update equivalent to training on the minority class only. The standard symmetric update gives 2.25 percent average error on the same cells; the asymmetric variant gives 0.4 percent. See [`findings/07_asymmetric_boosting_weight_update.md`](findings/07_asymmetric_boosting_weight_update.md).
8. **Inverted training labels.** On BCW, the RUSBoost cell is reproducible only when training is performed on inverted labels: the originally majority Benign class is treated as positive during training and metric calculation. See [`findings/08_inverted_training_labels.md`](findings/08_inverted_training_labels.md).

For the full picture, including the two side experiments (two uniform metric protocols on five datasets, and a 15-variation methodological sweep on the same five datasets), see [REPORT.md](REPORT.md) and `results/earlier_workspace_tests/README.md`. They found no single protocol or knob that closes the gap; their percentages are not match rates against the paper and are explained there.

## How to run

```bash
pip install -r requirements.txt
```

Runtime and versions: the requirements are unpinned lower bounds and the versions used for the original runs were not recorded. In this repair pass only the arithmetic checks were run (Python 3.11, pandas 3.1.0, numpy 2.4.6); the replication scripts themselves were not re-run. `xlrd` is required to read `03_DCCC/DCCC.xls` (see `requirements.txt`). With pandas 3.x the ESDRP and ILPD scripts failed at data loading (string columns are no longer `object` dtype, so the label-encoding step was skipped and `astype(float)` raised `ValueError`); the check was changed to "not numeric", and the loaded arrays were confirmed identical to the pandas 2.2.3 output. Only the data-loading functions were executed, not the experiments. The scripts in the numbered folders are the author's final versions; for the 42 rows that carry uniform-protocol values they do not regenerate the stored numbers (see "Evidence status").

Run the uniform-protocol (Track 1) engine, which reads the datasets from the numbered folders, for one or more datasets (the full run is long and was not repeated in this repair pass):

```bash
python exact_replication/exact_replication_local.py --datasets PSDAS BCW
```

(ESR and DCCC were run with the Colab variants `exact_replication_colab_ESR.py` and `exact_replication_colab_DCCC.py`.)

Run any tuned-track dataset script:

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
- AdaBoost: 50 estimators, learning_rate = 1.0. The paper pins no base-tree depth; the uniform-protocol track (Track 1) uses an unpruned decision tree (`max_depth=None`) as a uniform project default, as the author's final README states, and standardises AdaBoost and RUSBoost to 50 learners. Earlier versions of this README gave `max_depth = 1`, which is not what the engine does.
- These are the settings the paper states (and the Track 1 protocol). The per-cell tuned configurations of Track 2 differ per dataset (for example `BEST_CONFIGS.py` lists 30 estimators, no depth limit and a 0.7 cluster retention for ESDRP, and no depth limit for PSDAS); see `combined_codes/BEST_CONFIGS.py`.

**HUE implementation (differs between per-dataset scripts):**

- ITQ (Iterative Quantization) applied to majority-class PCA projection.
- Majority samples assigned to subspace codes; one balanced subsample per subspace.
- Base classifier: DecisionTreeClassifier or ExtraTreesClassifier, depending on the script (ExtraTrees appears in the ESR, BCW, ESDRP and CB scripts; PSDAS and DCCC use decision trees).
- **Which HUE ran where.** The ITQ/PCA code path (the hashing ensemble described above) is used in the PSDAS, ESR, DCCC, BCW, ESDRP and CB scripts and in the uniform-protocol engine. In the tuned scripts for GLASS, ILPD, SEED, WINE, YEAST5, YEAST5-ERL, CARGOOD, CARVGOOD and FLARE-F, HUE is a small bagged RandomForest approximation with no hashing step (SMOTE per bag in most, bootstrap bags in SEED and WINE); the author's own re-run script `results/hue_canonical/hue_runner.py` describes them that way. That script re-ran the canonical ITQ-HUE on those nine datasets over 100 folds (outputs in `results/hue_canonical/outputs/`): only WINE is within 3 percent (average error 2.29); the others are not (ILPD 3.21, SEED 3.97, GLASS 5.75, FLARE-F 8.56, YEAST5 10.54, CARVGOOD 14.74, CARGOOD 14.85, YEAST5-ERL 16.83). So the stored tuned HUE rows of eight of those datasets are matched by the RandomForest approximation, not by the HUE of the original reference. For WINE, the stored HUE row is the uniform-protocol (canonical ITQ) value, and the WINE script, which runs the RandomForest approximation, does not regenerate it. The ITQ description above therefore does not apply to every dataset's HUE row; read the `HUE` branch of each script. (Checked by reading the committed scripts and the re-run outputs; the HUE implementations were not re-run here.)

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

## Related extension: ERF-CSRBoost

Alongside this audit, I explored an ensemble extension of CSRBoost and built
FixYourDataset, a web application for examining class imbalance and comparing
model behavior. The extension is a research project that is not peer reviewed. Its methods,
evaluation protocol and limitations are documented separately from this audit.
The results in this repository (the audit tables, exact-replication runs and
findings) do not include the ERF extension and say nothing about it.

The separate ERF record (updated 7 November 2026; not peer reviewed) adds 2,000 new fits on five datasets to 15 historical datasets:

- ERF exceeds the local CSRBoost reimplementation on 16 of 20 datasets (+0.0685 mean AP; lower on 4) and is within 0.001 mean AP of a plain random forest (higher on 12, tied on 2, lower on 6).
- All 2,000 new fits completed with zero failures, and every table of that record is covered by a SHA-256 manifest and a read-only checker.
- Scope: counts of datasets, not significance tests, and the near-tie with a plain random forest does not show equivalence. The median paired fit-and-score time ratio of ERF against a plain random forest is 6.6 to 35.6 times across the five new datasets (15.0 to 25.1 times against the CSRBoost reimplementation). None of this changes the audit's 42/143 uniform-protocol finding.

See the [qualified result record](https://github.com/SherwanAli0/erf-csrboost/tree/main/research/results/november2026_aligned_extension).

- [ERF-CSRBoost implementation and results](https://github.com/SherwanAli0/erf-csrboost)
- [FixYourDataset application](https://fixyourdataset.fly.dev)


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
  author  = {Yadav, Seema and Joshi, Dhruvanshu and Mulye, Soham and Asari, Labib and Udmale, Sandeep S. and Bhole, Girish P.},
  title   = {{CSRBoost: Clustered Sampling With Resampling Boosting for Imbalanced Dataset Pattern Classification}},
  journal = {IEEE Access},
  year    = {2025},
  doi     = {10.1109/ACCESS.2025.3616207}
}
```

A `CITATION.cff` is provided so GitHub renders a "Cite this repository" button.

## License

This repository is dual-licensed (see [LICENSE](LICENSE)):

- Source code (Python files, `combined_codes/`, `results/verify_summary.py`): [MIT License](LICENSE).
- Documentation, results CSVs, findings, reports and research notes (`README.md`, `REPORT.md`, `findings/`, `results/`): [CC-BY 4.0](https://creativecommons.org/licenses/by/4.0/). The full legal code is included in [LICENSE-CC-BY-4.0.txt](LICENSE-CC-BY-4.0.txt).
- **Not covered by either license:** the original CSRBoost paper (Yadav et al., IEEE Access 2025, copyright of its authors and publisher; it is not redistributed here), the datasets in the numbered folders (UCI Machine Learning Repository and KEEL data, which keep their providers' terms), and the third-party libraries the code uses.

**License change.** Earlier versions of this repository shipped a `LICENSE` file containing Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 (CC BY-NC-ND 4.0). That contradicted the MIT and CC-BY 4.0 terms stated in this README and in `CITATION.cff`, and made the code unusable under the stated terms. The file was replaced with the dual MIT + CC-BY 4.0 layout the README and `CITATION.cff` already described, matching the other repositories of this author. Copies obtained before the change were distributed under the earlier text.

**Section B clarification.** An intermediate version of Section B of `LICENSE` added a condition that the supervision credit and the attribution of the original CSRBoost paper be kept when the material is reused. That condition was removed, because CC BY 4.0 does not permit additional restrictions. The credit is kept as an acknowledgement and a request (see [Acknowledgements](#acknowledgements)); who is credited has not changed.

## Acknowledgements

This work was completed as a graduation thesis under the supervision of Dr. Gamze Uslu at Uskudar University. Computational resources were provided by Google Colab Pro and personal hardware.

If you reuse this material, please keep this supervision credit (Dr. Gamze Uslu, Uskudar University) and the citation of the original paper that this audit examines: S. Yadav, D. Joshi, S. Mulye, L. Asari, S. S. Udmale, and G. P. Bhole, "CSRBoost: Clustered Sampling With Resampling Boosting for Imbalanced Dataset Pattern Classification," IEEE Access, 2025, [DOI 10.1109/ACCESS.2025.3616207](https://doi.org/10.1109/ACCESS.2025.3616207). This is a request made in good faith, not an additional condition of the licenses. The CC BY 4.0 license itself requires attribution to the licensor, as set out in its legal code.
