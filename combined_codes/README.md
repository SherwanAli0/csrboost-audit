# combined_codes/

Single-file unified replication framework for the per-cell tuned track (Track 2) of the CSRBoost audit on all 15 datasets.
It was written to mirror the section structure of an `exact_replication_local.py` script for the uniform-protocol track; that script is not included in this repository.

**Status (repair pass).** The framework has not been re-run end to end: only the dataset loaders were executed and checked (all 15 reproduce the sample counts, feature counts and class counts in the top-level README) and `BEST_CONFIGS.py` was run for its built-in self-check. The ESDRP, ILPD and CB loaders in `combined_replication.py` were corrected to match the per-dataset scripts: ESDRP read a file name that does not exist in the repository (`ESDRP.csv`; the shipped file is `diabetes_data_upload.csv`), ILPD tried to convert the string `Female`/`Male` column to float (the column is named `V2`, not `Gender`) and labelled the majority class as positive, and CB labelled the majority class (Mine) as positive whereas the per-dataset script and the README treat Rock (97) as the minority class 1. The HUE branch of this framework uses SMOTE-balanced RandomForest bags for every dataset, which is not the ITQ-based HUE that several per-dataset scripts run (see the HUE note in the top-level README), so HUE rows produced here are not expected to equal those scripts.

## Files

### Primary (the unified replication)

- **`combined_replication.py`** (~950 lines)
  Framework: imports, global config, metric utilities, universal protocol,
  shared GAN/CSRBoost/HUE models, all 15 dataset loaders, universal
  `run_fold` dispatcher, `evaluate_dataset`, and `main`. No code duplication
  across datasets.

- **`BEST_CONFIGS.py`** (~1900 lines)
  All 150 (dataset, method) configurations as one big dict. Captures sampler
  type/args, classifier hyperparameters (depth, n_est, lr), scaler choice,
  and per-metric (ACC, AUC, F1, AP, GMEAN) source/threshold/kind. The 7 N/A
  cells (matching the audit) are stored as the literal string `'N/A'`. Also
  exports `PAPER_TABLE` with the 143 corrected paper values.

- **`combined_replication.docx`** (~63 KB)
  Single-document concatenation of `combined_replication.py` and
  `BEST_CONFIGS.py` for review or LLM consumption.

### Reference (per-dataset scripts)

The standalone per-dataset scripts live in the numbered folders at the top level (for example `01_PSDAS/psdas_replication_v2.py`). Copies of them are not stored in this folder. The unified framework's logic was extracted from them and does not import them.

## Running

To run all 15 datasets via the unified framework:

```
python combined_replication.py
```

To run a subset:

```
python combined_replication.py PSDAS GLASS WINE
```

Output is written to `outputs/combined_run/<DATASET>_folds.csv`,
`<DATASET>_comparison.csv`, and `ALL_comparison.csv`.

## Why the rewrite?

The previous `combined_all_15_datasets.py` was a 9095-line file produced by
concatenating the 15 individual scripts verbatim. Every file redeclared the
same `gmean_score`, `make_adaboost`, `safe_ap`/`safe_auc`, `Generator`/
`Discriminator`/`NNClassifier`, GAN training loop, and metric protocol
functions, leading to ~30% duplication and no clear central point for
configuration.

The unified version moves everything shared into one section at the top,
then captures *only* the per-(dataset, method) tuning differences as data in
`BEST_CONFIGS`. This matches the structure of `exact_replication_local.py`
and is far easier to review.

| Metric                                | Old monolith  | New unified  |
|---------------------------------------|---------------|--------------|
| Lines (combined `.py`)                | 9,095         | ~2,820       |
| Code duplication across 15 datasets   | High          | None         |
| Single point of configuration         | No            | Yes (BEST_CONFIGS) |
| Section structure                     | 15 concatenated scripts | Mirrors exact_replication_local.py |

## Audit consistency

`BEST_CONFIGS` and `PAPER_TABLE` are aligned with the 143/143 PASS audit table `results/replication_results.csv` (the dated file name `replication_audit_final_20260425.csv` used in earlier notes is not in this repository; the paper columns of `results/replication_results.csv` equal `PAPER_TABLE` for all 143 rows, checked by `python results/verify_summary.py`). The per-dataset scripts carry some different paper constants (27 metric entries across 23 rows), see the top-level README:

- 143 evaluated cells (one config per cell)
- 7 N/A cells: PSDAS/RUSBoost, ESR/RUSBoost, DCCC/RUSBoost, CB/ADASYN,
  YEAST5-ERL/SMOTified-GAN, YEAST5-ERL/GAN, YEAST5-ERL/ADASYN
- 143 paper-value entries

Run `python BEST_CONFIGS.py` to invoke the bundled `_self_check()`, which
asserts every (dataset, algorithm) pair has a config or N/A marker.
