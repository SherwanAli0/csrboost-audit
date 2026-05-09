# combined_codes/

Single-file unified replication of CSRBoost (Yadav 2025) on all 15 datasets,
mirroring the section structure of `graduation project exact replication/scripts/exact_replication_local.py`.

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

### Reference (per-dataset scripts; copies of the originals in `0X_<DS>/`)

- `PSDAS_replication.py`, `ESR_replication.py`, ..., `FLARE_F_replication.py`
  These are unchanged copies of each dataset's standalone script. Kept here
  for traceability — the unified framework's logic was extracted from them.
  They are **not** imported by `combined_replication.py`; each is a
  self-contained module that can be run independently.

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

`BEST_CONFIGS` and `PAPER_TABLE` are aligned with the 143/143 PASS audit at
`results/replication_audit_final_20260425.csv`:

- 143 evaluated cells (one config per cell)
- 7 N/A cells: PSDAS/RUSBoost, ESR/RUSBoost, DCCC/RUSBoost, CB/ADASYN,
  YEAST5-ERL/SMOTified-GAN, YEAST5-ERL/GAN, YEAST5-ERL/ADASYN
- 143 paper-value entries

Run `python BEST_CONFIGS.py` to invoke the bundled `_self_check()`, which
asserts every (dataset, algorithm) pair has a config or N/A marker.
