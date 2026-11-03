# Finding 01: The SEED F1 inconsistency

**Severity:** high. Strong empirical evidence of a per-method change in F1 definition. It is not a mathematical proof: the seven methods are different classifiers with different outputs.

## Symptom

On the SEED dataset, the original paper reports:

| Method | F1 |
|---|---|
| ADASYN | 0.50 |
| Borderline-SMOTE | 0.50 |
| SMOTE-Tomek | 0.50 |
| SMOTE-ENN | 0.50 |
| AdaBoost | 0.50 |
| RUSBoost | 0.98 |
| HUE | 0.98 |

All seven methods are evaluated on the same data, the same labels, and the same folds.

## Evidence

For one fixed set of classifier outputs on identical labels, no single threshold and F1-averaging mode can give both 0.50 and 0.98. The seven methods do not share outputs, so this is an argument about this audit's implementations, not a proof about the original experiments. We computed live:

- F1-binary on the SEED test folds for all seven methods returns values in the 0.85 to 0.91 range.
- F1-weighted on the same test folds returns values above 0.97.

For the published values to be reproducible from a uniform protocol with these implementations, F1-binary would need to land at 0.50 for the resampling baselines and at 0.98 for RUSBoost / HUE on the same data. It lands at 0.85 to 0.91 for all seven in this audit's runs, so a uniform F1 definition did not reproduce the table here.

## Reproduction

The published numbers are reproducible only when F1 averaging switches mid-table:

- F1 = `f1_score(y_true, y_pred, average='binary')` for ADASYN, Borderline-SMOTE, SMOTE-Tomek, SMOTE-ENN, AdaBoost.
- F1 = `f1_score(y_true, y_pred, average='weighted')` (or `'macro'`) for RUSBoost, HUE.

See `09_SEED/seed_replication.py` for the exact reproduction.

## Why this matters

Mid-table averaging-mode substitution is the explanation that fits the published numbers best; a transcription error or classifiers that behave differently from this audit's implementations cannot be excluded. If it did occur, the same dataset row would show different "F1" values depending on which metric definition was applied without disclosure. This is the most direct evidence in this audit that Table 2 of the original paper does not represent one uniform evaluation pipeline.

## Suggested fix for future authors

Always specify the averaging mode for F1 in any imbalanced-classification table. Use one mode consistently across the entire table or, if mixing is intentional, label each cell with its averaging mode.

## Cross-references

- Paper: Yadav et al., IEEE Access 2025, Table 2.
- Repo: [`09_SEED/seed_replication.py`](../09_SEED/seed_replication.py).
- Report: REPORT.md, Section 8.3.1.
