# Finding 06: AP computed via roc_auc_score on CARGOOD GAN

**Severity:** medium-high. Function-level substitution of two mathematically distinct metrics.

## Symptom

On CARGOOD, the GAN cell's reported AP value is not reproducible using `average_precision_score`. The standard call yields a value outside the 3 percent tolerance.

## Evidence

The published number is reproducible only when AP is computed by calling `roc_auc_score` rather than `average_precision_score`:

```python
# Published "AP" for this cell is consistent with this:
roc_auc_score(y_true, y_score)

# Not with the standard AP call:
average_precision_score(y_true, y_score)
```

These are mathematically distinct quantities. AP is the area under the precision-recall curve; AUC is the area under the ROC curve. They measure different aspects of classifier performance and are not interchangeable.

## Reproduction

See `13_CARGOOD/cargood_replication.py`. The configuration matrix entry for this cell records `ap_func = "roc_auc_score"` instead of the default `"average_precision_score"`.

## Why this matters

Substituting one metric function for another, while reporting under the original metric name, breaks any cross-paper comparison that assumes consistent metric definitions.

## Suggested fix for future authors

Always use `average_precision_score` for AP and `roc_auc_score` for AUC. If both are interesting, report both in separate columns with their correct labels.

## Cross-references

- Paper: Yadav et al., IEEE Access 2025, Table 2 (CARGOOD GAN, AP column).
- Repo: [`13_CARGOOD/cargood_replication.py`](../13_CARGOOD/cargood_replication.py).
- Report: REPORT.md, Section 8.3.8.
