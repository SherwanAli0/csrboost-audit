# Finding 05: Majority-class AP reported as standard AP

**Severity:** medium-high. AP polarity inversion on four datasets without disclosure.

## Symptom

On BCW, CB, CARGOOD, and CARVGOOD, several reported AP cells are not reproducible using the standard minority-class AP. The standard `average_precision_score(y_true, y_pred)` differs from the published value by margins exceeding the 3 percent tolerance.

## Evidence

We verified that the published values are reproducible only when AP is computed against the majority class instead of the minority class:

```python
# Published AP is consistent with this, not with the standard call:
average_precision_score(y_true, 1 - y_pred, pos_label=0)
```

The standard convention in imbalanced-classification literature is that AP characterises detection of the rare class. Reporting majority-class AP under the AP column without disclosure conflates two distinct quantities.

## Reproduction

See `04_BCW/bcw_final_replication.py`, `06_CB/cb_replication_v2.py`, `13_CARGOOD/cargood_replication.py`, `14_CARVGOOD/carvgood_replication.py`. The configuration matrix records `ap_pos_label = 0` for the affected cells.

## Why this matters

Reported AP values for these cells are not directly comparable to AP values reported by other publications in the same problem area. Any meta-analysis or benchmark that aggregates AP values across studies inherits this incompatibility.

## Suggested fix for future authors

Always compute AP for the minority class (`pos_label = 1` if minority is encoded as 1). If majority-class precision-recall is informative, label it as a separate metric ("AP-majority") rather than reporting it under the AP column.

## Cross-references

- Paper: Yadav et al., IEEE Access 2025, Table 2 (BCW, CB, CARGOOD, CARVGOOD AP columns, multiple methods).
- Report: REPORT.md, Section 8.3.5.
