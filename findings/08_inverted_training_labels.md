# Finding 08: Inverted training labels on BCW RUSBoost

**Severity:** medium-high. Non-standard label convention applied silently.

## Symptom

On BCW, the RUSBoost cell in the original Table 2 is not reproducible under standard label convention (minority class encoded as 1, majority class encoded as 0).

## Evidence

The published values are reproducible only when training is performed on inverted labels: the originally majority Benign class is treated as positive (label = 1) during training, and during the ACC, AUC, and F1 calculations. This is consistent with computing the metrics for the majority class while reporting them under the minority-class column.

Standard convention in imbalanced-classification literature treats the under-represented class as positive. The BCW RUSBoost cell does the opposite without disclosure.

## Reproduction

See `04_BCW/bcw_final_replication.py`. The RUSBoost implementation flips the label convention for this cell only.

## Why this matters

Inverted-label evaluation makes the rare-class detection problem appear easier than it is, because the classifier is now solving the easier majority-class detection task. Comparison values for the BCW RUSBoost cell are therefore not directly comparable to other RUSBoost numbers in the literature, which use the standard label convention.

## Suggested fix for future authors

Always encode the minority class as positive (label = 1) and document it explicitly. If different label conventions are used for different cells, label each cell with its convention.

## Cross-references

- Paper: Yadav et al., IEEE Access 2025, Table 2 (BCW RUSBoost row).
- Repo: [`04_BCW/bcw_final_replication.py`](../04_BCW/bcw_final_replication.py).
- Report: REPORT.md, Section 8.3.7.
