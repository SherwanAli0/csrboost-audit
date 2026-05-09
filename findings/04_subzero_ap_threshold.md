# Finding 04: Sub-zero AP threshold on FLARE-F SMOTified-GAN

**Severity:** high. The cell is reproducible only by forcing every prediction positive, which collapses AP to a class-prevalence constant.

## Symptom

The FLARE-F SMOTified-GAN AP cell in the original Table 2 is not reproducible under any positive decision threshold using `average_precision_score`.

## Evidence

We swept the threshold from 1.0 down to 0.0 in steps of 0.001 on the trained classifier; no positive threshold produced an AP within 3 percent of the published value. Extending the sweep below zero (every prediction is forced positive) eventually lands the cell within tolerance. At this threshold the AP value reduces to the class prevalence of the augmented training set, which has no information content about the classifier.

This is the only operating point that lands the cell within the 3 percent tolerance.

## Reproduction

See `15_FLARE-F/flaref_replication.py`. The configuration matrix records `ap_threshold = -0.1` (or similar negative value) for this cell.

## Why this matters

Reporting AP at a sub-zero threshold means the published number is not characterising classifier performance at all. It is reporting the prevalence of the positive class in the training set under a different label.

## Suggested fix for future authors

AP is a curve-based metric and should be computed via `average_precision_score(y_true, y_score)` over predicted probabilities, without any threshold cutoff. If thresholded prediction is desired (for comparing against a fixed operating point), use precision-recall at threshold and label it as such, not AP.

## Cross-references

- Paper: Yadav et al., IEEE Access 2025, Table 2 (FLARE-F SMOTified-GAN, AP column).
- Repo: [`15_FLARE-F/flaref_replication.py`](../15_FLARE-F/flaref_replication.py).
- Report: REPORT.md, Section 8.3.6.
