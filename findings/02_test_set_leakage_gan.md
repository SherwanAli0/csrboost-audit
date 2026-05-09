# Finding 02: Test-set leakage on GAN-family methods

**Severity:** high. Constitutes data leakage on every GAN cell of the original Table 2.

## Symptom

The GAN and SMOTified-GAN cells in the original Table 2 cannot be reproduced under standard held-out test evaluation. Predictions on the test fold yield error margins exceeding the 3 percent tolerance on every (dataset, method) cell where these methods appear.

## Evidence

For each (dataset, GAN-method) cell we evaluated four candidate prediction sources:

1. Held-out test predictions (standard).
2. Resampled-training predictions (predictions on the SMOTE-balanced training set).
3. Original-training predictions (predictions on the unmodified training set).
4. Augmented-training predictions including synthetic GAN samples.

The published numbers are reproducible only when option 4 is used: evaluation on the augmented training set, including synthetic samples that the GAN itself generated and the classifier trained on. This is data leakage by definition: the classifier is being scored on data that contributed to its training.

In several datasets (ILPD, CARGOOD, CARVGOOD, YEAST5, FLARE-F), the GAN methods train two parallel classifiers per fold (one on raw features, one on standard-scaled features) and individual reported metrics are drawn from whichever copy aligns more closely with the paper. On PSDAS the same effect is achieved by extracting metrics from multiple output formats (raw scores and sigmoid-transformed probabilities) of a single classifier.

## Reproduction

See per-dataset scripts. The configuration matrix entry for any GAN cell records `eval_source = "augmented_train"` (or one of its variants).

## Why this matters

Comparison tables that include GAN-family methods evaluated on augmented training data are not directly comparable to GAN methods evaluated on held-out test data, which is the standard for the rest of the field. Any benchmark that uses CSRBoost's Table 2 as a baseline inherits this leakage.

## Suggested fix for future authors

Always evaluate on held-out test data only. If augmented evaluation is informative for diagnostic purposes, report it as a separate column with an explicit label (for example, "AP-on-augmented-train").

## Cross-references

- Paper: Yadav et al., IEEE Access 2025, Table 2 (GAN, SMOTified-GAN rows on PSDAS, ESR, DCCC, BCW, ILPD, CARGOOD, CARVGOOD, YEAST5, FLARE-F).
- Report: REPORT.md, Sections 8.2 and 8.3.4.
