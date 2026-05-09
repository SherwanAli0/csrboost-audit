# Finding 07: Asymmetric boosting weight update on BCW RUSBoost

**Severity:** high. Equivalent to training AdaBoost on the minority class only, while reporting under standard RUSBoost.

## Symptom

The BCW RUSBoost row in the original Table 2 is not reproducible under the standard symmetric AdaBoost weight update. Mean error across all five reported metrics is 2.25 percent under standard symmetric updates, exceeding the 3 percent tolerance only by virtue of one metric being lucky.

## Evidence

The standard symmetric boosting update is:

```python
w *= np.exp(-alpha * (2*y - 1) * (2*ypred - 1))
```

This scales weights for both classes based on agreement between true label `y` and prediction `ypred`.

The asymmetric update that reproduces the published BCW RUSBoost numbers within 0.4 percent average error is:

```python
w *= np.exp(-alpha * y * (2*ypred - 1))
```

This update applies only when `y == 1` (the minority class). Majority-class samples (`y == 0`) receive no weight update during boosting, which is mathematically equivalent to training AdaBoost on the minority class only.

The original paper reports a single set of BCW RUSBoost numbers without specifying which weight update was used. Only the non-standard asymmetric form lands all five metrics inside the 3 percent envelope.

## Reproduction

See `04_BCW/bcw_final_replication.py`. The RUSBoost implementation in that file uses the asymmetric form with a comment marking the deviation.

## Why this matters

Standard RUSBoost (Seiffert et al., 2010) uses the symmetric AdaBoost weight update on the random-undersampled training set. The asymmetric variant is a different algorithm with different training dynamics and different generalisation characteristics. Reporting the asymmetric variant under the RUSBoost column conflates two distinct algorithms.

## Suggested fix for future authors

When using non-standard boosting weight updates, label the algorithm explicitly (for example, "RUSBoost-asymmetric") and cite the source paper for the variant. If the variant is a contribution of the work itself, describe it in the methodology section.

## Cross-references

- Paper: Yadav et al., IEEE Access 2025, Table 2 (BCW RUSBoost row).
- Repo: [`04_BCW/bcw_final_replication.py`](../04_BCW/bcw_final_replication.py).
- Report: REPORT.md, Section 8.3.2.
