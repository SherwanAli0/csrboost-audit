# Finding 03: Per-metric decision-threshold switching on RUSBoost and HUE

**Severity:** high. Five different thresholds applied to the same model to produce five reported metrics within a single (dataset, method) cell.

## Symptom

The BCW and CB RUSBoost cells, and several HUE cells, require five different decision thresholds to produce the five reported metrics from the same trained model. Standard evaluation uses a single threshold (typically 0.5) for all threshold-dependent metrics.

Specifically:

- On CB, the reported AP = 0.97 alongside reported AUC = 0.72 is mathematically inconsistent under any single threshold.
- On BCW, the reported AP = 0.75 alongside AUC = 0.97 exhibits the same impossibility.

These mismatches between AP and AUC cannot be resolved by choosing any single threshold; they require AP and AUC to be computed at different operating points.

## Evidence

We verified the impossibility by simulating the full threshold space (every possible threshold from 0 to 1 in steps of 0.001) on the same classifier outputs and checking whether any threshold landed all five metrics within tolerance simultaneously. None did, on either BCW or CB RUSBoost.

The published numbers are reproducible only by selecting the threshold that minimises the gap to the published value for each metric independently. This is a 1:5 inflation of the parameter space relative to standard evaluation, applied silently in the original publication.

## Reproduction

See `04_BCW/bcw_final_replication.py` and `06_CB/cb_replication_v2.py`. The configuration matrix records five separate thresholds per cell, one per metric.

## Why this matters

Reporting five metrics from one model means the metrics characterise the model at one operating point. Reporting metrics from different operating points without disclosure conflates "best metric value reachable for this model" with "model performance at a fixed operating point" and inflates apparent performance.

## Suggested fix for future authors

Report all five threshold-dependent metrics (ACC, F1, G-Mean, AP at threshold) at the same threshold. If the threshold differs from 0.5, state it. If multiple thresholds are reported as a sensitivity analysis, label each with its threshold.

## Cross-references

- Paper: Yadav et al., IEEE Access 2025, Table 2 (RUSBoost rows on BCW, CB; HUE rows on multiple datasets).
- Repo: [`04_BCW/bcw_final_replication.py`](../04_BCW/bcw_final_replication.py), [`06_CB/cb_replication_v2.py`](../06_CB/cb_replication_v2.py).
- Report: REPORT.md, Sections 8.3.3 and 8.3.9.
