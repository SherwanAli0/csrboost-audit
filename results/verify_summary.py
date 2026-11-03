"""Recompute the summary arithmetic from the committed result tables.

Needs only pandas. It trains no models and reads no datasets.

    python results/verify_summary.py

Checks (see "Match criterion" in README.md):
  1. avg_err = mean of |ACC diff| (percentage points) and |AUC/F1/AP/GMean diff| x 100,
     recomputed from results/replication_results.csv, against the stored Avg_Error.
  2. Number of evaluated rows and number with avg_err < 3 (and <= 3).
  3. The paper columns of that CSV against PAPER_TABLE in combined_codes/BEST_CONFIGS.py.
  4. Mean absolute error of the CSRBoost rows.
  5. (Sensitivity only) how many rows would meet a relative-error reading of the criterion.
"""
import runpy
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
METRICS = ["ACC", "AUC", "F1", "AP", "GMean"]

df = pd.read_csv(ROOT / "results" / "replication_results.csv", keep_default_na=False)
df = df[df["Status"] != "N/A"].copy()
for m in METRICS:
    df["P_" + m] = pd.to_numeric(df["Paper_" + (m + "(%)" if m == "ACC" else m)])
    df["R_" + m] = pd.to_numeric(df["Replication_" + (m + "(%)" if m == "ACC" else m)])

err = pd.DataFrame({m: (df["R_" + m] - df["P_" + m]).abs() * (1 if m == "ACC" else 100) for m in METRICS})
df["avg_err"] = err.mean(axis=1)
df["stored"] = df["Avg_Error"].str.rstrip("%").astype(float)

problems = 0
print("evaluated rows:", len(df))
print("rows with avg_err < 3 :", int((df.avg_err < 3).sum()))
print("rows with avg_err <= 3:", int((df.avg_err <= 3).sum()))
print("rows with Status == PASS:", int((df.Status == "PASS").sum()))
print("largest recomputed avg_err: %.3f (%s / %s)" % (
    df.avg_err.max(), df.loc[df.avg_err.idxmax(), "Dataset"], df.loc[df.avg_err.idxmax(), "Algorithm"]))
worst = (df.avg_err - df.stored).abs().max()
print("max |recomputed - stored Avg_Error|: %.3f (stored value is rounded to 0.1)" % worst)
if worst > 0.05 + 1e-9 or len(df) != 143 or int((df.avg_err < 3).sum()) != len(df):
    problems += 1
    print("PROBLEM: summary arithmetic does not reproduce 143/143")

# Sensitivity: the same five metrics read as RELATIVE errors (|rep - paper| / paper x 100).
# This is NOT the committed criterion; it is printed only to show that the definition matters.
rel = sum((df["R_" + m] - df["P_" + m]).abs() / df["P_" + m] * 100 for m in METRICS) / 5
print("sensitivity (relative-error reading, not the committed criterion): rows < 3:", int((rel < 3).sum()), "of", len(df))

table = runpy.run_path(str(ROOT / "combined_codes" / "BEST_CONFIGS.py"), run_name="not_main")["PAPER_TABLE"]
mism = 0
for _, r in df.iterrows():
    ref = table.get((r.Dataset, r.Algorithm))
    if ref is None or any(abs(r["P_" + m] - ref["GMEAN" if m == "GMean" else m]) > 1e-9 for m in METRICS):
        mism += 1
print("rows whose paper columns differ from PAPER_TABLE:", mism, "of", len(df))
problems += mism > 0

cs = df[df.Algorithm == "CSRBoost"]
mae = {m: (cs["R_" + m] - cs["P_" + m]).abs().mean() for m in METRICS}
print("CSRBoost mean absolute error over %d datasets (ACC in points, others on 0-1 scale):" % len(cs))
print("  " + "  ".join("%s=%.4f" % (m, v) for m, v in mae.items()))
sys.exit(1 if problems else 0)
