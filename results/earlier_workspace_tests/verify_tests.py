"""Recompute the Test 1 and Test 2 figures from the copied result files.

    python results/earlier_workspace_tests/verify_tests.py

Needs only pandas. It trains nothing and imports no research script. Inputs are the files in this
folder (copied from an earlier workspace of the author, runs of 2026-04-29) and the engine's Table 2
constants. These are calculations from stored files, not reruns.

Definitions, taken from the two scripts in this folder:
  Test 1 (test1_uniform_protocol_test.py): per (dataset, method, metric) one comparison of two
    protocols, A (AUC/AP from probabilities) and B (AUC/AP from binary predictions); the protocol whose
    mean over 100 saved fold values is closer to the reference is the "closer" one; equal distances are ties.
  Test 2 (test2_comprehensive_protocol_test.py): per (variation, dataset, method) the mean over 20
    evaluations (5 splits x 4 repeats); a metric cell is within tolerance if |mean - reference| <= 3.0
    (ACC, in points) or <= 0.03 (AUC, F1, AP, G-Mean).
Every result is shown against two references:
  ORIGINAL reference  - the paper values embedded in the scripts at the time of the runs;
  CORRECTED reference - Table 2 as printed, i.e. PAPER_TABLE2 of exact_replication_local.py as corrected
                        in this repository (the repository's paper-value table, not a new independent
                        transcription of every paper cell).
"""
import ast
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
K = ["ACC", "AUC", "F1", "AP", "GMEAN"]
TOL = {"ACC": 3.0, "AUC": 0.03, "F1": 0.03, "AP": 0.03, "GMEAN": 0.03}
NAME = {"YEAST5ERL": "YEAST5-ERL", "FLAREF": "FLARE-F"}
ALG = {"SMOTifiedGAN": "SMOTified-GAN"}
problems = 0


def check(ok, text):
    global problems
    print(("ok   " if ok else "FAIL ") + text)
    problems += 0 if ok else 1


src = (ROOT / "exact_replication" / "exact_replication_local.py").read_text(encoding="utf-8")
table = None
for node in ast.parse(src).body:
    if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "PAPER_TABLE2":
        table = ast.literal_eval(node.value)
T = {(NAME.get(d, d), ALG.get(a, a)): m for d, per in table.items() for a, m in per.items() if m["ACC"] is not None}

# ------------------------------------------------------------------ Test 1
s = pd.read_csv(HERE / "test1_summary_20260429_135108.csv")
n = len(s) * 5
print("TEST 1: %d rows (%s), %d metric comparisons" % (len(s), ", ".join(sorted(s.dataset.unique())), n))
folds = pd.read_csv(HERE / "test1_fold_results_20260429_135108.csv")
groups = folds.groupby(["dataset", "method"])
check(len(folds) == 3900 and len(groups) == 39 and all(len(g) == 100 and g.fold.nunique() == 100 for _, g in groups),
      "fold file: %d saved fold rows, %d dataset-method combinations, 100 distinct fold IDs each (configured protocol 5 x 20; "
      "this shows the saved rows, not an independent check that each fold of a fresh run would complete)" % (len(folds), len(groups)))
bad = 0
for _, r in s.iterrows():
    g = groups.get_group((r.dataset, r.method))
    for p_ in "AB":
        for k in K:
            bad += abs(g[p_ + "_" + k].mean() - r[p_ + "_" + k]) > 1e-10
check(bad == 0, "the summary means equal the means of the saved 100 fold values (tolerance 1e-10; %d disagreements of %d)" % (bad, 2 * n))


def test1(paper_of):
    wins = {"A": 0, "B": 0, "tie": 0}
    within = {"A": 0, "B": 0}
    rows_pass = {"A": 0, "B": 0}
    for _, r in s.iterrows():
        errs = {"A": [], "B": []}
        for k in K:
            pv = paper_of(r, k)
            a, b = abs(r["A_" + k] - pv), abs(r["B_" + k] - pv)
            wins["tie" if abs(a - b) < 1e-6 else ("A" if a < b else "B")] += 1
            within["A"] += a <= TOL[k] + 1e-10
            within["B"] += b <= TOL[k] + 1e-10
            errs["A"].append(a if k == "ACC" else a * 100)
            errs["B"].append(b if k == "ACC" else b * 100)
        for p_ in "AB":
            rows_pass[p_] += sum(errs[p_]) / 5 < 3
    return wins, within, rows_pass


res1 = {}
for label, f in (("ORIGINAL reference", lambda r, k: r["A_%s_paper" % k]),
                 ("CORRECTED reference", lambda r, k: T[(r.dataset, r.method)][k])):
    w, wi, rp = test1(f)
    res1[label] = (w, wi, rp)
    print(" %s" % label)
    print("   relative closeness to the reference: A %d (%.1f%%), B %d (%.1f%%), equal distance %d (%.1f%%) of %d" % (
        w["A"], 100 * w["A"] / n, w["B"], 100 * w["B"] / n, w["tie"], 100 * w["tie"] / n, n))
    print("   among the %d AUC/AP comparisons: A %.1f%%, B %.1f%%" % (
        w["A"] + w["B"], 100 * w["A"] / (w["A"] + w["B"]), 100 * w["B"] / (w["A"] + w["B"])))
    print("   metric cells within the per-metric tolerance: A %d (%.1f%%), B %d (%.1f%%) of %d" % (
        wi["A"], 100 * wi["A"] / n, wi["B"], 100 * wi["B"] / n, n))
    print("   rows with average error < 3 points: A %d, B %d of %d" % (rp["A"], rp["B"], len(s)))
w, wi, rp = res1["ORIGINAL reference"]
check((w["A"], w["B"], w["tie"]) == (29, 49, 117), "relative closeness (original reference) equals the stated 29 / 49 / 117 = 14.9 / 25.1 / 60.0 percent")
check((wi["A"], wi["B"]) == (72, 75), "tolerance counts (original reference): A 72/195 (36.9%), B 75/195 (38.5%)")
wc, wic, _ = res1["CORRECTED reference"]
check((wic["A"], wic["B"]) == (67, 64), "tolerance counts (corrected reference): A 67/195 (34.4%), B 64/195 (32.8%)")
same = all(abs(r["A_" + k] - r["B_" + k]) < 1e-12 for _, r in s.iterrows() for k in ("ACC", "F1", "GMEAN"))
check(same and w["tie"] == 3 * len(s), "the %d equal-distance cases are structural: ACC, F1 and G-Mean are identical under A and B (3 x %d)" % (w["tie"], len(s)))
allA = [r["A_%s_diff" % k] for _, r in s.iterrows() for k in K]
allB = [r["B_%s_diff" % k] for _, r in s.iterrows() for k in K]
check(abs(sum(allA) / n - 0.7213) < 5e-5 and abs(sum(allB) / n - 0.7159) < 5e-5,
      "stated mean differences 0.7213 / 0.7159 reproduced (mean over all metrics, ACC in points; dominated by ACC)")
dis = {}
for _, r in s.iterrows():
    for k in K:
        if abs(r["A_%s_paper" % k] - T[(r.dataset, r.method)][k]) > 1e-9:
            dis[r.dataset] = dis.get(r.dataset, 0) + 1
check(sum(dis.values()) == 95, "reference disagreements (original vs corrected) over the 195 evaluated cells: %d %s" % (sum(dis.values()), dict(sorted(dis.items()))))

# ------------------------------------------------------------------ Test 2
raw = pd.read_csv(HERE / "test2_raw_20260429_142018.csv")
raw = raw[raw.mean_diff_normalized.notna()]
print("TEST 2: %d variations, %d rows per variation in the full sweep (20 evaluations per row: 5 splits x 4 repeats)" % (
    raw.variation.nunique(), (raw.variation == "V00_baseline").sum()))
res = {}
for v, g in raw.groupby("variation"):
    orig = int(g.pass_count_3pct.sum())
    cells = len(g) * 5
    corr = sum(abs(getattr(r, k) - T[(r.dataset, r.method)][k]) <= TOL[k] + 1e-10 for r in g.itertuples() for k in K)
    rows = sum((abs(r.ACC - T[(r.dataset, r.method)]["ACC"]) + sum(abs(getattr(r, k) - T[(r.dataset, r.method)][k]) * 100
               for k in K[1:])) / 5 < 3 for r in g.itertuples())
    res[v] = (orig, int(corr), cells, int(rows), len(g))
print(" %-20s %-20s %-22s %s" % ("variation", "ORIGINAL reference", "CORRECTED reference", "rows < 3 points (corrected)"))
for v in sorted(res, key=lambda x: -res[x][0] / res[x][2]):
    a, b, c, rw, nr = res[v]
    print(" %-20s %4d/%d = %5.1f%%    %4d/%d = %5.1f%%        %d of %d" % (v, a, c, 100 * a / c, b, c, 100 * b / c, rw, nr))
check([res[v][0] for v in ("V00_baseline", "V01_platt_calib", "V02_isotonic_calib")] == [74, 85, 85],
      "original reference: baseline 74, Platt 85, isotonic 85 of 195 (the tie exists only under this reference)")
check([res[v][1] for v in ("V00_baseline", "V01_platt_calib", "V02_isotonic_calib")] == [69, 78, 75],
      "corrected reference: baseline 69, Platt 78, isotonic 75 of 195 (no tie)")
check([res[v][3] for v in ("V00_baseline", "V01_platt_calib", "V02_isotonic_calib")] == [2, 4, 4],
      "corrected reference, row criterion (average error < 3 points): baseline 2, Platt 4, isotonic 4 of 39")
check(all(res[v][2] == 195 for v in res if v not in ("V11_retain_30", "V12_retain_70", "V13_class_redefine")),
      "13 variations on 39 rows (195 cells); the three CSRBoost-only variations on 5 rows (25 cells)")
print("Reading: none of the tested variations reproduced all evaluated entries; this does not rule out other uniform protocols.")
sys.exit(1 if problems else 0)
