"""Recompute the Test 1 and Test 2 percentages from the copied result files.

    python results/earlier_workspace_tests/verify_tests.py

Needs only pandas. It trains nothing. Inputs are the files in this folder (copied from an earlier
workspace of the author, runs of 2026-04-29) and the engine's Table 2 constants.

Definitions, taken from the two scripts in this folder:
  Test 1 (test1_uniform_protocol_test.py): per (dataset, method, metric) one comparison of two
    protocols, A (AUC/AP from probabilities) and B (AUC/AP from binary predictions); the "winner" is
    the protocol whose mean over the 100 folds is closer to the paper value; equal distances are ties.
  Test 2 (test2_comprehensive_protocol_test.py): per (variation, dataset, method) the mean over 20
    folds (5 splits x 4 repeats); a metric cell passes if |mean - paper| <= 3.0 (ACC, in points) or
    <= 0.03 (AUC, F1, AP, G-Mean).
Each result is printed twice: "as run" (paper values embedded in the scripts at the time) and
against Table 2 as printed (PAPER_TABLE2 of exact_replication_local.py, with the one override
described in the README).
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
T[("CARVGOOD", "ADASYN")] = dict(T[("CARVGOOD", "ADASYN")], AUC=0.98)  # printed value, see README

# ------------------------------------------------------------------ Test 1
s = pd.read_csv(HERE / "test1_summary_20260429_135108.csv")
print("TEST 1: %d rows (%s), %d metric comparisons" % (len(s), ", ".join(sorted(s.dataset.unique())), len(s) * 5))


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
            within["A"] += a <= TOL[k]
            within["B"] += b <= TOL[k]
            errs["A"].append(a if k == "ACC" else a * 100)
            errs["B"].append(b if k == "ACC" else b * 100)
        for p in "AB":
            rows_pass[p] += sum(errs[p]) / 5 < 3
    return wins, within, rows_pass


n = len(s) * 5
for label, f in (("as run (paper values embedded in the script)", lambda r, k: r["A_%s_paper" % k]),
                 ("against Table 2 as printed", lambda r, k: T[(r.dataset, r.method)][k])):
    w, wi, rp = test1(f)
    print(" %s" % label)
    print("   closer-to-paper counts: A %d (%.1f%%), B %d (%.1f%%), ties %d (%.1f%%)" % (
        w["A"], 100 * w["A"] / n, w["B"], 100 * w["B"] / n, w["tie"], 100 * w["tie"] / n))
    print("   among the %d AUC/AP comparisons: A %.1f%%, B %.1f%%" % (
        w["A"] + w["B"], 100 * w["A"] / (w["A"] + w["B"]), 100 * w["B"] / (w["A"] + w["B"])))
    print("   metric cells within tolerance: A %d (%.1f%%), B %d (%.1f%%) of %d" % (
        wi["A"], 100 * wi["A"] / n, wi["B"], 100 * wi["B"] / n, n))
    print("   rows with average error < 3 points: A %d, B %d of %d" % (rp["A"], rp["B"], len(s)))
    if label.startswith("as run"):
        check((w["A"], w["B"], w["tie"]) == (29, 49, 117), "as-run counts equal the stated 29 / 49 / 117 (14.9 / 25.1 / 60.0 percent)")
        same = all(abs(r["A_" + k] - r["B_" + k]) < 1e-12 for _, r in s.iterrows() for k in ("ACC", "F1", "GMEAN"))
        check(same and w["tie"] == 3 * len(s), "the 117 ties are structural: ACC, F1 and G-Mean are identical under A and B (3 x %d)" % len(s))
        allA = [r["A_%s_diff" % k] for _, r in s.iterrows() for k in K]
        allB = [r["B_%s_diff" % k] for _, r in s.iterrows() for k in K]
        check(abs(sum(allA) / n - 0.7213) < 5e-5 and abs(sum(allB) / n - 0.7159) < 5e-5,
              "stated mean differences 0.7213 / 0.7159 reproduced (mean over all metrics with ACC in points)")
n_paper_diff = sum(abs(r["A_%s_paper" % k] - T[(r.dataset, r.method)][k]) > 1e-9 for _, r in s.iterrows() for k in K)
print(" paper values in the Test 1 script that differ from Table 2 as printed: %d of %d cells" % (n_paper_diff, n))

# ------------------------------------------------------------------ Test 2
raw = pd.read_csv(HERE / "test2_raw_20260429_142018.csv")
raw = raw[raw.mean_diff_normalized.notna()]
print("TEST 2: variations %d, rows per variation (full sweep) %d" % (raw.variation.nunique(), (raw.variation == "V00_baseline").sum()))
res = {}
for v, g in raw.groupby("variation"):
    asrun = int(g.pass_count_3pct.sum())
    cells = len(g) * 5
    real = sum(abs(getattr(r, k) - T[(r.dataset, r.method)][k]) <= TOL[k] for r in g.itertuples() for k in K)
    res[v] = (asrun, int(real), cells)
print(" %-20s %12s %12s" % ("variation", "as run", "vs Table 2"))
for v in sorted(res, key=lambda x: (-res[x][0] / res[x][2])):
    a, b, c = res[v]
    print(" %-20s %4d/%d %5.1f%%  %4d/%d %5.1f%%" % (v, a, c, 100 * a / c, b, c, 100 * b / c))
check(res["V00_baseline"][:1] == (74,) and res["V01_platt_calib"][:1] == (85,) and res["V00_baseline"][2] == 195,
      "as-run counts equal the stated 74/195 = 37.9 percent (baseline) and 85/195 = 43.6 percent (Platt, isotonic)")
check(res["V02_isotonic_calib"][0] == 85, "isotonic calibration also 85/195 as run")
x = raw[raw.variation == "V00_baseline"]
print(" baseline: rows with average error < 3 points: %d of %d" % (
    sum((r.ACC_diff + sum(getattr(r, k + "_diff") * 100 for k in K[1:])) / 5 < 3 for r in x.itertuples()), len(x)))
sys.exit(1 if problems else 0)
