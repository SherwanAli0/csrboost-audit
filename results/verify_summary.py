"""Recompute the headline counts and error summaries from the committed tables.

Needs only pandas. It trains no models and reads no datasets.

    python results/verify_summary.py

What is checked (definitions: "Match criterion" in README.md):
  1. results/replication_audit_final_20260425.csv (tuned track, 143 evaluated rows):
     avg_err = mean(|ACC diff| in points, |AUC/F1/AP/GMean diff| x 100) recomputed from the
     row's own columns and compared with the stored Average_Error(%).
  2. Its paper columns against PAPER_TABLE2 in exact_replication/exact_replication_local.py,
     against the PAPER_TABLE in combined_codes/BEST_CONFIGS.py, and the one documented
     override (PAPER_OVERRIDES below).
  3. results/exact_replication/baseline_phase1_comparison.csv (uniform-protocol track):
     42 PASS / 101 FAIL / 7 N/A recomputed, equality with _42_pass_cells_spec.csv, per-algorithm
     and CSRBoost counts.
  4. That the 42 rows of the spec appear unchanged as the replication values in the final audit.
  5. Mean absolute error of the 15 CSRBoost rows and, for information only, how many rows would
     meet a relative-error reading of the criterion.
Exit status is 1 if a check that should hold does not.
"""
import ast
import runpy
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
METRICS = ["ACC", "AUC", "F1", "AP", "GMean"]
NAME_MAP = {"YEAST5ERL": "YEAST5-ERL", "FLAREF": "FLARE-F"}
ALG_MAP = {"SMOTifiedGAN": "SMOTified-GAN"}

# Paper constant that differs between the author's copies of Table 2 and the table image
# (CARVGOOD / ADASYN / AUC: 0.98 in the printed Table 2; 0.99 in the engine and audit CSVs).
PAPER_OVERRIDES = {("CARVGOOD", "ADASYN", "AUC"): 0.98}

problems = 0


def check(ok, text):
    global problems
    print(("ok   " if ok else "FAIL ") + text)
    if not ok:
        problems += 1


def avg_err(rep, pap):
    errs = [(rep["ACC"] - pap["ACC"]).abs()]
    errs += [(rep[m] - pap[m]).abs() * 100 for m in ("AUC", "F1", "AP", "GMean")]
    return pd.concat(errs, axis=1).mean(axis=1)


# ---------------------------------------------------------------- 1. final audit
fin = pd.read_csv(RES / "replication_audit_final_20260425.csv", keep_default_na=False)
print("final audit file rows:", len(fin))
fin = fin[~fin["Replication_ACC(%)"].isin(["", "N/A"])].copy()
rep = pd.DataFrame({m: pd.to_numeric(fin["Replication_" + ("ACC(%)" if m == "ACC" else m)]) for m in METRICS})
pap = pd.DataFrame({m: pd.to_numeric(fin["Paper_" + ("ACC(%)" if m == "ACC" else m)]) for m in METRICS})
fin["calc"] = avg_err(rep, pap)
fin["stored"] = pd.to_numeric(fin["Average_Error(%)"])
check(len(fin) == 143, "evaluated rows = %d (expected 143)" % len(fin))
check(int((fin.calc < 3).sum()) == 143, "rows with avg_err < 3: %d; <= 3: %d; largest %.3f (%s / %s)" % (
    (fin.calc < 3).sum(), (fin.calc <= 3).sum(), fin.calc.max(),
    fin.loc[fin.calc.idxmax(), "Dataset"], fin.loc[fin.calc.idxmax(), "Algorithm"]))
check((fin.calc - fin.stored).abs().max() < 0.005,
      "max |recomputed - stored Average_Error|: %.4f" % (fin.calc - fin.stored).abs().max())
check(int((fin.Status == "PASS").sum()) == 143, "rows with Status == PASS: %d" % (fin.Status == "PASS").sum())

# ---------------------------------------------------------------- 2. paper constants
src = (ROOT / "exact_replication" / "exact_replication_local.py").read_text(encoding="utf-8")
engine = None
for node in ast.parse(src).body:
    if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "PAPER_TABLE2":
        engine = ast.literal_eval(node.value)
table = {}
for ds, per in engine.items():
    for alg, m in per.items():
        if m["ACC"] is not None:
            table[(NAME_MAP.get(ds, ds), ALG_MAP.get(alg, alg))] = m
best = runpy.run_path(str(ROOT / "combined_codes" / "BEST_CONFIGS.py"), run_name="not_main")["PAPER_TABLE"]
KEYS = {"ACC": "ACC", "AUC": "AUC", "F1": "F1", "AP": "AP", "GMean": "GMEAN"}


def n_diff(a_tab, b_tab):
    out = []
    for key, a in a_tab.items():
        for m, k in KEYS.items():
            if abs(a[k] - b_tab[key][k]) > 1e-9:
                out.append((key, m, a[k], b_tab[key][k]))
    return out


fin_tab = {(r.Dataset, r.Algorithm): {KEYS[m]: pap.loc[i, m] for m in METRICS} for i, r in fin.iterrows()}
d1 = n_diff(fin_tab, table)
check(len(d1) == 0, "final-audit paper columns vs PAPER_TABLE2 in the engine: %d differing cells" % len(d1))
d2 = n_diff(fin_tab, best)
check(len(d2) <= len(PAPER_OVERRIDES), "final-audit paper columns vs BEST_CONFIGS PAPER_TABLE: %d differing cell(s) %s" % (
    len(d2), [(k, m, a, b) for k, m, a, b in d2]))

fin2 = fin.copy()
pap2 = pap.copy()
for (ds, alg, m), v in PAPER_OVERRIDES.items():
    idx = fin2[(fin2.Dataset == ds) & (fin2.Algorithm == alg)].index[0]
    pap2.loc[idx, m] = v
fin2["calc"] = avg_err(rep, pap2)
check(int((fin2.calc < 3).sum()) == 143, "with the documented paper override(s): rows < 3: %d, largest %.3f" % (
    (fin2.calc < 3).sum(), fin2.calc.max()))
for (ds, alg, m), v in PAPER_OVERRIDES.items():
    idx = fin2[(fin2.Dataset == ds) & (fin2.Algorithm == alg)].index[0]
    print("     override %s/%s %s: paper %.2f instead of %.2f -> row avg_err %.3f instead of %.3f" % (
        ds, alg, m, v, pap.loc[idx, m], fin2.loc[idx, "calc"], fin.loc[idx, "calc"]))

# ---------------------------------------------------------------- 3. uniform-protocol track
base = pd.read_csv(RES / "exact_replication" / "baseline_phase1_comparison.csv", keep_default_na=False)
na = base[base["repl_ACC"].astype(str).isin(["", "N/A", "nan"])]
ev = base.drop(na.index).copy()
for c in ("ACC", "AUC", "F1", "AP", "GMEAN"):
    for p in ("paper_", "repl_"):
        ev[p + c] = pd.to_numeric(ev[p + c])
ev["calc"] = (abs(ev.repl_ACC - ev.paper_ACC) + sum(abs(ev["repl_" + c] - ev["paper_" + c]) * 100
              for c in ("AUC", "F1", "AP", "GMEAN"))) / 5
check(len(ev) == 143 and len(na) == 7, "uniform-protocol table: %d evaluated rows, %d N/A" % (len(ev), len(na)))
n_pass = int((ev.calc < 3).sum())
check(n_pass == 42 and int((ev.calc <= 3).sum()) == 42, "uniform-protocol rows with avg_err < 3: %d of %d (%.1f percent); FAIL %d" % (
    n_pass, len(ev), 100.0 * n_pass / len(ev), len(ev) - n_pass))
check(int((ev.status == "PASS").sum()) == n_pass, "stored status PASS count equals the recomputed count")
check((ev.calc - pd.to_numeric(ev.avg_err_pp)).abs().max() < 1e-3, "stored avg_err_pp reproduced (max gap %.5f)" % (
    (ev.calc - pd.to_numeric(ev.avg_err_pp)).abs().max()))
spec = pd.read_csv(RES / "exact_replication" / "_42_pass_cells_spec.csv")
check(set(zip(spec.dataset, spec.algorithm)) == set(zip(ev[ev.calc < 3].dataset, ev[ev.calc < 3].algorithm)),
      "the 42-row spec equals the set of passing rows")
per_alg = ev[ev.calc < 3].groupby("algorithm").size().to_dict()
cs = ev[(ev.algorithm == "CSRBoost")]
cs_fail = sorted(NAME_MAP.get(d, d) for d in cs[cs.calc >= 3].dataset)
print("     passing rows per algorithm:", per_alg)
check(per_alg.get("CSRBoost") == 10, "CSRBoost passes on %d of %d datasets; fails: %s" % (
    per_alg.get("CSRBoost", 0), len(cs), cs_fail))
bt = {(NAME_MAP.get(r.dataset, r.dataset), ALG_MAP.get(r.algorithm, r.algorithm)):
      {k: r["paper_" + k] for k in ("ACC", "AUC", "F1", "AP", "GMEAN")} for _, r in ev.iterrows()}
check(len(n_diff(bt, table)) == 0, "uniform-protocol paper columns vs PAPER_TABLE2: 0 differing cells")
r = ev[(ev.dataset == "CARVGOOD") & (ev.algorithm == "ADASYN")].iloc[0]
alt = (abs(r.repl_ACC - r.paper_ACC) + abs(r.repl_AUC - 0.98) * 100 + abs(r.repl_F1 - r.paper_F1) * 100
       + abs(r.repl_AP - r.paper_AP) * 100 + abs(r.repl_GMEAN - r.paper_GMEAN) * 100) / 5
print("     CARVGOOD/ADASYN with the override: avg_err %.3f instead of %.3f (still PASS)" % (alt, r.calc))

# ---------------------------------------------------------------- 4. swapped rows
bad = 0
for _, s in spec.iterrows():
    key = (NAME_MAP.get(s.dataset, s.dataset), ALG_MAP.get(s.algorithm, s.algorithm))
    f = fin[(fin.Dataset == key[0]) & (fin.Algorithm == key[1])]
    if len(f) != 1:
        bad += 1
        continue
    f = f.iloc[0]
    for c, fc in (("ACC", "ACC(%)"), ("AUC", "AUC"), ("F1", "F1"), ("AP", "AP"), ("GMEAN", "GMean")):
        if abs(float(s["repl_" + c]) - float(f["Replication_" + fc])) > 0.0051:
            bad += 1
check(bad == 0, "the 42 spec rows appear unchanged as replication values in the final audit (%d differences)" % bad)

# ---------------------------------------------------------------- 5. summaries
csr = fin[fin.Algorithm == "CSRBoost"].index
mae = {m: (rep.loc[csr, m] - pap2.loc[csr, m]).abs().mean() for m in METRICS}
print("CSRBoost mean absolute error over %d datasets (ACC in points, others on the 0-1 scale):" % len(csr))
print("     " + "  ".join("%s=%.4f" % (m, v) for m, v in mae.items()))
relerr = sum((rep[m] - pap2[m]).abs() / pap2[m] * 100 for m in METRICS) / 5
print("information only: rows that would be < 3 if the same metrics were read as relative errors: %d of %d" % (
    int((relerr < 3).sum()), len(relerr)))
sys.exit(1 if problems else 0)
