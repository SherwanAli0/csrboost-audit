"""Compare combined_codes/BEST_CONFIGS.py with the configuration dictionaries of the final
per-dataset scripts, cell by cell.

    python combined_codes/compare_configs.py          (pandas not needed; writes the CSV next to this file)

Why this is partial: only seven of the fifteen final scripts keep their per-cell configuration in a
literal dictionary (04_BCW, 05_ESDRP, 06_CB, 07_GLASS, 08_ILPD, 12_YEAST5-ERL, 14_CARVGOOD). In the
other eight (PSDAS, ESR, DCCC, SEED, WINE, YEAST5, CARGOOD, FLARE-F) the configuration is written into
the run functions, so a machine comparison is not possible and the cell is reported as such.
Compared fields (when present on both sides): tree depth, number of boosting rounds, scaler,
CSRBoost cluster retention and p, and the decision threshold(s).
"""
import ast
import csv
import runpy
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
best = runpy.run_path(str(HERE / "BEST_CONFIGS.py"), run_name="not_main")["BEST_CONFIGS"]
NA = runpy.run_path(str(HERE / "BEST_CONFIGS.py"), run_name="not_main")["NA_CELLS"]

LITERAL = {  # folder -> (dataset key in BEST_CONFIGS, name of the dict in the script)
    "04_BCW": ("BCW", "BEST_CONFIGS"), "05_ESDRP": ("ESDRP", "BEST_CONFIGS"), "06_CB": ("CB", "BEST_CONFIGS"),
    "07_GLASS": ("GLASS", "BEST_CONFIGS"), "08_ILPD": ("ILPD", "BEST_CONFIGS"),
    "12_YEAST5-ERL": ("YEAST5-ERL", "BEST"), "14_CARVGOOD": ("CARVGOOD", "CONFIGS"),
}
HARDCODED = {"01_PSDAS": "PSDAS", "02_ESR": "ESR", "03_DCCC": "DCCC", "09_SEED": "SEED", "10_WINE": "WINE",
             "11_YEAST5": "YEAST5", "13_CARGOOD": "CARGOOD", "15_FLARE-F": "FLARE-F"}
ALG = {"SMOTifiedGAN": "SMOTified-GAN"}
swapped = set()
spec = (ROOT / "results" / "exact_replication" / "_42_pass_cells_spec.csv").read_text(encoding="utf-8").splitlines()[1:]
for ln in spec:
    d, a = ln.split(",")[:2]
    swapped.add(({"YEAST5ERL": "YEAST5-ERL", "FLAREF": "FLARE-F"}.get(d, d), ALG.get(a, a)))


def script_cfg(folder, name):
    f = next((ROOT / folder).glob("*.py"))
    for node in ast.parse(f.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == name:
            return ast.literal_eval(node.value)


def norm_scaler(x):
    return "none" if x in (None, "none") else x


def compare(b, s):
    """Return list of (field, best_value, script_value) for fields present on both sides."""
    out = []
    pairs = [("depth", b.get("depth"), s.get("depth", s.get("d", "<absent>"))),
             ("n_est", b.get("n_est"), s.get("n_est", s.get("n", "<absent>"))),
             ("scaler", norm_scaler(b.get("scaler")), norm_scaler(s["scaler"]) if "scaler" in s else "<absent>")]
    csr = b.get("csrboost_args")
    if csr:
        pairs.append(("cluster_pct", csr.get("cluster_pct"), s.get("samp", s.get("cluster_pct", "<absent>"))))
        pairs.append(("p", csr.get("p"), s.get("p", "<absent>")))
    ths = {v.get("th") for v in b["metrics"].values() if v.get("th") is not None}
    sth = {s[k] for k in ("thresh", "th1", "th2") if k in s}
    if sth:
        pairs.append(("threshold", sorted(ths), sorted(sth)))
    for field, bv, sv in pairs:
        if sv == "<absent>":
            continue
        out.append((field, bv, sv))
    return out


rows = []
for (ds, alg), cfg in sorted(best.items()):
    key = (ds, alg)
    folder = next((f for f, (d, _) in LITERAL.items() if d == ds), None)
    row = {"dataset": ds, "algorithm": alg, "swapped_to_uniform_protocol": key in swapped}
    if cfg == "N/A":
        row.update(status="N/A in the paper", detail="")
    elif folder is None:
        row.update(status="not machine-comparable (configuration is inside run functions)", detail="")
    else:
        sc = script_cfg(folder, LITERAL[folder][1])
        s = sc.get(alg) or sc.get(alg.replace("SMOTified-GAN", "SMOTifiedGAN"))
        if s is None:
            row.update(status="no entry in the script dictionary", detail="")
        else:
            diffs = [(f, b, v) for f, b, v in compare(cfg, s) if f != "threshold" and b != v
                     or (f == "threshold" and b != v)]
            fields = [f for f, _, _ in compare(cfg, s)]
            row.update(status="differs" if diffs else "match",
                       detail="; ".join("%s: BEST_CONFIGS %s / script %s" % (f, b, v) for f, b, v in diffs)
                       or "compared: " + ", ".join(fields))
    rows.append(row)

out = HERE / "best_configs_vs_final_scripts.csv"
with open(out, "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=["dataset", "algorithm", "swapped_to_uniform_protocol", "status", "detail"])
    w.writeheader()
    w.writerows(rows)

from collections import Counter
print("cells:", len(rows), " (143 evaluated + 7 N/A expected)")
print(Counter(r["status"] for r in rows))
cmp_rows = [r for r in rows if r["status"] in ("match", "differs")]
print("machine-compared cells:", len(cmp_rows), "| of which swapped to uniform protocol:",
      sum(r["swapped_to_uniform_protocol"] for r in cmp_rows))
print("differs among swapped:", sum(r["status"] == "differs" and r["swapped_to_uniform_protocol"] for r in rows),
      "| differs among not swapped:", sum(r["status"] == "differs" and not r["swapped_to_uniform_protocol"] for r in rows))
print("wrote", out)
