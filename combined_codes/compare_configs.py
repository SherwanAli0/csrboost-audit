"""Compare combined_codes/BEST_CONFIGS.py (a historical snapshot) with the final per-dataset scripts, cell by cell.

    python combined_codes/compare_configs.py        (standard library only; writes best_configs_vs_final_scripts.csv)

Three different verification levels exist and are kept apart:
  1. configuration agreement - this script: do the recorded configuration fields match what the final script
     contains? It is a STATIC comparison of source text. It does not run any model.
  2. CSV arithmetic - results/verify_summary.py: do the stored tables recompute to the stated counts?
  3. reproduced model run - not performed in this repair pass.
Passing level 1 says nothing about level 3.

Sources of the script side, by script:
  dictionary scripts (04_BCW, 05_ESDRP, 06_CB, 07_GLASS, 08_ILPD, 12_YEAST5-ERL, 14_CARVGOOD): the configuration
    dictionary is read with ast and compared field by field (status "match" / "differs").
  function or marker scripts (09_SEED, 10_WINE, 11_YEAST5, 13_CARGOOD, 15_FLARE-F): the source text of each
    method's run function or comment-delimited block is searched for tree depth, boosting rounds and decision
    thresholds written as literals (status "partial agreement (static text)" / "differs (static text)"). Reading of
    literals can miss values that are computed; these statuses are heuristic and do not establish agreement on unextracted fields.
  global-constant scripts (01_PSDAS, 02_ESR, 03_DCCC): the configuration is spread over module constants,
    classes and metric functions; not compared (status "unverified").
Compared fields (when present on both sides): tree depth, number of boosting rounds, decision threshold set,
and for dictionary scripts also scaler, CSRBoost cluster retention and p. GAN-family cells are compared
only through thresholds. A threshold set that is a strict subset of the other set is reported as
"unverified (threshold representation)", because the two sides may record the same thresholds differently.
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
FUNCS = {"09_SEED": "SEED", "10_WINE": "WINE", "11_YEAST5": "YEAST5"}          # one run function per method
MARKERS = {"13_CARGOOD": "CARGOOD", "15_FLARE-F": "FLARE-F"}                    # "# ---- Method:" blocks
GLOBALS = {"01_PSDAS": "PSDAS", "02_ESR": "ESR", "03_DCCC": "DCCC"}              # module constants and classes
import re
FUNC_NAMES = {"CSRBoost": "csrboost", "ADASYN": "adasyn", "Borderline-SMOTE": "bsmote", "SMOTE-Tomek": "smotetomek",
              "SMOTE-ENN": "smoteenn", "AdaBoost": "adaboost", "RUSBoost": "rusboost", "HUE": "hue",
              "GAN": "gan", "SMOTified-GAN": ("smotigan", "smotifiedgan")}
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


def best_thresholds(cfg):
    return {v.get("th") for v in cfg["metrics"].values() if v.get("th") is not None}


def static_blocks(folder):
    """method -> source text of its run function or marker block, for the function and marker scripts."""
    f = next((ROOT / folder).glob("*.py"))
    text = f.read_text(encoding="utf-8")
    blocks = {}
    if folder in FUNCS:
        tree = ast.parse(text)
        funcs = {n.name: ast.get_source_segment(text, n) for n in tree.body if isinstance(n, ast.FunctionDef)}
        for alg, key in FUNC_NAMES.items():
            keys = key if isinstance(key, tuple) else (key,)
            for k in keys:
                if "run_%s_fold" % k in funcs:
                    blocks[alg] = funcs["run_%s_fold" % k]
    else:
        marks = [(m.start(), m.group(1)) for m in re.finditer(r"# ---- ([A-Za-z-]+):", text)]
        for n_, (pos, name) in enumerate(marks):
            end = marks[n_ + 1][0] if n_ + 1 < len(marks) else len(text)
            blocks[name] = text[pos:end]
    return blocks


def literals(block):
    depth = set(re.findall(r"max_depth\s*=\s*(\d+|None)", block))
    n_est = set(re.findall(r"n_est(?:imators)?\s*=\s*(\d+)", block))
    ths = set()
    for m in re.finditer(r"^\s*(?:th\w*(?:\s*,\s*th\w*)*)\s*=\s*([0-9.]+(?:\s*,\s*[0-9.]+)*)\s*(?:#.*)?$", block, re.M):
        ths.update(float(x) for x in re.findall(r"[0-9]*\.[0-9]+", m.group(1)))
    ths.update(float(x) for x in re.findall(r">=\s*(0\.[0-9]+)", block))
    rf = {int(x) for x in re.findall(r"(?:rf_trees|n_rf)\s*=\s*(\d+)", block)}
    return {"depth": {None if d == "None" else int(d) for d in depth}, "n_est": {int(x) for x in n_est}, "th": ths, "rf": rf}


def classify_static(cfg, alg, block):
    lit = literals(block)
    gan = alg in ("GAN", "SMOTified-GAN")
    diffs, notes = [], []
    if alg == "HUE":
        # HUE has several depth-like parameters (bag forest depth, boosting depth): compare membership only
        ha = cfg.get("hue_args") or {}
        if ha.get("max_depth") is not None and lit["depth"]:
            (notes if ha["max_depth"] in lit["depth"] else diffs).append("hue max_depth: BEST_CONFIGS %s / script %s" % (ha["max_depth"], sorted(lit["depth"], key=str)))
        pool = lit["rf"] or lit["n_est"]
        if ha.get("rf_trees") is not None and pool:
            (notes if ha["rf_trees"] in pool else diffs).append("hue rf_trees: BEST_CONFIGS %s / script %s" % (ha["rf_trees"], sorted(pool)))
    elif not gan:
        if lit["depth"]:
            (diffs if lit["depth"] != {cfg.get("depth")} else notes).append("depth: BEST_CONFIGS %s / script %s" % (cfg.get("depth"), sorted(lit["depth"], key=str)))
        if lit["n_est"]:
            (diffs if lit["n_est"] != {cfg.get("n_est")} else notes).append("n_est: BEST_CONFIGS %s / script %s" % (cfg.get("n_est"), sorted(lit["n_est"])))
    bt, st = best_thresholds(cfg), lit["th"]
    thr_state = None
    if bt and st:
        if bt == st:
            notes.append("thresholds equal %s" % sorted(bt))
        elif bt < st or st < bt:
            thr_state = "subset"
        else:
            diffs.append("thresholds: BEST_CONFIGS %s / script %s" % (sorted(bt), sorted(st)))
    if diffs:
        return "differs (static text)", "; ".join(diffs)
    if thr_state:
        return "unverified (threshold representation)", "BEST_CONFIGS thresholds %s, script thresholds %s; one set contains the other" % (sorted(bt), sorted(st))
    if not (lit["depth"] or lit["n_est"] or st) or (gan and not (bt and st)):
        return "unverified (no comparable literal found)", ""
    return "partial agreement (static text)", "; ".join(notes)


rows = []
static_cache = {}
for (ds, alg), cfg in sorted(best.items()):
    key = (ds, alg)
    folder = next((f for f, (d, _) in LITERAL.items() if d == ds), None)
    sfolder = next((f for f, d in {**FUNCS, **MARKERS}.items() if d == ds), None)
    row = {"dataset": ds, "algorithm": alg, "swapped_to_uniform_protocol": key in swapped}
    if cfg == "N/A":
        row.update(status="N/A in the paper", detail="")
    elif folder is not None:
        sc = script_cfg(folder, LITERAL[folder][1])
        s_ = sc.get(alg) or sc.get(alg.replace("SMOTified-GAN", "SMOTifiedGAN"))
        if s_ is None:
            row.update(status="unverified (no entry in the script dictionary)", detail="")
        else:
            cmp_ = compare(cfg, s_)
            diffs = [(f, b, v) for f, b, v in cmp_ if b != v]
            thr_only = diffs and all(f == "threshold" for f, _, _ in diffs)
            if thr_only:
                row.update(status="unverified (threshold representation)",
                           detail="BEST_CONFIGS records per-metric thresholds %s, the script dictionary a nominal %s; "
                                  "whether the script applies per-metric thresholds elsewhere was not settled" % (diffs[0][1], diffs[0][2]))
            elif diffs:
                row.update(status="differs", detail="; ".join("%s: BEST_CONFIGS %s / script %s" % x for x in diffs))
            else:
                row.update(status="match", detail="compared: " + ", ".join(f for f, _, _ in cmp_))
    elif sfolder is not None:
        if sfolder not in static_cache:
            static_cache[sfolder] = static_blocks(sfolder)
        block = static_cache[sfolder].get(alg)
        if block is None:
            row.update(status="unverified (no run function or block found)", detail="")
        else:
            st, detail = classify_static(cfg, alg, block)
            row.update(status=st, detail=detail)
    else:
        row.update(status="unverified (configuration in module constants and classes; not compared)", detail="")
    rows.append(row)

out = HERE / "best_configs_vs_final_scripts.csv"
with open(out, "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=["dataset", "algorithm", "swapped_to_uniform_protocol", "status", "detail"])
    w.writeheader()
    w.writerows(rows)

from collections import Counter
print("cells:", len(rows), " (143 evaluated + 7 N/A expected)")
for k, v in sorted(Counter(r["status"] for r in rows).items()):
    print("  %3d  %s" % (v, k))
print("swapped to the uniform protocol (42): status counts:",
      dict(Counter(r["status"] for r in rows if r["swapped_to_uniform_protocol"])))
print("This is a static text comparison (level 1); it is not CSV arithmetic and not a reproduced run.")
print("wrote", out)
