"""
Canonical ITQ-HUE re-runner.

Replaces the bagged Random Forest HUE approximation (used in 9 datasets:
GLASS, ILPD, SEED, WINE, YEAST5, YEAST5-ERL, CARGOOD, CARVGOOD, FLARE-F)
with the canonical Hashing-based Undersampling Ensemble per Ng et al. 2022
(IEEE Transactions on Cybernetics, "Hashing-based undersampling ensemble for
imbalanced pattern classification problems").

Same RepeatedStratifiedKFold(5, 20, random_state=42) = 100 folds.
Same standard evaluation protocol: threshold 0.55 on test-set probabilities,
soft-probability averaging across the 2^B subspace base classifiers,
binary AUC/AP from probability scores, G-Mean from confusion matrix at 0.55.

Output: one CSV per dataset under fixes/hue_canonical/outputs/.
"""

import os
import sys
import math
import time
import warnings
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, roc_auc_score, f1_score,
                             average_precision_score, confusion_matrix)
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import check_random_state
from sklearn.decomposition import PCA

warnings.filterwarnings('ignore')

# Defaults to the repository root (this file lives in results/hue_canonical/).
PROJECT_ROOT = Path(os.environ.get(
    "PROJECT_ROOT", str(Path(__file__).resolve().parents[2])))
FIXES_DIR = PROJECT_ROOT / "results" / "hue_canonical"
OUTPUTS_DIR = FIXES_DIR / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
N_SPLITS = 5
REPEATS = 20
HUE_THRESHOLD = 0.55
HUE_N_ESTIMATORS = 10
HUE_MAX_DEPTH = 5
HUE_ITQ_ITERS = 50


# ============================================================================
# Canonical ITQ-HUE per Ng 2022 (extracted verbatim from 06_CB)
# ============================================================================

def itq_fit(X, B, iters, seed):
    rng = check_random_state(seed)
    mu = X.mean(axis=0, keepdims=True)
    Xc = X - mu
    n_comp = min(B, Xc.shape[1], Xc.shape[0] - 1)
    if n_comp < 1:
        n_comp = 1
    pca = PCA(n_components=n_comp, random_state=seed)
    V = pca.fit_transform(Xc)
    if V.shape[1] < B:
        V = np.hstack([V, np.zeros((V.shape[0], B - V.shape[1]))])
    R = rng.randn(B, B)
    U, _, VT = np.linalg.svd(R, full_matrices=False)
    R = U @ VT
    for _ in range(iters):
        Z = V @ R
        Bits = np.where(Z >= 0, 1.0, -1.0)
        U, _, VT = np.linalg.svd(Bits.T @ V, full_matrices=False)
        R = U @ VT
    return {'mu': mu, 'pca': pca, 'R': R, 'B': B}


def itq_transform(X, itq):
    Xc = X - itq['mu']
    V = itq['pca'].transform(Xc)
    B = itq['B']
    if V.shape[1] < B:
        V = np.hstack([V, np.zeros((V.shape[0], B - V.shape[1]))])
    return (V @ itq['R'] >= 0).astype(np.uint8)


def bits_to_uint(bits):
    B = bits.shape[1]
    w = (1 << np.arange(B, dtype=np.uint32))
    return (bits.astype(np.uint32) * w).sum(axis=1)


class HUEClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, n_estimators=10, max_depth=5, random_state=42,
                 itq_iters=50, threshold=0.55):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.itq_iters = itq_iters
        self.threshold = threshold

    def fit(self, X, y):
        rng = check_random_state(self.random_state)
        X_min = X[y == 1]
        X_maj = X[y == 0]
        n_min = len(X_min)
        n_maj = len(X_maj)
        if n_min < 1 or n_maj < 1:
            raise ValueError("Both classes must have at least one sample")
        B = max(1, int(math.ceil(math.log2(3.0 * n_maj / n_min))))
        n_sub = 2 ** B
        itq = itq_fit(X_maj, B, self.itq_iters, self.random_state)
        maj_codes = bits_to_uint(itq_transform(X_maj, itq))
        self.estimators_ = []
        for ref in range(n_sub):
            xor = np.bitwise_xor(maj_codes, np.uint32(ref))
            d = np.array([int(v).bit_count() for v in xor], dtype=np.int32)
            w = np.where(d == 0, 1.0, 1.0 / (d * n_sub))
            w = w / w.sum()
            idx = rng.choice(np.arange(n_maj), size=n_min, replace=True, p=w)
            Xs = np.vstack([X_min, X_maj[idx]])
            ys = np.hstack([np.ones(n_min, dtype=int), np.zeros(n_min, dtype=int)])
            clf = ExtraTreesClassifier(n_estimators=self.n_estimators,
                                       max_depth=self.max_depth,
                                       random_state=rng.randint(0, 1_000_000))
            clf.fit(Xs, ys)
            self.estimators_.append(clf)
        return self

    def predict_proba(self, X):
        prob = np.zeros((len(X), 2))
        for est in self.estimators_:
            p = est.predict_proba(X)
            t = np.zeros_like(prob)
            for j, c in enumerate(est.classes_):
                t[:, int(c)] = p[:, j]
            prob += t
        return prob / len(self.estimators_)

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= self.threshold).astype(int)


# ============================================================================
# Metric helpers
# ============================================================================

def gmean_score(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    if cm.shape != (2, 2):
        return 0.0
    tn, fp, fn, tp = cm.ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return float(np.sqrt(spec * sens))


def safe_auc(y_true, scores):
    try:
        return float(roc_auc_score(y_true, scores))
    except Exception:
        return 0.0


def safe_ap(y_true, scores):
    try:
        return float(average_precision_score(y_true, scores))
    except Exception:
        return 0.0


def evaluate_hue_fold(Xtr, ytr, Xte, yte, seed, threshold=HUE_THRESHOLD,
                     scaler_mode='std'):
    """Standard evaluation: train canonical ITQ-HUE on (Xtr,ytr), evaluate on
    (Xte,yte) with threshold-0.55 binary predictions. AUC and AP from
    probabilities."""
    if scaler_mode == 'std':
        sc = StandardScaler().fit(Xtr)
        Xtr_p = sc.transform(Xtr)
        Xte_p = sc.transform(Xte)
    else:
        Xtr_p, Xte_p = Xtr, Xte

    clf = HUEClassifier(n_estimators=HUE_N_ESTIMATORS,
                        max_depth=HUE_MAX_DEPTH,
                        random_state=seed,
                        itq_iters=HUE_ITQ_ITERS,
                        threshold=threshold)
    clf.fit(Xtr_p, ytr)
    proba = clf.predict_proba(Xte_p)[:, 1]
    yp = (proba >= threshold).astype(int)
    return {
        'ACC': accuracy_score(yte, yp) * 100.0,
        'AUC': safe_auc(yte, proba),
        'F1':  f1_score(yte, yp, zero_division=0),
        'AP':  safe_ap(yte, proba),
        'GMEAN': gmean_score(yte, yp),
    }


# ============================================================================
# Per-dataset configuration: loader path + paper values
# ============================================================================

DATASETS = [
    ('GLASS',      '07_GLASS/glass_replication.py',         'load_glass',     'glass.csv',         {'ACC': 94.88, 'AUC': 0.95, 'F1': 0.82, 'AP': 0.71, 'GMEAN': 0.93}),
    ('ILPD',       '08_ILPD/ilpd_replication_v2.py',        'load_ilpd',      'ilpd.csv',          {'ACC': 66.55, 'AUC': 0.71, 'F1': 0.58, 'AP': 0.42, 'GMEAN': 0.70}),
    ('SEED',       '09_SEED/seed_replication.py',           'load_seed',      'seeds_dataset.txt', {'ACC': 98.57, 'AUC': 0.98, 'F1': 0.98, 'AP': 0.97, 'GMEAN': 0.99}),
    ('WINE',       '10_WINE/wine_replication.py',           'load_wine',      'wine.data',         {'ACC': 94.43, 'AUC': 0.98, 'F1': 0.97, 'AP': 0.95, 'GMEAN': 0.98}),
    ('YEAST5',     '11_YEAST5/yeast5_replication.py',       'load_yeast5',    'yeast5.dat',        {'ACC': 95.96, 'AUC': 0.97, 'F1': 0.59, 'AP': 0.43, 'GMEAN': 0.97}),
    ('YEAST5-ERL', '12_YEAST5-ERL/yeast5erl_replication.py','load_data',      'yeast5erl.dat',     {'ACC': 99.46, 'AUC': 0.99, 'F1': 0.65, 'AP': 0.52, 'GMEAN': 1.00}),
    ('CARGOOD',    '13_CARGOOD/cargood_replication.py',     'load_cargood',   'car-good.dat',      {'ACC': 95.49, 'AUC': 0.97, 'F1': 0.64, 'AP': 0.47, 'GMEAN': 0.98}),
    ('CARVGOOD',   '14_CARVGOOD/carvgood_replication.py',   'load_data',      'car-vgood.dat',     {'ACC': 98.44, 'AUC': 0.99, 'F1': 0.83, 'AP': 0.72, 'GMEAN': 0.99}),
    ('FLARE-F',    '15_FLARE-F/flaref_replication.py',      'load_data',      'flare-F.dat',       {'ACC': 81.43, 'AUC': 0.86, 'F1': 0.28, 'AP': 0.15, 'GMEAN': 0.86}),
]


def import_loader(rel_script_path, loader_name):
    """Import a function from a sibling per-dataset script via importlib."""
    full_path = PROJECT_ROOT / rel_script_path
    spec = importlib.util.spec_from_file_location(
        f"loader_{rel_script_path.replace('/', '_').replace('.py', '')}",
        full_path)
    module = importlib.util.module_from_spec(spec)
    # Ensure loader's relative DATA_PATH resolves correctly
    saved_cwd = os.getcwd()
    os.chdir(full_path.parent)
    try:
        spec.loader.exec_module(module)
    finally:
        os.chdir(saved_cwd)
    return getattr(module, loader_name), module


def run_dataset(name, rel_script_path, loader_name, data_filename, paper):
    print(f"\n{'='*72}")
    print(f"Canonical ITQ-HUE on {name}")
    print(f"{'='*72}")
    loader_fn, _ = import_loader(rel_script_path, loader_name)
    full_data_path = PROJECT_ROOT / Path(rel_script_path).parent / data_filename
    X, y = loader_fn(str(full_data_path))
    print(f"Loaded: n={len(X)}, features={X.shape[1]}, "
          f"minority={int((y==1).sum())}, majority={int((y==0).sum())}")

    cv = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=REPEATS,
                                 random_state=SEED)
    folds = list(cv.split(X, y))
    fold_metrics = []
    t0 = time.time()
    for fi, (tr, te) in enumerate(folds):
        seed = SEED + fi
        try:
            m = evaluate_hue_fold(X[tr], y[tr], X[te], y[te], seed)
            fold_metrics.append(m)
        except Exception as e:
            print(f"  Fold {fi+1} failed: {e}")
        if (fi + 1) % 10 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (fi + 1) * (len(folds) - (fi + 1))
            print(f"  Fold {fi+1}/{len(folds)} done | elapsed={elapsed:.1f}s "
                  f"| ETA={eta:.1f}s")

    if not fold_metrics:
        print(f"{name}: ALL FOLDS FAILED")
        return None

    avg = {k: float(np.mean([m[k] for m in fold_metrics])) for k in fold_metrics[0]}
    err_acc = abs(avg['ACC'] - paper['ACC'])
    err_auc = abs(avg['AUC'] - paper['AUC']) * 100
    err_f1 = abs(avg['F1'] - paper['F1']) * 100
    err_ap = abs(avg['AP'] - paper['AP']) * 100
    err_gm = abs(avg['GMEAN'] - paper['GMEAN']) * 100
    avg_err = (err_acc + err_auc + err_f1 + err_ap + err_gm) / 5
    status = 'PASS' if avg_err < 3.0 else 'FAIL'
    print(f"\n{name} HUE (canonical ITQ): "
          f"ACC={avg['ACC']:.2f} AUC={avg['AUC']:.4f} F1={avg['F1']:.4f} "
          f"AP={avg['AP']:.4f} GMean={avg['GMEAN']:.4f}")
    print(f"vs paper:                   "
          f"ACC={paper['ACC']:.2f} AUC={paper['AUC']:.2f} F1={paper['F1']:.2f} "
          f"AP={paper['AP']:.2f} GMean={paper['GMEAN']:.2f}")
    print(f"Avg error: {avg_err:.4f}% | Status: {status} "
          f"| folds: {len(fold_metrics)}/{len(folds)}")

    out = OUTPUTS_DIR / f"hue_canonical_{name.replace('-','')}.csv"
    pd.DataFrame([{
        'Dataset': name,
        'Method': 'HUE',
        'Paper_ACC(%)': paper['ACC'],
        'Paper_AUC':    paper['AUC'],
        'Paper_F1':     paper['F1'],
        'Paper_AP':     paper['AP'],
        'Paper_GMean':  paper['GMEAN'],
        'Replication_ACC(%)': avg['ACC'],
        'Replication_AUC':    avg['AUC'],
        'Replication_F1':     avg['F1'],
        'Replication_AP':     avg['AP'],
        'Replication_GMean':  avg['GMEAN'],
        'ACC_Error(pp)': err_acc,
        'AUC_Error(pp)': err_auc,
        'F1_Error(pp)':  err_f1,
        'AP_Error(pp)':  err_ap,
        'GMean_Error(pp)': err_gm,
        'Average_Error(%)': avg_err,
        'Status': status,
        'N_folds_completed': len(fold_metrics),
        'Source': 'fixes/hue_canonical (ITQ-HUE per Ng 2022)',
    }]).to_csv(out, index=False)
    print(f"Saved: {out}")
    return out


def main():
    print(f"Canonical ITQ-HUE re-runner")
    print(f"Output dir: {OUTPUTS_DIR}")
    print(f"CV: RepeatedStratifiedKFold(5, 20, random_state={SEED}) = 100 folds")
    print(f"HUE config: n_estimators={HUE_N_ESTIMATORS}, max_depth={HUE_MAX_DEPTH}, "
          f"itq_iters={HUE_ITQ_ITERS}, threshold={HUE_THRESHOLD}")
    print(f"Datasets: {len(DATASETS)}\n")

    t_start = time.time()
    for name, script, loader, datafile, paper in DATASETS:
        try:
            run_dataset(name, script, loader, datafile, paper)
        except Exception as e:
            import traceback
            print(f"\n{name}: EXCEPTION")
            traceback.print_exc()
    print(f"\nTotal elapsed: {(time.time()-t_start)/60:.1f} min")


if __name__ == '__main__':
    main()
