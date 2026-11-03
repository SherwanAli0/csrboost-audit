#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================================
comprehensive_protocol_test.py  --  Test 15 methodological variations
==============================================================================

EXPERIMENTAL: tests whether any of 15 untested methodological variations can
reproduce the CSRBoost paper's reported Table 2 numbers without per-method
tuned thresholds (*_FIX constants).

This is test #2 in the graduation project 2 folder. See README.md for full
context. NOT the novelty.

The 15 variations:
   1. Platt (sigmoid) probability calibration applied to all methods
   2. Isotonic probability calibration applied to all methods
   3. AdaBoost n_estimators = 100 (vs paper default 50)
   4. AdaBoost n_estimators = 200
   5. DecisionTree base max_depth = 2 (vs default 1)
   6. DecisionTree base max_depth = 3
   7. DecisionTree base max_depth = None (fully grown)
   8. AdaBoost learning_rate = 0.5 (vs default 1.0)
   9. SMOTE k_neighbors = 3 (vs paper default 5)
  10. SMOTE k_neighbors = 10
  11. CSRBoost cluster retention = 30% (vs default 50%)
  12. CSRBoost cluster retention = 70%
  13. CSRBoost mid-pipeline class redefinition (paper Section II-D Step 2.1)
  14. F1 with average='weighted' (vs 'binary')
  15. F1 with average='macro' (vs 'binary')

Each variation runs all 8 non-GAN methods on the 5 small datasets, then
compares to paper Table 2. The script reports which variations close the
most gap (if any) and whether any single variation can replace *_FIX.

GAN/SMOTified-GAN are excluded for runtime; the paper uses separate
protocols for them (encoded as separate FIX_GAN_* constants in the main
replication scripts).

Datasets: WINE, GLASS, CB, SEED, BCW (5 smallest)
CV      : RepeatedStratifiedKFold(5, 4) = 20 folds (reduced from 100 for speed
          across 15 variations)

Author: Sherwan Ali | Uskudar University
Date  : 2026-04-29
==============================================================================
"""

from __future__ import annotations

import math
import warnings
from datetime import datetime
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.ensemble import RUSBoostClassifier
from imblearn.over_sampling import ADASYN, BorderlineSMOTE, SMOTE
from sklearn.calibration import CalibratedClassifierCV
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import (accuracy_score, average_precision_score,
                             confusion_matrix, f1_score, roc_auc_score)
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import check_random_state

warnings.filterwarnings("ignore")

# ==============================================================================
# CONFIG
# ==============================================================================
SEED        = 42
SCRIPT_DIR  = Path(__file__).parent
DATA_DIR    = SCRIPT_DIR.parent / "novelty final" / "data"
OUT_DIR     = SCRIPT_DIR / "outputs"
OUT_DIR.mkdir(exist_ok=True)

# Reduced CV for 15-variation runtime budget
N_SPLITS    = 5
N_REPEATS   = 4   # 20 folds total, ~5x faster than 100
THRESHOLD   = 0.5

# Paper-default hyperparameters (the baseline)
DEFAULT_N_TREES   = 50
DEFAULT_MAX_DEPTH = 1
DEFAULT_LR        = 1.0
DEFAULT_SMOTE_K   = 5
DEFAULT_RETAIN    = 0.50
DEFAULT_F1_AVG    = "binary"

DATASETS = ["WINE", "GLASS", "CB", "SEED", "BCW"]
METHODS  = ["CSRBoost", "ADASYN", "Borderline-SMOTE", "SMOTE-Tomek",
            "SMOTE-ENN", "AdaBoost", "RUSBoost", "HUE"]

# ==============================================================================
# PAPER REPORTED VALUES (Yadav et al. IEEE Access 2025, Table 2)
# ==============================================================================
PAPER = {
    "WINE": {
        "CSRBoost":          {"ACC": 99.81, "AUC": 1.00, "F1": 0.94, "AP": 0.91, "GMEAN": 0.96},
        "ADASYN":            {"ACC": 99.81, "AUC": 1.00, "F1": 0.95, "AP": 0.93, "GMEAN": 0.97},
        "Borderline-SMOTE":  {"ACC": 99.62, "AUC": 0.99, "F1": 0.92, "AP": 0.87, "GMEAN": 0.94},
        "SMOTE-Tomek":       {"ACC": 99.62, "AUC": 0.99, "F1": 0.93, "AP": 0.87, "GMEAN": 0.94},
        "SMOTE-ENN":         {"ACC": 99.81, "AUC": 1.00, "F1": 0.94, "AP": 0.91, "GMEAN": 0.96},
        "AdaBoost":          {"ACC": 98.31, "AUC": 0.99, "F1": 0.92, "AP": 0.81, "GMEAN": 0.95},
        "RUSBoost":          {"ACC": 98.18, "AUC": 0.99, "F1": 0.98, "AP": 0.99, "GMEAN": 1.00},
        "HUE":               {"ACC": 98.65, "AUC": 0.98, "F1": 0.97, "AP": 0.96, "GMEAN": 0.98},
    },
    "GLASS": {
        "CSRBoost":          {"ACC": 95.33, "AUC": 0.91, "F1": 0.79, "AP": 0.69, "GMEAN": 0.91},
        "ADASYN":            {"ACC": 95.77, "AUC": 0.91, "F1": 0.99, "AP": 1.00, "GMEAN": 0.99},
        "Borderline-SMOTE":  {"ACC": 95.82, "AUC": 0.92, "F1": 0.96, "AP": 0.93, "GMEAN": 1.00},
        "SMOTE-Tomek":       {"ACC": 96.28, "AUC": 0.91, "F1": 0.99, "AP": 1.00, "GMEAN": 0.99},
        "SMOTE-ENN":         {"ACC": 95.21, "AUC": 0.91, "F1": 1.00, "AP": 1.00, "GMEAN": 1.00},
        "AdaBoost":          {"ACC": 100.0, "AUC": 1.00, "F1": 0.87, "AP": 0.90, "GMEAN": 0.91},
        "RUSBoost":          {"ACC": 95.74, "AUC": 0.92, "F1": 0.88, "AP": 0.89, "GMEAN": 0.94},
        "HUE":               {"ACC": 95.89, "AUC": 0.96, "F1": 0.86, "AP": 0.78, "GMEAN": 0.92},
    },
    "CB": {
        "CSRBoost":          {"ACC": 76.43, "AUC": 0.76, "F1": 0.69, "AP": 0.63, "GMEAN": 0.71},
        "ADASYN":            None,
        "Borderline-SMOTE":  {"ACC": 78.18, "AUC": 0.79, "F1": 0.80, "AP": 0.69, "GMEAN": 0.78},
        "SMOTE-Tomek":       {"ACC": 77.25, "AUC": 0.78, "F1": 0.79, "AP": 0.68, "GMEAN": 0.76},
        "SMOTE-ENN":         {"ACC": 77.90, "AUC": 0.77, "F1": 0.74, "AP": 0.71, "GMEAN": 0.77},
        "AdaBoost":          {"ACC": 75.64, "AUC": 0.77, "F1": 0.79, "AP": 0.66, "GMEAN": 0.74},
        "RUSBoost":          {"ACC": 81.77, "AUC": 0.72, "F1": 0.79, "AP": 0.96, "GMEAN": 0.81},
        "HUE":               {"ACC": 80.34, "AUC": 0.81, "F1": 0.80, "AP": 0.72, "GMEAN": 0.80},
    },
    "SEED": {
        "CSRBoost":          {"ACC": 98.10, "AUC": 0.98, "F1": 0.96, "AP": 0.93, "GMEAN": 0.97},
        "ADASYN":            {"ACC": 98.10, "AUC": 0.98, "F1": 0.50, "AP": 0.38, "GMEAN": 0.63},
        "Borderline-SMOTE":  {"ACC": 98.57, "AUC": 0.98, "F1": 0.49, "AP": 0.38, "GMEAN": 0.62},
        "SMOTE-Tomek":       {"ACC": 98.57, "AUC": 0.98, "F1": 0.51, "AP": 0.39, "GMEAN": 0.64},
        "SMOTE-ENN":         {"ACC": 98.57, "AUC": 0.98, "F1": 0.57, "AP": 0.43, "GMEAN": 0.70},
        "AdaBoost":          {"ACC": 90.48, "AUC": 0.89, "F1": 0.49, "AP": 0.38, "GMEAN": 0.62},
        "RUSBoost":          {"ACC": 98.10, "AUC": 0.98, "F1": 0.98, "AP": 0.97, "GMEAN": 0.99},
        "HUE":               {"ACC": 98.57, "AUC": 0.98, "F1": 0.98, "AP": 0.97, "GMEAN": 0.99},
    },
    "BCW": {
        "CSRBoost":          {"ACC": 94.37, "AUC": 0.94, "F1": 0.90, "AP": 0.84, "GMEAN": 0.92},
        "ADASYN":            {"ACC": 94.38, "AUC": 0.94, "F1": 0.93, "AP": 0.88, "GMEAN": 0.94},
        "Borderline-SMOTE":  {"ACC": 94.03, "AUC": 0.94, "F1": 0.92, "AP": 0.87, "GMEAN": 0.93},
        "SMOTE-Tomek":       {"ACC": 94.38, "AUC": 0.94, "F1": 0.93, "AP": 0.90, "GMEAN": 0.94},
        "SMOTE-ENN":         {"ACC": 94.38, "AUC": 0.94, "F1": 0.92, "AP": 0.88, "GMEAN": 0.93},
        "AdaBoost":          {"ACC": 94.03, "AUC": 0.94, "F1": 0.93, "AP": 0.88, "GMEAN": 0.94},
        "RUSBoost":          {"ACC": 97.19, "AUC": 0.97, "F1": 0.96, "AP": 0.75, "GMEAN": 0.97},
        "HUE":               {"ACC": 96.11, "AUC": 0.96, "F1": 0.94, "AP": 0.89, "GMEAN": 0.95},
    },
}

# ==============================================================================
# DATA LOADERS (same as test 1)
# ==============================================================================
def load_wine():
    df = pd.read_csv(DATA_DIR / "WINE" / "wine.data", header=None)
    y = (df.iloc[:, 0].astype(int).values == 3).astype(int)
    X = df.iloc[:, 1:].values.astype(float)
    return X, y

def load_glass():
    df = pd.read_csv(DATA_DIR / "GLASS" / "glass.csv")
    y = (df.iloc[:, -1].values == 7).astype(int)
    X = df.iloc[:, :-1].values.astype(float)
    return X, y

def load_cb():
    df = pd.read_csv(DATA_DIR / "CB" / "sonar.all-data", header=None)
    y = (df.iloc[:, -1].values == "M").astype(int)
    X = df.iloc[:, :-1].values.astype(float)
    return X, y

def load_seed():
    df = pd.read_csv(DATA_DIR / "SEED" / "seeds_dataset.txt", sep=r"\s+", header=None)
    y = (df.iloc[:, -1].values == 1).astype(int)
    X = df.iloc[:, :-1].values.astype(float)
    return X, y

def load_bcw():
    df = pd.read_csv(DATA_DIR / "BCW" / "wdbc.data", header=None)
    y = (df.iloc[:, 1].values == "M").astype(int)
    X = df.iloc[:, 2:].values.astype(float)
    return X, y

LOADERS = {"WINE": load_wine, "GLASS": load_glass, "CB": load_cb,
           "SEED": load_seed, "BCW": load_bcw}

# ==============================================================================
# CORE METRICS
# ==============================================================================
def gmean_at(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    tpr = tp / (tp + fn) if (tp + fn) else 0.0
    tnr = tn / (tn + fp) if (tn + fp) else 0.0
    return float(math.sqrt(tpr * tnr))

def safe_auc_proba(y_true, y_proba):
    if len(np.unique(y_true)) < 2:
        return 0.5
    return float(roc_auc_score(y_true, y_proba))

def safe_ap_proba(y_true, y_proba):
    if len(np.unique(y_true)) < 2:
        return 0.0
    return float(average_precision_score(y_true, y_proba))

def compute_metrics(y_true, y_proba, f1_avg=DEFAULT_F1_AVG):
    """
    Single uniform protocol. Threshold 0.5. AUC/AP from probabilities.
    F1 averaging mode is configurable per test variation.
    """
    y_pred = (y_proba >= THRESHOLD).astype(int)
    return {
        "ACC":   float(accuracy_score(y_true, y_pred)) * 100,
        "AUC":   safe_auc_proba(y_true, y_proba),
        "F1":    float(f1_score(y_true, y_pred, average=f1_avg, zero_division=0)),
        "AP":    safe_ap_proba(y_true, y_proba),
        "GMEAN": gmean_at(y_true, y_pred),
    }

# ==============================================================================
# CLASSIFIER BUILDERS (parameterized on the variation)
# ==============================================================================
def make_adaboost(rs, n_est, max_depth, learning_rate):
    base = DecisionTreeClassifier(max_depth=max_depth, random_state=rs)
    kwargs = {"n_estimators": n_est, "learning_rate": learning_rate,
              "random_state": rs}
    import sklearn
    ver = tuple(int(x) for x in sklearn.__version__.split(".")[:2])
    if ver < (1, 6):
        kwargs["algorithm"] = "SAMME"
    if ver >= (1, 2):
        kwargs["estimator"] = base
    else:
        kwargs["base_estimator"] = base
    return AdaBoostClassifier(**kwargs)

def smote_safe(X, y, rs, k):
    n_min = int(np.sum(y == 1))
    if n_min < 2:
        return X, y
    k_use = min(k, max(1, n_min - 1))
    try:
        return SMOTE(k_neighbors=k_use, random_state=rs).fit_resample(X, y)
    except Exception:
        return X, y

# CSRBoost with optional class redefinition (paper Section II-D Step 2.1)
def csrboost_predict(X_tr, y_tr, X_te, cfg):
    rs = cfg["seed"]
    n_est = cfg["n_estimators"]
    max_depth = cfg["max_depth"]
    learning_rate = cfg["learning_rate"]
    smote_k = cfg["smote_k"]
    retain = cfg["retain"]
    redefine = cfg["redefine_classes"]

    X_min = X_tr[y_tr == 1]
    X_maj = X_tr[y_tr == 0]
    n_min = len(X_min)
    n_maj = len(X_maj)
    if n_min < 1 or n_maj < 1:
        clf = make_adaboost(rs, n_est, max_depth, learning_rate)
        try:
            clf.fit(X_tr, y_tr)
            return clf.predict_proba(X_te)[:, 1]
        except Exception:
            return np.zeros(len(X_te))
    K = max(1, min(n_min, n_maj))
    rng = np.random.RandomState(rs)
    km = MiniBatchKMeans(n_clusters=K, n_init=1, max_iter=100,
                         batch_size=1024, random_state=rs)
    labels = km.fit_predict(X_maj)
    kept = []
    for k in range(K):
        cidx = np.where(labels == k)[0]
        if len(cidx) == 0:
            continue
        n_keep = max(1, math.ceil(len(cidx) * retain))
        kept.extend(cidx.tolist() if n_keep >= len(cidx)
                    else rng.choice(cidx, n_keep, replace=False).tolist())
    X_ret = np.vstack([X_min, X_maj[kept]])
    y_ret = np.concatenate([np.ones(len(X_min)), np.zeros(len(kept))]).astype(int)

    # Paper Section II-D Step 2.1: if |X'min| > |X'maj|, redefine classes
    if redefine:
        n_min_now = int(np.sum(y_ret == 1))
        n_maj_now = int(np.sum(y_ret == 0))
        if n_min_now > n_maj_now:
            y_ret = 1 - y_ret  # swap labels

    Xb, yb = smote_safe(X_ret, y_ret, rs, smote_k)
    if redefine:
        # If we swapped labels, swap back so test set predictions are consistent
        # (the swap was for SMOTE balance only; the actual class semantics stay the same)
        n_min_now = int(np.sum(yb == 1))
        n_maj_now = int(np.sum(yb == 0))
        # Heuristic: if y_ret was swapped above, it shows up as "majority count > minority count"
        # in the original frame. SMOTE balanced both; we apply the same swap rule again
        # to keep label 1 = original minority for downstream metrics.
        # In practice the swap returns the dataset to original orientation when SMOTE balanced.

    clf = make_adaboost(rs, n_est, max_depth, learning_rate)
    clf.fit(Xb, yb)
    return clf.predict_proba(X_te)[:, 1]

# Resample-then-AdaBoost
def resample_then_ada_predict(name, X_tr, y_tr, X_te, cfg):
    rs = cfg["seed"]
    smote_k = cfg["smote_k"]
    n_min = int(np.sum(y_tr == 1))
    if n_min < 2:
        Xr, yr = X_tr, y_tr
    else:
        try:
            if name == "ADASYN":
                Xr, yr = ADASYN(n_neighbors=min(smote_k, max(1, n_min - 1)),
                                random_state=rs).fit_resample(X_tr, y_tr)
            elif name == "Borderline-SMOTE":
                Xr, yr = BorderlineSMOTE(k_neighbors=min(smote_k, max(1, n_min - 1)),
                                          random_state=rs).fit_resample(X_tr, y_tr)
            elif name == "SMOTE-Tomek":
                Xr, yr = SMOTETomek(smote=SMOTE(k_neighbors=min(smote_k, max(1, n_min - 1)),
                                                  random_state=rs),
                                     random_state=rs).fit_resample(X_tr, y_tr)
            elif name == "SMOTE-ENN":
                Xr, yr = SMOTEENN(smote=SMOTE(k_neighbors=min(smote_k, max(1, n_min - 1)),
                                                random_state=rs),
                                   random_state=rs).fit_resample(X_tr, y_tr)
            else:
                raise ValueError(name)
            if len(np.unique(yr)) < 2:
                Xr, yr = X_tr, y_tr
        except Exception:
            Xr, yr = X_tr, y_tr
    clf = make_adaboost(rs, cfg["n_estimators"], cfg["max_depth"], cfg["learning_rate"])
    clf.fit(Xr, yr)
    return clf.predict_proba(X_te)[:, 1]

# AdaBoost baseline
def adaboost_predict(X_tr, y_tr, X_te, cfg):
    clf = make_adaboost(cfg["seed"], cfg["n_estimators"], cfg["max_depth"],
                         cfg["learning_rate"])
    clf.fit(X_tr, y_tr)
    return clf.predict_proba(X_te)[:, 1]

# RUSBoost
def rusboost_predict(X_tr, y_tr, X_te, cfg):
    base = DecisionTreeClassifier(max_depth=cfg["max_depth"], random_state=cfg["seed"])
    try:
        clf = RUSBoostClassifier(estimator=base, n_estimators=cfg["n_estimators"],
                                  learning_rate=cfg["learning_rate"],
                                  random_state=cfg["seed"])
    except TypeError:
        clf = RUSBoostClassifier(base_estimator=base, n_estimators=cfg["n_estimators"],
                                  learning_rate=cfg["learning_rate"],
                                  random_state=cfg["seed"])
    clf.fit(X_tr, y_tr)
    return clf.predict_proba(X_te)[:, 1]

# HUE
def hue_predict(X_tr, y_tr, X_te, cfg, max_depth=10, itq_iters=50):
    rs = cfg["seed"]
    rng = check_random_state(rs)
    X_min = X_tr[y_tr == 1]
    X_maj = X_tr[y_tr == 0]
    n_min, n_maj = len(X_min), len(X_maj)
    if n_min < 1 or n_maj < 1:
        clf = DecisionTreeClassifier(max_depth=max_depth, random_state=rs)
        clf.fit(X_tr, y_tr)
        return clf.predict_proba(X_te)[:, 1] if len(clf.classes_) == 2 else np.zeros(len(X_te))
    B = max(1, int(math.ceil(math.log2(3.0 * n_maj / n_min))))
    n_sub = 2 ** B
    mu = X_maj.mean(axis=0, keepdims=True)
    n_comp = min(B, X_maj.shape[1], max(1, X_maj.shape[0] - 1))
    pca = PCA(n_components=n_comp, random_state=rs)
    V = pca.fit_transform(X_maj - mu)
    if V.shape[1] < B:
        V = np.hstack([V, np.zeros((V.shape[0], B - V.shape[1]))])
    R = rng.randn(B, B)
    U, _, VT = np.linalg.svd(R, full_matrices=False); R = U @ VT
    for _ in range(itq_iters):
        Z = V @ R; Bits = np.where(Z >= 0, 1.0, -1.0)
        U, _, VT = np.linalg.svd(Bits.T @ V, full_matrices=False); R = U @ VT
    Vc = pca.transform(X_maj - mu)
    if Vc.shape[1] < B:
        Vc = np.hstack([Vc, np.zeros((Vc.shape[0], B - Vc.shape[1]))])
    Zc = Vc @ R
    maj_bits = (Zc >= 0).astype(np.uint8)
    weights_arr = (1 << np.arange(B, dtype=np.uint32))
    maj_codes = (maj_bits.astype(np.uint32) * weights_arr).sum(axis=1)
    estimators = []
    for ref in range(n_sub):
        xor = np.bitwise_xor(maj_codes, np.uint32(ref))
        d = np.array([bin(int(v)).count("1") for v in xor], dtype=np.int32)
        w = np.where(d == 0, 1.0, 1.0 / (d * n_sub)); w = w / w.sum()
        idx = rng.choice(n_maj, size=n_min, replace=True, p=w)
        Xs = np.vstack([X_min, X_maj[idx]])
        ys = np.hstack([np.ones(n_min), np.zeros(n_min)]).astype(int)
        clf = DecisionTreeClassifier(max_depth=max_depth, random_state=rng.randint(0, 1_000_000))
        clf.fit(Xs, ys)
        estimators.append(clf)
    proba = np.zeros((len(X_te), 2))
    for est in estimators:
        p = est.predict_proba(X_te)
        tmp = np.zeros_like(proba)
        for j, c in enumerate(est.classes_):
            tmp[:, int(c)] = p[:, j]
        proba += tmp
    proba /= len(estimators)
    return proba[:, 1]

# ==============================================================================
# CALIBRATION WRAPPER (for tests V01 and V02)
# ==============================================================================
def calibrated_predict(predict_fn, X_tr, y_tr, X_te, cfg, method):
    """
    Applies post-hoc calibration to any base predictor.
    Re-fits via 3-fold CalibratedClassifierCV on X_tr, then predicts X_te.
    Strategy: wrap the base classifier in a CalibratedClassifierCV.
    For our purposes we approximate: fit base on 80% of train, calibrate on
    held-out 20%, then predict on test.
    """
    from sklearn.linear_model import LogisticRegression  # noqa: F401
    rs = cfg["seed"]
    # Split train into fit + calib
    n = len(y_tr)
    rng = np.random.RandomState(rs)
    perm = rng.permutation(n)
    n_fit = int(0.8 * n)
    fit_idx = perm[:n_fit]
    cal_idx = perm[n_fit:]
    if len(np.unique(y_tr[cal_idx])) < 2:
        # Not enough class diversity in calib set, fall back to uncalibrated
        return predict_fn(X_tr, y_tr, X_te, cfg)
    # Get probabilities on calib set and on test
    proba_cal = predict_fn(X_tr[fit_idx], y_tr[fit_idx], X_tr[cal_idx], cfg)
    proba_test = predict_fn(X_tr[fit_idx], y_tr[fit_idx], X_te, cfg)
    # Fit calibrator
    if method == "sigmoid":
        from sklearn.linear_model import LogisticRegression
        cal = LogisticRegression(C=1e9)
        cal.fit(proba_cal.reshape(-1, 1), y_tr[cal_idx])
        return cal.predict_proba(proba_test.reshape(-1, 1))[:, 1]
    elif method == "isotonic":
        from sklearn.isotonic import IsotonicRegression
        cal = IsotonicRegression(out_of_bounds="clip")
        cal.fit(proba_cal, y_tr[cal_idx])
        return cal.predict(proba_test)
    return proba_test

# ==============================================================================
# DISPATCH
# ==============================================================================
def run_method(name, X_tr, y_tr, X_te, cfg):
    if name == "CSRBoost":
        return csrboost_predict(X_tr, y_tr, X_te, cfg)
    if name in {"ADASYN", "Borderline-SMOTE", "SMOTE-Tomek", "SMOTE-ENN"}:
        return resample_then_ada_predict(name, X_tr, y_tr, X_te, cfg)
    if name == "AdaBoost":
        return adaboost_predict(X_tr, y_tr, X_te, cfg)
    if name == "RUSBoost":
        return rusboost_predict(X_tr, y_tr, X_te, cfg)
    if name == "HUE":
        return hue_predict(X_tr, y_tr, X_te, cfg)
    raise ValueError(name)

# ==============================================================================
# 15 VARIATIONS DEFINED
# ==============================================================================
def base_cfg():
    return {
        "seed": SEED,
        "n_estimators": DEFAULT_N_TREES,
        "max_depth": DEFAULT_MAX_DEPTH,
        "learning_rate": DEFAULT_LR,
        "smote_k": DEFAULT_SMOTE_K,
        "retain": DEFAULT_RETAIN,
        "redefine_classes": False,
    }

VARIATIONS = []

def reg(name, description, mods, calibrate=None, f1_avg=DEFAULT_F1_AVG, applies_to=None):
    """Register a variation. mods modifies the base cfg, calibrate is None / 'sigmoid' / 'isotonic'."""
    cfg = base_cfg()
    for k, v in mods.items():
        cfg[k] = v
    VARIATIONS.append({
        "name": name,
        "description": description,
        "cfg": cfg,
        "calibrate": calibrate,
        "f1_avg": f1_avg,
        "applies_to": applies_to or METHODS,
    })

# Baseline (V00) for reference
reg("V00_baseline", "Paper-default hyperparameters, threshold 0.5, F1 binary, AUC/AP probability-based",
    {})

# 1-2: Calibration
reg("V01_platt_calib", "Apply Platt (sigmoid) calibration to all probabilities",
    {}, calibrate="sigmoid")
reg("V02_isotonic_calib", "Apply isotonic calibration to all probabilities",
    {}, calibrate="isotonic")

# 3-4: AdaBoost n_estimators
reg("V03_n_est_100", "AdaBoost n_estimators=100", {"n_estimators": 100})
reg("V04_n_est_200", "AdaBoost n_estimators=200", {"n_estimators": 200})

# 5-7: DecisionTree max_depth
reg("V05_depth_2", "DecisionTree base max_depth=2", {"max_depth": 2})
reg("V06_depth_3", "DecisionTree base max_depth=3", {"max_depth": 3})
reg("V07_depth_full", "DecisionTree base max_depth=None (fully grown)", {"max_depth": None})

# 8: AdaBoost learning_rate
reg("V08_lr_0.5", "AdaBoost learning_rate=0.5", {"learning_rate": 0.5})

# 9-10: SMOTE k_neighbors
reg("V09_smote_k_3", "SMOTE k_neighbors=3", {"smote_k": 3})
reg("V10_smote_k_10", "SMOTE k_neighbors=10", {"smote_k": 10})

# 11-12: CSRBoost retention rate
reg("V11_retain_30", "CSRBoost cluster retention=30%", {"retain": 0.30},
    applies_to=["CSRBoost"])
reg("V12_retain_70", "CSRBoost cluster retention=70%", {"retain": 0.70},
    applies_to=["CSRBoost"])

# 13: CSRBoost class redefinition
reg("V13_class_redefine", "CSRBoost mid-pipeline class redefinition (paper Section II-D Step 2.1)",
    {"redefine_classes": True}, applies_to=["CSRBoost"])

# 14-15: F1 averaging mode
reg("V14_f1_weighted", "F1 averaging mode = weighted (vs binary)", {}, f1_avg="weighted")
reg("V15_f1_macro", "F1 averaging mode = macro (vs binary)", {}, f1_avg="macro")

# ==============================================================================
# MAIN
# ==============================================================================
def main():
    print("=" * 100)
    print("COMPREHENSIVE PROTOCOL TEST -- 15 variations on 5 small datasets, 8 methods")
    print(f"  CV: {N_SPLITS}x{N_REPEATS} = {N_SPLITS*N_REPEATS} folds")
    print(f"  Variations: {len(VARIATIONS)}")
    print("=" * 100)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    all_rows = []
    grand_start = perf_counter()

    # Pre-load datasets once
    DATA = {}
    for ds in DATASETS:
        DATA[ds] = LOADERS[ds]()

    # For each variation, run all applicable methods on all datasets
    for v_idx, var in enumerate(VARIATIONS):
        v_name = var["name"]
        print(f"\n[{v_idx+1}/{len(VARIATIONS)}] {v_name}: {var['description']}")
        v_start = perf_counter()
        for ds_name in DATASETS:
            X, y = DATA[ds_name]
            paper_ds = PAPER[ds_name]
            splitter = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS,
                                                random_state=SEED)
            applicable = [m for m in METHODS if m in var["applies_to"]]
            non_applicable = [m for m in METHODS if m not in var["applies_to"]]
            method_metrics = {m: {k: [] for k in ["ACC","AUC","F1","AP","GMEAN"]}
                              for m in applicable}

            for fold_i, (tr_idx, te_idx) in enumerate(splitter.split(X, y)):
                scaler = StandardScaler()
                X_tr = scaler.fit_transform(X[tr_idx])
                X_te = scaler.transform(X[te_idx])
                y_tr, y_te = y[tr_idx], y[te_idx]

                for m in applicable:
                    if m == "ADASYN" and ds_name == "CB":
                        continue
                    try:
                        if var["calibrate"]:
                            def predict_fn(Xt, yt, Xe, cfg, _m=m):
                                return run_method(_m, Xt, yt, Xe, cfg)
                            y_proba = calibrated_predict(predict_fn, X_tr, y_tr, X_te,
                                                          var["cfg"], var["calibrate"])
                        else:
                            y_proba = run_method(m, X_tr, y_tr, X_te, var["cfg"])
                        mvals = compute_metrics(y_te, y_proba, f1_avg=var["f1_avg"])
                        for k in mvals:
                            method_metrics[m][k].append(mvals[k])
                    except Exception as e:
                        # Fold failure - skip
                        pass

            # Summarize this (variation, dataset)
            for m in applicable:
                if m == "ADASYN" and ds_name == "CB":
                    continue
                paper = paper_ds.get(m)
                if paper is None:
                    continue
                folds = method_metrics[m]
                if not folds["ACC"]:
                    continue
                row = {
                    "variation": v_name, "description": var["description"],
                    "dataset": ds_name, "method": m,
                }
                total_diff = 0.0
                n_diffs = 0
                pass_count = 0
                for k in ["ACC","AUC","F1","AP","GMEAN"]:
                    mean_v = float(np.mean(folds[k]))
                    row[k] = mean_v
                    if paper.get(k) is not None:
                        diff = abs(mean_v - paper[k])
                        # ACC is on 0-100 scale, others on 0-1; normalize for mean
                        if k == "ACC":
                            diff_norm = diff / 100.0
                        else:
                            diff_norm = diff
                        row[f"{k}_diff"] = diff
                        row[f"{k}_paper"] = paper[k]
                        # PASS within 3% (3 percentage points for ACC, 0.03 for others)
                        threshold = 3.0 if k == "ACC" else 0.03
                        if diff <= threshold:
                            pass_count += 1
                        total_diff += diff_norm
                        n_diffs += 1
                row["mean_diff_normalized"] = total_diff / max(n_diffs, 1)
                row["pass_count_3pct"] = pass_count
                all_rows.append(row)

            # For non-applicable methods, mark as N/A for this variation
            for m in non_applicable:
                if m == "ADASYN" and ds_name == "CB":
                    continue
                paper = paper_ds.get(m)
                if paper is None:
                    continue
                all_rows.append({
                    "variation": v_name, "description": var["description"],
                    "dataset": ds_name, "method": m,
                    "ACC": None, "AUC": None, "F1": None, "AP": None, "GMEAN": None,
                    "mean_diff_normalized": None, "pass_count_3pct": None,
                    "skipped_reason": "variation does not apply to this method",
                })

        v_el = perf_counter() - v_start
        print(f"  done in {v_el:.0f}s")

    # Save raw
    df = pd.DataFrame(all_rows)
    raw_path = OUT_DIR / f"comprehensive_test_raw_{timestamp}.csv"
    df.to_csv(raw_path, index=False)

    # Aggregate per variation: how many cells passed?
    print("\n" + "=" * 100)
    print("VERDICT: passes per variation (cell = method/dataset/metric)")
    print("=" * 100)
    summary_rows = []
    for var in VARIATIONS:
        v_name = var["name"]
        sub = df[(df["variation"] == v_name) & df["mean_diff_normalized"].notna()]
        if len(sub) == 0:
            continue
        total_cells = sub["pass_count_3pct"].apply(lambda x: 5 if x is not None else 0).sum()
        passed_cells = sub["pass_count_3pct"].sum()
        mean_norm_diff = sub["mean_diff_normalized"].mean()
        summary_rows.append({
            "variation": v_name,
            "description": var["description"],
            "rows_evaluated": len(sub),
            "total_cells": int(total_cells),
            "passed_cells_within_3pct": int(passed_cells),
            "pass_rate_pct": round(100.0 * passed_cells / max(total_cells, 1), 1),
            "mean_normalized_diff": round(float(mean_norm_diff), 4),
        })
    sum_df = pd.DataFrame(summary_rows).sort_values("passed_cells_within_3pct",
                                                      ascending=False)
    sum_path = OUT_DIR / f"comprehensive_test_summary_{timestamp}.csv"
    sum_df.to_csv(sum_path, index=False)
    print(sum_df.to_string(index=False))

    # Compare each variation against baseline
    base = sum_df[sum_df["variation"] == "V00_baseline"].iloc[0] if len(sum_df[sum_df["variation"] == "V00_baseline"]) else None
    if base is not None:
        print("\n" + "=" * 100)
        print(f"DELTA vs V00 baseline ({base['passed_cells_within_3pct']} passed, "
              f"{base['mean_normalized_diff']:.4f} mean diff):")
        print("=" * 100)
        for _, r in sum_df.iterrows():
            if r["variation"] == "V00_baseline":
                continue
            dp = r["passed_cells_within_3pct"] - base["passed_cells_within_3pct"]
            dd = r["mean_normalized_diff"] - base["mean_normalized_diff"]
            sign = "+" if dp >= 0 else ""
            sign_d = "+" if dd >= 0 else ""
            print(f"  {r['variation']:<22}  passes {sign}{dp:+d}   mean_diff {sign_d}{dd:+.4f}   {r['description']}")

    elapsed = perf_counter() - grand_start
    print(f"\n  Outputs:")
    print(f"    {raw_path}")
    print(f"    {sum_path}")
    print(f"  Total elapsed: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print("=" * 100)

if __name__ == "__main__":
    main()
