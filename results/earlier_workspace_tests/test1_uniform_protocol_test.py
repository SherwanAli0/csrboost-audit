#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================================
uniform_protocol_test.py  --  Test uniform metric protocol for CSRBoost paper
==============================================================================

EXPERIMENTAL: tests whether a single uniform metric protocol can reproduce the
CSRBoost paper's reported numbers without per-method tuned thresholds.

This is a side-investigation. It is NOT the novelty. See README.md.

Methods (8 non-GAN):
  CSRBoost, ADASYN, Borderline-SMOTE, SMOTE-Tomek, SMOTE-ENN,
  AdaBoost (baseline), RUSBoost, HUE

Two metric protocols computed in parallel for every method:
  A: probability-based AUC and AP (modern standard)
  B: binary-prediction-based AUC and AP (paper hypothesis)

Both protocols use threshold=0.5, no other tuning.

Datasets: 5 smallest - WINE, GLASS, CB, SEED, BCW.

Author: Sherwan Ali | Uskudar University
Date: 2026-04-29
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
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import (accuracy_score, average_precision_score,
                             confusion_matrix, f1_score, roc_auc_score)
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import check_random_state

warnings.filterwarnings("ignore")

# ==============================================================================
# CONFIG
# ==============================================================================
SEED        = 42
SCRIPT_DIR  = Path(__file__).parent
# Read data from the parent novelty folder. Do not modify.
DATA_DIR    = SCRIPT_DIR.parent / "novelty final" / "data"
OUT_DIR     = SCRIPT_DIR / "outputs"
OUT_DIR.mkdir(exist_ok=True)

N_SPLITS    = 5
N_REPEATS   = 20
N_TREES     = 50          # AdaBoost / RUSBoost estimators (paper default)
SMOTE_K     = 5           # paper says "five nearest neighbors"
THRESHOLD   = 0.5         # paper default for all metrics

# Datasets to test: 5 smallest
DATASETS = ["WINE", "GLASS", "CB", "SEED", "BCW"]

METHODS = [
    "CSRBoost",
    "ADASYN",
    "Borderline-SMOTE",
    "SMOTE-Tomek",
    "SMOTE-ENN",
    "AdaBoost",
    "RUSBoost",
    "HUE",
]

# ==============================================================================
# PAPER REPORTED VALUES (from the CSRBoost paper Table 2)
# ==============================================================================
# These are the numbers Yadav et al. (IEEE Access 2025) report. Goal is to see
# whether either protocol reproduces these without per-method tuning.
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
        "AdaBoost":          {"ACC": 100.00, "AUC": 1.00, "F1": 0.87, "AP": 0.90, "GMEAN": 0.91},
        "RUSBoost":          {"ACC": 95.74, "AUC": 0.92, "F1": 0.88, "AP": 0.89, "GMEAN": 0.94},
        "HUE":               {"ACC": 95.89, "AUC": 0.96, "F1": 0.86, "AP": 0.78, "GMEAN": 0.92},
    },
    "CB": {
        "CSRBoost":          {"ACC": 76.43, "AUC": 0.76, "F1": 0.69, "AP": 0.63, "GMEAN": 0.71},
        "ADASYN":            {"ACC": None, "AUC": None, "F1": None, "AP": None, "GMEAN": None},  # not applicable
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
# DATA LOADERS
# ==============================================================================
def _label_encode_objects(df):
    out = df.copy()
    for col in out.select_dtypes(include="object").columns:
        values = sorted(out[col].dropna().astype(str).unique())
        mapping = {value: i for i, value in enumerate(values)}
        out[col] = out[col].astype(str).map(mapping).fillna(0)
    return out

def load_wine():
    p = DATA_DIR / "WINE" / "wine.data"
    df = pd.read_csv(p, header=None)
    y = (df.iloc[:, 0].astype(int).values == 3).astype(int)
    X = df.iloc[:, 1:].values.astype(float)
    return X, y

def load_glass():
    p = DATA_DIR / "GLASS" / "glass.csv"
    df = pd.read_csv(p)
    y = (df.iloc[:, -1].values == 7).astype(int)
    X = df.iloc[:, :-1].values.astype(float)
    return X, y

def load_cb():
    p = DATA_DIR / "CB" / "sonar.all-data"
    df = pd.read_csv(p, header=None)
    y = (df.iloc[:, -1].values == "M").astype(int)
    X = df.iloc[:, :-1].values.astype(float)
    return X, y

def load_seed():
    p = DATA_DIR / "SEED" / "seeds_dataset.txt"
    df = pd.read_csv(p, sep=r"\s+", header=None)
    y = (df.iloc[:, -1].values == 1).astype(int)
    X = df.iloc[:, :-1].values.astype(float)
    return X, y

def load_bcw():
    p = DATA_DIR / "BCW" / "wdbc.data"
    df = pd.read_csv(p, header=None)
    y = (df.iloc[:, 1].values == "M").astype(int)
    X = df.iloc[:, 2:].values.astype(float)
    return X, y

LOADERS = {
    "WINE":  load_wine,
    "GLASS": load_glass,
    "CB":    load_cb,
    "SEED":  load_seed,
    "BCW":   load_bcw,
}

# ==============================================================================
# METRICS
# ==============================================================================
def gmean_at(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    tpr = tp / (tp + fn) if (tp + fn) else 0.0
    tnr = tn / (tn + fp) if (tn + fp) else 0.0
    return float(math.sqrt(tpr * tnr))

def safe_auc(y_true, score):
    if len(np.unique(y_true)) < 2:
        return 0.5
    return float(roc_auc_score(y_true, score))

def safe_ap(y_true, score):
    if len(np.unique(y_true)) < 2:
        return 0.0
    return float(average_precision_score(y_true, score))

def metrics_protocol_A(y_true, y_proba):
    """Probability-based AUC and AP (modern standard)."""
    y_pred = (y_proba >= THRESHOLD).astype(int)
    return {
        "ACC":   float(accuracy_score(y_true, y_pred)) * 100,
        "AUC":   safe_auc(y_true, y_proba),
        "F1":    float(f1_score(y_true, y_pred, zero_division=0)),
        "AP":    safe_ap(y_true, y_proba),
        "GMEAN": gmean_at(y_true, y_pred),
    }

def metrics_protocol_B(y_true, y_proba):
    """Binary-prediction-based AUC and AP (paper hypothesis)."""
    y_pred = (y_proba >= THRESHOLD).astype(int)
    return {
        "ACC":   float(accuracy_score(y_true, y_pred)) * 100,
        "AUC":   safe_auc(y_true, y_pred),
        "F1":    float(f1_score(y_true, y_pred, zero_division=0)),
        "AP":    safe_ap(y_true, y_pred),
        "GMEAN": gmean_at(y_true, y_pred),
    }

# ==============================================================================
# CLASSIFIERS (paper-default hyperparameters, no FIX)
# ==============================================================================
def make_adaboost(rs):
    base = DecisionTreeClassifier(max_depth=1, random_state=rs)
    kwargs = {"n_estimators": N_TREES, "random_state": rs}
    import sklearn
    ver = tuple(int(x) for x in sklearn.__version__.split(".")[:2])
    if ver < (1, 6):
        kwargs["algorithm"] = "SAMME"
    if ver >= (1, 2):
        kwargs["estimator"] = base
    else:
        kwargs["base_estimator"] = base
    return AdaBoostClassifier(**kwargs)

def smote_safe(X, y, rs, k=SMOTE_K):
    n_min = int(np.sum(y == 1))
    if n_min < 2:
        return X, y
    k_use = min(k, max(1, n_min - 1))
    try:
        return SMOTE(k_neighbors=k_use, random_state=rs).fit_resample(X, y)
    except Exception:
        return X, y

# CSRBoost
def csrboost_fit_predict(X_tr, y_tr, X_te, rs):
    X_min = X_tr[y_tr == 1]
    X_maj = X_tr[y_tr == 0]
    n_min = len(X_min)
    n_maj = len(X_maj)
    K = max(1, min(n_min, n_maj))
    rng = np.random.RandomState(rs)
    km = MiniBatchKMeans(n_clusters=K, n_init=1, max_iter=100,
                         batch_size=1024, random_state=rs)
    labels = km.fit_predict(X_maj)
    kept = []
    for k in range(K):
        cidx = np.where(labels == k)[0]
        n_keep = max(1, math.ceil(len(cidx) * 0.50))
        kept.extend(cidx.tolist() if n_keep >= len(cidx)
                    else rng.choice(cidx, n_keep, replace=False).tolist())
    X_ret = np.vstack([X_min, X_maj[kept]])
    y_ret = np.concatenate([np.ones(len(X_min)), np.zeros(len(kept))]).astype(int)
    Xb, yb = smote_safe(X_ret, y_ret, rs)
    clf = make_adaboost(rs)
    clf.fit(Xb, yb)
    return clf.predict_proba(X_te)[:, 1]

# Resample-then-AdaBoost wrappers
def make_resampler(name, rs):
    if name == "ADASYN":
        return ADASYN(n_neighbors=SMOTE_K, random_state=rs)
    if name == "Borderline-SMOTE":
        return BorderlineSMOTE(k_neighbors=SMOTE_K, random_state=rs)
    if name == "SMOTE-Tomek":
        return SMOTETomek(smote=SMOTE(k_neighbors=SMOTE_K, random_state=rs),
                          random_state=rs)
    if name == "SMOTE-ENN":
        return SMOTEENN(smote=SMOTE(k_neighbors=SMOTE_K, random_state=rs),
                        random_state=rs)
    raise ValueError(name)

def resample_then_ada_fit_predict(name, X_tr, y_tr, X_te, rs):
    sampler = make_resampler(name, rs)
    n_min = int(np.sum(y_tr == 1))
    if n_min < 2:
        Xr, yr = X_tr, y_tr
    else:
        try:
            Xr, yr = sampler.fit_resample(X_tr, y_tr)
            if len(np.unique(yr)) < 2:
                Xr, yr = X_tr, y_tr
        except Exception:
            Xr, yr = X_tr, y_tr
    clf = make_adaboost(rs)
    clf.fit(Xr, yr)
    return clf.predict_proba(X_te)[:, 1]

# AdaBoost baseline
def adaboost_fit_predict(X_tr, y_tr, X_te, rs):
    clf = make_adaboost(rs)
    clf.fit(X_tr, y_tr)
    return clf.predict_proba(X_te)[:, 1]

# RUSBoost
def rusboost_fit_predict(X_tr, y_tr, X_te, rs):
    base = DecisionTreeClassifier(max_depth=1, random_state=rs)
    try:
        clf = RUSBoostClassifier(estimator=base, n_estimators=N_TREES,
                                 random_state=rs)
    except TypeError:
        clf = RUSBoostClassifier(base_estimator=base, n_estimators=N_TREES,
                                 random_state=rs)
    clf.fit(X_tr, y_tr)
    return clf.predict_proba(X_te)[:, 1]

# HUE  (ITQ-based hash undersampling ensemble, Ng et al. 2022)
def hue_fit_predict(X_tr, y_tr, X_te, rs, max_depth=10, itq_iters=50):
    rng = check_random_state(rs)
    X_min = X_tr[y_tr == 1]
    X_maj = X_tr[y_tr == 0]
    n_min = len(X_min)
    n_maj = len(X_maj)
    if n_min < 1 or n_maj < 1:
        clf = DecisionTreeClassifier(max_depth=max_depth, random_state=rs)
        clf.fit(X_tr, y_tr)
        return clf.predict_proba(X_te)[:, 1] if len(clf.classes_) == 2 else \
               np.full(len(X_te), float(clf.classes_[0]))
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
        clf = DecisionTreeClassifier(max_depth=max_depth,
                                     random_state=rng.randint(0, 1_000_000))
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
# DISPATCH
# ==============================================================================
def run_method(name, X_tr, y_tr, X_te, rs):
    if name == "CSRBoost":
        return csrboost_fit_predict(X_tr, y_tr, X_te, rs)
    if name in {"ADASYN", "Borderline-SMOTE", "SMOTE-Tomek", "SMOTE-ENN"}:
        return resample_then_ada_fit_predict(name, X_tr, y_tr, X_te, rs)
    if name == "AdaBoost":
        return adaboost_fit_predict(X_tr, y_tr, X_te, rs)
    if name == "RUSBoost":
        return rusboost_fit_predict(X_tr, y_tr, X_te, rs)
    if name == "HUE":
        return hue_fit_predict(X_tr, y_tr, X_te, rs)
    raise ValueError(name)

# ==============================================================================
# MAIN
# ==============================================================================
def main():
    print("=" * 100)
    print("UNIFORM PROTOCOL TEST -- 5 smallest datasets, 8 non-GAN methods")
    print(f"  Threshold: {THRESHOLD}  |  CV: {N_SPLITS}x{N_REPEATS} = {N_SPLITS*N_REPEATS} folds")
    print(f"  Datasets : {', '.join(DATASETS)}")
    print(f"  Methods  : {', '.join(METHODS)}")
    print("=" * 100)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fold_rows = []   # per-fold metrics
    summary_rows = []  # per-dataset / per-method summary

    grand_start = perf_counter()
    for ds_name in DATASETS:
        loader = LOADERS[ds_name]
        X, y = loader()
        n_min = int(np.sum(y == 1))
        n_maj = int(np.sum(y == 0))
        ir = n_maj / max(n_min, 1)
        print(f"\n--- {ds_name}: n={len(y)}  features={X.shape[1]}  IR={ir:.2f}  minority={n_min}")
        ds_start = perf_counter()
        splitter = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS,
                                            random_state=SEED)

        # collect per-fold metrics for both protocols, all methods
        per_method_A = {m: {k: [] for k in ["ACC","AUC","F1","AP","GMEAN"]} for m in METHODS}
        per_method_B = {m: {k: [] for k in ["ACC","AUC","F1","AP","GMEAN"]} for m in METHODS}

        for fold_i, (tr_idx, te_idx) in enumerate(splitter.split(X, y)):
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X[tr_idx])
            X_te = scaler.transform(X[te_idx])
            y_tr, y_te = y[tr_idx], y[te_idx]

            for m in METHODS:
                # Skip ADASYN on CB (paper excludes it)
                if m == "ADASYN" and ds_name == "CB":
                    continue
                try:
                    y_proba = run_method(m, X_tr, y_tr, X_te, SEED)
                    mA = metrics_protocol_A(y_te, y_proba)
                    mB = metrics_protocol_B(y_te, y_proba)
                    for k in mA:
                        per_method_A[m][k].append(mA[k])
                        per_method_B[m][k].append(mB[k])
                    fold_rows.append({
                        "dataset": ds_name, "method": m, "fold": fold_i,
                        **{f"A_{k}": mA[k] for k in mA},
                        **{f"B_{k}": mB[k] for k in mB},
                    })
                except Exception as e:
                    print(f"  [{ds_name} fold {fold_i} {m}] error: {e}")

            if (fold_i + 1) % 20 == 0:
                el = perf_counter() - ds_start
                eta = (N_SPLITS*N_REPEATS - fold_i - 1) / ((fold_i+1)/el) if el > 0 else 0
                print(f"    fold {fold_i+1}/{N_SPLITS*N_REPEATS}  elapsed={el:.0f}s  ETA={eta:.0f}s")

        # Summarize this dataset
        for m in METHODS:
            if m == "ADASYN" and ds_name == "CB":
                continue
            paper = PAPER[ds_name][m]
            row = {
                "dataset": ds_name, "method": m, "ir": round(ir, 2), "n_min": n_min,
            }
            for proto, store, label in [("A", per_method_A, "Protocol A (proba)"),
                                          ("B", per_method_B, "Protocol B (binary)")]:
                for k in ["ACC","AUC","F1","AP","GMEAN"]:
                    if not store[m][k]:
                        continue
                    row[f"{proto}_{k}"] = float(np.mean(store[m][k]))
                    if paper[k] is not None:
                        row[f"{proto}_{k}_paper"]    = paper[k]
                        row[f"{proto}_{k}_diff"]     = abs(row[f"{proto}_{k}"] - paper[k])
            summary_rows.append(row)

    # Write outputs
    folds_df   = pd.DataFrame(fold_rows)
    summary_df = pd.DataFrame(summary_rows)
    folds_path   = OUT_DIR / f"fold_results_{timestamp}.csv"
    summary_path = OUT_DIR / f"summary_{timestamp}.csv"
    folds_df.to_csv(folds_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    # Print verdict table
    print("\n" + "=" * 100)
    print("VERDICT: which protocol better matches paper?")
    print("=" * 100)
    headers = ["Dataset", "Method", "Metric", "Paper", "Proto A (proba)",
               "Proto B (binary)", "A diff", "B diff", "Better"]
    rows_print = []
    for r in summary_rows:
        ds = r["dataset"]; m = r["method"]
        for k in ["ACC","AUC","F1","AP","GMEAN"]:
            ka_diff = r.get(f"A_{k}_diff")
            kb_diff = r.get(f"B_{k}_diff")
            paper_v = r.get(f"A_{k}_paper")
            if paper_v is None:
                continue
            a_v = r.get(f"A_{k}", float("nan"))
            b_v = r.get(f"B_{k}", float("nan"))
            better = "A" if (ka_diff or 99) < (kb_diff or 99) else "B"
            if ka_diff is None or kb_diff is None or abs(ka_diff - kb_diff) < 1e-6:
                better = "tie"
            rows_print.append([ds, m, k, f"{paper_v:.2f}", f"{a_v:.4f}",
                                f"{b_v:.4f}", f"{ka_diff:.4f}", f"{kb_diff:.4f}",
                                better])
    print(f"{'Dataset':<7} {'Method':<18} {'Metric':<6} {'Paper':>7} "
          f"{'Proto A':>9} {'Proto B':>9} {'A diff':>8} {'B diff':>8} {'Better':>7}")
    print("-" * 100)
    for r in rows_print:
        print(f"{r[0]:<7} {r[1]:<18} {r[2]:<6} {r[3]:>7} "
              f"{r[4]:>9} {r[5]:>9} {r[6]:>8} {r[7]:>8} {r[8]:>7}")

    # Aggregate
    n_A_better = sum(1 for r in rows_print if r[-1] == "A")
    n_B_better = sum(1 for r in rows_print if r[-1] == "B")
    n_tie      = sum(1 for r in rows_print if r[-1] == "tie")
    total      = len(rows_print)

    print("\n" + "=" * 100)
    print(f"  Total comparisons : {total}")
    print(f"  Protocol A wins   : {n_A_better} ({100*n_A_better/total:.1f}%)")
    print(f"  Protocol B wins   : {n_B_better} ({100*n_B_better/total:.1f}%)")
    print(f"  Ties              : {n_tie} ({100*n_tie/total:.1f}%)")
    print("=" * 100)

    # Mean absolute error per protocol
    A_diffs = [r.get(f"A_{k}_diff") for r in summary_rows for k in ["ACC","AUC","F1","AP","GMEAN"]
                if r.get(f"A_{k}_diff") is not None]
    B_diffs = [r.get(f"B_{k}_diff") for r in summary_rows for k in ["ACC","AUC","F1","AP","GMEAN"]
                if r.get(f"B_{k}_diff") is not None]
    print(f"  Protocol A mean diff (vs paper): {np.mean(A_diffs):.4f}")
    print(f"  Protocol B mean diff (vs paper): {np.mean(B_diffs):.4f}")
    print()
    print(f"  Outputs:\n    {folds_path}\n    {summary_path}")

    elapsed = perf_counter() - grand_start
    print(f"  Total elapsed: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print("=" * 100)

if __name__ == "__main__":
    main()
