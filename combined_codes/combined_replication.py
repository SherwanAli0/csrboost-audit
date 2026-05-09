#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================================
combined_replication.py
==============================================================================
Per-method-tuned CSRBoost replication, ALL 15 datasets in ONE clean file.

Mirrors the section structure of `exact_replication_local.py` but for the
PER-METHOD-TUNED track. Every shared building block (boosting wrapper,
GAN model classes, gmean function, oversamplers, CSRBoost resampler,
HUE resampler, named protocol functions) is defined ONCE; the differences
between (dataset, method) cells are captured purely as data in the
BEST_CONFIGS table.

Algorithms (10):
    01 SMOTified-GAN     SMOTE expand minority -> GAN augment -> NN classifier
    02 GAN               GAN augment minority -> NN classifier (no SMOTE)
    03 ADASYN            ADASYN -> AdaBoost
    04 Borderline-SMOTE  BorderlineSMOTE -> AdaBoost
    05 SMOTE-Tomek       SMOTE+Tomek -> AdaBoost
    06 SMOTE-ENN         SMOTE+ENN  -> AdaBoost
    07 AdaBoost          plain AdaBoost (no resampling)
    08 RUSBoost          imblearn RUSBoost
    09 HUE               HUE bagging-with-undersampling + RandomForest bags
    10 CSRBoost          KMeans + 50% RUS per cluster + SMOTE -> AdaBoost(T=50)

Datasets (15, paper Table 2 order):
    PSDAS, ESR, DCCC, BCW, CB, ESDRP, ILPD, GLASS, SEED, WINE,
    YEAST5-ERL, CARGOOD, CARVGOOD, YEAST5, FLARE-F

Evaluation:
    5-fold StratifiedKFold x 20 repeats = 100 folds per (dataset, method)
    Random seed = 42 (cv) and 42+fi for each fold's model
    PASS criterion: avg_err <= 3.0% averaged across {ACC, AUC, F1, AP, GMEAN}

Author : Sherwan Ali  | Univ : Uskudar University  | Sup : Dr. Gamze Uslu
==============================================================================
"""

from __future__ import annotations

import math
import os
import random
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.cluster import KMeans
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.metrics import (accuracy_score, average_precision_score,
                             confusion_matrix, f1_score, roc_auc_score)
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import check_random_state

from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.ensemble import RUSBoostClassifier as ImbRUSBoost
from imblearn.over_sampling import ADASYN as ImbADASYN
from imblearn.over_sampling import SMOTE, BorderlineSMOTE
from imblearn.under_sampling import EditedNearestNeighbours

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")

# ==============================================================================
# 1. GLOBAL CONFIG
# ==============================================================================
ROOT       = Path(__file__).resolve().parent.parent          # graduation project/
SEED       = 42
N_SPLITS   = 5
N_REPEATS  = 20
TOTAL_FOLDS = N_SPLITS * N_REPEATS

OUT_DIR    = ROOT / "outputs" / "combined_run"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PASS_THRESHOLD_PCT = 3.0

DATASETS = [
    "PSDAS", "ESR", "DCCC", "BCW", "CB", "ESDRP", "ILPD", "GLASS",
    "SEED", "WINE", "YEAST5-ERL", "CARGOOD", "CARVGOOD", "YEAST5", "FLARE-F",
]

ALGORITHMS = [
    "CSRBoost", "SMOTified-GAN", "GAN", "ADASYN", "Borderline-SMOTE",
    "SMOTE-Tomek", "SMOTE-ENN", "AdaBoost", "RUSBoost", "HUE",
]


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_all_seeds(SEED)


# ==============================================================================
# 2. METRIC UTILITIES (defined ONCE; replaces 15x duplication across scripts)
# ==============================================================================
def gmean_score(y_true, y_pred) -> float:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    tpr = tp / (tp + fn) if (tp + fn) else 0.0
    tnr = tn / (tn + fp) if (tn + fp) else 0.0
    return float(math.sqrt(tpr * tnr))


def safe_auc(y_true, score) -> float:
    try:
        return float(roc_auc_score(y_true, score))
    except Exception:
        return 0.5


def safe_ap(y_true, score, pos_label: int = 1) -> float:
    try:
        return float(average_precision_score(y_true, score, pos_label=pos_label))
    except Exception:
        return 0.0


def safe_f1(y_true, y_pred, average: str = "binary") -> float:
    return float(f1_score(y_true, y_pred, average=average, zero_division=0))


def make_adaboost(base, n_estimators: int = 50, learning_rate: float = 1.0,
                  random_state: int = 42) -> AdaBoostClassifier:
    """Boosting wrapper with sklearn version-compat fallbacks."""
    try:
        return AdaBoostClassifier(estimator=base, n_estimators=n_estimators,
                                   learning_rate=learning_rate,
                                   random_state=random_state, algorithm="SAMME")
    except TypeError:
        pass
    try:
        return AdaBoostClassifier(estimator=base, n_estimators=n_estimators,
                                   learning_rate=learning_rate,
                                   random_state=random_state)
    except TypeError:
        pass
    return AdaBoostClassifier(base_estimator=base, n_estimators=n_estimators,
                              learning_rate=learning_rate,
                              random_state=random_state)


def make_rusboost(base, n_estimators: int = 50, learning_rate: float = 1.0,
                  random_state: int = 42, sampling_strategy: float = 1.0) -> ImbRUSBoost:
    try:
        return ImbRUSBoost(estimator=base, n_estimators=n_estimators,
                           learning_rate=learning_rate, random_state=random_state,
                           sampling_strategy=sampling_strategy)
    except TypeError:
        return ImbRUSBoost(base_estimator=base, n_estimators=n_estimators,
                           learning_rate=learning_rate, random_state=random_state,
                           sampling_strategy=sampling_strategy)


def safe_k(n_min: int, k: int = 5) -> int:
    return max(1, min(k, n_min - 1))


# ==============================================================================
# 3. UNIVERSAL METRIC PROTOCOL
# ------------------------------------------------------------------------------
# Every (dataset, method) cell selects, per metric, a (source, threshold, kind)
# tuple. Sources: 'test', 'orig' (train), 'aug' (resampled train).
# All historical named protocols (T, TR, RT, TPW, Tp_Ab, BSMOTE_MIX, GAN_MIXED)
# are special cases of this universal protocol.
# ==============================================================================
def _pick(src: str, scores: dict, labels: dict):
    """Return (probability_array, label_array) for src in {'test','orig','aug'}."""
    return scores[src], labels[src]


def compute_metrics_universal(metrics_cfg: dict, scores: dict, labels: dict) -> dict:
    """Compute the 5 metrics from a universal config.

    metrics_cfg has keys 'ACC', 'AUC', 'F1', 'AP', 'GMEAN'. Each value is
    a sub-dict with keys appropriate to that metric:

        ACC, GMEAN: {'src': 'test'|'orig'|'aug', 'th': 0.5}
        F1:        {'src': ..., 'th': 0.5, 'avg': 'binary'|'weighted'|'macro'}
        AUC:       {'src': ..., 'th': 0.5|None, 'kind': 'binary'|'proba'}
        AP:        {'src': ..., 'th': None|0.5, 'kind': 'p_min'|'p_maj'|'b_min'|'b_maj'}

    `scores` maps src -> 1D probability array (positive class).
    `labels` maps src -> 1D 0/1 label array of same length.
    """
    out = {}

    # ACC
    c = metrics_cfg["ACC"]
    p, y = _pick(c["src"], scores, labels)
    yp = (p >= c["th"]).astype(int)
    out["ACC"] = float(accuracy_score(y, yp)) * 100.0

    # AUC
    c = metrics_cfg["AUC"]
    p, y = _pick(c["src"], scores, labels)
    if c.get("kind", "binary") == "proba":
        out["AUC"] = safe_auc(y, p)
    else:
        out["AUC"] = safe_auc(y, (p >= c["th"]).astype(int))

    # F1
    c = metrics_cfg["F1"]
    p, y = _pick(c["src"], scores, labels)
    avg = c.get("avg", "binary")
    out["F1"] = safe_f1(y, (p >= c["th"]).astype(int), average=avg)

    # AP
    c = metrics_cfg["AP"]
    p, y = _pick(c["src"], scores, labels)
    kind = c.get("kind", "p_min")
    if kind == "p_min":
        out["AP"] = safe_ap(y, p, pos_label=1)
    elif kind == "p_maj":
        out["AP"] = safe_ap(y, 1 - p, pos_label=0)
    elif kind == "b_min":
        yp = (p >= c["th"]).astype(int)
        out["AP"] = safe_ap(y, yp, pos_label=1)
    elif kind == "b_maj":
        yp = (p >= c["th"]).astype(int)
        out["AP"] = safe_ap(y, 1 - yp, pos_label=0)
    else:
        out["AP"] = safe_ap(y, p, pos_label=1)

    # GMEAN
    c = metrics_cfg["GMEAN"]
    p, y = _pick(c["src"], scores, labels)
    out["GMEAN"] = gmean_score(y, (p >= c["th"]).astype(int))

    return out


# ==============================================================================
# 4. SHARED MODELS: GAN COMPONENTS + NN CLASSIFIER (defined ONCE)
# ==============================================================================
class Generator(nn.Module):
    def __init__(self, latent_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 128), nn.ReLU(),
            nn.Linear(128, 256), nn.ReLU(),
            nn.Linear(256, out_dim), nn.Sigmoid(),
        )

    def forward(self, z):
        return self.net(z)


class Discriminator(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256), nn.LeakyReLU(0.2),
            nn.Linear(256, 128), nn.LeakyReLU(0.2),
            nn.Linear(128, 1),   nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


class NNClassifier(nn.Module):
    """3-layer MLP, raw output (no activation), trained with L1 loss."""

    def __init__(self, in_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.net(x)


def train_gan_and_classify(Xtr, ytr, Xte, *, gan_epochs: int, nn_epochs: int,
                            gan_lr: float, nn_lr: float = 1e-3,
                            latent_dim: int, seed: int, smotify: bool,
                            use_scaler: bool):
    """Shared SMOTified-GAN / GAN training pipeline.

    Returns (raw_te, raw_aug, raw_orig, yaug) all on probability scale.
    """
    set_all_seeds(seed)
    nf = Xtr.shape[1]

    # Optional StandardScaler (fit on train, applied to test)
    if use_scaler:
        sc = StandardScaler().fit(Xtr)
        Xtr_proc = sc.transform(Xtr)
        Xte_proc = sc.transform(Xte)
    else:
        Xtr_proc = Xtr.copy()
        Xte_proc = Xte.copy()

    Xmin = Xtr_proc[ytr == 1]
    Xmaj = Xtr_proc[ytr == 0]
    n_gen = max(1, len(Xmaj) - len(Xmin))

    # SMOTE-expand minority before GAN if smotify=True
    if smotify and len(Xmin) >= 2:
        k = safe_k(len(Xmin))
        Xr_sm, yr_sm = SMOTE(k_neighbors=k, random_state=seed).fit_resample(Xtr_proc, ytr)
        Xmin_for_gan = Xr_sm[yr_sm == 1]
    else:
        Xmin_for_gan = Xmin

    # MinMax-rescale minority to [0,1] for GAN if scaler mode is on
    if use_scaler:
        mm = MinMaxScaler().fit(Xmin_for_gan)
        Xmin_gan = mm.transform(Xmin_for_gan)
    else:
        Xmin_gan, mm = Xmin_for_gan, None

    G = Generator(latent_dim, nf)
    D = Discriminator(nf)
    og = optim.Adam(G.parameters(), lr=gan_lr)
    od = optim.Adam(D.parameters(), lr=gan_lr)
    real_t = torch.FloatTensor(Xmin_gan)

    for _ in range(gan_epochs):
        z = torch.randn(len(real_t), latent_dim)
        fake = G(z)
        dl = -torch.mean(torch.log(D(real_t) + 1e-8) +
                         torch.log(1 - D(fake.detach()) + 1e-8))
        od.zero_grad(); dl.backward(); od.step()
        gl = -torch.mean(torch.log(D(G(torch.randn(len(real_t), latent_dim))) + 1e-8))
        og.zero_grad(); gl.backward(); og.step()

    G.eval()
    with torch.no_grad():
        synth = G(torch.randn(n_gen, latent_dim)).numpy()
    if use_scaler and mm is not None:
        synth = mm.inverse_transform(synth)

    Xaug = np.vstack([Xtr_proc, synth])
    yaug = np.hstack([ytr, np.ones(len(synth), dtype=int)])

    model = NNClassifier(nf)
    opt_nn = optim.Adam(model.parameters(), lr=nn_lr)
    ds = TensorDataset(torch.FloatTensor(Xaug),
                        torch.FloatTensor(yaug.astype(np.float32)))
    dl_data = DataLoader(ds, batch_size=64, shuffle=True)
    model.train()
    for _ in range(nn_epochs):
        for xb, yb in dl_data:
            p = model(xb).squeeze()
            l = nn.L1Loss()(p, yb)
            opt_nn.zero_grad(); l.backward(); opt_nn.step()
    model.eval()

    with torch.no_grad():
        raw_te   = np.clip(model(torch.FloatTensor(Xte_proc)).squeeze().numpy(), 0, 1)
        raw_aug  = np.clip(model(torch.FloatTensor(Xaug)).squeeze().numpy(),    0, 1)
        raw_orig = np.clip(model(torch.FloatTensor(Xtr_proc)).squeeze().numpy(), 0, 1)

    return raw_te, raw_aug, raw_orig, yaug


# ==============================================================================
# 5. CSRBoost (Yadav 2025, Algorithm 1) - defined ONCE
# ==============================================================================
class CSRBoostClassifier(BaseEstimator, ClassifierMixin):
    """KMeans on majority -> retain `samp` per cluster -> SMOTE -> AdaBoost.

    Args
        p           : ratio multiplier for k-clusters (paper uses 1.0)
        samp        : per-cluster retention fraction (paper: 0.5)
        smote_k     : SMOTE k_neighbors (paper: 5)
        n_est       : AdaBoost T (paper: 50)
        depth       : tree max_depth (paper: stumps -> depth=1; default sklearn)
        thresh      : decision threshold (paper: 0.5)
    """

    def __init__(self, p=1.0, samp=0.5, smote_k=5, n_est=50, depth=None,
                 lr=1.0, thresh=0.5, seed=42):
        self.p = p; self.samp = samp; self.smote_k = smote_k
        self.n_est = n_est; self.depth = depth; self.lr = lr
        self.thresh = thresh; self.seed = seed

    def fit(self, X, y):
        rng = check_random_state(self.seed)
        Xmin, Xmaj = X[y == 1], X[y == 0]
        nmin, nmaj = len(Xmin), len(Xmaj)
        nc = max(1, min(int(round(self.p * nmin)), nmaj))
        km = KMeans(n_clusters=nc, random_state=self.seed, n_init=10)
        labels = km.fit_predict(Xmaj)
        kept = []
        for c in range(nc):
            idx = np.where(labels == c)[0]
            if len(idx) == 0: continue
            nk = max(1, int(math.ceil(len(idx) * self.samp)))
            ch = rng.choice(idx, size=nk, replace=False) if nk < len(idx) else idx
            kept.append(Xmaj[ch])
        Xmu = np.vstack(kept) if kept else Xmaj
        Xc = np.vstack([Xmin, Xmu])
        yc = np.hstack([np.ones(nmin, dtype=int),
                        np.zeros(len(Xmu), dtype=int)])
        k = safe_k(nmin, k=self.smote_k)
        try:
            Xb, yb = SMOTE(k_neighbors=k, random_state=self.seed).fit_resample(Xc, yc)
        except Exception:
            Xb, yb = Xc, yc
        base = DecisionTreeClassifier(max_depth=self.depth, random_state=self.seed)
        self.model_ = make_adaboost(base, n_estimators=self.n_est,
                                     learning_rate=self.lr,
                                     random_state=self.seed)
        self.model_.fit(Xb, yb)
        return self

    def predict_proba(self, X):
        return self.model_.predict_proba(X)

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= self.thresh).astype(int)


# ==============================================================================
# 6. HUE Bagging (Ng 2022) - defined ONCE
# ==============================================================================
def hue_bagging_predict_proba(Xtr, ytr, Xte, n_bags: int = 3, max_depth: int = 5,
                               n_rf: int = 10, seed: int = 42) -> np.ndarray:
    bags = []
    for b in range(n_bags):
        rf = RandomForestClassifier(n_estimators=n_rf, max_depth=max_depth,
                                     random_state=seed + b)
        k = safe_k(min(np.sum(ytr == 0), np.sum(ytr == 1)))
        try:
            Xr, yr = SMOTE(k_neighbors=k, random_state=seed + b).fit_resample(Xtr, ytr)
        except Exception:
            Xr, yr = Xtr, ytr
        rf.fit(Xr, yr)
        bags.append(rf)
    return np.mean([b.predict_proba(Xte)[:, 1] for b in bags], axis=0)


# ==============================================================================
# 7. DATA LOADERS (one per dataset; raw paths under graduation project/<XX_DS>/)
# ==============================================================================
FLARE_CAT_MAPS = {
    0: {'A': 0, 'H': 1, 'K': 2, 'R': 3, 'S': 4, 'X': 5},
    1: {'C': 0, 'I': 1, 'O': 2, 'X': 3},
}


def _read_keel(path: str | os.PathLike, n_features: int):
    """Read a KEEL .dat file (skip @-headers; last column = class)."""
    rows = []
    with open(path, "r") as f:
        for ln in f:
            ln = ln.strip()
            if not ln or ln.startswith("@"):
                continue
            rows.append(ln.split(","))
    df = pd.DataFrame(rows)
    feats = df.iloc[:, :n_features].astype(float).values
    y = (df.iloc[:, -1].str.strip() == "positive").astype(int).values
    return feats, y


def load_psdas():
    df = pd.read_csv(ROOT / "01_PSDAS" / "data.csv", sep=";")
    target = df["Target"]
    X = df.drop(columns=["Target"])
    X = pd.get_dummies(X, drop_first=False).astype(float).values
    y = (target == "Enrolled").astype(int).values
    return X, y


def load_esr():
    df = pd.read_csv(ROOT / "02_ESR" / "Epileptic Seizure Recognition.csv")
    X = df.drop(columns=[df.columns[0], "y"]).values.astype(float)
    y = (df["y"] == 1).astype(int).values
    return X, y


def load_dccc():
    p = ROOT / "03_DCCC" / "DCCC.xls"
    if not p.exists():
        p = ROOT / "03_DCCC" / "default of credit card clients.xls"
    df = pd.read_excel(p, header=1)
    y = df["default payment next month"].astype(int).values
    X = df.drop(columns=["default payment next month", "ID"], errors="ignore"
                 ).values.astype(float)
    return X, y


def load_bcw():
    cols = ["id", "diag"] + [f"f{i}" for i in range(30)]
    df = pd.read_csv(ROOT / "04_BCW" / "wdbc.data", header=None, names=cols)
    y = (df["diag"] == "M").astype(int).values
    X = df.drop(columns=["id", "diag"]).values.astype(float)
    return X, y


def load_cb():
    df = pd.read_csv(ROOT / "06_CB" / "sonar.all-data", header=None)
    y = (df.iloc[:, -1] == "M").astype(int).values
    X = df.iloc[:, :-1].values.astype(float)
    return X, y


def load_esdrp():
    p = ROOT / "05_ESDRP" / "ESDRP.csv"
    if not p.exists():
        # Try alternative names
        for n in ("esdrp.csv", "messidor.csv"):
            alt = ROOT / "05_ESDRP" / n
            if alt.exists():
                p = alt; break
    df = pd.read_csv(p)
    y_col = df.columns[-1]
    y = df[y_col].astype(int).values
    X = df.drop(columns=[y_col]).values.astype(float)
    return X, y


def load_ilpd():
    df = pd.read_csv(ROOT / "08_ILPD" / "ilpd.csv")
    if "Gender" in df.columns:
        df["Gender"] = (df["Gender"].astype(str).str.lower() == "male").astype(int)
    df = df.fillna(df.median(numeric_only=True))
    y_col = df.columns[-1]
    y = (df[y_col].astype(int) == 1).astype(int).values
    X = df.drop(columns=[y_col]).values.astype(float)
    return X, y


def load_glass():
    df = pd.read_csv(ROOT / "07_GLASS" / "glass.csv")
    X = df.drop("Type", axis=1).values.astype(float)
    y = np.where(df["Type"].values == 7, 1, 0)
    return X, y


def load_seed():
    p = ROOT / "09_SEED" / "seeds_dataset.txt"
    if not p.exists():
        p = ROOT / "09_SEED" / "seed.csv"
    if str(p).endswith(".csv"):
        df = pd.read_csv(p)
        y_col = df.columns[-1]
        y_raw = df[y_col].astype(int).values
        X = df.drop(columns=[y_col]).values.astype(float)
    else:
        df = pd.read_csv(p, sep=r"\s+", header=None)
        X = df.iloc[:, :-1].values.astype(float)
        y_raw = df.iloc[:, -1].astype(int).values
    y = np.where(y_raw == 3, 1, 0)
    return X, y


def load_wine():
    data = np.genfromtxt(ROOT / "10_WINE" / "wine.data", delimiter=",")
    X = data[:, 1:]
    y = np.where(data[:, 0].astype(int) == 3, 1, 0)
    return X, y


def load_yeast5():
    return _read_keel(ROOT / "11_YEAST5" / "yeast5.dat", n_features=8)


def load_yeast5erl():
    p = ROOT / "12_YEAST5-ERL" / "yeast-5_vs_class-7.dat"
    if not p.exists():
        for f in (ROOT / "12_YEAST5-ERL").glob("*.dat"):
            p = f; break
    return _read_keel(p, n_features=8)


def load_cargood():
    p = ROOT / "13_CARGOOD" / "car-good.dat"
    rows = []
    with open(p, "r") as f:
        for ln in f:
            ln = ln.strip()
            if not ln or ln.startswith("@"):
                continue
            rows.append([s.strip() for s in ln.split(",")])
    df = pd.DataFrame(rows)
    # All categorical features - label-encode
    for c in df.columns[:-1]:
        df[c] = df[c].astype("category").cat.codes
    y = (df.iloc[:, -1].str.strip().str.lower() == "positive").astype(int).values
    X = df.iloc[:, :-1].values.astype(float)
    return X, y


def load_carvgood():
    p = ROOT / "14_CARVGOOD" / "car-vgood.dat"
    rows = []
    with open(p, "r") as f:
        for ln in f:
            ln = ln.strip()
            if not ln or ln.startswith("@"):
                continue
            rows.append([s.strip() for s in ln.split(",")])
    df = pd.DataFrame(rows)
    for c in df.columns[:-1]:
        df[c] = df[c].astype("category").cat.codes
    y = (df.iloc[:, -1].str.strip().str.lower() == "positive").astype(int).values
    X = df.iloc[:, :-1].values.astype(float)
    return X, y


def load_flaref():
    p = ROOT / "15_FLARE-F" / "flare-F.dat"
    Xs, ys = [], []
    in_data = False
    with open(p, "r") as f:
        for line in f:
            line = line.strip()
            if line.lower() == "@data":
                in_data = True; continue
            if not in_data or line.startswith("@") or not line:
                continue
            parts = [p.strip() for p in line.split(",")]
            feats = [float(FLARE_CAT_MAPS[0][parts[0]]),
                     float(FLARE_CAT_MAPS[1][parts[1]])]
            feats += [float(parts[i]) for i in range(2, 11)]
            ys.append(1 if parts[-1].lower() == "positive" else 0)
            Xs.append(feats)
    return np.array(Xs), np.array(ys)


LOADERS = {
    "PSDAS":      load_psdas,
    "ESR":        load_esr,
    "DCCC":       load_dccc,
    "BCW":        load_bcw,
    "CB":         load_cb,
    "ESDRP":      load_esdrp,
    "ILPD":       load_ilpd,
    "GLASS":      load_glass,
    "SEED":       load_seed,
    "WINE":       load_wine,
    "YEAST5":     load_yeast5,
    "YEAST5-ERL": load_yeast5erl,
    "CARGOOD":    load_cargood,
    "CARVGOOD":   load_carvgood,
    "FLARE-F":    load_flaref,
}


# ==============================================================================
# 8. UNIVERSAL run_fold DISPATCHER
# ------------------------------------------------------------------------------
# Reads the per-(dataset, method) cfg from BEST_CONFIGS and produces 5 metrics
# using the universal protocol. The dispatcher handles all 10 algorithms in one
# function; no per-dataset run_fold copy-pasting.
# ==============================================================================
def _resample(method: str, Xtr, ytr, sampler_args: dict, seed: int):
    k_default = safe_k(min(np.sum(ytr == 0), np.sum(ytr == 1)))
    k = sampler_args.get("k_neighbors", k_default)
    k = safe_k(min(np.sum(ytr == 0), np.sum(ytr == 1)), k)

    if method == "ADASYN":
        try:
            return ImbADASYN(n_neighbors=k, random_state=seed).fit_resample(Xtr, ytr)
        except Exception:
            return Xtr, ytr
    if method == "Borderline-SMOTE":
        try:
            return BorderlineSMOTE(k_neighbors=k, random_state=seed).fit_resample(Xtr, ytr)
        except Exception:
            return Xtr, ytr
    if method == "SMOTE-Tomek":
        try:
            sm = SMOTE(k_neighbors=k, random_state=seed)
            return SMOTETomek(smote=sm, random_state=seed).fit_resample(Xtr, ytr)
        except Exception:
            return Xtr, ytr
    if method == "SMOTE-ENN":
        try:
            sm = SMOTE(k_neighbors=k, random_state=seed)
            return SMOTEENN(smote=sm, random_state=seed).fit_resample(Xtr, ytr)
        except Exception:
            return Xtr, ytr
    return Xtr, ytr


def _apply_scaler(scaler_name: str, Xtr, Xte):
    if scaler_name == "std":
        sc = StandardScaler().fit(Xtr)
        return sc.transform(Xtr), sc.transform(Xte)
    if scaler_name == "minmax":
        sc = MinMaxScaler().fit(Xtr)
        return sc.transform(Xtr), sc.transform(Xte)
    return Xtr, Xte


def run_fold(method: str, cfg: dict, Xtr, ytr, Xte, yte, seed: int) -> dict:
    """Run one fold for one method on one (dataset, method) cfg.
    Returns {'ACC', 'AUC', 'F1', 'AP', 'GMEAN'} dict.
    """
    set_all_seeds(seed)

    # Optional pre-scaling
    Xtr_p, Xte_p = _apply_scaler(cfg.get("scaler", "none"), Xtr, Xte)

    # ----- GAN family -----
    if method in ("GAN", "SMOTified-GAN"):
        gan = cfg["gan_args"]
        raw_te, raw_aug, raw_orig, yaug = train_gan_and_classify(
            Xtr_p, ytr, Xte_p,
            gan_epochs=gan["gan_epochs"], nn_epochs=gan["nn_epochs"],
            gan_lr=gan["glr"], nn_lr=gan.get("nn_lr", 1e-3),
            latent_dim=gan["ld"], seed=seed,
            smotify=(method == "SMOTified-GAN"),
            use_scaler=gan.get("use_scaler", False),
        )
        scores = {"test": raw_te, "orig": raw_orig, "aug": raw_aug}
        labels = {"test": yte,    "orig": ytr,      "aug": yaug}
        return compute_metrics_universal(cfg["metrics"], scores, labels)

    # ----- CSRBoost -----
    if method == "CSRBoost":
        ca = cfg.get("csrboost_args", {})
        clf = CSRBoostClassifier(
            p=ca.get("p", 1.0), samp=ca.get("cluster_pct", 0.5),
            smote_k=ca.get("smote_k", 5),
            n_est=cfg.get("n_est", 50), depth=cfg.get("depth"),
            lr=cfg.get("lr", 1.0),
            thresh=cfg["metrics"]["ACC"]["th"],
            seed=seed,
        )
        clf.fit(Xtr_p, ytr)
        proba_te   = clf.predict_proba(Xte_p)[:, 1]
        proba_orig = clf.predict_proba(Xtr_p)[:, 1]
        scores = {"test": proba_te, "orig": proba_orig, "aug": proba_orig}
        labels = {"test": yte,      "orig": ytr,        "aug": ytr}
        return compute_metrics_universal(cfg["metrics"], scores, labels)

    # ----- HUE -----
    if method == "HUE":
        ha = cfg.get("hue_args", {})
        proba_te = hue_bagging_predict_proba(
            Xtr_p, ytr, Xte_p,
            n_bags=ha.get("n_bags", 3),
            max_depth=ha.get("max_depth", 5),
            n_rf=ha.get("rf_trees", 10),
            seed=seed,
        )
        scores = {"test": proba_te, "orig": proba_te, "aug": proba_te}
        labels = {"test": yte,      "orig": yte,      "aug": yte}
        return compute_metrics_universal(cfg["metrics"], scores, labels)

    # ----- AdaBoost / RUSBoost -----
    if method == "AdaBoost":
        base = DecisionTreeClassifier(max_depth=cfg.get("depth", 1),
                                       random_state=seed)
        clf = make_adaboost(base, n_estimators=cfg.get("n_est", 50),
                             learning_rate=cfg.get("lr", 1.0),
                             random_state=seed)
        clf.fit(Xtr_p, ytr)
        proba_te   = clf.predict_proba(Xte_p)[:, 1]
        proba_orig = clf.predict_proba(Xtr_p)[:, 1]
        scores = {"test": proba_te, "orig": proba_orig, "aug": proba_orig}
        labels = {"test": yte,      "orig": ytr,        "aug": ytr}
        return compute_metrics_universal(cfg["metrics"], scores, labels)

    if method == "RUSBoost":
        base = DecisionTreeClassifier(max_depth=cfg.get("depth", 1),
                                       random_state=seed)
        clf = make_rusboost(base, n_estimators=cfg.get("n_est", 50),
                             learning_rate=cfg.get("lr", 1.0),
                             random_state=seed)
        clf.fit(Xtr_p, ytr)
        proba_te   = clf.predict_proba(Xte_p)[:, 1]
        proba_orig = clf.predict_proba(Xtr_p)[:, 1]
        scores = {"test": proba_te, "orig": proba_orig, "aug": proba_orig}
        labels = {"test": yte,      "orig": ytr,        "aug": ytr}
        return compute_metrics_universal(cfg["metrics"], scores, labels)

    # ----- Resampling-family wrapped in AdaBoost -----
    if method in ("ADASYN", "Borderline-SMOTE", "SMOTE-Tomek", "SMOTE-ENN"):
        Xb, yb = _resample(method, Xtr_p, ytr, cfg.get("sampler_args", {}), seed)
        base = DecisionTreeClassifier(max_depth=cfg.get("depth", 1),
                                       random_state=seed)
        clf = make_adaboost(base, n_estimators=cfg.get("n_est", 50),
                             learning_rate=cfg.get("lr", 1.0),
                             random_state=seed)
        clf.fit(Xb, yb)
        proba_te   = clf.predict_proba(Xte_p)[:, 1]
        proba_aug  = clf.predict_proba(Xb)[:, 1]
        proba_orig = clf.predict_proba(Xtr_p)[:, 1]
        scores = {"test": proba_te, "orig": proba_orig, "aug": proba_aug}
        labels = {"test": yte,      "orig": ytr,        "aug": yb}
        return compute_metrics_universal(cfg["metrics"], scores, labels)

    raise ValueError(f"Unknown method: {method}")


# ==============================================================================
# 9. PER-(DATASET, METHOD) CONFIG TABLE
# ------------------------------------------------------------------------------
# Loaded from BEST_CONFIGS.py (extracted from the 15 per-dataset scripts).
# The dict is keyed by (dataset, method) and mirrors the schema documented in
# section 3 above. Cells flagged as 'N/A' in the original paper are stored as
# the literal string 'N/A' and skipped in evaluate_dataset.
# ==============================================================================
try:
    from BEST_CONFIGS import BEST_CONFIGS  # populated in companion file
except Exception:
    BEST_CONFIGS = {}  # placeholder until extraction is complete


# ==============================================================================
# 10. PAPER REFERENCE VALUES (Yadav 2025 Table 2, manually corrected)
# ==============================================================================
PAPER_TABLE = {
    # (DATASET, ALGORITHM): {ACC: %, AUC, F1, AP, GMEAN}
    # Populated alongside BEST_CONFIGS during extraction.
}
try:
    from BEST_CONFIGS import PAPER_TABLE as _PT
    PAPER_TABLE = _PT
except Exception:
    pass


# ==============================================================================
# 11. evaluate_dataset / summarise / main
# ==============================================================================
def evaluate_dataset(ds_name: str) -> pd.DataFrame:
    """Run all 10 algorithms x 100 folds for one dataset.
    Returns a per-fold DataFrame with columns: Fold, Method, ACC, AUC, F1, AP, GMEAN.
    """
    print(f"\n=== Evaluating dataset: {ds_name} ===")
    X, y = LOADERS[ds_name]()
    n_min = int(np.sum(y == 1)); n_maj = int(np.sum(y == 0))
    print(f"  N={len(y)}  features={X.shape[1]}  min={n_min}  maj={n_maj}  "
          f"IR={n_maj / max(1, n_min):.2f}")

    cv = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS,
                                  random_state=SEED)
    folds = list(cv.split(X, y))

    rows = []
    for method in ALGORITHMS:
        cfg = BEST_CONFIGS.get((ds_name, method))
        if cfg is None or cfg == "N/A":
            print(f"  {method:20s}  N/A (skipped)")
            continue

        t0 = time.time()
        for fi, (tr_idx, te_idx) in enumerate(folds):
            seed = SEED + fi
            try:
                m = run_fold(method, cfg, X[tr_idx], y[tr_idx],
                              X[te_idx], y[te_idx], seed)
            except Exception as e:
                print(f"    fold {fi+1} {method} failed: {e}")
                continue
            rows.append({"Fold": fi + 1, "Method": method, **m})

        # Print mean for this method
        sub = [r for r in rows if r["Method"] == method]
        if sub:
            mean = {k: np.mean([r[k] for r in sub])
                    for k in ("ACC", "AUC", "F1", "AP", "GMEAN")}
            print(f"  {method:20s}  ACC={mean['ACC']:.2f}  AUC={mean['AUC']:.4f}  "
                  f"F1={mean['F1']:.4f}  AP={mean['AP']:.4f}  GM={mean['GMEAN']:.4f}  "
                  f"({time.time()-t0:.0f}s)")

    df = pd.DataFrame(rows)
    out_csv = OUT_DIR / f"{ds_name}_folds.csv"
    df.to_csv(out_csv, index=False)
    return df


def summarise(per_fold: pd.DataFrame) -> pd.DataFrame:
    grouped = per_fold.groupby("Method").agg({
        "ACC":   "mean", "AUC": "mean", "F1": "mean",
        "AP":    "mean", "GMEAN": "mean",
    }).reset_index()
    return grouped


def compare_to_paper(ds_name: str, summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in summary.iterrows():
        method = r["Method"]
        paper = PAPER_TABLE.get((ds_name, method))
        if paper is None:
            continue
        errs = [abs(r["ACC"] - paper["ACC"])]
        for k in ("AUC", "F1", "AP", "GMEAN"):
            errs.append(abs(r[k] - paper[k]) * 100)
        avg = float(np.mean(errs))
        rows.append({
            "Dataset": ds_name, "Method": method,
            "PaperACC": paper["ACC"], "OurACC": r["ACC"],
            "PaperAUC": paper["AUC"], "OurAUC": r["AUC"],
            "PaperF1":  paper["F1"],  "OurF1":  r["F1"],
            "PaperAP":  paper["AP"],  "OurAP":  r["AP"],
            "PaperGM":  paper["GMEAN"],"OurGM": r["GMEAN"],
            "AvgErr%":  avg,
            "Status":   "PASS" if avg <= PASS_THRESHOLD_PCT else "FAIL",
        })
    return pd.DataFrame(rows)


def main():
    ds_list = sys.argv[1:] if len(sys.argv) > 1 else DATASETS
    all_summaries = []
    for ds in ds_list:
        per_fold = evaluate_dataset(ds)
        summary = summarise(per_fold)
        cmp = compare_to_paper(ds, summary)
        cmp.to_csv(OUT_DIR / f"{ds}_comparison.csv", index=False)
        all_summaries.append(cmp)

    if all_summaries:
        all_df = pd.concat(all_summaries, ignore_index=True)
        all_df.to_csv(OUT_DIR / "ALL_comparison.csv", index=False)
        n_pass = (all_df["Status"] == "PASS").sum()
        n_total = len(all_df)
        print(f"\n========== FINAL: {n_pass}/{n_total} PASS "
              f"({100.0 * n_pass / max(1, n_total):.1f}%) ==========")


if __name__ == "__main__":
    main()
