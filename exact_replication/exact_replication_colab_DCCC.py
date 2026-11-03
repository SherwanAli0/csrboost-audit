#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================================
exact_replication_colab_DCCC.py
==============================================================================
Self-contained exact-as-prescribed replication for the DCCC dataset.
Works on Google Colab, Kaggle, and any local Python with PyTorch installed.

QUICKEST KAGGLE WORKFLOW (zero setup):
  1. New Kaggle notebook -> Settings -> Accelerator -> GPU T4.
  2. Add the DCCC dataset (Default of Credit Card Clients) to the notebook
     using "Add Data" in the right sidebar. The script auto-scans
     /kaggle/input/ for any DCCC.xls/.xlsx/.csv.
  3. Paste this entire file into a single notebook cell and click Run.
  4. Walk away. Checkpoints save every fold to /kaggle/working/.
     Resume by re-running the cell after a crash.

QUICKEST COLAB WORKFLOW:
  1. New Colab notebook -> Runtime -> GPU (T4 free tier is fine).
  2. Upload DCCC.xls to /content/ or /content/data/DCCC/ (or to Drive).
  3. Paste this entire file into a single cell and click Run.
  4. Drive auto-mounts; checkpoints land in
         /content/drive/MyDrive/exact_replication_DCCC/

WHAT THIS SCRIPT DOES:
  * mounts Google Drive on Colab (skips on Kaggle) and writes checkpoints
  * scans Kaggle inputs, Colab paths, and local directories for the dataset
  * auto-resumes from the last checkpoint after a crash
  * prints a heartbeat every method/fold to stay under the idle timeout
  * auto-installs missing pip packages
  * uses GPU automatically when available

GAN epochs default = 200 (matches local run).
For full Sharma 2022 prescription, set GAN_EPOCHS=2000 before running.

Algorithms (all 11 from CSRBoost paper Table 2):
    SMOTE, ADASYN, Borderline-SMOTE, SMOTE-Tomek, SMOTE-ENN, AdaBoost (T=100),
    RUSBoost (T=10), HUE, SMOTified-GAN, GAN, CSRBoost.

Evaluation : 5-fold StratifiedKFold x 20 repeats = 100 folds, threshold 0.5,
             AUC and AP from probabilities, F1 binary, seed=42.

Author : Sherwan Ali  |  Uskudar University  |  Sup: Dr. Gamze Uslu
==============================================================================
"""

from __future__ import annotations

# ==============================================================================
# COLAB SETUP CELL  --  runs first, before any heavy imports
# ==============================================================================
import os
import sys
import subprocess
import time

DATASET = "DCCC"

# 1. Choose checkpoint directory based on environment:
#      Colab  -> Google Drive (persistent across sessions)
#      Kaggle -> /kaggle/working/ (persistent if you click Save Version)
#      local  -> ./checkpoints_<DATASET>/
IN_COLAB = False
try:
    from google.colab import drive  # type: ignore
    if not os.path.exists("/content/drive/MyDrive"):
        drive.mount("/content/drive")
    DRIVE_DIR = f"/content/drive/MyDrive/exact_replication_{DATASET}"
    os.makedirs(DRIVE_DIR, exist_ok=True)
    IN_COLAB = True
    print(f"[SETUP] Colab + Drive mounted. Checkpoint dir: {DRIVE_DIR}", flush=True)
except Exception:
    if os.path.exists("/kaggle/working"):
        DRIVE_DIR = f"/kaggle/working/exact_replication_{DATASET}"
        os.makedirs(DRIVE_DIR, exist_ok=True)
        print(f"[SETUP] Kaggle. Checkpoint dir: {DRIVE_DIR} (Save Version to "
              f"persist across notebook closures)", flush=True)
    else:
        DRIVE_DIR = f"./checkpoints_{DATASET}"
        os.makedirs(DRIVE_DIR, exist_ok=True)
        print(f"[SETUP] Local run. Checkpoint dir: {DRIVE_DIR}", flush=True)

# 2. Auto-install dependencies (Colab + Kaggle; skip locally if user's env is set).
print("[SETUP] Installing/checking dependencies...", flush=True)
DEPS = ["imbalanced-learn", "scikit-learn", "pandas", "numpy", "xlrd>=2.0.1",
        "openpyxl"]
IN_KAGGLE = os.path.exists("/kaggle/input") or os.path.exists("/kaggle/working")
if IN_COLAB or IN_KAGGLE:
    for pkg in DEPS:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", pkg],
            check=False,
        )
print("[SETUP] Dependencies OK.", flush=True)

# 3. Detect GPU once and surface it as a global.
import torch  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[SETUP] PyTorch device: {DEVICE}", flush=True)
if DEVICE == "cuda":
    print(f"[SETUP] GPU: {torch.cuda.get_device_name(0)}", flush=True)

# ==============================================================================
# Heavy imports
# ==============================================================================
import math  # noqa: E402
import pickle  # noqa: E402
import shutil  # noqa: E402
import warnings  # noqa: E402
from datetime import datetime  # noqa: E402
from pathlib import Path  # noqa: E402
from time import perf_counter  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from imblearn.combine import SMOTEENN, SMOTETomek  # noqa: E402
from imblearn.ensemble import RUSBoostClassifier  # noqa: E402
from imblearn.over_sampling import ADASYN, SMOTE, BorderlineSMOTE  # noqa: E402
from imblearn.under_sampling import EditedNearestNeighbours, TomekLinks  # noqa: E402
from sklearn.cluster import KMeans  # noqa: E402  (vanilla KMeans per paper, n_init=10)
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.ensemble import AdaBoostClassifier  # noqa: E402
from sklearn.metrics import (accuracy_score, average_precision_score,  # noqa: E402
                             confusion_matrix, f1_score, roc_auc_score)
from sklearn.model_selection import RepeatedStratifiedKFold  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.tree import DecisionTreeClassifier  # noqa: E402
from sklearn.utils import check_random_state  # noqa: E402

warnings.filterwarnings("ignore")

# ==============================================================================
# CONFIG
# ==============================================================================
SEED = 42

# Search several plausible data locations -- helpful on Colab/Kaggle where the
# user may upload to /content/, /content/data/DCCC/, /kaggle/input/..., or Drive.
_DATA_CANDIDATES = [
    Path(os.environ.get("DCCC_DATA_DIR", "")) if os.environ.get("DCCC_DATA_DIR") else None,
    # Colab paths
    Path("/content/data/DCCC"),
    Path("/content/DCCC"),
    Path("/content/drive/MyDrive/data/DCCC"),
    Path("/content/drive/MyDrive/DCCC"),
    # Kaggle paths -- glob wildcards do not work in Path() literals so we
    # enumerate the most common Kaggle dataset prefixes at runtime below.
    # Generic local
    Path("./data/DCCC"),
    Path("./DCCC"),
    Path("."),
]
# Kaggle: scan /kaggle/input for any folder containing a DCCC .xls/.xlsx/.csv
_KAGGLE_INPUT = Path("/kaggle/input")
if _KAGGLE_INPUT.exists():
    for sub in _KAGGLE_INPUT.rglob("*"):
        if sub.is_dir():
            files = list(sub.glob("DCCC*.xls*")) + list(sub.glob("DCCC*.csv")) + \
                    list(sub.glob("dccc*.xls*")) + list(sub.glob("dccc*.csv"))
            if files:
                _DATA_CANDIDATES.insert(1, sub)  # high priority
                break
DATA_DIR = next((p for p in _DATA_CANDIDATES if p is not None and p.exists()),
                Path("./data/DCCC"))

OUT_DIR = Path(f"./outputs_{DATASET}")
OUT_DIR.mkdir(exist_ok=True)

N_SPLITS = 5
N_REPEATS = 20
SMOTE_K = 5
THRESHOLD = 0.5

T_ADABOOST = 100
T_RUSBOOST = 10
T_CSRBOOST = 50
T_SAMPLERS = 50
GAN_EPOCHS = int(os.environ.get("GAN_EPOCHS", "200"))

METHODS = [
    "CSRBoost",
    "SMOTifiedGAN",
    "GAN",
    "ADASYN",
    "Borderline-SMOTE",
    "SMOTE-Tomek",
    "SMOTE-ENN",
    "SMOTE",
    "AdaBoost",
    "RUSBoost",
    "HUE",
]

# CSRBoost paper Table 2 reported values for DCCC (hardcoded)
PAPER_TABLE2_DCCC = {
    "CSRBoost":         {"ACC": 68.32, "AUC": 0.64, "F1": 0.42, "AP": 0.29, "GMEAN": 0.62},
    "SMOTifiedGAN":     {"ACC": 81.68, "AUC": 0.95, "F1": 0.82, "AP": 0.88, "GMEAN": 0.82},
    "GAN":              {"ACC": 80.89, "AUC": 0.94, "F1": 0.81, "AP": 0.87, "GMEAN": 0.81},
    "ADASYN":           {"ACC": 72.27, "AUC": 0.62, "F1": 0.41, "AP": 0.29, "GMEAN": 0.59},
    "Borderline-SMOTE": {"ACC": 72.45, "AUC": 0.62, "F1": 0.41, "AP": 0.29, "GMEAN": 0.59},
    "SMOTE-Tomek":      {"ACC": 72.49, "AUC": 0.62, "F1": 0.41, "AP": 0.29, "GMEAN": 0.59},
    "SMOTE-ENN":        {"ACC": 70.52, "AUC": 0.65, "F1": 0.45, "AP": 0.31, "GMEAN": 0.64},
    "SMOTE":            {"ACC": 67.00, "AUC": 0.62, "F1": 0.41, "AP": 0.28, "GMEAN": 0.61},
    "AdaBoost":         {"ACC": 72.99, "AUC": 0.62, "F1": 0.41, "AP": 0.3, "GMEAN": 0.59},
    "RUSBoost":         {"ACC": None, "AUC": None, "F1": None, "AP": None, "GMEAN": None},
    "HUE":              {"ACC": 68.66, "AUC": 0.68, "F1": 0.49, "AP": 0.33, "GMEAN": 0.69},
}

# ==============================================================================
# CHECKPOINTING  --  resume-after-crash logic for Colab free tier
# ==============================================================================
CHECKPOINT_FILE = os.path.join(DRIVE_DIR, "checkpoint.pkl")


def _now() -> str:
    return time.strftime("%H:%M:%S")


def save_checkpoint(state: dict) -> None:
    """Atomically pickle the run state to Drive. Safe across crashes."""
    tmp = CHECKPOINT_FILE + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(state, f)
    os.replace(tmp, CHECKPOINT_FILE)
    print(f"[{_now()}] [CHECKPOINT] Saved at fold {state.get('fold_idx', '?')} "
          f"-> {CHECKPOINT_FILE}", flush=True)


def load_checkpoint() -> dict | None:
    """Return the previous run state if a valid checkpoint exists, else None."""
    if os.path.exists(CHECKPOINT_FILE):
        try:
            with open(CHECKPOINT_FILE, "rb") as f:
                state = pickle.load(f)
            print(f"[{_now()}] [RESUME] Loaded checkpoint at fold "
                  f"{state.get('fold_idx', '?')} from {CHECKPOINT_FILE}",
                  flush=True)
            return state
        except Exception as e:
            print(f"[{_now()}] [RESUME] Could not read checkpoint "
                  f"({type(e).__name__}). Starting fresh.", flush=True)
    return None


# ==============================================================================
# DATA LOADER -- DCCC
# ==============================================================================
def _to_float(df):
    return df.apply(pd.to_numeric, errors="coerce").fillna(0).values


def load_dccc():
    """DCCC loader: skip ID column, target is last column. Class==1 is minority."""
    xls = list(DATA_DIR.glob("*.xls")) + list(DATA_DIR.glob("*.xlsx"))
    csv = list(DATA_DIR.glob("*.csv")) if not xls else []
    if not xls and not csv:
        for p in [Path("/content/data/DCCC"), Path("/content/DCCC"),
                  Path("/content/drive/MyDrive/data/DCCC"),
                  Path("/content/drive/MyDrive/DCCC"),
                  Path("./data/DCCC"), Path("./DCCC"), Path(".")]:
            xls = list(p.glob("*.xls")) + list(p.glob("*.xlsx"))
            if xls:
                break
            csv = list(p.glob("*.csv"))
            if csv:
                break

    if xls:
        print(f"[{_now()}] [DATA] Loading {xls[0]}", flush=True)
        df = pd.read_excel(xls[0], header=1)
    elif csv:
        print(f"[{_now()}] [DATA] Loading {csv[0]}", flush=True)
        df = pd.read_csv(csv[0], header=1)
    else:
        raise FileNotFoundError(
            f"No DCCC .xls/.xlsx/.csv found. Looked in {DATA_DIR} and standard "
            f"fallbacks. On Kaggle: click 'Add Data' in the right sidebar and "
            f"add a DCCC dataset. On Colab: upload DCCC.xls to /content/data/DCCC/. "
            f"Or set DCCC_DATA_DIR in os.environ before running.")

    df = df.iloc[:, 1:]   # drop ID column
    y = pd.to_numeric(df.iloc[:, -1], errors="coerce").fillna(0).astype(int).values
    X = _to_float(df.iloc[:, :-1])
    if y.mean() > 0.5:
        y = 1 - y
    return X, y


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


def metrics(y_true, y_proba):
    y_pred = (y_proba >= THRESHOLD).astype(int)
    return {
        "ACC":   float(accuracy_score(y_true, y_pred)) * 100,
        "AUC":   safe_auc(y_true, y_proba),
        "F1":    float(f1_score(y_true, y_pred, zero_division=0, average="binary")),
        "AP":    safe_ap(y_true, y_proba),
        "GMEAN": gmean_at(y_true, y_pred),
    }


# ==============================================================================
# CLASSIFIERS / RESAMPLERS (source-paper prescribed)
# ==============================================================================
def make_adaboost_freund1996(rs, n_estimators=T_ADABOOST):
    base = DecisionTreeClassifier(max_depth=1, random_state=rs)
    kwargs = {"estimator": base, "n_estimators": n_estimators,
              "learning_rate": 1.0, "random_state": rs}
    import sklearn
    ver = tuple(int(x) for x in sklearn.__version__.split(".")[:2])
    if ver < (1, 6):
        kwargs["algorithm"] = "SAMME"
    if ver < (1, 2):
        kwargs["base_estimator"] = kwargs.pop("estimator")
    return AdaBoostClassifier(**kwargs)


def make_adaboost_csrboost(rs, n_estimators=T_CSRBOOST):
    base = DecisionTreeClassifier(random_state=rs)
    kwargs = {"estimator": base, "n_estimators": n_estimators,
              "learning_rate": 1.0, "random_state": rs}
    import sklearn
    ver = tuple(int(x) for x in sklearn.__version__.split(".")[:2])
    if ver < (1, 6):
        kwargs["algorithm"] = "SAMME"
    if ver < (1, 2):
        kwargs["base_estimator"] = kwargs.pop("estimator")
    return AdaBoostClassifier(**kwargs)


def make_adaboost_sampler_wrapper(rs, n_estimators=T_SAMPLERS):
    base = DecisionTreeClassifier(random_state=rs)
    kwargs = {"estimator": base, "n_estimators": n_estimators,
              "learning_rate": 1.0, "random_state": rs}
    import sklearn
    ver = tuple(int(x) for x in sklearn.__version__.split(".")[:2])
    if ver < (1, 6):
        kwargs["algorithm"] = "SAMME"
    if ver < (1, 2):
        kwargs["base_estimator"] = kwargs.pop("estimator")
    return AdaBoostClassifier(**kwargs)


def make_rusboost_seiffert2010(rs, n_estimators=T_RUSBOOST):
    # Seiffert 2010: unpruned C4.5-style trees (no max_depth limit).
    base = DecisionTreeClassifier(random_state=rs)
    try:
        return RUSBoostClassifier(estimator=base, n_estimators=n_estimators,
                                  learning_rate=1.0, sampling_strategy=1.0,
                                  random_state=rs)
    except TypeError:
        return RUSBoostClassifier(base_estimator=base, n_estimators=n_estimators,
                                  learning_rate=1.0, sampling_strategy=1.0,
                                  random_state=rs)


def safe_k(n_min, k=SMOTE_K):
    return max(1, min(k, max(1, n_min - 1)))


def make_smote(rs, n_min):
    return SMOTE(k_neighbors=safe_k(n_min), random_state=rs)


def make_adasyn(rs, n_min):
    return ADASYN(n_neighbors=safe_k(n_min), random_state=rs)


def make_borderline_smote(rs, n_min):
    return BorderlineSMOTE(k_neighbors=safe_k(n_min), m_neighbors=10,
                           kind="borderline-1", random_state=rs)


def make_smote_tomek(rs, n_min):
    return SMOTETomek(smote=SMOTE(k_neighbors=safe_k(n_min), random_state=rs),
                      tomek=TomekLinks(sampling_strategy="all"),
                      random_state=rs)


def make_smote_enn(rs, n_min):
    return SMOTEENN(smote=SMOTE(k_neighbors=safe_k(n_min), random_state=rs),
                    enn=EditedNearestNeighbours(n_neighbors=3, kind_sel="all",
                                                sampling_strategy="all"),
                    random_state=rs)


def fit_resample_safe(sampler, X, y):
    n_min = int(np.sum(y == 1))
    if n_min < 2:
        return X, y
    try:
        Xr, yr = sampler.fit_resample(X, y)
        if len(np.unique(yr)) < 2:
            return X, y
        return Xr, yr
    except Exception:
        return X, y


# ==============================================================================
# CSRBoost
# ==============================================================================
def csrboost_fit_predict(X_tr, y_tr, X_te, rs):
    X_min = X_tr[y_tr == 1]
    X_maj = X_tr[y_tr == 0]
    n_min = len(X_min)
    n_maj = len(X_maj)
    K = max(1, min(n_min, n_maj))
    rng = np.random.RandomState(rs)
    km = KMeans(n_clusters=K, n_init=10, random_state=rs)
    labels = km.fit_predict(X_maj)
    kept = []
    for k in range(K):
        cidx = np.where(labels == k)[0]
        n_keep = max(1, math.ceil(len(cidx) * 0.50))
        if n_keep >= len(cidx):
            kept.extend(cidx.tolist())
        else:
            kept.extend(rng.choice(cidx, n_keep, replace=False).tolist())
    X_ret = np.vstack([X_min, X_maj[kept]])
    y_ret = np.concatenate([np.ones(len(X_min)), np.zeros(len(kept))]).astype(int)
    if int(np.sum(y_ret == 1)) > int(np.sum(y_ret == 0)):
        y_ret = 1 - y_ret
    Xb, yb = fit_resample_safe(make_smote(rs, int(np.sum(y_ret == 1))),
                               X_ret, y_ret)
    clf = make_adaboost_csrboost(rs)
    clf.fit(Xb, yb)
    return clf.predict_proba(X_te)[:, 1]


# ==============================================================================
# HUE (Ng 2022)
# ==============================================================================
def hue_fit_predict(X_tr, y_tr, X_te, rs, itq_iters=50):
    rng = check_random_state(rs)
    X_min = X_tr[y_tr == 1]
    X_maj = X_tr[y_tr == 0]
    n_min = len(X_min)
    n_maj = len(X_maj)
    if n_min < 1 or n_maj < 1:
        clf = DecisionTreeClassifier(random_state=rs)
        clf.fit(X_tr, y_tr)
        if len(clf.classes_) < 2:
            return np.full(len(X_te), float(clf.classes_[0]))
        return clf.predict_proba(X_te)[:, 1]
    B = max(1, int(math.ceil(math.log2(3.0 * n_maj / max(n_min, 1)))))
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
        Z = V @ R
        Bits = np.where(Z >= 0, 1.0, -1.0)
        U, _, VT = np.linalg.svd(Bits.T @ V, full_matrices=False); R = U @ VT
    maj_bits = (V @ R >= 0).astype(np.uint8)
    weights_arr = (1 << np.arange(B, dtype=np.uint32))
    maj_codes = (maj_bits.astype(np.uint32) * weights_arr).sum(axis=1)
    estimators = []
    for ref in range(n_sub):
        xor = np.bitwise_xor(maj_codes, np.uint32(ref))
        d = np.array([bin(int(v)).count("1") for v in xor], dtype=np.int32)
        w = np.where(d == 0, 1.0, 1.0 / (d * n_sub))
        w = w / w.sum()
        idx = rng.choice(n_maj, size=n_min, replace=True, p=w)
        Xs = np.vstack([X_min, X_maj[idx]])
        ys = np.hstack([np.ones(n_min), np.zeros(n_min)]).astype(int)
        clf = DecisionTreeClassifier(criterion="gini",
                                     random_state=rng.randint(0, 1_000_000))
        clf.fit(Xs, ys)
        estimators.append(clf)
    # Paper Algorithm 1 step 3: H(x) = sign(sum_i C_i(x)) where C_i in {-1,+1}.
    # We use the unsigned weighted sum as the continuous score for AUC/AP and the
    # sign convention for hard prediction; both come from the same vote tally.
    score = np.zeros(len(X_te))
    for est in estimators:
        if len(est.classes_) < 2:
            vote = np.full(len(X_te), 1.0 if int(est.classes_[0]) == 1 else -1.0)
        else:
            pred = est.predict(X_te)
            vote = np.where(pred == 1, 1.0, -1.0)
        score += vote
    return (score / len(estimators) + 1.0) / 2.0


# ==============================================================================
# SMOTified-GAN / GAN (Sharma 2022)  --  uses global DEVICE for GPU on Colab
# ==============================================================================
class GANResult:
    def __init__(self, ok, prob):
        self.ok = ok
        self.prob = prob


def gan_fit_predict(X_tr, y_tr, X_te, rs, mode="smotified", epochs=GAN_EPOCHS,
                    fold_i=None, ds_name=None):
    """GAN/SMOTified-GAN training. Heartbeats every ~5 percent epochs to keep Colab alive."""
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import DataLoader, TensorDataset
    except Exception:
        return GANResult(False, None)
    torch.manual_seed(rs)
    np.random.seed(rs)
    dev = torch.device(DEVICE)
    n_features = X_tr.shape[1]
    n_min = int(np.sum(y_tr == 1))
    n_maj = int(np.sum(y_tr == 0))
    n_generate = n_maj - n_min
    if n_min < 2 or n_generate <= 0:
        return GANResult(False, None)

    Xmin = X_tr[y_tr == 1].astype(np.float32)
    if mode == "smotified":
        from sklearn.neighbors import NearestNeighbors
        rng = np.random.RandomState(rs)
        k = min(SMOTE_K, len(Xmin) - 1)
        nn_model = NearestNeighbors(n_neighbors=k + 1).fit(Xmin)
        pool_size = max(n_generate * 2, 2048)
        pool = []
        for _ in range(pool_size):
            i = rng.randint(0, len(Xmin))
            neigh = nn_model.kneighbors(Xmin[i:i+1], return_distance=False)[0][1:]
            j = rng.choice(neigh)
            alpha = rng.rand()
            pool.append(Xmin[i] + alpha * (Xmin[j] - Xmin[i]))
        smote_pool = np.array(pool, dtype=np.float32)
        latent_dim = n_features

        def sample_latent(n):
            idx = np.random.choice(len(smote_pool), n, replace=True)
            return smote_pool[idx]
    else:
        latent_dim = n_features

        def sample_latent(n):
            return np.random.randn(n, latent_dim).astype(np.float32)

    g_hidden = [128, 256, 512, 1024]
    layers = []; in_dim = latent_dim
    for h in g_hidden:
        layers += [nn.Linear(in_dim, h), nn.BatchNorm1d(h), nn.ReLU()]
        in_dim = h
    layers += [nn.Linear(in_dim, n_features), nn.Tanh()]
    G = nn.Sequential(*layers).to(dev)

    d_hidden = [512, 256, 128]
    layers = []; in_dim = n_features
    for h in d_hidden:
        layers += [nn.Linear(in_dim, h), nn.LeakyReLU(0.2)]
        in_dim = h
    layers += [nn.Linear(in_dim, 1)]
    D = nn.Sequential(*layers).to(dev)

    g_opt = optim.Adam(G.parameters(), lr=1e-5, betas=(0.9, 0.999))
    d_opt = optim.Adam(D.parameters(), lr=1e-5, betas=(0.9, 0.999))
    bce = nn.BCEWithLogitsLoss()

    batch_size = min(128, len(Xmin))
    Xmin_t = torch.FloatTensor(Xmin)
    dataset = TensorDataset(Xmin_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    G.train(); D.train()
    hb_every = max(1, epochs // 20)
    gan_t0 = time.time()
    for ep in range(epochs):
        for (real_batch,) in loader:
            real_batch = real_batch.to(dev)
            bs = real_batch.size(0)
            d_opt.zero_grad()
            d_real = bce(D(real_batch), torch.ones(bs, 1, device=dev))
            z = torch.FloatTensor(sample_latent(bs)).to(dev)
            fake = G(z)
            d_fake = bce(D(fake.detach()), torch.zeros(bs, 1, device=dev))
            (d_real + d_fake).backward()
            d_opt.step()
            g_opt.zero_grad()
            z = torch.FloatTensor(sample_latent(bs)).to(dev)
            g_loss = bce(D(G(z)), torch.ones(bs, 1, device=dev))
            g_loss.backward()
            g_opt.step()
        if (ep + 1) % hb_every == 0 or ep == epochs - 1:
            tag = f"fold {fold_i+1}" if fold_i is not None else ""
            ds_tag = ds_name or DATASET
            print(f"[{_now()}] [HB] {mode}-GAN {ds_tag} {tag} "
                  f"epoch {ep+1}/{epochs} dev={DEVICE} "
                  f"({time.time()-gan_t0:.0f}s)", flush=True)

    G.eval()
    with torch.no_grad():
        z = torch.FloatTensor(sample_latent(n_generate)).to(dev)
        synth = G(z).cpu().numpy().astype(np.float32)

    X_aug = np.vstack([X_tr, synth])
    y_aug = np.hstack([y_tr, np.ones(len(synth), dtype=int)])

    class CLF(nn.Module):
        def __init__(self, in_dim):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim, 256), nn.ReLU(),
                nn.Linear(256, 128), nn.ReLU(),
                nn.Linear(128, 1),
            )

        def forward(self, x):
            return self.net(x)

    C = CLF(n_features).to(dev)
    c_opt = optim.Adam(C.parameters(), lr=1e-5)
    mae = nn.L1Loss()
    Xa_t = torch.FloatTensor(X_aug); ya_t = torch.FloatTensor(y_aug).unsqueeze(1)
    cds = TensorDataset(Xa_t, ya_t)
    cloader = DataLoader(cds, batch_size=128, shuffle=True)
    C.train()
    for _ in range(30):
        for xb, yb in cloader:
            xb, yb = xb.to(dev), yb.to(dev)
            c_opt.zero_grad()
            out = torch.sigmoid(C(xb))
            loss = mae(out, yb)
            loss.backward()
            c_opt.step()

    C.eval()
    with torch.no_grad():
        prob = torch.sigmoid(C(torch.FloatTensor(X_te).to(dev))).cpu().numpy().ravel()
    return GANResult(True, prob)


# ==============================================================================
# DISPATCH
# ==============================================================================
def run_method(name, X_tr, y_tr, X_te, y_te, rs, fold_i=None):
    nan_metrics = {"ACC": float("nan"), "AUC": float("nan"),
                   "F1": float("nan"), "AP": float("nan"),
                   "GMEAN": float("nan")}
    n_min = int(np.sum(y_tr == 1))

    if name == "CSRBoost":
        return metrics(y_te, csrboost_fit_predict(X_tr, y_tr, X_te, rs))
    if name == "SMOTifiedGAN":
        res = gan_fit_predict(X_tr, y_tr, X_te, rs, mode="smotified",
                              fold_i=fold_i, ds_name=DATASET)
        return metrics(y_te, res.prob) if res.ok else nan_metrics
    if name == "GAN":
        res = gan_fit_predict(X_tr, y_tr, X_te, rs, mode="gan",
                              fold_i=fold_i, ds_name=DATASET)
        return metrics(y_te, res.prob) if res.ok else nan_metrics
    if name == "ADASYN":
        Xr, yr = fit_resample_safe(make_adasyn(rs, n_min), X_tr, y_tr)
    elif name == "Borderline-SMOTE":
        Xr, yr = fit_resample_safe(make_borderline_smote(rs, n_min), X_tr, y_tr)
    elif name == "SMOTE-Tomek":
        Xr, yr = fit_resample_safe(make_smote_tomek(rs, n_min), X_tr, y_tr)
    elif name == "SMOTE-ENN":
        Xr, yr = fit_resample_safe(make_smote_enn(rs, n_min), X_tr, y_tr)
    elif name == "SMOTE":
        Xr, yr = fit_resample_safe(make_smote(rs, n_min), X_tr, y_tr)
    elif name == "AdaBoost":
        clf = make_adaboost_freund1996(rs); clf.fit(X_tr, y_tr)
        return metrics(y_te, clf.predict_proba(X_te)[:, 1])
    elif name == "RUSBoost":
        clf = make_rusboost_seiffert2010(rs); clf.fit(X_tr, y_tr)
        return metrics(y_te, clf.predict_proba(X_te)[:, 1])
    elif name == "HUE":
        return metrics(y_te, hue_fit_predict(X_tr, y_tr, X_te, rs))
    else:
        raise ValueError(f"Unknown method: {name}")

    clf = make_adaboost_sampler_wrapper(rs); clf.fit(Xr, yr)
    return metrics(y_te, clf.predict_proba(X_te)[:, 1])


# ==============================================================================
# RUNNER
# ==============================================================================
def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print("=" * 100, flush=True)
    print(f"  EXACT-AS-PRESCRIBED REPLICATION  --  {DATASET} (Colab free-tier hardened)",
          flush=True)
    print("=" * 100, flush=True)
    print(f"  Methods    : {', '.join(METHODS)}", flush=True)
    print(f"  CV         : {N_SPLITS}-fold x {N_REPEATS} repeats = "
          f"{N_SPLITS * N_REPEATS} folds", flush=True)
    print(f"  GAN epochs : {GAN_EPOCHS} (matches local run; Sharma 2022 prescribes 2000 but the original CSRBoost paper's reported runtime indicates effective use of ~100 epochs)", flush=True)
    print(f"  Device     : {DEVICE}", flush=True)
    print(f"  Data dir   : {DATA_DIR}", flush=True)
    print(f"  Output dir : {OUT_DIR}", flush=True)
    print(f"  Drive dir  : {DRIVE_DIR}", flush=True)
    print("=" * 100, flush=True)

    print(f"\n[{_now()}] Loading {DATASET} data...", flush=True)
    X, y = load_dccc()
    n_min = int(y.sum()); n_maj = int((1 - y).sum())
    print(f"[{_now()}] n={len(y)}  features={X.shape[1]}  "
          f"IR={n_maj/max(n_min,1):.2f}  minority={n_min}  majority={n_maj}",
          flush=True)

    splitter = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS,
                                       random_state=SEED)
    total_folds = N_SPLITS * N_REPEATS

    # ----------- Resume from checkpoint if one exists ------------------------
    state = load_checkpoint() or {
        "fold_idx": -1,
        "fold_rows": [],
        "per_method": {m: [] for m in METHODS},
        "timestamp": timestamp,
    }
    timestamp = state.get("timestamp", timestamp)
    start_fold = state["fold_idx"] + 1
    fold_rows = state["fold_rows"]
    per_method = state["per_method"]
    for m in METHODS:
        per_method.setdefault(m, [])

    if start_fold > 0:
        print(f"[{_now()}] [RESUME] Skipping first {start_fold} folds (already done).",
              flush=True)
    run_start = time.time()

    print(f"\n[{_now()}] Beginning folds {start_fold+1}..{total_folds}. "
          f"This may take several hours; checkpoints save to Drive after every fold.",
          flush=True)
    for fold_i, (tr_idx, te_idx) in enumerate(splitter.split(X, y)):
        if fold_i < start_fold:
            continue

        fold_start = time.time()
        print(f"[{_now()}] [FOLD] Starting fold {fold_i+1}/{total_folds} "
              f"(train n={len(tr_idx)}, test n={len(te_idx)})", flush=True)

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[tr_idx])
        X_te = scaler.transform(X[te_idx])
        y_tr, y_te = y[tr_idx], y[te_idx]

        for m in METHODS:
            t_m0 = time.time()
            print(f"[{_now()}] [HB] training {m} on {DATASET} fold {fold_i+1}",
                  flush=True)
            try:
                result = run_method(m, X_tr, y_tr, X_te, y_te, SEED, fold_i=fold_i)
                status = "ok"
            except Exception as e:
                result = {k: float("nan") for k in ("ACC", "AUC", "F1", "AP", "GMEAN")}
                status = f"ERR:{type(e).__name__}"
            per_method[m].append(result)
            fold_rows.append({"dataset": DATASET, "method": m, "fold": fold_i,
                              "status": status, **result})
            print(f"[{_now()}] [HB] {m} fold {fold_i+1} done "
                  f"({time.time()-t_m0:.0f}s)  status={status}", flush=True)

        fold_elapsed = time.time() - fold_start
        elapsed_total = (time.time() - run_start) / 60
        remaining_folds = total_folds - fold_i - 1
        completed_in_run = fold_i - start_fold + 1
        per_fold_avg = (time.time() - run_start) / max(completed_in_run, 1)
        eta_min = remaining_folds * per_fold_avg / 60
        print(f"[{_now()}] [PROGRESS] Fold {fold_i+1}/{total_folds} done "
              f"({fold_elapsed:.0f}s)  Total: {elapsed_total:.1f}m  "
              f"ETA: {eta_min:.0f}m  Run alive", flush=True)

        # Save state every fold (covers the 10-15 min user constraint easily).
        state = {
            "fold_idx": fold_i,
            "fold_rows": fold_rows,
            "per_method": per_method,
            "timestamp": timestamp,
            "last_save": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        save_checkpoint(state)

        partial_csv = os.path.join(DRIVE_DIR,
                                   f"fold_results_{DATASET}_{timestamp}_partial.csv")
        try:
            pd.DataFrame(fold_rows).to_csv(partial_csv, index=False)
        except Exception as e:
            print(f"[{_now()}] [WARN] partial CSV write failed: {e}", flush=True)

    # ==========================================================================
    # Final outputs
    # ==========================================================================
    fold_path = OUT_DIR / f"fold_results_{DATASET}_{timestamp}.csv"
    sum_path = OUT_DIR / f"dataset_summary_{DATASET}_{timestamp}.csv"
    cmp_path = OUT_DIR / f"comparison_to_paper_{DATASET}_{timestamp}.csv"
    pd.DataFrame(fold_rows).to_csv(fold_path, index=False)

    summary = {"dataset": DATASET}
    for m in METHODS:
        for k in ("ACC", "AUC", "F1", "AP", "GMEAN"):
            vals = [f[k] for f in per_method[m] if not np.isnan(f.get(k, float("nan")))]
            summary[f"{m}_{k}_mean"] = float(np.mean(vals)) if vals else float("nan")
            summary[f"{m}_{k}_std"] = float(np.std(vals)) if vals else float("nan")
    pd.DataFrame([summary]).to_csv(sum_path, index=False)

    cmp_rows = []
    for m in METHODS:
        ours = {k: float(np.nanmean([f[k] for f in per_method[m]]))
                if per_method[m] else float("nan")
                for k in ("ACC", "AUC", "F1", "AP", "GMEAN")}
        p = PAPER_TABLE2_DCCC.get(m, {})
        cmp_rows.append({"dataset": DATASET, "method": m,
                         **{f"{k}_paper": p.get(k) for k in ("ACC", "AUC", "F1", "AP", "GMEAN")},
                         **{f"{k}_ours":  ours[k] for k in ("ACC", "AUC", "F1", "AP", "GMEAN")}})
    pd.DataFrame(cmp_rows).to_csv(cmp_path, index=False)

    # Comparison printout
    print(f"\n[{_now()}] COMPARISON vs paper Table 2 -- {DATASET}", flush=True)
    hdr = f"  {'METHOD':<18} {'ACC pap/ours':<22} {'AUC pap/ours':<18} " \
          f"{'F1 pap/ours':<18} {'AP pap/ours':<18} {'GMEAN pap/ours':<18}"
    print(hdr, flush=True); print("  " + "-" * (len(hdr) - 2), flush=True)
    for m in METHODS:
        p = PAPER_TABLE2_DCCC.get(m, {})
        ours = {k: float(np.nanmean([f[k] for f in per_method[m]]))
                if per_method[m] else float("nan")
                for k in ("ACC", "AUC", "F1", "AP", "GMEAN")}

        def fmt(x):
            return "  N/A " if x is None or (isinstance(x, float) and np.isnan(x)) \
                else f"{x:.2f}"
        print(f"  {m:<18} "
              f"{fmt(p.get('ACC'))}/{fmt(ours['ACC']):<10} "
              f"{fmt(p.get('AUC'))}/{fmt(ours['AUC']):<8} "
              f"{fmt(p.get('F1'))}/{fmt(ours['F1']):<8} "
              f"{fmt(p.get('AP'))}/{fmt(ours['AP']):<8} "
              f"{fmt(p.get('GMEAN'))}/{fmt(ours['GMEAN']):<8}", flush=True)

    # Mirror final outputs to Drive so the user can grab them after Colab dies.
    try:
        drive_outputs = os.path.join(DRIVE_DIR, "final_outputs")
        if os.path.exists(drive_outputs):
            shutil.rmtree(drive_outputs)
        shutil.copytree(str(OUT_DIR), drive_outputs)
        print(f"[{_now()}] [FINAL] Outputs copied to {drive_outputs}", flush=True)
    except Exception as e:
        print(f"[{_now()}] [WARN] could not copy outputs to Drive: {e}",
              flush=True)

    elapsed = time.time() - run_start
    print("\n" + "=" * 100, flush=True)
    print(f"  Total elapsed (this run) : {elapsed/60:.1f} minutes "
          f"({elapsed/3600:.2f} hours)", flush=True)
    print(f"  Fold CSV      : {fold_path}", flush=True)
    print(f"  Summary CSV   : {sum_path}", flush=True)
    print(f"  Comparison    : {cmp_path}", flush=True)
    print(f"  Drive mirror  : {DRIVE_DIR}/final_outputs", flush=True)
    print("=" * 100, flush=True)


if __name__ == "__main__":
    main()
