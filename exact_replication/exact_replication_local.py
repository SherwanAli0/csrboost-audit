#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================================
exact_replication_local.py
==============================================================================
EXACT-AS-PRESCRIBED replication of the 10 algorithms reported in the CSRBoost
paper (Yadav et al., IEEE Access 2025). Each algorithm follows its source-paper
prescription where the paper specifies it, and standard library defaults applied
uniformly (identical for every method, with no tuning toward Table 2) where the
CSRBoost paper is silent - e.g. base-tree depth, boosting iteration count, feature
scaling, and decision threshold.

This script runs LOCALLY on 13 of the 15 CSRBoost datasets. The two largest
datasets (ESR and DCCC) are run separately on Colab via:
    exact_replication_colab_ESR.py
    exact_replication_colab_DCCC.py

Algorithms (EXACT source-paper prescription, no FIX, no per-method threshold
tuning, no target-value sweeps):

    01 SMOTE             (Chawla 2002)            k=5
    02 ADASYN            (He 2008)                K=5, beta=1.0
    03 Borderline-SMOTE  (Han 2005)               k=5, m=10, kind='borderline-1'
    04 SMOTE-Tomek       (Batista 2004)           SMOTE k=5 + TomekLinks
    05 SMOTE-ENN         (Batista 2004)           SMOTE k=5 + ENN n=3
    06 AdaBoost          (Freund 1996)            T=50, unpruned decision-tree base (max_depth=None)
    07 RUSBoost          (Seiffert 2010)          T=50, N=50% via imblearn
    08 HUE               (Ng 2022)                B=ceil(log2(3*Nmaj/Nmin)), ITQ
    09 SMOTified-GAN     (Sharma 2022)            SMOTE+GAN, lr=1e-5, batch=128
    10 CSRBoost          (Yadav 2025)             KMeans+50% RUS+SMOTE+AdaBoost(T=50)

Wrapped classifier for SMOTE-family (01-05): each oversampler is followed by an
AdaBoost(T=50) classifier - this is a PROJECT convention (the CSRBoost paper does
NOT state the SMOTE-family baselines are wrapped in AdaBoost) for the
sampling baselines. The boosting-paper prescriptions (06, 07) use their OWN T.

Evaluation protocol (uniform across all 10 algorithms):
    5-fold StratifiedKFold repeated 20 times = 100 folds
    Threshold 0.5 for all threshold-dependent metrics
    AUC and AP computed from probability scores (modern standard)
    F1 with average='binary'
    StandardScaler fit on train fold, applied to test fold
    Random seed = 42

Datasets (13): PSDAS, BCW, CB, ESDRP, ILPD, GLASS, SEED, WINE, YEAST5-ERL,
               CARGOOD, CARVGOOD, YEAST5, FLARE-F.

Author : Sherwan Ali
Univ   : Uskudar University
Sup    : Dr. Gamze Uslu
==============================================================================
"""

from __future__ import annotations

import argparse
import math
import warnings
from datetime import datetime
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.ensemble import RUSBoostClassifier
from imblearn.over_sampling import ADASYN, SMOTE, BorderlineSMOTE
from imblearn.under_sampling import EditedNearestNeighbours, TomekLinks
from sklearn.cluster import KMeans
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
# GLOBAL CONFIG
# ==============================================================================
SEED         = 42
SCRIPT_DIR   = Path(__file__).parent
REPL_ROOT    = SCRIPT_DIR.parent
# After 2026-05-04 reorganization: dataset folders live at root as
# 01_PSDAS, 02_ESR, ... 15_FLARE-F. DATA_DIR points at the project root;
# per-loader paths use the numbered folder names.
DATA_DIR     = REPL_ROOT
OUT_DIR      = REPL_ROOT / "outputs" / "run_2026-05-05_local_full"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR = OUT_DIR / "_checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

N_SPLITS     = 5
N_REPEATS    = 20
SMOTE_K      = 5
THRESHOLD    = 0.5

# Per-algorithm prescribed iteration counts
T_ADABOOST   = 50    # Uniform project default (sklearn n_estimators=50, unpruned base); paper pins no T/depth
T_RUSBOOST   = 50    # Uniform project default; paper pins no RUSBoost T (Seiffert's original used 10)
T_CSRBOOST   = 50    # Uniform project default (sklearn AdaBoost T=50); paper pins no T
T_SAMPLERS   = 50    # AdaBoost wrapping SMOTE-family (PROJECT convention; not stated by the paper)

# GAN training epochs - 200 fixed (no early stopping). The val-F1 early-stopping
# logic in gan_fit_predict is preserved but disabled by default; flip the flag
# below to True to re-enable it.
GAN_EPOCHS_LOCAL = 200
GAN_EARLY_STOPPING_ENABLED = False
GAN_EARLY_STOP_PATIENCE = 5
GAN_WARMUP_EPOCHS = 50  # only used when GAN_EARLY_STOPPING_ENABLED = True

# DCCC-only pass. The other 14 datasets are already complete in the same
# OUT_DIR (PSDAS via psdas_recovered_*.csv; the remaining 13 from the
# 2026-05-06 local run with full per_dataset/ checkpoints).
DATASETS = ["DCCC"]

# Algorithms in CSRBoost paper Table 2 order
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

# Paper Section III-D explicit N/A entries
# - ADASYN, GAN, SMOTified-GAN on CB and YEAST5-ERL ("no neighbours" - minority
#   too small for k-nearest-neighbour synthesis).
# - RUSBoost on PSDAS, ESR, DCCC (time-series datasets, RUSBoost loses too much
#   information via random undersampling).
# These skip sets are enforced at runtime so the raw fold_results CSV reflects
# the paper N/A status (NaN with status="skip(N/A)") instead of relying on the
# downstream markdown builder to mask numbers.
SKIP_ADASYN_DATASETS = {"CB", "YEAST5ERL"}
# GAN and SMOTified-GAN are N/A on YEAST5ERL only (paper Table 2 actually
# reports values for them on CB despite the Section III-D text saying otherwise).
SKIP_GAN_DATASETS = {"YEAST5ERL"}
SKIP_RUSBOOST_DATASETS = {"PSDAS", "ESR", "DCCC"}

# ==============================================================================
# CSRBoost paper Table 2 reported values (hardcoded -- same values as the
# existing replication scripts use). Used for a side-by-side comparison.
# Values are: ACC (percent), AUC, F1, AP, GMEAN.
# ==============================================================================
PAPER_TABLE2 = {
    "PSDAS": {
        "CSRBoost": {"ACC": 72.85, "AUC": 0.66, "F1": 0.4, "AP": 0.25, "GMEAN": 0.63},
        "SMOTifiedGAN": {"ACC": 64.46, "AUC": 0.82, "F1": 0.64, "AP": 0.7, "GMEAN": 0.32},
        "GAN": {"ACC": 63.8, "AUC": 0.82, "F1": 0.64, "AP": 0.69, "GMEAN": 0.32},
        "ADASYN": {"ACC": 76.11, "AUC": 0.64, "F1": 0.38, "AP": 0.25, "GMEAN": 0.58},
        "Borderline-SMOTE": {"ACC": 75.99, "AUC": 0.62, "F1": 0.38, "AP": 0.25, "GMEAN": 0.59},
        "SMOTE-Tomek": {"ACC": 75.99, "AUC": 0.61, "F1": 0.39, "AP": 0.26, "GMEAN": 0.59},
        "SMOTE-ENN": {"ACC": 72.67, "AUC": 0.65, "F1": 0.42, "AP": 0.27, "GMEAN": 0.64},
        "AdaBoost": {"ACC": 75.72, "AUC": 0.65, "F1": 0.38, "AP": 0.25, "GMEAN": 0.58},
        "RUSBoost": {"ACC": None, "AUC": None, "F1": None, "AP": None, "GMEAN": None},
        "HUE": {"ACC": 74.25, "AUC": 0.71, "F1": 0.49, "AP": 0.32, "GMEAN": 0.74},
    },
    "ESR": {
        "CSRBoost": {"ACC": 92.05, "AUC": 0.9, "F1": 0.8, "AP": 0.67, "GMEAN": 0.89},
        "SMOTifiedGAN": {"ACC": 95.84, "AUC": 0.97, "F1": 0.96, "AP": 0.95, "GMEAN": 0.92},
        "GAN": {"ACC": 95.89, "AUC": 0.97, "F1": 0.96, "AP": 0.95, "GMEAN": 0.92},
        "ADASYN": {"ACC": 89.95, "AUC": 0.89, "F1": 0.77, "AP": 0.63, "GMEAN": 0.89},
        "Borderline-SMOTE": {"ACC": 91.93, "AUC": 0.89, "F1": 0.82, "AP": 0.69, "GMEAN": 0.9},
        "SMOTE-Tomek": {"ACC": 91.81, "AUC": 0.89, "F1": 0.81, "AP": 0.69, "GMEAN": 0.9},
        "SMOTE-ENN": {"ACC": 91.72, "AUC": 0.89, "F1": 0.81, "AP": 0.68, "GMEAN": 0.9},
        "AdaBoost": {"ACC": 94.18, "AUC": 0.91, "F1": 0.85, "AP": 0.76, "GMEAN": 0.91},
        "RUSBoost": {"ACC": None, "AUC": None, "F1": None, "AP": None, "GMEAN": None},
        "HUE": {"ACC": 95.55, "AUC": 0.94, "F1": 0.87, "AP": 0.77, "GMEAN": 0.96},
    },
    "DCCC": {
        "CSRBoost": {"ACC": 68.32, "AUC": 0.64, "F1": 0.42, "AP": 0.29, "GMEAN": 0.62},
        "SMOTifiedGAN": {"ACC": 81.68, "AUC": 0.95, "F1": 0.82, "AP": 0.88, "GMEAN": 0.82},
        "GAN": {"ACC": 80.89, "AUC": 0.94, "F1": 0.81, "AP": 0.87, "GMEAN": 0.81},
        "ADASYN": {"ACC": 72.27, "AUC": 0.62, "F1": 0.41, "AP": 0.29, "GMEAN": 0.59},
        "Borderline-SMOTE": {"ACC": 72.45, "AUC": 0.62, "F1": 0.41, "AP": 0.29, "GMEAN": 0.59},
        "SMOTE-Tomek": {"ACC": 72.49, "AUC": 0.62, "F1": 0.41, "AP": 0.29, "GMEAN": 0.59},
        "SMOTE-ENN": {"ACC": 70.52, "AUC": 0.65, "F1": 0.45, "AP": 0.31, "GMEAN": 0.64},
        "AdaBoost": {"ACC": 72.99, "AUC": 0.62, "F1": 0.41, "AP": 0.3, "GMEAN": 0.59},
        "RUSBoost": {"ACC": None, "AUC": None, "F1": None, "AP": None, "GMEAN": None},
        "HUE": {"ACC": 68.66, "AUC": 0.68, "F1": 0.49, "AP": 0.33, "GMEAN": 0.69},
    },
    "BCW": {
        "CSRBoost": {"ACC": 94.37, "AUC": 0.94, "F1": 0.9, "AP": 0.84, "GMEAN": 0.92},
        "SMOTifiedGAN": {"ACC": 94.17, "AUC": 0.99, "F1": 0.94, "AP": 0.99, "GMEAN": 0.94},
        "GAN": {"ACC": 89.78, "AUC": 0.99, "F1": 0.9, "AP": 0.99, "GMEAN": 0.9},
        "ADASYN": {"ACC": 94.38, "AUC": 0.94, "F1": 0.93, "AP": 0.88, "GMEAN": 0.94},
        "Borderline-SMOTE": {"ACC": 94.03, "AUC": 0.94, "F1": 0.92, "AP": 0.87, "GMEAN": 0.93},
        "SMOTE-Tomek": {"ACC": 94.38, "AUC": 0.94, "F1": 0.93, "AP": 0.9, "GMEAN": 0.94},
        "SMOTE-ENN": {"ACC": 94.38, "AUC": 0.94, "F1": 0.92, "AP": 0.88, "GMEAN": 0.93},
        "AdaBoost": {"ACC": 94.03, "AUC": 0.94, "F1": 0.93, "AP": 0.88, "GMEAN": 0.94},
        "RUSBoost": {"ACC": 97.19, "AUC": 0.97, "F1": 0.96, "AP": 0.75, "GMEAN": 0.97},
        "HUE": {"ACC": 96.11, "AUC": 0.96, "F1": 0.94, "AP": 0.89, "GMEAN": 0.95},
    },
    "CB": {
        "CSRBoost": {"ACC": 76.43, "AUC": 0.76, "F1": 0.69, "AP": 0.63, "GMEAN": 0.71},
        "SMOTifiedGAN": {"ACC": 87.62, "AUC": 0.96, "F1": 0.88, "AP": 0.93, "GMEAN": 0.88},
        "GAN": {"ACC": 86.79, "AUC": 0.96, "F1": 0.87, "AP": 0.94, "GMEAN": 0.86},
        "ADASYN": {"ACC": None, "AUC": None, "F1": None, "AP": None, "GMEAN": None},
        "Borderline-SMOTE": {"ACC": 77.44, "AUC": 0.77, "F1": 0.77, "AP": 0.71, "GMEAN": 0.78},
        "SMOTE-Tomek": {"ACC": 78.37, "AUC": 0.78, "F1": 0.74, "AP": 0.67, "GMEAN": 0.75},
        "SMOTE-ENN": {"ACC": 75.01, "AUC": 0.74, "F1": 0.73, "AP": 0.68, "GMEAN": 0.75},
        "AdaBoost": {"ACC": 76.42, "AUC": 0.76, "F1": 0.74, "AP": 0.66, "GMEAN": 0.75},
        "RUSBoost": {"ACC": 82.26, "AUC": 0.72, "F1": 0.8, "AP": 0.97, "GMEAN": 0.82},
        "HUE": {"ACC": 78.81, "AUC": 0.78, "F1": 0.8, "AP": 0.73, "GMEAN": 0.81},
    },
    "ESDRP": {
        "CSRBoost": {"ACC": 97.31, "AUC": 0.97, "F1": 0.95, "AP": 0.92, "GMEAN": 0.96},
        "SMOTifiedGAN": {"ACC": 92.98, "AUC": 0.99, "F1": 0.93, "AP": 0.99, "GMEAN": 0.92},
        "GAN": {"ACC": 91.97, "AUC": 0.98, "F1": 0.92, "AP": 0.99, "GMEAN": 0.91},
        "ADASYN": {"ACC": 97.69, "AUC": 0.98, "F1": 0.97, "AP": 0.95, "GMEAN": 0.98},
        "Borderline-SMOTE": {"ACC": 97.12, "AUC": 0.97, "F1": 0.97, "AP": 0.95, "GMEAN": 0.98},
        "SMOTE-Tomek": {"ACC": 97.5, "AUC": 0.97, "F1": 0.97, "AP": 0.95, "GMEAN": 0.98},
        "SMOTE-ENN": {"ACC": 93.27, "AUC": 0.94, "F1": 0.93, "AP": 0.88, "GMEAN": 0.95},
        "AdaBoost": {"ACC": 97.5, "AUC": 0.97, "F1": 0.97, "AP": 0.95, "GMEAN": 0.97},
        "RUSBoost": {"ACC": 98.65, "AUC": 0.99, "F1": 0.98, "AP": 0.94, "GMEAN": 0.98},
        "HUE": {"ACC": 97.19, "AUC": 0.97, "F1": 0.97, "AP": 0.94, "GMEAN": 0.98},
    },
    "ILPD": {
        "CSRBoost": {"ACC": 66.72, "AUC": 0.66, "F1": 0.48, "AP": 0.36, "GMEAN": 0.62},
        "SMOTifiedGAN": {"ACC": 70.09, "AUC": 0.76, "F1": 0.77, "AP": 0.88, "GMEAN": 0.55},
        "GAN": {"ACC": 67.52, "AUC": 0.74, "F1": 0.74, "AP": 0.86, "GMEAN": 0.54},
        "ADASYN": {"ACC": 67.76, "AUC": 0.64, "F1": 0.97, "AP": 0.95, "GMEAN": 0.97},
        "Borderline-SMOTE": {"ACC": 69.48, "AUC": 0.65, "F1": 0.98, "AP": 0.97, "GMEAN": 0.98},
        "SMOTE-Tomek": {"ACC": 69.12, "AUC": 0.63, "F1": 0.97, "AP": 0.94, "GMEAN": 0.99},
        "SMOTE-ENN": {"ACC": 68.63, "AUC": 0.69, "F1": 0.95, "AP": 0.92, "GMEAN": 0.96},
        "AdaBoost": {"ACC": 67.92, "AUC": 0.62, "F1": 0.98, "AP": 0.96, "GMEAN": 0.99},
        "RUSBoost": {"ACC": 71.52, "AUC": 0.66, "F1": 0.53, "AP": 0.41, "GMEAN": 0.66},
        "HUE": {"ACC": 66.55, "AUC": 0.71, "F1": 0.58, "AP": 0.42, "GMEAN": 0.7},
    },
    "GLASS": {
        "CSRBoost": {"ACC": 95.8, "AUC": 0.93, "F1": 0.79, "AP": 0.67, "GMEAN": 0.9},
        "SMOTifiedGAN": {"ACC": 55.47, "AUC": 0.89, "F1": 0.55, "AP": 0.65, "GMEAN": 0.88},
        "GAN": {"ACC": 56.74, "AUC": 0.89, "F1": 0.56, "AP": 0.65, "GMEAN": 0.88},
        "ADASYN": {"ACC": 96.74, "AUC": 0.94, "F1": 0.96, "AP": 0.95, "GMEAN": 0.97},
        "Borderline-SMOTE": {"ACC": 96.28, "AUC": 0.93, "F1": 0.96, "AP": 0.93, "GMEAN": 0.97},
        "SMOTE-Tomek": {"ACC": 96.74, "AUC": 0.93, "F1": 0.97, "AP": 0.96, "GMEAN": 0.97},
        "SMOTE-ENN": {"ACC": 96.27, "AUC": 0.94, "F1": 0.98, "AP": 0.97, "GMEAN": 0.98},
        "AdaBoost": {"ACC": 99.53, "AUC": 0.99, "F1": 0.9, "AP": 0.86, "GMEAN": 0.92},
        "RUSBoost": {"ACC": 97.21, "AUC": 0.94, "F1": 0.91, "AP": 0.86, "GMEAN": 0.94},
        "HUE": {"ACC": 94.88, "AUC": 0.95, "F1": 0.82, "AP": 0.71, "GMEAN": 0.93},
    },
    "SEED": {
        "CSRBoost": {"ACC": 98.1, "AUC": 0.98, "F1": 0.96, "AP": 0.93, "GMEAN": 0.97},
        "SMOTifiedGAN": {"ACC": 86.31, "AUC": 0.98, "F1": 0.86, "AP": 0.96, "GMEAN": 0.85},
        "GAN": {"ACC": 87.14, "AUC": 0.98, "F1": 0.87, "AP": 0.96, "GMEAN": 0.86},
        "ADASYN": {"ACC": 98.1, "AUC": 0.98, "F1": 0.5, "AP": 0.38, "GMEAN": 0.63},
        "Borderline-SMOTE": {"ACC": 98.57, "AUC": 0.98, "F1": 0.49, "AP": 0.38, "GMEAN": 0.62},
        "SMOTE-Tomek": {"ACC": 98.57, "AUC": 0.98, "F1": 0.51, "AP": 0.39, "GMEAN": 0.64},
        "SMOTE-ENN": {"ACC": 98.57, "AUC": 0.98, "F1": 0.57, "AP": 0.43, "GMEAN": 0.7},
        "AdaBoost": {"ACC": 90.48, "AUC": 0.89, "F1": 0.49, "AP": 0.38, "GMEAN": 0.62},
        "RUSBoost": {"ACC": 98.1, "AUC": 0.98, "F1": 0.98, "AP": 0.97, "GMEAN": 0.99},
        "HUE": {"ACC": 98.57, "AUC": 0.98, "F1": 0.98, "AP": 0.97, "GMEAN": 0.99},
    },
    "WINE": {
        "CSRBoost": {"ACC": 98.86, "AUC": 0.98, "F1": 0.93, "AP": 0.88, "GMEAN": 0.95},
        "SMOTifiedGAN": {"ACC": 96.67, "AUC": 0.97, "F1": 0.77, "AP": 0.88, "GMEAN": 0.55},
        "GAN": {"ACC": 93.89, "AUC": 0.96, "F1": 0.74, "AP": 0.85, "GMEAN": 0.54},
        "ADASYN": {"ACC": 98.89, "AUC": 0.98, "F1": 0.87, "AP": 0.78, "GMEAN": 0.94},
        "Borderline-SMOTE": {"ACC": 98.89, "AUC": 0.98, "F1": 0.88, "AP": 0.8, "GMEAN": 0.94},
        "SMOTE-Tomek": {"ACC": 98.3, "AUC": 0.98, "F1": 0.88, "AP": 0.81, "GMEAN": 0.92},
        "SMOTE-ENN": {"ACC": 97.21, "AUC": 0.97, "F1": 0.88, "AP": 0.8, "GMEAN": 0.93},
        "AdaBoost": {"ACC": 98.89, "AUC": 0.98, "F1": 0.87, "AP": 0.8, "GMEAN": 0.94},
        "RUSBoost": {"ACC": 99.43, "AUC": 0.99, "F1": 0.99, "AP": 0.99, "GMEAN": 0.99},
        "HUE": {"ACC": 94.43, "AUC": 0.98, "F1": 0.97, "AP": 0.95, "GMEAN": 0.98},
    },
    "YEAST5ERL": {
        "CSRBoost": {"ACC": 99.93, "AUC": 0.6, "F1": 0.47, "AP": 0.43, "GMEAN": 0.55},
        "SMOTifiedGAN": {"ACC": None, "AUC": None, "F1": None, "AP": None, "GMEAN": None},
        "GAN": {"ACC": None, "AUC": None, "F1": None, "AP": None, "GMEAN": None},
        "ADASYN": {"ACC": None, "AUC": None, "F1": None, "AP": None, "GMEAN": None},
        "Borderline-SMOTE": {"ACC": 99.93, "AUC": 0.99, "F1": 0.73, "AP": 0.7, "GMEAN": 0.8},
        "SMOTE-Tomek": {"ACC": 99.8, "AUC": 0.99, "F1": 0.93, "AP": 0.9, "GMEAN": 1.0},
        "SMOTE-ENN": {"ACC": 99.93, "AUC": 0.99, "F1": 0.93, "AP": 0.9, "GMEAN": 1.0},
        "AdaBoost": {"ACC": 99.93, "AUC": 0.99, "F1": 0.93, "AP": 0.9, "GMEAN": 1.0},
        "RUSBoost": {"ACC": 99.87, "AUC": 0.99, "F1": 0.87, "AP": 0.8, "GMEAN": 1.0},
        "HUE": {"ACC": 99.46, "AUC": 0.99, "F1": 0.65, "AP": 0.52, "GMEAN": 1.0},
    },
    "CARGOOD": {
        "CSRBoost": {"ACC": 98.9, "AUC": 0.99, "F1": 0.8, "AP": 0.66, "GMEAN": 0.92},
        "SMOTifiedGAN": {"ACC": 93.09, "AUC": 0.98, "F1": 0.93, "AP": 0.98, "GMEAN": 0.85},
        "GAN": {"ACC": 94.25, "AUC": 0.99, "F1": 0.94, "AP": 0.99, "GMEAN": 0.9},
        "ADASYN": {"ACC": 99.25, "AUC": 0.96, "F1": 0.91, "AP": 0.84, "GMEAN": 0.94},
        "Borderline-SMOTE": {"ACC": 99.31, "AUC": 0.95, "F1": 0.84, "AP": 0.73, "GMEAN": 0.94},
        "SMOTE-Tomek": {"ACC": 99.48, "AUC": 0.96, "F1": 0.88, "AP": 0.78, "GMEAN": 0.94},
        "SMOTE-ENN": {"ACC": 98.73, "AUC": 0.95, "F1": 0.84, "AP": 0.72, "GMEAN": 0.95},
        "AdaBoost": {"ACC": 99.88, "AUC": 0.99, "F1": 0.99, "AP": 0.99, "GMEAN": 0.99},
        "RUSBoost": {"ACC": 99.19, "AUC": 0.98, "F1": 0.84, "AP": 0.73, "GMEAN": 0.99},
        "HUE": {"ACC": 95.49, "AUC": 0.97, "F1": 0.64, "AP": 0.47, "GMEAN": 0.98},
    },
    "CARVGOOD": {
        "CSRBoost": {"ACC": 99.83, "AUC": 0.99, "F1": 0.96, "AP": 0.92, "GMEAN": 0.99},
        "SMOTifiedGAN": {"ACC": 93.14, "AUC": 0.99, "F1": 0.93, "AP": 0.98, "GMEAN": 0.82},
        "GAN": {"ACC": 94.78, "AUC": 0.99, "F1": 0.95, "AP": 0.99, "GMEAN": 0.93},
        "ADASYN": {"ACC": 99.88, "AUC": 0.99, "F1": 0.98, "AP": 0.97, "GMEAN": 0.98},
        "Borderline-SMOTE": {"ACC": 99.88, "AUC": 0.99, "F1": 0.99, "AP": 0.99, "GMEAN": 1.0},
        "SMOTE-Tomek": {"ACC": 99.94, "AUC": 0.99, "F1": 0.98, "AP": 0.97, "GMEAN": 0.99},
        "SMOTE-ENN": {"ACC": 99.88, "AUC": 0.99, "F1": 0.96, "AP": 0.93, "GMEAN": 0.98},
        "AdaBoost": {"ACC": 99.94, "AUC": 0.99, "F1": 0.98, "AP": 0.97, "GMEAN": 0.99},
        "RUSBoost": {"ACC": 99.02, "AUC": 0.99, "F1": 0.88, "AP": 0.78, "GMEAN": 0.99},
        "HUE": {"ACC": 98.44, "AUC": 0.99, "F1": 0.83, "AP": 0.72, "GMEAN": 0.99},
    },
    "YEAST5": {
        "CSRBoost": {"ACC": 98.32, "AUC": 0.93, "F1": 0.69, "AP": 0.5, "GMEAN": 0.89},
        "SMOTifiedGAN": {"ACC": 96.23, "AUC": 0.97, "F1": 0.96, "AP": 0.45, "GMEAN": 0.55},
        "GAN": {"ACC": 96.77, "AUC": 0.95, "F1": 0.97, "AP": 0.46, "GMEAN": 0.66},
        "ADASYN": {"ACC": 98.59, "AUC": 0.93, "F1": 0.75, "AP": 0.57, "GMEAN": 0.88},
        "Borderline-SMOTE": {"ACC": 98.38, "AUC": 0.93, "F1": 0.71, "AP": 0.53, "GMEAN": 0.9},
        "SMOTE-Tomek": {"ACC": 98.31, "AUC": 0.91, "F1": 0.76, "AP": 0.59, "GMEAN": 0.9},
        "SMOTE-ENN": {"ACC": 98.18, "AUC": 0.96, "F1": 0.75, "AP": 0.6, "GMEAN": 0.97},
        "AdaBoost": {"ACC": 98.52, "AUC": 0.85, "F1": 0.72, "AP": 0.54, "GMEAN": 0.85},
        "RUSBoost": {"ACC": 97.03, "AUC": 0.97, "F1": 0.67, "AP": 0.5, "GMEAN": 0.97},
        "HUE": {"ACC": 95.96, "AUC": 0.97, "F1": 0.59, "AP": 0.43, "GMEAN": 0.97},
    },
    "FLAREF": {
        "CSRBoost": {"ACC": 93.43, "AUC": 0.67, "F1": 0.22, "AP": 0.1, "GMEAN": 0.48},
        "SMOTifiedGAN": {"ACC": 95.93, "AUC": 0.94, "F1": 0.96, "AP": 0.4, "GMEAN": 0.44},
        "GAN": {"ACC": 95.58, "AUC": 0.95, "F1": 0.96, "AP": 0.44, "GMEAN": 0.53},
        "ADASYN": {"ACC": 94.56, "AUC": 0.61, "F1": 0.26, "AP": 0.11, "GMEAN": 0.51},
        "Borderline-SMOTE": {"ACC": 94.65, "AUC": 0.62, "F1": 0.26, "AP": 0.11, "GMEAN": 0.47},
        "SMOTE-Tomek": {"ACC": 94.09, "AUC": 0.57, "F1": 0.27, "AP": 0.11, "GMEAN": 0.49},
        "SMOTE-ENN": {"ACC": 93.34, "AUC": 0.72, "F1": 0.35, "AP": 0.18, "GMEAN": 0.65},
        "AdaBoost": {"ACC": 94.94, "AUC": 0.55, "F1": 0.19, "AP": 0.08, "GMEAN": 0.37},
        "RUSBoost": {"ACC": 82.55, "AUC": 0.83, "F1": 0.28, "AP": 0.15, "GMEAN": 0.84},
        "HUE": {"ACC": 81.43, "AUC": 0.86, "F1": 0.28, "AP": 0.15, "GMEAN": 0.86},
    },
}

# Categorical encodings (matches the per-dataset replication scripts)
CARGOOD_CAT_MAPS = {
    0: {'low': 0, 'med': 1, 'high': 2, 'vhigh': 3},
    1: {'low': 0, 'med': 1, 'high': 2, 'vhigh': 3},
    2: {'2': 0, '3': 1, '4': 2, '5more': 3},
    3: {'2': 0, '4': 1, 'more': 2},
    4: {'small': 0, 'med': 1, 'big': 2},
    5: {'low': 0, 'med': 1, 'high': 2},
}
CARVGOOD_CAT_MAPS = CARGOOD_CAT_MAPS
FLAREF_CAT_MAPS = {
    0: {'A': 0, 'H': 1, 'K': 2, 'R': 3, 'S': 4, 'X': 5},
    1: {'C': 0, 'I': 1, 'O': 2, 'X': 3},
}

# ==============================================================================
# DATASET LOADERS
# ==============================================================================
def _to_float(df):
    return df.apply(pd.to_numeric, errors="coerce").fillna(0).values

def _dummies(df):
    cats = df.select_dtypes(include="object").columns.tolist()
    return pd.get_dummies(df, columns=cats, drop_first=False) if cats else df

def _label_encode_objects(df):
    out = df.copy()
    for col in out.select_dtypes(include="object").columns:
        values = sorted(out[col].dropna().astype(str).unique())
        mapping = {value: i for i, value in enumerate(values)}
        out[col] = out[col].astype(str).map(mapping).fillna(0)
    return out


def load_psdas():
    p = DATA_DIR / "01_PSDAS" / "data.csv"
    df = pd.read_csv(p, sep=";")
    y = (df["Target"] == "Enrolled").astype(int).values
    X = _to_float(_dummies(df.drop(columns=["Target"])))
    return X, y

def load_bcw():
    p = DATA_DIR / "04_BCW" / "wdbc.data"
    df = pd.read_csv(p, header=None)
    y = (df.iloc[:, 1].values == "M").astype(int)
    X = df.iloc[:, 2:].values.astype(float)
    return X, y

def load_cb():
    p = DATA_DIR / "06_CB" / "sonar.all-data"
    df = pd.read_csv(p, header=None)
    # Polarity fix: R (Rock) is the minority class (n=97); M (Metal) is majority (n=111).
    # Project-wide convention is y=1 == minority. Previous code used "M" which inverted it.
    y = (df.iloc[:, -1].values == "R").astype(int)
    X = df.iloc[:, :-1].values.astype(float)
    return X, y

def load_esdrp():
    p = DATA_DIR / "05_ESDRP" / "diabetes_data_upload.csv"
    df = pd.read_csv(p)
    y = (df["class"].astype(str).str.lower() == "negative").astype(int).values
    X = _to_float(_label_encode_objects(df.drop(columns=["class"])))
    return X, y

def load_ilpd():
    p = DATA_DIR / "08_ILPD" / "ilpd.csv"
    df = pd.read_csv(p)
    y = (df["Class"].values == 2).astype(int)
    X = _to_float(_label_encode_objects(df.drop(columns=["Class"])))
    return X, y

def load_glass():
    p = DATA_DIR / "07_GLASS" / "glass.csv"
    df = pd.read_csv(p)
    y = (df.iloc[:, -1].values == 7).astype(int)
    X = df.iloc[:, :-1].values.astype(float)
    return X, y

def load_seed():
    p = DATA_DIR / "09_SEED" / "seeds_dataset.txt"
    df = pd.read_csv(p, sep=r"\s+", header=None)
    y = (df.iloc[:, -1].values == 1).astype(int)
    X = df.iloc[:, :-1].values.astype(float)
    return X, y

def load_wine():
    p = DATA_DIR / "10_WINE" / "wine.data"
    df = pd.read_csv(p, header=None)
    y = (df.iloc[:, 0].astype(int).values == 3).astype(int)
    X = df.iloc[:, 1:].values.astype(float)
    return X, y

def _load_keel(p, n_features):
    lines = [l for l in p.read_text().splitlines()
             if l.strip() and not l.strip().startswith("@")]
    arr = np.array([[v.strip() for v in l.split(",")] for l in lines if "," in l])
    return arr

def load_yeast5erl():
    p = DATA_DIR / "12_YEAST5-ERL" / "yeast5erl.dat"
    arr = _load_keel(p, 8)
    y = (arr[:, -1] == "positive").astype(int)
    X = arr[:, :-1].astype(float)
    return X, y

def load_yeast5():
    p = DATA_DIR / "11_YEAST5" / "yeast5.dat"
    arr = _load_keel(p, 8)
    y = (arr[:, -1] == "positive").astype(int)
    X = arr[:, :-1].astype(float)
    return X, y

def load_cargood():
    p = DATA_DIR / "13_CARGOOD" / "car-good.dat"
    arr = _load_keel(p, 6)
    y = (arr[:, -1] == "positive").astype(int)
    X = np.empty((arr.shape[0], 6), dtype=float)
    for col in range(6):
        X[:, col] = [CARGOOD_CAT_MAPS[col][v] for v in arr[:, col]]
    return X, y

def load_carvgood():
    p = DATA_DIR / "14_CARVGOOD" / "car-vgood.dat"
    arr = _load_keel(p, 6)
    y = (arr[:, -1] == "positive").astype(int)
    X = np.empty((arr.shape[0], 6), dtype=float)
    for col in range(6):
        X[:, col] = [CARVGOOD_CAT_MAPS[col][v] for v in arr[:, col]]
    return X, y

def load_flaref():
    p = DATA_DIR / "15_FLARE-F" / "flare-F.dat"
    arr = _load_keel(p, 11)
    y = (arr[:, -1] == "positive").astype(int)
    X = np.empty((arr.shape[0], 11), dtype=float)
    X[:, 0] = [FLAREF_CAT_MAPS[0][v] for v in arr[:, 0]]
    X[:, 1] = [FLAREF_CAT_MAPS[1][v] for v in arr[:, 1]]
    for col in range(2, 11):
        X[:, col] = arr[:, col].astype(float)
    return X, y


def load_esr():
    """ESR loader (mirrors the Colab script). y==1 is the seizure class (minority)."""
    p = DATA_DIR / "02_ESR" / "Epileptic Seizure Recognition.csv"
    df = pd.read_csv(p)
    drop = [c for c in df.columns if str(c).lower().startswith("unnamed")]
    if drop:
        df = df.drop(columns=drop)
    tc = "y" if "y" in df.columns else df.columns[-1]
    y = (pd.to_numeric(df[tc], errors="coerce").fillna(0).astype(int) == 1).astype(int).values
    X = df.drop(columns=[tc]).apply(pd.to_numeric, errors="coerce").fillna(0).values.astype(float)
    return X, y


def load_dccc():
    """DCCC loader (mirrors the Colab script). Drops ID col, uses header row 1, y==1 is default."""
    p = DATA_DIR / "03_DCCC" / "DCCC.xls"
    df = pd.read_excel(p, header=1)
    df = df.iloc[:, 1:]  # drop ID column
    y = pd.to_numeric(df.iloc[:, -1], errors="coerce").fillna(0).astype(int).values
    X = df.iloc[:, :-1].apply(pd.to_numeric, errors="coerce").fillna(0).values.astype(float)
    if y.mean() > 0.5:
        y = 1 - y
    return X, y


LOADERS = {
    "PSDAS":     load_psdas,
    "ESR":       load_esr,
    "DCCC":      load_dccc,
    "BCW":       load_bcw,
    "CB":        load_cb,
    "ESDRP":     load_esdrp,
    "ILPD":      load_ilpd,
    "GLASS":     load_glass,
    "SEED":      load_seed,
    "WINE":      load_wine,
    "YEAST5ERL": load_yeast5erl,
    "CARGOOD":   load_cargood,
    "CARVGOOD":  load_carvgood,
    "YEAST5":    load_yeast5,
    "FLAREF":    load_flaref,
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

def metrics(y_true, y_proba):
    """Probability-based modern metrics, threshold 0.5, F1 binary."""
    y_pred = (y_proba >= THRESHOLD).astype(int)
    return {
        "ACC":   float(accuracy_score(y_true, y_pred)) * 100,
        "AUC":   safe_auc(y_true, y_proba),
        "F1":    float(f1_score(y_true, y_pred, zero_division=0, average="binary")),
        "AP":    safe_ap(y_true, y_proba),
        "GMEAN": gmean_at(y_true, y_pred),
    }

# ==============================================================================
# CLASSIFIER FACTORIES (source-paper prescribed)
# ==============================================================================
def make_adaboost_freund1996(rs, n_estimators=T_ADABOOST):
    """Yadav 2025 AdaBoost baseline: 'The base classifier is a decision tree' --
    sklearn-default DecisionTreeClassifier (unpruned, max_depth=None) with
    n_estimators=50. This matches CSRBoost's own baseline configuration."""
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

def make_adaboost_csrboost(rs, n_estimators=T_CSRBOOST):
    """CSRBoost (Yadav 2025): AdaBoost wrapping a default decision tree, T=50."""
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
    """AdaBoost wrapping the SMOTE-family resamplers (CSRBoost paper baseline)."""
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
    """Yadav 2025 RUSBoost baseline: T=50 (matches boosting iterations across
    the table for fair comparison), sampling_strategy=1.0 (50% post-balance).
    Base classifier is unpruned standard DecisionTreeClassifier."""
    base = DecisionTreeClassifier(random_state=rs)
    try:
        return RUSBoostClassifier(estimator=base, n_estimators=n_estimators,
                                  learning_rate=1.0, sampling_strategy=1.0,
                                  random_state=rs)
    except TypeError:
        return RUSBoostClassifier(base_estimator=base, n_estimators=n_estimators,
                                  learning_rate=1.0, sampling_strategy=1.0,
                                  random_state=rs)

# ==============================================================================
# RESAMPLERS (source-paper prescribed)
# ==============================================================================
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
    # Batista 2004: ENN removes a sample when its class differs from at least 2 of its 3 nearest
    # neighbours -- that is the majority-disagreement (Wilson 1972) rule, which corresponds to
    # imblearn's kind_sel="mode". The earlier kind_sel="all" was more aggressive (any single
    # neighbour disagreement removes the sample) and not faithful to the source paper.
    return SMOTEENN(smote=SMOTE(k_neighbors=safe_k(n_min), random_state=rs),
                    enn=EditedNearestNeighbours(n_neighbors=3, kind_sel="mode",
                                                sampling_strategy="all"),
                    random_state=rs)

def fit_resample_safe(sampler, X, y):
    n_min = int(np.sum(y == 1))
    if n_min < 2:
        warnings.warn(
            f"[fit_resample_safe] {type(sampler).__name__}: n_minority={n_min} < 2, "
            f"returning unresampled data (no synthesis possible).",
            RuntimeWarning, stacklevel=2,
        )
        return X, y
    try:
        Xr, yr = sampler.fit_resample(X, y)
        if len(np.unique(yr)) < 2:
            warnings.warn(
                f"[fit_resample_safe] {type(sampler).__name__}: post-resample y has "
                f"<2 classes, returning unresampled data.",
                RuntimeWarning, stacklevel=2,
            )
            return X, y
        return Xr, yr
    except Exception as e:
        warnings.warn(
            f"[fit_resample_safe] {type(sampler).__name__} raised "
            f"{type(e).__name__}: {e}. Returning unresampled data.",
            RuntimeWarning, stacklevel=2,
        )
        return X, y

# ==============================================================================
# CSRBoost: KMeans + 50% per-cluster RUS + SMOTE + AdaBoost(50 trees)
# ==============================================================================
def csrboost_fit_predict(X_tr, y_tr, X_te, rs):
    X_min = X_tr[y_tr == 1]
    X_maj = X_tr[y_tr == 0]
    n_min = len(X_min)
    n_maj = len(X_maj)
    K = max(1, min(n_min, n_maj))    # Eq.6 with p=100%, capped at n_majority
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
    swapped = False
    if int(np.sum(y_ret == 1)) > int(np.sum(y_ret == 0)):
        y_ret = 1 - y_ret  # class redefinition guard (paper Sec II-D, scoped to SMOTE)
        swapped = True
    Xb, yb = fit_resample_safe(make_smote(rs, int(np.sum(y_ret == 1))),
                               X_ret, y_ret)
    clf = make_adaboost_csrboost(rs)
    clf.fit(Xb, yb)
    proba = clf.predict_proba(X_te)[:, 1]
    if swapped:
        # un-swap so the returned score is always P(true minority). The guard does
        # not fire on any of these datasets, so this is a no-op for the reported run.
        proba = 1.0 - proba
    return proba

# ==============================================================================
# HUE: Hashing-based Undersampling Ensemble (Ng 2022) -- manual implementation
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
        # Scheme IV reciprocal weights
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
    # Normalise to [0, 1] so it is a usable score for AUC/AP and so that
    # threshold=0.5 reproduces the paper's sign() decision rule.
    return (score / len(estimators) + 1.0) / 2.0

# ==============================================================================
# SMOTified-GAN / GAN (Sharma 2022) -- manual PyTorch implementation
# ==============================================================================
class GANResult:
    def __init__(self, ok, prob):
        self.ok = ok
        self.prob = prob

def _import_torch():
    """Lazy import - allows the script to load even without torch installed."""
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    return torch, nn, optim, DataLoader, TensorDataset

def gan_fit_predict(X_tr, y_tr, X_te, rs, mode="smotified",
                    epochs=GAN_EPOCHS_LOCAL):
    """
    Sharma 2022 SMOTified-GAN / naive GAN.
    Generator   : Linear(latent, 128) -> ReLU -> BN ->
                  Linear(128, 256) -> ReLU -> BN ->
                  Linear(256, 512) -> ReLU -> BN ->
                  Linear(512, 1024) -> ReLU -> BN ->
                  Linear(1024, n_features) -> Tanh
    Discriminator: Linear(n_features, 512) -> LeakyReLU(0.2) ->
                   Linear(512, 256) -> LeakyReLU(0.2) ->
                   Linear(256, 128) -> LeakyReLU(0.2) ->
                   Linear(128, 1)
    Classifier  : Linear(n_features, 256) -> ReLU -> Linear(256, 128) -> ReLU ->
                  Linear(128, 1)
    Adam(lr=1e-5), batch=128, BCEWithLogitsLoss for GAN, MAE for classifier.
    """
    try:
        torch, nn, optim, DataLoader, TensorDataset = _import_torch()
    except Exception:
        return GANResult(False, None)
    torch.manual_seed(rs)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(rs)
    np.random.seed(rs)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_features = X_tr.shape[1]
    n_min = int(np.sum(y_tr == 1))
    n_maj = int(np.sum(y_tr == 0))
    n_generate = n_maj - n_min

    if n_min < 2 or n_generate <= 0:
        # Fall back: too few minority for GAN training - use AdaBoost on raw
        return GANResult(False, None)

    # 90/10 stratified split for early-stopping validation. The GAN trains on
    # the sub-train minority only; val is held out for F1-driven early stop.
    # Final classifier (after GAN converges) still trains on the FULL X_tr.
    has_val = False
    X_sub_tr, y_sub_tr = X_tr, y_tr
    X_val_es, y_val_es = None, None
    if n_min >= 4:
        try:
            from sklearn.model_selection import train_test_split as _tts
            X_sub_tr, X_val_es, y_sub_tr, y_val_es = _tts(
                X_tr, y_tr, test_size=0.10, stratify=y_tr, random_state=rs)
            if int(np.sum(y_val_es == 1)) >= 1 and int(np.sum(y_val_es == 0)) >= 1 \
                    and int(np.sum(y_sub_tr == 1)) >= 2:
                has_val = True
            else:
                X_sub_tr, y_sub_tr = X_tr, y_tr
                X_val_es, y_val_es = None, None
                has_val = False
        except ValueError:
            X_sub_tr, y_sub_tr = X_tr, y_tr
            X_val_es, y_val_es = None, None
            has_val = False

    Xmin = X_sub_tr[y_sub_tr == 1].astype(np.float32)

    # --- Stage 1: latent samples ---
    if mode == "smotified":
        # Use SMOTE-generated samples as latent input to G (paper's key idea)
        from sklearn.neighbors import NearestNeighbors
        rng = np.random.RandomState(rs)
        k = min(SMOTE_K, len(Xmin) - 1)
        nn_model = NearestNeighbors(n_neighbors=k + 1).fit(Xmin)
        pool_size = max(n_generate * 2, 1024)
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

    # --- Generator (Sharma 2022 Table 2) ---
    g_hidden = [128, 256, 512, 1024]
    layers = []
    in_dim = latent_dim
    for h in g_hidden:
        layers += [nn.Linear(in_dim, h), nn.BatchNorm1d(h), nn.ReLU()]
        in_dim = h
    layers += [nn.Linear(in_dim, n_features), nn.Tanh()]
    G = nn.Sequential(*layers).to(DEVICE)

    # --- Discriminator (Sharma 2022 Table 2) ---
    d_hidden = [512, 256, 128]
    layers = []
    in_dim = n_features
    for h in d_hidden:
        layers += [nn.Linear(in_dim, h), nn.LeakyReLU(0.2)]
        in_dim = h
    layers += [nn.Linear(in_dim, 1)]
    D = nn.Sequential(*layers).to(DEVICE)

    g_opt = optim.Adam(G.parameters(), lr=1e-5, betas=(0.9, 0.999))
    d_opt = optim.Adam(D.parameters(), lr=1e-5, betas=(0.9, 0.999))
    bce = nn.BCEWithLogitsLoss()
    mae = nn.L1Loss()

    batch_size = min(128, len(Xmin))
    Xmin_t = torch.FloatTensor(Xmin)
    dataset = TensorDataset(Xmin_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Downstream classifier architecture (defined here so early-stopping can use it)
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

    # Early stopping state (only used when has_val=True)
    es_best_f1 = -1.0
    es_no_improve = 0
    es_n_gen_val = max(1, int(np.sum(y_sub_tr == 0) - np.sum(y_sub_tr == 1)))

    G.train(); D.train()
    for ep in range(epochs):
        for (real_batch,) in loader:
            real_batch = real_batch.to(DEVICE)
            bs = real_batch.size(0)

            d_opt.zero_grad()
            d_real = bce(D(real_batch), torch.ones(bs, 1, device=DEVICE))
            z = torch.FloatTensor(sample_latent(bs)).to(DEVICE)
            fake = G(z)
            d_fake = bce(D(fake.detach()), torch.zeros(bs, 1, device=DEVICE))
            (d_real + d_fake).backward()
            d_opt.step()

            g_opt.zero_grad()
            z = torch.FloatTensor(sample_latent(bs)).to(DEVICE)
            g_loss = bce(D(G(z)), torch.ones(bs, 1, device=DEVICE))
            g_loss.backward()
            g_opt.step()

        # ---- Early stopping: evaluate F1 on held-out 10% val every epoch ----
        if has_val and GAN_EARLY_STOPPING_ENABLED:
            G.eval()
            with torch.no_grad():
                z_val = torch.FloatTensor(sample_latent(es_n_gen_val)).to(DEVICE)
                synth_val = G(z_val).cpu().numpy().astype(np.float32)
            G.train()

            X_aug_es = np.vstack([X_sub_tr, synth_val])
            y_aug_es = np.hstack([y_sub_tr, np.ones(len(synth_val), dtype=int)])

            Cv = CLF(n_features).to(DEVICE)
            cv_opt = optim.Adam(Cv.parameters(), lr=1e-5)
            Xv_t = torch.FloatTensor(X_aug_es)
            yv_t = torch.FloatTensor(y_aug_es).unsqueeze(1)
            cv_loader = DataLoader(TensorDataset(Xv_t, yv_t),
                                   batch_size=128, shuffle=True)
            Cv.train()
            for _ in range(5):  # short classifier training for early-stop check
                for xb, yb in cv_loader:
                    xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                    cv_opt.zero_grad()
                    out = torch.sigmoid(Cv(xb))
                    cv_loss = mae(out, yb)
                    cv_loss.backward()
                    cv_opt.step()

            Cv.eval()
            with torch.no_grad():
                p_val = torch.sigmoid(Cv(torch.FloatTensor(X_val_es).to(DEVICE))).cpu().numpy().ravel()
            y_pred_val = (p_val >= 0.5).astype(int)
            cur_f1 = float(f1_score(y_val_es, y_pred_val, zero_division=0, average="binary"))

            # Always track best F1 (including during warm-up). Patience counter
            # only increments after the warm-up period, so the GAN gets at least
            # GAN_WARMUP_EPOCHS to learn before early stopping can trigger.
            if cur_f1 > es_best_f1:
                es_best_f1 = cur_f1
                es_no_improve = 0
            elif ep >= GAN_WARMUP_EPOCHS:
                es_no_improve += 1

            if es_no_improve >= GAN_EARLY_STOP_PATIENCE:
                break

    G.eval()
    with torch.no_grad():
        z = torch.FloatTensor(sample_latent(n_generate)).to(DEVICE)
        synth = G(z).cpu().numpy().astype(np.float32)

    X_aug = np.vstack([X_tr, synth])
    y_aug = np.hstack([y_tr, np.ones(len(synth), dtype=int)])

    # --- Downstream classifier (NN, MAE loss, 30 epochs) ---
    # CLF class and `mae` are defined above (before the GAN training loop).
    C = CLF(n_features).to(DEVICE)
    c_opt = optim.Adam(C.parameters(), lr=1e-5)
    Xa_t = torch.FloatTensor(X_aug); ya_t = torch.FloatTensor(y_aug).unsqueeze(1)
    cds = TensorDataset(Xa_t, ya_t)
    cloader = DataLoader(cds, batch_size=128, shuffle=True)
    C.train()
    for _ in range(30):
        for xb, yb in cloader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            c_opt.zero_grad()
            out = torch.sigmoid(C(xb))
            loss = mae(out, yb)
            loss.backward()
            c_opt.step()

    C.eval()
    with torch.no_grad():
        prob = torch.sigmoid(C(torch.FloatTensor(X_te).to(DEVICE))).cpu().numpy().ravel()
    return GANResult(True, prob)

# ==============================================================================
# DISPATCH
# ==============================================================================
def run_method(name, X_tr, y_tr, X_te, y_te, rs):
    """Returns metrics dict, or NaN dict if the method failed/skipped."""
    nan_metrics = {"ACC": float("nan"), "AUC": float("nan"),
                   "F1": float("nan"), "AP": float("nan"),
                   "GMEAN": float("nan")}
    n_min = int(np.sum(y_tr == 1))

    if name == "CSRBoost":
        prob = csrboost_fit_predict(X_tr, y_tr, X_te, rs)
        return metrics(y_te, prob)

    if name == "SMOTifiedGAN":
        res = gan_fit_predict(X_tr, y_tr, X_te, rs, mode="smotified")
        if not res.ok:
            return nan_metrics
        return metrics(y_te, res.prob)

    if name == "GAN":
        res = gan_fit_predict(X_tr, y_tr, X_te, rs, mode="gan")
        if not res.ok:
            return nan_metrics
        return metrics(y_te, res.prob)

    if name == "ADASYN":
        sampler = make_adasyn(rs, n_min)
        Xr, yr = fit_resample_safe(sampler, X_tr, y_tr)
        clf = make_adaboost_sampler_wrapper(rs); clf.fit(Xr, yr)
        return metrics(y_te, clf.predict_proba(X_te)[:, 1])

    if name == "Borderline-SMOTE":
        sampler = make_borderline_smote(rs, n_min)
        Xr, yr = fit_resample_safe(sampler, X_tr, y_tr)
        clf = make_adaboost_sampler_wrapper(rs); clf.fit(Xr, yr)
        return metrics(y_te, clf.predict_proba(X_te)[:, 1])

    if name == "SMOTE-Tomek":
        sampler = make_smote_tomek(rs, n_min)
        Xr, yr = fit_resample_safe(sampler, X_tr, y_tr)
        clf = make_adaboost_sampler_wrapper(rs); clf.fit(Xr, yr)
        return metrics(y_te, clf.predict_proba(X_te)[:, 1])

    if name == "SMOTE-ENN":
        sampler = make_smote_enn(rs, n_min)
        Xr, yr = fit_resample_safe(sampler, X_tr, y_tr)
        clf = make_adaboost_sampler_wrapper(rs); clf.fit(Xr, yr)
        return metrics(y_te, clf.predict_proba(X_te)[:, 1])

    if name == "SMOTE":
        sampler = make_smote(rs, n_min)
        Xr, yr = fit_resample_safe(sampler, X_tr, y_tr)
        clf = make_adaboost_sampler_wrapper(rs); clf.fit(Xr, yr)
        return metrics(y_te, clf.predict_proba(X_te)[:, 1])

    if name == "AdaBoost":
        clf = make_adaboost_freund1996(rs); clf.fit(X_tr, y_tr)
        return metrics(y_te, clf.predict_proba(X_te)[:, 1])

    if name == "RUSBoost":
        clf = make_rusboost_seiffert2010(rs); clf.fit(X_tr, y_tr)
        return metrics(y_te, clf.predict_proba(X_te)[:, 1])

    if name == "HUE":
        prob = hue_fit_predict(X_tr, y_tr, X_te, rs)
        return metrics(y_te, prob)

    raise ValueError(f"Unknown method: {name}")

# ==============================================================================
# EXPERIMENT
# ==============================================================================
def evaluate_dataset(ds_name, X, y):
    n_min = int(y.sum())
    n_maj = int((1 - y).sum())
    ir = n_maj / max(n_min, 1)
    print(f"\n{'-'*100}\n  DATASET: {ds_name}")
    print(f"  n={len(y)}  features={X.shape[1]}  IR={ir:.2f}  "
          f"minority={n_min}  majority={n_maj}\n{'-'*100}")

    splitter = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS,
                                       random_state=SEED)
    total_folds = N_SPLITS * N_REPEATS
    fold_rows = []
    per_method = {m: [] for m in METHODS}
    ds_start = perf_counter()

    # ---- Per-fold checkpointing: append to disk after EVERY completed fold so
    # a crash / power loss / software update etc. cannot lose more than the
    # in-flight fold. On restart we read the checkpoint and skip done folds. ----
    ckpt_csv = CHECKPOINT_DIR / f"{ds_name}_folds.csv"
    completed_folds = set()
    if ckpt_csv.exists():
        try:
            ckpt_df = pd.read_csv(ckpt_csv)
            for fold_i in sorted(ckpt_df["fold"].unique()):
                fold_df = ckpt_df[ckpt_df["fold"] == fold_i]
                if set(fold_df["method"].astype(str)) >= set(METHODS):
                    completed_folds.add(int(fold_i))
                    for _, r in fold_df.iterrows():
                        m = r["method"]
                        result = {k: r[k] for k in ("ACC","AUC","F1","AP","GMEAN")}
                        per_method[m].append(result)
                        fold_rows.append({"dataset": ds_name, "method": m,
                                          "fold": int(fold_i),
                                          "status": r.get("status", "ok"),
                                          **result})
            if completed_folds:
                print(f"  RESUMING: {len(completed_folds)} folds already in "
                      f"checkpoint ({ckpt_csv.name})")
        except Exception as e:
            print(f"  WARNING: could not read checkpoint {ckpt_csv}: {e}")
            completed_folds = set()
    ckpt_header_written = ckpt_csv.exists() and ckpt_csv.stat().st_size > 0

    for fold_i, (tr_idx, te_idx) in enumerate(splitter.split(X, y)):
        if fold_i in completed_folds:
            continue
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[tr_idx])
        X_te = scaler.transform(X[te_idx])
        y_tr, y_te = y[tr_idx], y[te_idx]

        new_rows_for_fold = []
        for m in METHODS:
            # Paper Section III-D explicit N/A skips (enforced at runtime so the
            # CSV row is NaN with status='skip(N/A)' rather than a real number that
            # would later need to be masked by the markdown builder).
            skip_reason = None
            if m == "ADASYN" and ds_name in SKIP_ADASYN_DATASETS:
                skip_reason = "ADASYN N/A on this dataset"
            elif m in ("GAN", "SMOTifiedGAN") and ds_name in SKIP_GAN_DATASETS:
                skip_reason = f"{m} N/A on this dataset"
            elif m == "RUSBoost" and ds_name in SKIP_RUSBOOST_DATASETS:
                skip_reason = "RUSBoost N/A on time-series dataset"

            if skip_reason is not None:
                result = {k: float("nan") for k in ("ACC","AUC","F1","AP","GMEAN")}
                status = "skip(N/A)"
            else:
                try:
                    result = run_method(m, X_tr, y_tr, X_te, y_te, SEED)
                    status = "ok"
                except Exception as e:
                    result = {k: float("nan") for k in ("ACC","AUC","F1","AP","GMEAN")}
                    status = f"ERR:{type(e).__name__}"
            per_method[m].append(result)
            row = {"dataset": ds_name, "method": m, "fold": fold_i,
                   "status": status, **result}
            fold_rows.append(row)
            new_rows_for_fold.append(row)

        # Append this fold's 11 rows to the checkpoint CSV right now, while
        # they are still hot. Worst case: in-flight fold lost on crash.
        try:
            df_new = pd.DataFrame(new_rows_for_fold)
            df_new.to_csv(ckpt_csv, mode="a",
                          header=not ckpt_header_written, index=False)
            ckpt_header_written = True
        except Exception as e:
            print(f"  WARNING: could not write checkpoint after fold {fold_i}: {e}")

        if (fold_i + 1) % 10 == 0 or fold_i == total_folds - 1:
            e = perf_counter() - ds_start
            eta = (total_folds - fold_i - 1) / ((fold_i + 1) / e) if e > 0 else 0
            print(f"    fold {fold_i+1:3d}/{total_folds}  "
                  f"elapsed={e:.0f}s  ETA={eta:.0f}s")

    return fold_rows, per_method

def summarise_dataset(ds_name, per_method):
    summary = {"dataset": ds_name}
    for m in METHODS:
        for k in ("ACC","AUC","F1","AP","GMEAN"):
            vals = [f[k] for f in per_method[m] if not np.isnan(f.get(k, float("nan")))]
            summary[f"{m}_{k}_mean"] = float(np.mean(vals)) if vals else float("nan")
            summary[f"{m}_{k}_std"]  = float(np.std(vals))  if vals else float("nan")
    return summary

def comparison_to_paper(ds_name, per_method):
    paper = PAPER_TABLE2.get(ds_name, {})
    rows = []
    for m in METHODS:
        ours = {}
        for k in ("ACC","AUC","F1","AP","GMEAN"):
            vals = [f[k] for f in per_method[m] if not np.isnan(f.get(k, float("nan")))]
            ours[k] = float(np.mean(vals)) if vals else float("nan")
        p = paper.get(m, {})
        row = {"method": m,
               "ACC_paper":  p.get("ACC"),
               "ACC_ours":   ours["ACC"],
               "AUC_paper":  p.get("AUC"),
               "AUC_ours":   ours["AUC"],
               "F1_paper":   p.get("F1"),
               "F1_ours":    ours["F1"],
               "AP_paper":   p.get("AP"),
               "AP_ours":    ours["AP"],
               "GMEAN_paper":p.get("GMEAN"),
               "GMEAN_ours": ours["GMEAN"]}
        rows.append(row)
    return rows

def print_comparison(ds_name, rows):
    print(f"\n  COMPARISON vs paper Table 2 -- {ds_name}")
    hdr = f"  {'METHOD':<18} {'ACC pap/ours':<22} {'AUC pap/ours':<18} " \
          f"{'F1 pap/ours':<18} {'AP pap/ours':<18} {'GMEAN pap/ours':<18}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for r in rows:
        def f(x):
            return "  N/A " if x is None or (isinstance(x, float) and np.isnan(x)) \
                else f"{x:.2f}"
        print(f"  {r['method']:<18} "
              f"{f(r['ACC_paper'])}/{f(r['ACC_ours']):<10} "
              f"{f(r['AUC_paper'])}/{f(r['AUC_ours']):<8} "
              f"{f(r['F1_paper'])}/{f(r['F1_ours']):<8} "
              f"{f(r['AP_paper'])}/{f(r['AP_ours']):<8} "
              f"{f(r['GMEAN_paper'])}/{f(r['GMEAN_ours']):<8}")


def run_all(datasets):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print("=" * 100)
    print("  EXACT-AS-PRESCRIBED REPLICATION  (LOCAL, 13 datasets)")
    print("=" * 100)
    print(f"  Datasets : {', '.join(datasets)}")
    print(f"  Methods  : {', '.join(METHODS)}")
    print(f"  CV       : {N_SPLITS}-fold x {N_REPEATS} repeats = "
          f"{N_SPLITS * N_REPEATS} folds")
    print(f"  GAN epochs (this run): {GAN_EPOCHS_LOCAL} (Sharma 2022 prescribes 2000; "
          f"reduced for CPU runtime)")
    print(f"  Data dir : {DATA_DIR}")
    print(f"  Output   : {OUT_DIR}")
    print("=" * 100)

    all_fold_rows = []
    all_summaries = []
    all_comparisons = {}

    t0 = perf_counter()
    for ds in datasets:
        loader = LOADERS.get(ds)
        if loader is None:
            print(f"  SKIP {ds}: no loader registered")
            continue
        try:
            X, y = loader()
        except Exception as e:
            print(f"  SKIP {ds}: load error - {e}")
            continue

        fold_rows, per_method = evaluate_dataset(ds, X, y)
        all_fold_rows.extend(fold_rows)
        all_summaries.append(summarise_dataset(ds, per_method))
        rows = comparison_to_paper(ds, per_method)
        all_comparisons[ds] = rows
        print_comparison(ds, rows)

        # ---- Per-dataset checkpoint: write what we have so far so an interrupt
        # never wipes completed datasets ----
        ckpt_dir = OUT_DIR / "per_dataset"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ds_fold_path = ckpt_dir / f"{ds}_fold_results.csv"
        ds_sum_path  = ckpt_dir / f"{ds}_summary.csv"
        ds_cmp_path  = ckpt_dir / f"{ds}_comparison.csv"
        pd.DataFrame([r for r in fold_rows]).to_csv(ds_fold_path, index=False)
        pd.DataFrame([summarise_dataset(ds, per_method)]).to_csv(ds_sum_path, index=False)
        pd.DataFrame([{"dataset": ds, **r} for r in rows]).to_csv(ds_cmp_path, index=False)
        print(f"  CHECKPOINT: wrote {ds}_fold_results.csv, {ds}_summary.csv, {ds}_comparison.csv")

    fold_path = OUT_DIR / f"fold_results_{timestamp}.csv"
    sum_path  = OUT_DIR / f"dataset_summary_{timestamp}.csv"
    cmp_path  = OUT_DIR / f"comparison_to_paper_{timestamp}.csv"

    pd.DataFrame(all_fold_rows).to_csv(fold_path, index=False)
    pd.DataFrame(all_summaries).to_csv(sum_path, index=False)
    cmp_rows = []
    for ds, rows in all_comparisons.items():
        for r in rows:
            cmp_rows.append({"dataset": ds, **r})
    pd.DataFrame(cmp_rows).to_csv(cmp_path, index=False)

    elapsed = perf_counter() - t0
    print("\n" + "=" * 100)
    print(f"  Total elapsed : {elapsed:.0f}s ({elapsed/3600:.2f}h)")
    print(f"  Fold CSV      : {fold_path}")
    print(f"  Summary CSV   : {sum_path}")
    print(f"  Comparison    : {cmp_path}")
    print("=" * 100)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=DATASETS,
                        help="Dataset keys to run (default: all 13 local)")
    args = parser.parse_args()
    run_all(args.datasets)


if __name__ == "__main__":
    main()
