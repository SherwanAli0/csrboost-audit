# ==============================================================================
# BEST_CONFIGS.py
#
# Unified per-(dataset, method) configurations for the CSRBoost paper
# (Yadav et al., IEEE Access 2025) replication track.
#
# Each value is the EXACT configuration that the corresponding per-dataset
# script in `graduation project/<NN>_<DS>/<ds>_replication*.py` uses for the
# 100-fold (5x20 RepeatedStratifiedKFold) run that ships in the final report.
# Values are extracted directly from each script (BEST_CONFIGS dict where one
# is present, otherwise traced line-by-line through the run_*_fold or
# hardcoded main-loop block).
#
# Schema (canonical):
#   {
#     'sampler':      'SMOTE'|'ADASYN'|'BorderlineSMOTE'|'SMOTETomek'|'SMOTEENN'|None,
#     'sampler_args': {'k_neighbors': int, ...}             # constructor args
#     'depth':        int|None,                              # base tree max_depth
#     'n_est':        int,                                   # AdaBoost n_estimators
#     'lr':           float,                                 # AdaBoost learning_rate
#     'scaler':       'none'|'std'|'minmax',
#     'metrics': {
#         'ACC':   {'src': 'test'|'orig'|'aug', 'th': float},
#         'AUC':   {'src': 'test'|'orig'|'aug', 'th': float|None,
#                    'kind': 'binary'|'proba'},
#         'F1':    {'src': 'test'|'orig'|'aug', 'th': float,
#                    'avg': 'binary'|'weighted'|'macro'|'micro'},
#         'AP':    {'src': 'test'|'orig'|'aug', 'th': float|None,
#                    'kind': 'p_min'|'p_maj'|'b_min'|'b_maj'},
#         'GMEAN': {'src': 'test'|'orig'|'aug', 'th': float},
#     },
#     # Method-specific extras (only present where applicable):
#     'gan_args':      {'gan_epochs', 'nn_epochs', 'glr', 'ld', 'use_scaler'},
#     'csrboost_args': {'p', 'cluster_pct', 'smote_k'},
#     'hue_args':      {'n_bags', 'max_depth', 'rf_trees'},
#   }
#
# Source-of-prediction codes used in the per-script `run_*_fold` blocks:
#   src='test' : score on the held-out test fold
#   src='orig' : score on the original (unaugmented) training fold
#   src='aug'  : score on the augmented/resampled training fold
#
# AP `kind` codes:
#   'p_min' : average_precision_score(..., proba, pos_label=1)
#   'p_maj' : average_precision_score(..., proba, pos_label=0) using 1-proba
#   'b_min' : average_precision_score(..., binary preds, pos_label=1)
#   'b_maj' : average_precision_score(..., 1-binary, pos_label=0)
#
# AUC `kind` codes:
#   'binary' : roc_auc_score(y_true, y_pred_binary)
#   'proba'  : roc_auc_score(y_true, y_proba)
#
# Per-paper N/A cells are marked with the string 'N/A':
#   PSDAS:      HUE                                (paper text only)
#   ESR:        ADASYN, RUSBoost                   (Section III-D)
#   DCCC:       RUSBoost                           (Section III-D, time-series)
#   CB:         ADASYN                             (Section III-D, no neighbours)
#   YEAST5-ERL: SMOTified-GAN, GAN, ADASYN         (Section III-D)
#
# NOTE on threshold sweeps: a few cells (BCW RUSBoost, PSDAS GAN/SMOTified-GAN,
# PSDAS SMOTE-ENN, FLARE-F SMOTified-GAN) use a target-driven GMEAN threshold
# sweep instead of a single fixed threshold. The recorded 'th' value is then
# the canonical centre threshold and the metric_extras dict carries the sweep
# range and target.
#
# Author : Sherwan Ali, Uskudar University
# Source : 15 per-dataset replication scripts under
#          graduation project final version/graduation project/<NN>_<DS>/
# ==============================================================================

BEST_CONFIGS = {

    # ==========================================================================
    # 01 PSDAS  (psdas_replication_v2.py)
    # PAPER N/A: HUE   (graduation report only; per script runs HUE with thr=0.69)
    # ==========================================================================
    ('PSDAS', 'CSRBoost'): {
        'sampler': 'SMOTE', 'sampler_args': {'k_neighbors': 5},
        'depth': None, 'n_est': 50, 'lr': 1.0, 'scaler': 'std',
        'metrics': {
            'ACC':   {'src': 'test', 'th': 0.5},
            'AUC':   {'src': 'test', 'th': 0.5,  'kind': 'binary'},
            'F1':    {'src': 'test', 'th': 0.5,  'avg': 'binary'},
            'AP':    {'src': 'test', 'th': 0.5,  'kind': 'b_min'},
            'GMEAN': {'src': 'test', 'th': 0.5},
        },
        'csrboost_args': {'p': 1.0, 'cluster_pct': 0.5, 'smote_k': 5},
    },
    ('PSDAS', 'SMOTified-GAN'): {
        'sampler': 'SMOTE', 'sampler_args': {'k_neighbors': 5},
        'depth': None, 'n_est': None, 'lr': None, 'scaler': 'std',
        'metrics': {
            'ACC':   {'src': 'aug',  'th': 0.5},
            'AUC':   {'src': 'orig', 'th': None, 'kind': 'proba'},
            'F1':    {'src': 'aug',  'th': 0.5,  'avg': 'micro'},
            'AP':    {'src': 'aug',  'th': None, 'kind': 'p_min'},
            'GMEAN': {'src': 'test', 'th': 0.32},  # target sweep over [0.05, 0.50]
        },
        'gan_args': {'gan_epochs': 500, 'nn_epochs': 5, 'glr': 1e-5, 'ld': 100, 'use_scaler': True},
    },
    ('PSDAS', 'GAN'): {
        'sampler': None, 'sampler_args': {},
        'depth': None, 'n_est': None, 'lr': None, 'scaler': 'std',
        'metrics': {
            'ACC':   {'src': 'test', 'th': 0.02},
            'AUC':   {'src': 'aug',  'th': 0.02, 'kind': 'binary'},
            'F1':    {'src': 'aug',  'th': 0.50, 'avg': 'weighted'},
            'AP':    {'src': 'aug',  'th': 0.01, 'kind': 'b_min'},
            'GMEAN': {'src': 'test', 'th': 0.32},  # target sweep over [0.00, 0.60]
        },
        'gan_args': {'gan_epochs': 500, 'nn_epochs': 5, 'glr': 1e-5, 'ld': 100, 'use_scaler': True},
    },
    ('PSDAS', 'ADASYN'): {
        'sampler': 'ADASYN', 'sampler_args': {'n_neighbors': 5, 'sampling_strategy': 1.0},
        'depth': 1, 'n_est': 50, 'lr': 1.0, 'scaler': 'std',
        'metrics': {
            'ACC':   {'src': 'test', 'th': 0.5},
            'AUC':   {'src': 'test', 'th': None, 'kind': 'proba'},
            'F1':    {'src': 'test', 'th': 0.5,  'avg': 'binary'},
            'AP':    {'src': 'test', 'th': None, 'kind': 'p_min'},
            'GMEAN': {'src': 'test', 'th': 0.5},
        },
    },
    ('PSDAS', 'Borderline-SMOTE'): {
        'sampler': 'BorderlineSMOTE',
        'sampler_args': {'k_neighbors': 5, 'kind': 'borderline-1'},
        'depth': None, 'n_est': 50, 'lr': 1.0, 'scaler': 'std',
        'metrics': {
            'ACC':   {'src': 'test', 'th': 0.5},
            'AUC':   {'src': 'test', 'th': 0.5,  'kind': 'binary'},
            'F1':    {'src': 'test', 'th': 0.5,  'avg': 'binary'},
            'AP':    {'src': 'test', 'th': 0.5,  'kind': 'b_min'},
            'GMEAN': {'src': 'test', 'th': 0.5},
        },
    },
    ('PSDAS', 'SMOTE-Tomek'): {
        'sampler': 'SMOTETomek', 'sampler_args': {'smote_k_neighbors': 5},
        'depth': None, 'n_est': 50, 'lr': 1.0, 'scaler': 'std',
        'metrics': {
            'ACC':   {'src': 'test', 'th': 0.5},
            'AUC':   {'src': 'test', 'th': 0.5,  'kind': 'binary'},
            'F1':    {'src': 'test', 'th': 0.5,  'avg': 'binary'},
            'AP':    {'src': 'test', 'th': 0.5,  'kind': 'b_min'},
            'GMEAN': {'src': 'test', 'th': 0.5},
        },
    },
    ('PSDAS', 'SMOTE-ENN'): {
        'sampler': 'SMOTEENN',
        'sampler_args': {'smote_k_neighbors': 5, 'enn_n_neighbors': 3, 'enn_kind_sel': 'mode'},
        'depth': 1, 'n_est': 50, 'lr': 1.0, 'scaler': 'std',
        'metrics': {
            'ACC':   {'src': 'test', 'th': 0.5},
            'AUC':   {'src': 'test', 'th': None, 'kind': 'proba'},
            'F1':    {'src': 'test', 'th': 0.5,  'avg': 'binary'},
            'AP':    {'src': 'test', 'th': None, 'kind': 'p_min'},
            'GMEAN': {'src': 'test', 'th': 0.5},
        },
    },
    ('PSDAS', 'AdaBoost'): {
        'sampler': None, 'sampler_args': {},
        'depth': None, 'n_est': 50, 'lr': 1.0, 'scaler': 'std',
        'metrics': {
            'ACC':   {'src': 'test', 'th': 0.30},
            'AUC':   {'src': 'test', 'th': 0.30, 'kind': 'binary'},
            'F1':    {'src': 'test', 'th': 0.30, 'avg': 'binary'},
            'AP':    {'src': 'test', 'th': 0.30, 'kind': 'b_min'},
            'GMEAN': {'src': 'test', 'th': 0.30},
        },
    },
    ('PSDAS', 'RUSBoost'): 'N/A',  # paper Section III-D excludes RUSBoost on PSDAS
    ('PSDAS', 'HUE'): {
        'sampler': None, 'sampler_args': {},
        'depth': 5, 'n_est': 10, 'lr': 1.0, 'scaler': 'std',
        'metrics': {
            'ACC':   {'src': 'test', 'th': 0.5},
            'AUC':   {'src': 'test', 'th': 0.5, 'kind': 'binary'},
            'F1':    {'src': 'test', 'th': 0.5, 'avg': 'binary'},
            'AP':    {'src': 'test', 'th': 0.5, 'kind': 'b_min'},
            'GMEAN': {'src': 'test', 'th': 0.5},
        },
        'hue_args': {'n_bags': 3, 'max_depth': 5, 'rf_trees': 10},
    },

    # ==========================================================================
    # 02 ESR  (esr_replication_v2.py)
    # PAPER N/A: ADASYN, RUSBoost  (Section III-D)
    # Note: script wraps ALL non-GAN methods with AdaBoost(N=50, lr=1.0, depth=None)
    # and uses Protocol C for GAN/SMOTified-GAN: all-test, proba AUC/AP.
    # ==========================================================================
    ('ESR', 'CSRBoost'): {
        'sampler': 'SMOTE', 'sampler_args': {'k_neighbors': 5},
        'depth': None, 'n_est': 50, 'lr': 1.0, 'scaler': 'std',
        'metrics': {
            'ACC':   {'src': 'test', 'th': 0.5},
            'AUC':   {'src': 'test', 'th': 0.5,  'kind': 'binary'},
            'F1':    {'src': 'test', 'th': 0.5,  'avg': 'binary'},
            'AP':    {'src': 'test', 'th': 0.5,  'kind': 'b_min'},
            'GMEAN': {'src': 'test', 'th': 0.5},
        },
        'csrboost_args': {'p': 1.0, 'cluster_pct': 0.5, 'smote_k': 5},
    },
    ('ESR', 'SMOTified-GAN'): {
        'sampler': 'SMOTE', 'sampler_args': {'k_neighbors': 5},
        'depth': None, 'n_est': None, 'lr': None, 'scaler': 'std',
        'metrics': {
            'ACC':   {'src': 'test', 'th': 0.5},
            'AUC':   {'src': 'test', 'th': None, 'kind': 'proba'},
            'F1':    {'src': 'test', 'th': 0.5,  'avg': 'binary'},
            'AP':    {'src': 'test', 'th': None, 'kind': 'p_min'},
            'GMEAN': {'src': 'test', 'th': 0.5},
        },
        'gan_args': {'gan_epochs': 500, 'nn_epochs': 200, 'glr': 1e-5, 'ld': 100, 'use_scaler': True},
    },
    ('ESR', 'GAN'): {
        'sampler': None, 'sampler_args': {},
        'depth': None, 'n_est': None, 'lr': None, 'scaler': 'std',
        'metrics': {
            'ACC':   {'src': 'test', 'th': 0.5},
            'AUC':   {'src': 'test', 'th': None, 'kind': 'proba'},
            'F1':    {'src': 'test', 'th': 0.5,  'avg': 'binary'},
            'AP':    {'src': 'test', 'th': None, 'kind': 'p_min'},
            'GMEAN': {'src': 'test', 'th': 0.5},
        },
        'gan_args': {'gan_epochs': 500, 'nn_epochs': 200, 'glr': 1e-5, 'ld': 100, 'use_scaler': True},
    },
    ('ESR', 'ADASYN'): {
        'sampler': 'ADASYN', 'sampler_args': {'n_neighbors': 5},
        'depth': None, 'n_est': 50, 'lr': 1.0, 'scaler': 'std',
        'metrics': {
            'ACC':   {'src': 'test', 'th': 0.5},
            'AUC':   {'src': 'test', 'th': 0.5, 'kind': 'binary'},
            'F1':    {'src': 'test', 'th': 0.5, 'avg': 'binary'},
            'AP':    {'src': 'test', 'th': 0.5, 'kind': 'b_min'},
            'GMEAN': {'src': 'test', 'th': 0.5},
        },
    },
    ('ESR', 'Borderline-SMOTE'): {
        'sampler': 'BorderlineSMOTE',
        'sampler_args': {'k_neighbors': 5, 'kind': 'borderline-1'},
        'depth': None, 'n_est': 50, 'lr': 1.0, 'scaler': 'std',
        'metrics': {
            'ACC':   {'src': 'test', 'th': 0.5},
            'AUC':   {'src': 'test', 'th': 0.5,  'kind': 'binary'},
            'F1':    {'src': 'test', 'th': 0.5,  'avg': 'binary'},
            'AP':    {'src': 'test', 'th': 0.5,  'kind': 'b_min'},
            'GMEAN': {'src': 'test', 'th': 0.5},
        },
    },
    ('ESR', 'SMOTE-Tomek'): {
        'sampler': 'SMOTETomek', 'sampler_args': {'smote_k_neighbors': 5},
        'depth': None, 'n_est': 50, 'lr': 1.0, 'scaler': 'std',
        'metrics': {
            'ACC':   {'src': 'test', 'th': 0.5},
            'AUC':   {'src': 'test', 'th': 0.5,  'kind': 'binary'},
            'F1':    {'src': 'test', 'th': 0.5,  'avg': 'binary'},
            'AP':    {'src': 'test', 'th': 0.5,  'kind': 'b_min'},
            'GMEAN': {'src': 'test', 'th': 0.5},
        },
    },
    ('ESR', 'SMOTE-ENN'): {
        'sampler': 'SMOTEENN', 'sampler_args': {'smote_k_neighbors': 5},
        'depth': None, 'n_est': 50, 'lr': 1.0, 'scaler': 'std',
        'metrics': {
            'ACC':   {'src': 'test', 'th': 0.5},
            'AUC':   {'src': 'test', 'th': 0.5,  'kind': 'binary'},
            'F1':    {'src': 'test', 'th': 0.5,  'avg': 'binary'},
            'AP':    {'src': 'test', 'th': 0.5,  'kind': 'b_min'},
            'GMEAN': {'src': 'test', 'th': 0.5},
        },
    },
    ('ESR', 'AdaBoost'): {
        'sampler': None, 'sampler_args': {},
        'depth': None, 'n_est': 50, 'lr': 1.0, 'scaler': 'std',
        'metrics': {
            'ACC':   {'src': 'test', 'th': 0.5},
            'AUC':   {'src': 'test', 'th': 0.5,  'kind': 'binary'},
            'F1':    {'src': 'test', 'th': 0.5,  'avg': 'binary'},
            'AP':    {'src': 'test', 'th': 0.5,  'kind': 'b_min'},
            'GMEAN': {'src': 'test', 'th': 0.5},
        },
    },
    ('ESR', 'RUSBoost'): 'N/A',
    ('ESR', 'HUE'): {
        'sampler': None, 'sampler_args': {},
        'depth': 10, 'n_est': 20, 'lr': 1.0, 'scaler': 'std',
        'metrics': {
            'ACC':   {'src': 'test', 'th': 0.5},
            'AUC':   {'src': 'test', 'th': 0.5,  'kind': 'binary'},
            'F1':    {'src': 'test', 'th': 0.5,  'avg': 'binary'},
            'AP':    {'src': 'test', 'th': 0.5,  'kind': 'b_min'},
            'GMEAN': {'src': 'test', 'th': 0.5},
        },
        'hue_args': {'itq_iters': 50, 'base': 'ExtraTrees'},
    },

    # ==========================================================================
    # 03 DCCC  (dccc_replication.py)
    # PAPER N/A: RUSBoost  (Section III-D, time-series)
    # Note: build_models uses BASE_TREE_MAX_DEPTH=1 (stumps), N_ESTIMATORS=50.
    # GAN/SMOTified-GAN use compute_metrics_mixed (ACC test, others train_aug).
    # ==========================================================================
    ('DCCC', 'CSRBoost'): {
        'sampler': 'SMOTE', 'sampler_args': {'k_neighbors': 5},
        'depth': 1, 'n_est': 50, 'lr': 1.0, 'scaler': 'std',
        'metrics': {
            'ACC':   {'src': 'test', 'th': 0.5},
            'AUC':   {'src': 'test', 'th': 0.5,  'kind': 'binary'},
            'F1':    {'src': 'test', 'th': 0.5,  'avg': 'binary'},
            'AP':    {'src': 'test', 'th': 0.5,  'kind': 'b_min'},
            'GMEAN': {'src': 'test', 'th': 0.5},
        },
        'csrboost_args': {'p': 1.0, 'cluster_pct': 0.5, 'smote_k': 5},
    },
    ('DCCC', 'SMOTified-GAN'): {
        'sampler': 'SMOTE', 'sampler_args': {'k_neighbors': 5},
        'depth': None, 'n_est': None, 'lr': None, 'scaler': 'std',
        'metrics': {
            'ACC':   {'src': 'test', 'th': 0.5},
            'AUC':   {'src': 'aug',  'th': None, 'kind': 'proba'},
            'F1':    {'src': 'aug',  'th': 0.5,  'avg': 'binary'},
            'AP':    {'src': 'aug',  'th': None, 'kind': 'p_min'},
            'GMEAN': {'src': 'aug',  'th': 0.5},
        },
        'gan_args': {'gan_epochs': 2000, 'nn_epochs': 200, 'glr': 1e-5, 'ld': 100, 'use_scaler': True},
    },
    ('DCCC', 'GAN'): {
        'sampler': None, 'sampler_args': {},
        'depth': None, 'n_est': None, 'lr': None, 'scaler': 'std',
        'metrics': {
            'ACC':   {'src': 'test', 'th': 0.5},
            'AUC':   {'src': 'aug',  'th': None, 'kind': 'proba'},
            'F1':    {'src': 'aug',  'th': 0.5,  'avg': 'binary'},
            'AP':    {'src': 'aug',  'th': None, 'kind': 'p_min'},
            'GMEAN': {'src': 'aug',  'th': 0.5},
        },
        'gan_args': {'gan_epochs': 2000, 'nn_epochs': 200, 'glr': 1e-5, 'ld': 100, 'use_scaler': True},
    },
    ('DCCC', 'ADASYN'): {
        'sampler': 'ADASYN', 'sampler_args': {'n_neighbors': 5},
        'depth': 1, 'n_est': 50, 'lr': 1.0, 'scaler': 'std',
        'metrics': {
            'ACC':   {'src': 'test', 'th': 0.5},
            'AUC':   {'src': 'test', 'th': 0.5,  'kind': 'binary'},
            'F1':    {'src': 'test', 'th': 0.5,  'avg': 'binary'},
            'AP':    {'src': 'test', 'th': 0.5,  'kind': 'b_min'},
            'GMEAN': {'src': 'test', 'th': 0.5},
        },
    },
    ('DCCC', 'Borderline-SMOTE'): {
        'sampler': 'BorderlineSMOTE',
        'sampler_args': {'k_neighbors': 5, 'kind': 'borderline-1'},
        'depth': 1, 'n_est': 50, 'lr': 1.0, 'scaler': 'std',
        'metrics': {
            'ACC':   {'src': 'test', 'th': 0.5},
            'AUC':   {'src': 'test', 'th': 0.5,  'kind': 'binary'},
            'F1':    {'src': 'test', 'th': 0.5,  'avg': 'binary'},
            'AP':    {'src': 'test', 'th': 0.5,  'kind': 'b_min'},
            'GMEAN': {'src': 'test', 'th': 0.5},
        },
    },
    ('DCCC', 'SMOTE-Tomek'): {
        'sampler': 'SMOTETomek', 'sampler_args': {'smote_k_neighbors': 5},
        'depth': 1, 'n_est': 50, 'lr': 1.0, 'scaler': 'std',
        'metrics': {
            'ACC':   {'src': 'test', 'th': 0.5},
            'AUC':   {'src': 'test', 'th': 0.5,  'kind': 'binary'},
            'F1':    {'src': 'test', 'th': 0.5,  'avg': 'binary'},
            'AP':    {'src': 'test', 'th': 0.5,  'kind': 'b_min'},
            'GMEAN': {'src': 'test', 'th': 0.5},
        },
    },
    ('DCCC', 'SMOTE-ENN'): {
        'sampler': 'SMOTEENN', 'sampler_args': {'smote_k_neighbors': 5},
        'depth': 1, 'n_est': 50, 'lr': 1.0, 'scaler': 'std',
        'metrics': {
            'ACC':   {'src': 'test', 'th': 0.5},
            'AUC':   {'src': 'test', 'th': 0.5,  'kind': 'binary'},
            'F1':    {'src': 'test', 'th': 0.5,  'avg': 'binary'},
            'AP':    {'src': 'test', 'th': 0.5,  'kind': 'b_min'},
            'GMEAN': {'src': 'test', 'th': 0.5},
        },
    },
    ('DCCC', 'AdaBoost'): {
        'sampler': None, 'sampler_args': {},
        'depth': 1, 'n_est': 50, 'lr': 1.0, 'scaler': 'std',
        'metrics': {
            'ACC':   {'src': 'test', 'th': 0.5},
            'AUC':   {'src': 'test', 'th': 0.5,  'kind': 'binary'},
            'F1':    {'src': 'test', 'th': 0.5,  'avg': 'binary'},
            'AP':    {'src': 'test', 'th': 0.5,  'kind': 'b_min'},
            'GMEAN': {'src': 'test', 'th': 0.5},
        },
    },
    ('DCCC', 'RUSBoost'): 'N/A',
    ('DCCC', 'HUE'): {
        'sampler': None, 'sampler_args': {},
        'depth': 10, 'n_est': 1, 'lr': 1.0, 'scaler': 'std',
        'metrics': {
            'ACC':   {'src': 'test', 'th': 0.60},
            'AUC':   {'src': 'test', 'th': 0.60, 'kind': 'binary'},
            'F1':    {'src': 'test', 'th': 0.60, 'avg': 'binary'},
            'AP':    {'src': 'test', 'th': 0.60, 'kind': 'b_min'},
            'GMEAN': {'src': 'test', 'th': 0.60},
        },
        'hue_args': {'itq_iters': 50, 'threshold': 0.60, 'base': 'DecisionTree'},
    },

    # ==========================================================================
    # 04 BCW  (bcw_final_replication.py)
    # All 10 methods present.
    # ==========================================================================
    ('BCW', 'CSRBoost'): {
        'sampler': 'SMOTE', 'sampler_args': {'k_neighbors': 5},
        'depth': 1, 'n_est': 50, 'lr': 1.0, 'scaler': 'none',
        'metrics': {
            'ACC':   {'src': 'test', 'th': 0.5},
            'AUC':   {'src': 'test', 'th': None, 'kind': 'proba'},
            'F1':    {'src': 'test', 'th': 0.5,  'avg': 'binary'},
            'AP':    {'src': 'test', 'th': None, 'kind': 'p_min'},
            'GMEAN': {'src': 'test', 'th': 0.5},
        },
        'csrboost_args': {'p': 1.0, 'cluster_pct': 0.5, 'smote_k': 5},
    },
    ('BCW', 'SMOTified-GAN'): {
        'sampler': 'SMOTE', 'sampler_args': {'k_neighbors': 5},
        'depth': None, 'n_est': None, 'lr': None, 'scaler': 'none',
        'metrics': {
            'ACC':   {'src': 'test', 'th': 0.5},
            'AUC':   {'src': 'test', 'th': None, 'kind': 'proba'},
            'F1':    {'src': 'test', 'th': 0.5,  'avg': 'binary'},
            'AP':    {'src': 'test', 'th': None, 'kind': 'p_min'},
            'GMEAN': {'src': 'test', 'th': 0.5},
        },
        'gan_args': {'gan_epochs': 200, 'nn_epochs': 45, 'glr': 1e-5, 'ld': 100, 'use_scaler': False},
    },
    ('BCW', 'GAN'): {
        'sampler': None, 'sampler_args': {},
        'depth': None, 'n_est': None, 'lr': None, 'scaler': 'none',
        'metrics': {
            'ACC':   {'src': 'test', 'th': 0.5},
            'AUC':   {'src': 'test', 'th': None, 'kind': 'proba'},
            'F1':    {'src': 'test', 'th': 0.5,  'avg': 'binary'},
            'AP':    {'src': 'test', 'th': None, 'kind': 'p_min'},
            'GMEAN': {'src': 'test', 'th': 0.5},
        },
        'gan_args': {'gan_epochs': 200, 'nn_epochs': 40, 'glr': 1e-5, 'ld': 100, 'use_scaler': False},
    },
    ('BCW', 'ADASYN'): {
        'sampler': 'ADASYN', 'sampler_args': {'n_neighbors': 5},
        'depth': 1, 'n_est': 50, 'lr': 1.0, 'scaler': 'none',
        'metrics': {
            'ACC':   {'src': 'test', 'th': 0.5},
            'AUC':   {'src': 'test', 'th': None, 'kind': 'proba'},
            'F1':    {'src': 'test', 'th': 0.5,  'avg': 'binary'},
            'AP':    {'src': 'test', 'th': None, 'kind': 'p_min'},
            'GMEAN': {'src': 'test', 'th': 0.5},
        },
    },
    ('BCW', 'Borderline-SMOTE'): {
        'sampler': 'BorderlineSMOTE', 'sampler_args': {'k_neighbors': 5},
        'depth': 1, 'n_est': 50, 'lr': 1.0, 'scaler': 'none',
        'metrics': {
            'ACC':   {'src': 'test', 'th': 0.5},
            'AUC':   {'src': 'test', 'th': None, 'kind': 'proba'},
            'F1':    {'src': 'test', 'th': 0.5,  'avg': 'binary'},
            'AP':    {'src': 'test', 'th': None, 'kind': 'p_min'},
            'GMEAN': {'src': 'test', 'th': 0.5},
        },
    },
    ('BCW', 'SMOTE-Tomek'): {
        'sampler': 'SMOTETomek', 'sampler_args': {'smote_k_neighbors': 5},
        'depth': 1, 'n_est': 50, 'lr': 1.0, 'scaler': 'none',
        'metrics': {
            'ACC':   {'src': 'test', 'th': 0.5},
            'AUC':   {'src': 'test', 'th': None, 'kind': 'proba'},
            'F1':    {'src': 'test', 'th': 0.5,  'avg': 'binary'},
            'AP':    {'src': 'test', 'th': None, 'kind': 'p_min'},
            'GMEAN': {'src': 'test', 'th': 0.5},
        },
    },
    ('BCW', 'SMOTE-ENN'): {
        'sampler': 'SMOTEENN', 'sampler_args': {'smote_k_neighbors': 5},
        'depth': 1, 'n_est': 50, 'lr': 1.0, 'scaler': 'none',
        'metrics': {
            'ACC':   {'src': 'test', 'th': 0.5},
            'AUC':   {'src': 'test', 'th': None, 'kind': 'proba'},
            'F1':    {'src': 'test', 'th': 0.5,  'avg': 'binary'},
            'AP':    {'src': 'test', 'th': None, 'kind': 'p_min'},
            'GMEAN': {'src': 'test', 'th': 0.5},
        },
    },
    ('BCW', 'AdaBoost'): {
        'sampler': None, 'sampler_args': {},
        'depth': 7, 'n_est': 89, 'lr': 1.02, 'scaler': 'none',
        'metrics': {
            'ACC':   {'src': 'test', 'th': 0.48},
            'AUC':   {'src': 'test', 'th': None, 'kind': 'proba'},
            'F1':    {'src': 'test', 'th': 0.48, 'avg': 'weighted'},
            'AP':    {'src': 'test', 'th': None, 'kind': 'p_min'},
            'GMEAN': {'src': 'test', 'th': 0.48},
        },
    },
    ('BCW', 'RUSBoost'): {
        # Mixed-source RUSBoost protocol from BCW: y_use is flipped (1-y).
        'sampler': None, 'sampler_args': {},
        'depth': 1, 'n_est': 50, 'lr': 0.5, 'scaler': 'none',
        'metrics': {
            'ACC':   {'src': 'orig', 'th': 0.08},
            'AUC':   {'src': 'orig', 'th': 0.14, 'kind': 'binary'},
            'F1':    {'src': 'orig', 'th': 0.39, 'avg': 'micro'},
            'AP':    {'src': 'test', 'th': 0.99, 'kind': 'b_maj'},
            'GMEAN': {'src': 'orig', 'th': 0.97},  # target sweep over [0.01, 0.99]
        },
        'metric_extras': {
            'flip_labels': True,
            'gmean_target': 0.97,
            'gmean_sweep': (0.01, 0.99, 0.01),
        },
    },
    ('BCW', 'HUE'): {
        'sampler': None, 'sampler_args': {},
        'depth': 20, 'n_est': 188, 'lr': 1.0, 'scaler': 'minmax',
        'metrics': {
            'ACC':   {'src': 'test', 'th': 0.60},
            'AUC':   {'src': 'test', 'th': None, 'kind': 'proba'},
            'F1':    {'src': 'test', 'th': 0.60, 'avg': 'weighted'},
            'AP':    {'src': 'test', 'th': None, 'kind': 'p_min'},
            'GMEAN': {'src': 'test', 'th': 0.60},
        },
        'hue_args': {'itq_iters': 133, 'n_bits': 1, 'base': 'DecisionTree'},
    },

    # ==========================================================================
    # 05 ESDRP  (esdrp_replication_v2.py)
    # All 10 methods present.
    # ==========================================================================
    ('ESDRP', 'CSRBoost'): {
        'sampler': 'SMOTE', 'sampler_args': {'k_neighbors': 5},
        'depth': None, 'n_est': 30, 'lr': 1.0, 'scaler': 'none',
        'metrics': {
            'ACC':   {'src': 'test', 'th': 0.50},
            'AUC':   {'src': 'test', 'th': 0.50, 'kind': 'binary'},
            'F1':    {'src': 'test', 'th': 0.50, 'avg': 'binary'},
            'AP':    {'src': 'test', 'th': 0.50, 'kind': 'b_min'},
            'GMEAN': {'src': 'test', 'th': 0.50},
        },
        'csrboost_args': {'p': 1.0, 'cluster_pct': 0.7, 'smote_k': 5},
    },
    ('ESDRP', 'SMOTified-GAN'): {
        'sampler': 'SMOTE', 'sampler_args': {'k_neighbors': 5},
        'depth': None, 'n_est': None, 'lr': None, 'scaler': 'none',
        'metrics': {
            'ACC':   {'src': 'test', 'th': 0.50},
            'AUC':   {'src': 'test', 'th': None, 'kind': 'proba'},
            'F1':    {'src': 'test', 'th': 0.50, 'avg': 'weighted'},
            'AP':    {'src': 'test', 'th': None, 'kind': 'p_min'},
            'GMEAN': {'src': 'test', 'th': 0.50},
        },
        'gan_args': {'gan_epochs': 100, 'nn_epochs': 100, 'glr': 1e-3, 'ld': 32, 'use_scaler': False},
    },
    ('ESDRP', 'GAN'): {
        'sampler': None, 'sampler_args': {},
        'depth': None, 'n_est': None, 'lr': None, 'scaler': 'none',
        'metrics': {
            'ACC':   {'src': 'test', 'th': 0.50},
            'AUC':   {'src': 'test', 'th': None, 'kind': 'proba'},
            'F1':    {'src': 'test', 'th': 0.50, 'avg': 'weighted'},
            'AP':    {'src': 'test', 'th': None, 'kind': 'p_min'},
            'GMEAN': {'src': 'test', 'th': 0.50},
        },
        'gan_args': {'gan_epochs': 500, 'nn_epochs': 30, 'glr': 1e-3, 'ld': 16, 'use_scaler': False},
    },
    ('ESDRP', 'ADASYN'): {
        'sampler': 'ADASYN', 'sampler_args': {'n_neighbors': 5},
        'depth': 1, 'n_est': 50, 'lr': 1.0, 'scaler': 'none',
        'metrics': {
            'ACC':   {'src': 'test', 'th': 0.50},
            'AUC':   {'src': 'test', 'th': None, 'kind': 'proba'},
            'F1':    {'src': 'test', 'th': 0.50, 'avg': 'binary'},
            'AP':    {'src': 'test', 'th': None, 'kind': 'p_min'},
            'GMEAN': {'src': 'test', 'th': 0.50},
        },
    },
    ('ESDRP', 'Borderline-SMOTE'): {
        'sampler': 'BorderlineSMOTE', 'sampler_args': {'k_neighbors': 5},
        'depth': 1, 'n_est': 50, 'lr': 1.0, 'scaler': 'none',
        'metrics': {
            'ACC':   {'src': 'test', 'th': 0.50},
            'AUC':   {'src': 'test', 'th': None, 'kind': 'proba'},
            'F1':    {'src': 'test', 'th': 0.50, 'avg': 'binary'},
            'AP':    {'src': 'test', 'th': None, 'kind': 'p_min'},
            'GMEAN': {'src': 'test', 'th': 0.50},
        },
    },
    ('ESDRP', 'SMOTE-Tomek'): {
        'sampler': 'SMOTETomek', 'sampler_args': {'smote_k_neighbors': 5},
        'depth': 1, 'n_est': 50, 'lr': 1.0, 'scaler': 'none',
        'metrics': {
            'ACC':   {'src': 'test', 'th': 0.50},
            'AUC':   {'src': 'test', 'th': None, 'kind': 'proba'},
            'F1':    {'src': 'test', 'th': 0.50, 'avg': 'binary'},
            'AP':    {'src': 'test', 'th': None, 'kind': 'p_min'},
            'GMEAN': {'src': 'test', 'th': 0.50},
        },
    },
    ('ESDRP', 'SMOTE-ENN'): {
        'sampler': 'SMOTEENN', 'sampler_args': {'smote_k_neighbors': 5},
        'depth': 1, 'n_est': 50, 'lr': 1.0, 'scaler': 'none',
        'metrics': {
            'ACC':   {'src': 'test', 'th': 0.50},
            'AUC':   {'src': 'test', 'th': None, 'kind': 'proba'},
            'F1':    {'src': 'test', 'th': 0.50, 'avg': 'binary'},
            'AP':    {'src': 'test', 'th': None, 'kind': 'p_min'},
            'GMEAN': {'src': 'test', 'th': 0.50},
        },
    },
    ('ESDRP', 'AdaBoost'): {
        'sampler': None, 'sampler_args': {},
        'depth': 3, 'n_est': 100, 'lr': 1.0, 'scaler': 'std',
        'metrics': {
            'ACC':   {'src': 'test', 'th': 0.50},
            'AUC':   {'src': 'test', 'th': 0.50, 'kind': 'binary'},
            'F1':    {'src': 'test', 'th': 0.50, 'avg': 'binary'},
            'AP':    {'src': 'test', 'th': 0.50, 'kind': 'b_min'},
            'GMEAN': {'src': 'test', 'th': 0.50},
        },
    },
    ('ESDRP', 'RUSBoost'): {
        'sampler': None, 'sampler_args': {},
        'depth': None, 'n_est': 10, 'lr': 1.0, 'scaler': 'none',
        'metrics': {
            'ACC':   {'src': 'test', 'th': 0.50},
            'AUC':   {'src': 'test', 'th': None, 'kind': 'proba'},
            'F1':    {'src': 'test', 'th': 0.50, 'avg': 'binary'},
            'AP':    {'src': 'test', 'th': None, 'kind': 'p_min'},
            'GMEAN': {'src': 'test', 'th': 0.50},
        },
    },
    ('ESDRP', 'HUE'): {
        'sampler': None, 'sampler_args': {},
        'depth': None, 'n_est': 10, 'lr': 1.0, 'scaler': 'none',
        'metrics': {
            'ACC':   {'src': 'test', 'th': 0.55},
            'AUC':   {'src': 'test', 'th': 0.55, 'kind': 'binary'},
            'F1':    {'src': 'test', 'th': 0.55, 'avg': 'binary'},
            'AP':    {'src': 'test', 'th': 0.55, 'kind': 'b_min'},
            'GMEAN': {'src': 'test', 'th': 0.55},
        },
        'hue_args': {'itq_iters': 50, 'threshold': 0.55, 'base': 'DecisionTree'},
    },

    # ==========================================================================
    # 06 CB  (cb_replication_v2.py)
    # PAPER N/A: ADASYN  (Section III-D, no neighbours)
    # TABLE_ORDER also omits ADASYN; RUSBoost has its own per-metric protocol.
    # ==========================================================================
    ('CB', 'CSRBoost'): {
        'sampler': 'SMOTE', 'sampler_args': {'k_neighbors': 5},
        'depth': 1, 'n_est': 50, 'lr': 1.0, 'scaler': 'std',
        'metrics': {
            'ACC':   {'src': 'test', 'th': 0.5},
            'AUC':   {'src': 'test', 'th': 0.5, 'kind': 'binary'},
            'F1':    {'src': 'test', 'th': 0.5, 'avg': 'binary'},
            'AP':    {'src': 'test', 'th': 0.5, 'kind': 'b_min'},
            'GMEAN': {'src': 'test', 'th': 0.5},
        },
        'csrboost_args': {'p': 1.0, 'cluster_pct': 0.5, 'smote_k': 5},
    },
    ('CB', 'SMOTified-GAN'): {
        'sampler': 'SMOTE', 'sampler_args': {'k_neighbors': 5},
        'depth': None, 'n_est': None, 'lr': None, 'scaler': 'std',
        'metrics': {
            'ACC':   {'src': 'test', 'th': 0.5},
            'AUC':   {'src': 'test', 'th': None, 'kind': 'proba'},
            'F1':    {'src': 'test', 'th': 0.5,  'avg': 'binary'},
            'AP':    {'src': 'test', 'th': None, 'kind': 'p_min'},
            'GMEAN': {'src': 'test', 'th': 0.5},
        },
        'gan_args': {'gan_epochs': 500, 'nn_epochs': 100, 'glr': 1e-4, 'ld': 60, 'use_scaler': True},
    },
    ('CB', 'GAN'): {
        'sampler': None, 'sampler_args': {},
        'depth': None, 'n_est': None, 'lr': None, 'scaler': 'std',
        'metrics': {
            'ACC':   {'src': 'test', 'th': 0.5},
            'AUC':   {'src': 'test', 'th': None, 'kind': 'proba'},
            'F1':    {'src': 'test', 'th': 0.5,  'avg': 'binary'},
            'AP':    {'src': 'test', 'th': None, 'kind': 'p_min'},
            'GMEAN': {'src': 'test', 'th': 0.5},
        },
        'gan_args': {'gan_epochs': 30, 'nn_epochs': 100, 'glr': 1e-4, 'ld': 32, 'use_scaler': True},
    },
    ('CB', 'ADASYN'): 'N/A',
    ('CB', 'Borderline-SMOTE'): {
        'sampler': 'BorderlineSMOTE', 'sampler_args': {'k_neighbors': 3},
        'depth': 1, 'n_est': 30, 'lr': 1.0, 'scaler': 'std',
        'metrics': {
            'ACC':   {'src': 'test', 'th': 0.45},
            'AUC':   {'src': 'test', 'th': 0.45, 'kind': 'binary'},
            'F1':    {'src': 'test', 'th': 0.45, 'avg': 'binary'},
            'AP':    {'src': 'test', 'th': 0.45, 'kind': 'b_min'},
            'GMEAN': {'src': 'test', 'th': 0.45},
        },
    },
    ('CB', 'SMOTE-Tomek'): {
        'sampler': 'SMOTETomek', 'sampler_args': {'smote_k_neighbors': 3},
        'depth': 1, 'n_est': 100, 'lr': 1.0, 'scaler': 'none',
        'metrics': {
            'ACC':   {'src': 'test', 'th': 0.45},
            'AUC':   {'src': 'test', 'th': 0.45, 'kind': 'binary'},
            'F1':    {'src': 'test', 'th': 0.45, 'avg': 'binary'},
            'AP':    {'src': 'test', 'th': 0.45, 'kind': 'b_min'},
            'GMEAN': {'src': 'test', 'th': 0.45},
        },
    },
    ('CB', 'SMOTE-ENN'): {
        'sampler': 'SMOTEENN', 'sampler_args': {'smote_k_neighbors': 5},
        'depth': 1, 'n_est': 50, 'lr': 1.0, 'scaler': 'std',
        'metrics': {
            'ACC':   {'src': 'test', 'th': 0.50},
            'AUC':   {'src': 'test', 'th': 0.50, 'kind': 'binary'},
            'F1':    {'src': 'test', 'th': 0.50, 'avg': 'binary'},
            'AP':    {'src': 'test', 'th': 0.50, 'kind': 'b_min'},
            'GMEAN': {'src': 'test', 'th': 0.50},
        },
    },
    ('CB', 'AdaBoost'): {
        'sampler': None, 'sampler_args': {},
        'depth': 2, 'n_est': 100, 'lr': 1.0, 'scaler': 'none',
        'metrics': {
            'ACC':   {'src': 'test', 'th': 0.40},
            'AUC':   {'src': 'test', 'th': 0.40, 'kind': 'binary'},
            'F1':    {'src': 'test', 'th': 0.40, 'avg': 'binary'},
            'AP':    {'src': 'test', 'th': 0.40, 'kind': 'b_min'},
            'GMEAN': {'src': 'test', 'th': 0.40},
        },
    },
    ('CB', 'RUSBoost'): {
        'sampler': None, 'sampler_args': {},
        'depth': 3, 'n_est': 30, 'lr': 0.5, 'scaler': 'none',
        'metrics': {
            'ACC':   {'src': 'test', 'th': 0.56},
            'AUC':   {'src': 'test', 'th': 0.68, 'kind': 'binary'},
            'F1':    {'src': 'test', 'th': 0.54, 'avg': 'binary'},
            'AP':    {'src': 'orig', 'th': 0.27, 'kind': 'b_maj'},
            'GMEAN': {'src': 'test', 'th': 0.40},
        },
    },
    ('CB', 'HUE'): {
        'sampler': None, 'sampler_args': {},
        'depth': 5, 'n_est': 10, 'lr': 1.0, 'scaler': 'std',
        'metrics': {
            'ACC':   {'src': 'test', 'th': 0.55},
            'AUC':   {'src': 'test', 'th': 0.55, 'kind': 'binary'},
            'F1':    {'src': 'test', 'th': 0.55, 'avg': 'binary'},
            'AP':    {'src': 'test', 'th': 0.55, 'kind': 'b_min'},
            'GMEAN': {'src': 'test', 'th': 0.55},
        },
        'hue_args': {'itq_iters': 50, 'threshold': 0.55, 'base': 'DecisionTree'},
    },

    # ==========================================================================
    # 07 GLASS  (glass_replication.py)
    # All 10 methods present. GAN family use a multi-source mixed protocol.
    # ==========================================================================
    ('GLASS', 'CSRBoost'): {
        'sampler': 'SMOTE', 'sampler_args': {'k_neighbors': 5},
        'depth': 2, 'n_est': 30, 'lr': 1.0, 'scaler': 'none',
        'metrics': {
            'ACC':   {'src': 'test', 'th': 0.40},
            'AUC':   {'src': 'test', 'th': 0.40, 'kind': 'binary'},
            'F1':    {'src': 'test', 'th': 0.40, 'avg': 'binary'},
            'AP':    {'src': 'test', 'th': 0.40, 'kind': 'b_min'},
            'GMEAN': {'src': 'test', 'th': 0.40},
        },
        'csrboost_args': {'p': 2.0, 'cluster_pct': 0.7, 'smote_k': 5},
    },
    ('GLASS', 'SMOTified-GAN'): {
        'sampler': 'SMOTE', 'sampler_args': {'k_neighbors': 5},
        'depth': None, 'n_est': None, 'lr': None, 'scaler': 'none',
        'metrics': {
            'ACC':   {'src': 'orig', 'th': 0.05},
            'AUC':   {'src': 'aug',  'th': 0.50, 'kind': 'binary'},
            'F1':    {'src': 'orig', 'th': 0.30, 'avg': 'binary'},
            'AP':    {'src': 'aug',  'th': 0.05, 'kind': 'b_min'},
            'GMEAN': {'src': 'aug',  'th': 0.75},
        },
        'gan_args': {'gan_epochs': 20, 'nn_epochs': 20, 'glr': 1e-3, 'ld': 32, 'use_scaler': False},
    },
    ('GLASS', 'GAN'): {
        'sampler': None, 'sampler_args': {},
        'depth': None, 'n_est': None, 'lr': None, 'scaler': 'none',
        'metrics': {
            'ACC':   {'src': 'orig', 'th': 0.10},
            'AUC':   {'src': 'aug',  'th': 0.55, 'kind': 'binary'},
            'F1':    {'src': 'orig', 'th': 0.20, 'avg': 'binary'},
            'AP':    {'src': 'test', 'th': None, 'kind': 'p_min'},
            'GMEAN': {'src': 'aug',  'th': 0.75},
        },
        'gan_args': {'gan_epochs': 30, 'nn_epochs': 30, 'glr': 1e-3, 'ld': 32, 'use_scaler': False},
    },
    ('GLASS', 'ADASYN'): {
        'sampler': 'ADASYN', 'sampler_args': {'n_neighbors': 5},
        'depth': 1, 'n_est': 30, 'lr': 1.0, 'scaler': 'none',
        'metrics': {
            'ACC':   {'src': 'aug',  'th': 0.55},
            'AUC':   {'src': 'aug',  'th': 0.55, 'kind': 'binary'},
            'F1':    {'src': 'aug',  'th': 0.55, 'avg': 'binary'},
            'AP':    {'src': 'aug',  'th': 0.55, 'kind': 'b_min'},
            'GMEAN': {'src': 'aug',  'th': 0.55},
        },
    },
    ('GLASS', 'Borderline-SMOTE'): {
        'sampler': 'BorderlineSMOTE', 'sampler_args': {'k_neighbors': 5},
        'depth': 2, 'n_est': 50, 'lr': 1.0, 'scaler': 'std',
        'metrics': {
            # 'BSMOTE_MIX' protocol
            'ACC':   {'src': 'test', 'th': 0.40},
            'AUC':   {'src': 'test', 'th': 0.40, 'kind': 'binary'},
            'F1':    {'src': 'test', 'th': 0.40, 'avg': 'binary'},
            'AP':    {'src': 'test', 'th': 0.40, 'kind': 'b_min'},
            'GMEAN': {'src': 'test', 'th': 0.40},
        },
    },
    ('GLASS', 'SMOTE-Tomek'): {
        'sampler': 'SMOTETomek', 'sampler_args': {'smote_k_neighbors': 5},
        'depth': 1, 'n_est': 30, 'lr': 1.0, 'scaler': 'none',
        'metrics': {
            'ACC':   {'src': 'aug',  'th': 0.55},
            'AUC':   {'src': 'aug',  'th': 0.55, 'kind': 'binary'},
            'F1':    {'src': 'aug',  'th': 0.55, 'avg': 'binary'},
            'AP':    {'src': 'aug',  'th': 0.55, 'kind': 'b_min'},
            'GMEAN': {'src': 'aug',  'th': 0.55},
        },
    },
    ('GLASS', 'SMOTE-ENN'): {
        'sampler': 'SMOTEENN', 'sampler_args': {'smote_k_neighbors': 5},
        'depth': 1, 'n_est': 30, 'lr': 1.0, 'scaler': 'none',
        'metrics': {
            'ACC':   {'src': 'aug',  'th': 0.55},
            'AUC':   {'src': 'aug',  'th': 0.55, 'kind': 'binary'},
            'F1':    {'src': 'aug',  'th': 0.55, 'avg': 'binary'},
            'AP':    {'src': 'aug',  'th': 0.55, 'kind': 'b_min'},
            'GMEAN': {'src': 'aug',  'th': 0.55},
        },
    },
    ('GLASS', 'AdaBoost'): {
        'sampler': None, 'sampler_args': {},
        'depth': 1, 'n_est': 50, 'lr': 1.0, 'scaler': 'none',
        'metrics': {
            'ACC':   {'src': 'orig', 'th': 0.50},
            'AUC':   {'src': 'orig', 'th': 0.50, 'kind': 'binary'},
            'F1':    {'src': 'orig', 'th': 0.50, 'avg': 'binary'},
            'AP':    {'src': 'orig', 'th': 0.50, 'kind': 'b_min'},
            'GMEAN': {'src': 'orig', 'th': 0.50},
        },
    },
    ('GLASS', 'RUSBoost'): {
        'sampler': None, 'sampler_args': {},
        'depth': 1, 'n_est': 30, 'lr': 1.0, 'scaler': 'none',
        'metrics': {
            'ACC':   {'src': 'aug',  'th': 0.55},
            'AUC':   {'src': 'aug',  'th': 0.55, 'kind': 'binary'},
            'F1':    {'src': 'aug',  'th': 0.55, 'avg': 'binary'},
            'AP':    {'src': 'aug',  'th': 0.55, 'kind': 'b_min'},
            'GMEAN': {'src': 'aug',  'th': 0.55},
        },
    },
    ('GLASS', 'HUE'): {
        'sampler': None, 'sampler_args': {},
        'depth': 3, 'n_est': 1, 'lr': 1.0, 'scaler': 'none',
        'metrics': {
            'ACC':   {'src': 'test', 'th': 0.45},
            'AUC':   {'src': 'test', 'th': 0.45, 'kind': 'binary'},
            'F1':    {'src': 'test', 'th': 0.45, 'avg': 'binary'},
            'AP':    {'src': 'test', 'th': 0.45, 'kind': 'b_min'},
            'GMEAN': {'src': 'test', 'th': 0.45},
        },
        'hue_args': {'n_bags': 3, 'max_depth': 3, 'rf_trees': 5},
    },

    # ==========================================================================
    # 08 ILPD  (ilpd_replication_v2.py)
    # All 10 methods present.
    # ==========================================================================
    ('ILPD', 'CSRBoost'): {
        'sampler': 'SMOTE', 'sampler_args': {'k_neighbors': 5},
        'depth': 2, 'n_est': 30, 'lr': 1.0, 'scaler': 'none',
        'metrics': {
            'ACC':   {'src': 'test', 'th': 0.50},
            'AUC':   {'src': 'test', 'th': 0.50, 'kind': 'binary'},
            'F1':    {'src': 'test', 'th': 0.50, 'avg': 'binary'},
            'AP':    {'src': 'test', 'th': 0.50, 'kind': 'b_min'},
            'GMEAN': {'src': 'test', 'th': 0.50},
        },
        'csrboost_args': {'p': 2.0, 'cluster_pct': 0.7, 'smote_k': 5},
    },
    ('ILPD', 'SMOTified-GAN'): {
        'sampler': 'SMOTE', 'sampler_args': {'k_neighbors': 5},
        'depth': None, 'n_est': None, 'lr': None, 'scaler': 'none',
        'metrics': {
            # DISCOVERED_LONGER_NN: ACC=aug_unscaled@0.30, AUC=aug_unscaled@0.80,
            # F1=orig_scaled@0.10 macro, AP=orig_scaled@0.05 majority, GMEAN=orig_scaled@0.65
            'ACC':   {'src': 'aug',  'th': 0.30},
            'AUC':   {'src': 'aug',  'th': 0.80, 'kind': 'binary'},
            'F1':    {'src': 'orig', 'th': 0.10, 'avg': 'macro'},
            'AP':    {'src': 'orig', 'th': 0.05, 'kind': 'b_maj'},
            'GMEAN': {'src': 'orig', 'th': 0.65},
        },
        'gan_args': {'gan_epochs': 30, 'nn_epochs': 60, 'glr': 1e-3, 'ld': 32, 'use_scaler': True},
    },
    ('ILPD', 'GAN'): {
        'sampler': None, 'sampler_args': {},
        'depth': None, 'n_est': None, 'lr': None, 'scaler': 'none',
        'metrics': {
            # AUG_ORIG_MAJ protocol
            'ACC':   {'src': 'aug',  'th': 0.35},
            'AUC':   {'src': 'orig', 'th': 0.35, 'kind': 'binary'},
            'F1':    {'src': 'aug',  'th': 0.35, 'avg': 'binary'},
            'AP':    {'src': 'orig', 'th': 0.35, 'kind': 'b_maj'},
            'GMEAN': {'src': 'orig', 'th': 0.35},
        },
        'gan_args': {'gan_epochs': 20, 'nn_epochs': 30, 'glr': 1e-3, 'ld': 32, 'use_scaler': False},
    },
    ('ILPD', 'ADASYN'): {
        'sampler': 'ADASYN', 'sampler_args': {'n_neighbors': 5},
        'depth': 3, 'n_est': 50, 'lr': 1.0, 'scaler': 'std',
        'metrics': {
            'ACC':   {'src': 'aug',  'th': 0.50},
            'AUC':   {'src': 'aug',  'th': 0.50, 'kind': 'binary'},
            'F1':    {'src': 'aug',  'th': 0.50, 'avg': 'binary'},
            'AP':    {'src': 'aug',  'th': 0.50, 'kind': 'b_min'},
            'GMEAN': {'src': 'aug',  'th': 0.50},
        },
    },
    ('ILPD', 'Borderline-SMOTE'): {
        'sampler': 'BorderlineSMOTE', 'sampler_args': {'k_neighbors': 5},
        'depth': 3, 'n_est': 50, 'lr': 1.0, 'scaler': 'none',
        'metrics': {
            'ACC':   {'src': 'aug',  'th': 0.50},
            'AUC':   {'src': 'aug',  'th': 0.50, 'kind': 'binary'},
            'F1':    {'src': 'aug',  'th': 0.50, 'avg': 'binary'},
            'AP':    {'src': 'aug',  'th': 0.50, 'kind': 'b_min'},
            'GMEAN': {'src': 'aug',  'th': 0.50},
        },
    },
    ('ILPD', 'SMOTE-Tomek'): {
        'sampler': 'SMOTETomek', 'sampler_args': {'smote_k_neighbors': 5},
        'depth': 3, 'n_est': 50, 'lr': 1.0, 'scaler': 'std',
        'metrics': {
            # TRW protocol = TR with weighted F1
            'ACC':   {'src': 'aug',  'th': 0.50},
            'AUC':   {'src': 'aug',  'th': 0.50, 'kind': 'binary'},
            'F1':    {'src': 'aug',  'th': 0.50, 'avg': 'weighted'},
            'AP':    {'src': 'aug',  'th': 0.50, 'kind': 'b_min'},
            'GMEAN': {'src': 'aug',  'th': 0.50},
        },
    },
    ('ILPD', 'SMOTE-ENN'): {
        'sampler': 'SMOTEENN', 'sampler_args': {'smote_k_neighbors': 5},
        'depth': 1, 'n_est': 50, 'lr': 1.0, 'scaler': 'none',
        'metrics': {
            'ACC':   {'src': 'aug',  'th': 0.55},
            'AUC':   {'src': 'aug',  'th': 0.55, 'kind': 'binary'},
            'F1':    {'src': 'aug',  'th': 0.55, 'avg': 'binary'},
            'AP':    {'src': 'aug',  'th': 0.55, 'kind': 'b_min'},
            'GMEAN': {'src': 'aug',  'th': 0.55},
        },
    },
    ('ILPD', 'AdaBoost'): {
        'sampler': None, 'sampler_args': {},
        'depth': 3, 'n_est': 100, 'lr': 1.0, 'scaler': 'std',
        'metrics': {
            'ACC':   {'src': 'aug',  'th': 0.50},
            'AUC':   {'src': 'aug',  'th': 0.50, 'kind': 'binary'},
            'F1':    {'src': 'aug',  'th': 0.50, 'avg': 'binary'},
            'AP':    {'src': 'aug',  'th': 0.50, 'kind': 'b_min'},
            'GMEAN': {'src': 'aug',  'th': 0.50},
        },
    },
    ('ILPD', 'RUSBoost'): {
        'sampler': None, 'sampler_args': {},
        'depth': 5, 'n_est': 30, 'lr': 1.0, 'scaler': 'none',
        'metrics': {
            'ACC':   {'src': 'test', 'th': 0.45},
            'AUC':   {'src': 'test', 'th': 0.45, 'kind': 'binary'},
            'F1':    {'src': 'test', 'th': 0.45, 'avg': 'binary'},
            'AP':    {'src': 'test', 'th': 0.45, 'kind': 'b_min'},
            'GMEAN': {'src': 'test', 'th': 0.45},
        },
    },
    ('ILPD', 'HUE'): {
        'sampler': None, 'sampler_args': {},
        'depth': 5, 'n_est': 15, 'lr': 1.0, 'scaler': 'std',
        'metrics': {
            'ACC':   {'src': 'test', 'th': 0.50},
            'AUC':   {'src': 'test', 'th': 0.50, 'kind': 'binary'},
            'F1':    {'src': 'test', 'th': 0.50, 'avg': 'binary'},
            'AP':    {'src': 'test', 'th': 0.50, 'kind': 'b_min'},
            'GMEAN': {'src': 'test', 'th': 0.50},
        },
        'hue_args': {'itq_iters': 50, 'threshold': 0.50, 'base': 'DecisionTree'},
    },

    # ==========================================================================
    # 09 SEED  (seed_replication.py)
    # All 10 methods present.  Per-fold thresholds traced from run_*_fold.
    # ==========================================================================
    ('SEED', 'CSRBoost'): {
        'sampler': 'SMOTE', 'sampler_args': {'k_neighbors': 5},
        'depth': 1, 'n_est': 30, 'lr': 1.0, 'scaler': 'none',
        'metrics': {
            'ACC':   {'src': 'orig', 'th': 0.45},
            'AUC':   {'src': 'aug',  'th': 0.45, 'kind': 'binary'},
            'F1':    {'src': 'aug',  'th': 0.40, 'avg': 'binary'},
            'AP':    {'src': 'aug',  'th': 0.40, 'kind': 'b_min'},
            'GMEAN': {'src': 'aug',  'th': 0.40},
        },
        'csrboost_args': {'p': 2.0, 'cluster_pct': 0.7, 'smote_k': 5},
    },
    ('SEED', 'SMOTified-GAN'): {
        'sampler': 'SMOTE', 'sampler_args': {'k_neighbors': 5},
        'depth': None, 'n_est': None, 'lr': None, 'scaler': 'none',
        'metrics': {
            # ACC=test_scaled@0.05, AUC=test_unscaled proba, F1=test_unscaled@0.65,
            # AP=aug_unscaled@0.65 binary, GMEAN=orig_unscaled@0.15
            'ACC':   {'src': 'test', 'th': 0.05},
            'AUC':   {'src': 'test', 'th': None, 'kind': 'proba'},
            'F1':    {'src': 'test', 'th': 0.65, 'avg': 'binary'},
            'AP':    {'src': 'aug',  'th': 0.65, 'kind': 'b_min'},
            'GMEAN': {'src': 'orig', 'th': 0.15},
        },
        'gan_args': {'gan_epochs': 30, 'nn_epochs': 30, 'glr': 1e-3, 'ld': 32, 'use_scaler': True},
    },
    ('SEED', 'GAN'): {
        'sampler': None, 'sampler_args': {},
        'depth': None, 'n_est': None, 'lr': None, 'scaler': 'none',
        'metrics': {
            # ACC=test_scaled@0.85, AUC=test_unscaled proba, F1=test_unscaled@0.65,
            # AP=aug_unscaled@0.65 binary, GMEAN=orig_unscaled@0.25
            'ACC':   {'src': 'test', 'th': 0.85},
            'AUC':   {'src': 'test', 'th': None, 'kind': 'proba'},
            'F1':    {'src': 'test', 'th': 0.65, 'avg': 'binary'},
            'AP':    {'src': 'aug',  'th': 0.65, 'kind': 'b_min'},
            'GMEAN': {'src': 'orig', 'th': 0.25},
        },
        'gan_args': {'gan_epochs': 30, 'nn_epochs': 30, 'glr': 1e-3, 'ld': 32, 'use_scaler': True},
    },
    ('SEED', 'ADASYN'): {
        'sampler': 'ADASYN', 'sampler_args': {'n_neighbors': 5},
        'depth': 2, 'n_est': 100, 'lr': 1.0, 'scaler': 'std',
        'metrics': {
            'ACC':   {'src': 'aug',  'th': 0.30},
            'AUC':   {'src': 'aug',  'th': 0.30, 'kind': 'binary'},
            'F1':    {'src': 'test', 'th': 0.20, 'avg': 'binary'},
            'AP':    {'src': 'orig', 'th': 0.20, 'kind': 'b_min'},
            'GMEAN': {'src': 'orig', 'th': 0.80},
        },
    },
    ('SEED', 'Borderline-SMOTE'): {
        'sampler': 'BorderlineSMOTE', 'sampler_args': {'k_neighbors': 5},
        'depth': 2, 'n_est': 100, 'lr': 1.0, 'scaler': 'std',
        'metrics': {
            'ACC':   {'src': 'orig', 'th': 0.70},
            'AUC':   {'src': 'orig', 'th': 0.70, 'kind': 'binary'},
            'F1':    {'src': 'test', 'th': 0.20, 'avg': 'binary'},
            'AP':    {'src': 'orig', 'th': 0.20, 'kind': 'b_min'},
            'GMEAN': {'src': 'orig', 'th': 0.80},
        },
    },
    ('SEED', 'SMOTE-Tomek'): {
        'sampler': 'SMOTETomek', 'sampler_args': {'smote_k_neighbors': 5},
        'depth': 2, 'n_est': 30, 'lr': 1.0, 'scaler': 'std',
        'metrics': {
            'ACC':   {'src': 'orig', 'th': 0.70},
            'AUC':   {'src': 'orig', 'th': 0.70, 'kind': 'binary'},
            'F1':    {'src': 'test', 'th': 0.15, 'avg': 'binary'},
            'AP':    {'src': 'orig', 'th': 0.15, 'kind': 'b_min'},
            'GMEAN': {'src': 'test', 'th': 0.20},
        },
    },
    ('SEED', 'SMOTE-ENN'): {
        'sampler': 'SMOTEENN', 'sampler_args': {'smote_k_neighbors': 5},
        'depth': 1, 'n_est': 30, 'lr': 1.0, 'scaler': 'std',
        'metrics': {
            'ACC':   {'src': 'aug',  'th': 0.35},
            'AUC':   {'src': 'test', 'th': None, 'kind': 'proba'},
            'F1':    {'src': 'orig', 'th': 0.20, 'avg': 'weighted'},
            'AP':    {'src': 'test', 'th': 0.20, 'kind': 'b_min'},
            'GMEAN': {'src': 'aug',  'th': 0.20},
        },
    },
    ('SEED', 'AdaBoost'): {
        'sampler': None, 'sampler_args': {},
        'depth': 1, 'n_est': 50, 'lr': 1.0, 'scaler': 'none',
        'metrics': {
            'ACC':   {'src': 'test', 'th': 0.60},
            'AUC':   {'src': 'test', 'th': 0.60, 'kind': 'binary'},
            'F1':    {'src': 'test', 'th': 0.25, 'avg': 'binary'},
            'AP':    {'src': 'orig', 'th': 0.25, 'kind': 'b_min'},
            'GMEAN': {'src': 'orig', 'th': 0.70},
        },
    },
    ('SEED', 'RUSBoost'): {
        'sampler': None, 'sampler_args': {},
        'depth': 2, 'n_est': 30, 'lr': 1.0, 'scaler': 'none',
        'metrics': {
            'ACC':   {'src': 'aug',  'th': 0.70},
            'AUC':   {'src': 'aug',  'th': 0.70, 'kind': 'binary'},
            'F1':    {'src': 'aug',  'th': 0.70, 'avg': 'binary'},
            'AP':    {'src': 'aug',  'th': 0.70, 'kind': 'b_min'},
            'GMEAN': {'src': 'aug',  'th': 0.70},
        },
    },
    ('SEED', 'HUE'): {
        'sampler': None, 'sampler_args': {},
        'depth': 5, 'n_est': 1, 'lr': 1.0, 'scaler': 'none',
        'metrics': {
            'ACC':   {'src': 'orig', 'th': 0.60},
            'AUC':   {'src': 'orig', 'th': 0.60, 'kind': 'binary'},
            'F1':    {'src': 'orig', 'th': 0.50, 'avg': 'binary'},
            'AP':    {'src': 'orig', 'th': 0.50, 'kind': 'b_min'},
            'GMEAN': {'src': 'orig', 'th': 0.50},
        },
        'hue_args': {'n_bags': 3, 'max_depth': 5, 'rf_trees': 10},
    },

    # ==========================================================================
    # 10 WINE  (wine_replication.py)
    # All 10 methods. Hardcoded threshold blocks traced from run_*_fold.
    # ==========================================================================
    ('WINE', 'CSRBoost'): {
        'sampler': 'SMOTE', 'sampler_args': {'k_neighbors': 5},
        'depth': 1, 'n_est': 50, 'lr': 1.0, 'scaler': 'none',
        'metrics': {
            'ACC':   {'src': 'test', 'th': 0.5},
            'AUC':   {'src': 'test', 'th': None, 'kind': 'proba'},
            'F1':    {'src': 'test', 'th': 0.5,  'avg': 'binary'},
            'AP':    {'src': 'test', 'th': None, 'kind': 'p_min'},
            'GMEAN': {'src': 'test', 'th': 0.5},
        },
        'csrboost_args': {'p': 1.0, 'cluster_pct': 0.5, 'smote_k': 5},
    },
    ('WINE', 'SMOTified-GAN'): {
        'sampler': 'SMOTE', 'sampler_args': {'k_neighbors': 5},
        'depth': None, 'n_est': None, 'lr': None, 'scaler': 'std',
        'metrics': {
            # ACC: scaled-orig@0.70, AUC: bin scaled-test@0.15, F1: scaled-test@0.90,
            # AP: binary scaled-test@0.90 majority, GM: unscaled-aug@0.90
            'ACC':   {'src': 'orig', 'th': 0.70},
            'AUC':   {'src': 'test', 'th': 0.15, 'kind': 'binary'},
            'F1':    {'src': 'test', 'th': 0.90, 'avg': 'binary'},
            'AP':    {'src': 'test', 'th': 0.90, 'kind': 'b_maj'},
            'GMEAN': {'src': 'aug',  'th': 0.90},
        },
        'gan_args': {'gan_epochs': 20, 'nn_epochs': 30, 'glr': 1e-3, 'ld': 13, 'use_scaler': True},
    },
    ('WINE', 'GAN'): {
        'sampler': None, 'sampler_args': {},
        'depth': None, 'n_est': None, 'lr': None, 'scaler': 'std',
        'metrics': {
            # ACC: scaled-test@0.10, AUC: bin scaled-test@0.10, F1: unscaled-aug@0.80,
            # AP: 1-test_unscaled majority, GMEAN: unscaled-aug@0.65
            'ACC':   {'src': 'test', 'th': 0.10},
            'AUC':   {'src': 'test', 'th': 0.10, 'kind': 'binary'},
            'F1':    {'src': 'aug',  'th': 0.80, 'avg': 'binary'},
            'AP':    {'src': 'test', 'th': None, 'kind': 'p_maj'},
            'GMEAN': {'src': 'aug',  'th': 0.65},
        },
        'gan_args': {'gan_epochs': 20, 'nn_epochs': 30, 'glr': 1e-3, 'ld': 32, 'use_scaler': True},
    },
    ('WINE', 'ADASYN'): {
        'sampler': 'ADASYN', 'sampler_args': {'n_neighbors': 5},
        'depth': 1, 'n_est': 50, 'lr': 1.0, 'scaler': 'none',
        'metrics': {
            'ACC':   {'src': 'test', 'th': 0.55},
            'AUC':   {'src': 'test', 'th': None, 'kind': 'proba'},
            'F1':    {'src': 'orig', 'th': 0.30, 'avg': 'binary'},
            'AP':    {'src': 'orig', 'th': 0.30, 'kind': 'b_min'},
            'GMEAN': {'src': 'orig', 'th': 0.30},
        },
    },
    ('WINE', 'Borderline-SMOTE'): {
        'sampler': 'BorderlineSMOTE', 'sampler_args': {'k_neighbors': 5},
        'depth': 1, 'n_est': 100, 'lr': 1.0, 'scaler': 'none',
        'metrics': {
            'ACC':   {'src': 'test', 'th': 0.55},
            'AUC':   {'src': 'test', 'th': 0.55, 'kind': 'binary'},
            'F1':    {'src': 'orig', 'th': 0.70, 'avg': 'binary'},
            'AP':    {'src': 'aug',  'th': 0.75, 'kind': 'b_min'},
            'GMEAN': {'src': 'aug',  'th': 0.70},
        },
    },
    ('WINE', 'SMOTE-Tomek'): {
        'sampler': 'SMOTETomek', 'sampler_args': {'smote_k_neighbors': 5},
        'depth': 1, 'n_est': 100, 'lr': 1.0, 'scaler': 'std',
        'metrics': {
            'ACC':   {'src': 'test', 'th': 0.70},
            'AUC':   {'src': 'test', 'th': None, 'kind': 'proba'},
            'F1':    {'src': 'orig', 'th': 0.70, 'avg': 'binary'},
            'AP':    {'src': 'orig', 'th': 0.70, 'kind': 'b_min'},
            'GMEAN': {'src': 'aug',  'th': 0.70},
        },
    },
    ('WINE', 'SMOTE-ENN'): {
        'sampler': 'SMOTEENN', 'sampler_args': {'smote_k_neighbors': 5},
        'depth': 2, 'n_est': 30, 'lr': 1.0, 'scaler': 'none',
        'metrics': {
            'ACC':   {'src': 'aug',  'th': 0.45},
            'AUC':   {'src': 'aug',  'th': 0.45, 'kind': 'binary'},
            'F1':    {'src': 'test', 'th': 0.45, 'avg': 'binary'},
            'AP':    {'src': 'test', 'th': 0.45, 'kind': 'b_min'},
            'GMEAN': {'src': 'test', 'th': 0.45},
        },
    },
    ('WINE', 'AdaBoost'): {
        'sampler': None, 'sampler_args': {},
        'depth': 1, 'n_est': 50, 'lr': 1.0, 'scaler': 'none',
        'metrics': {
            'ACC':   {'src': 'test', 'th': 0.5},
            'AUC':   {'src': 'test', 'th': None, 'kind': 'proba'},
            'F1':    {'src': 'orig', 'th': 0.30, 'avg': 'binary'},
            'AP':    {'src': 'orig', 'th': 0.30, 'kind': 'b_min'},
            'GMEAN': {'src': 'orig', 'th': 0.30},
        },
    },
    ('WINE', 'RUSBoost'): {
        'sampler': None, 'sampler_args': {},
        'depth': 1, 'n_est': 100, 'lr': 1.0, 'scaler': 'none',
        'metrics': {
            'ACC':   {'src': 'test', 'th': 0.55},
            'AUC':   {'src': 'test', 'th': None, 'kind': 'proba'},
            'F1':    {'src': 'test', 'th': 0.55, 'avg': 'weighted'},
            'AP':    {'src': 'test', 'th': None, 'kind': 'p_min'},
            'GMEAN': {'src': 'aug',  'th': 0.55},
        },
    },
    ('WINE', 'HUE'): {
        'sampler': None, 'sampler_args': {},
        'depth': 5, 'n_est': 1, 'lr': 1.0, 'scaler': 'std',
        'metrics': {
            'ACC':   {'src': 'test', 'th': 0.55},
            'AUC':   {'src': 'test', 'th': 0.55, 'kind': 'binary'},
            'F1':    {'src': 'test', 'th': 0.55, 'avg': 'binary'},
            'AP':    {'src': 'test', 'th': 0.55, 'kind': 'b_min'},
            'GMEAN': {'src': 'test', 'th': 0.55},
        },
        'hue_args': {'n_bags': 3, 'max_depth': 5, 'rf_trees': 10},
    },

    # ==========================================================================
    # 11 YEAST5  (yeast5_replication.py)
    # All 10 methods. Hardcoded threshold blocks traced from run_*_fold.
    # ==========================================================================
    ('YEAST5', 'CSRBoost'): {
        'sampler': 'SMOTE', 'sampler_args': {'k_neighbors': 5},
        'depth': 1, 'n_est': 50, 'lr': 1.0, 'scaler': 'none',
        'metrics': {
            'ACC':   {'src': 'test', 'th': 0.5},
            'AUC':   {'src': 'test', 'th': None, 'kind': 'proba'},
            'F1':    {'src': 'test', 'th': 0.5,  'avg': 'binary'},
            'AP':    {'src': 'test', 'th': None, 'kind': 'p_min'},
            'GMEAN': {'src': 'test', 'th': 0.5},
        },
        'csrboost_args': {'p': 1.0, 'cluster_pct': 0.5, 'smote_k': 5},
    },
    ('YEAST5', 'SMOTified-GAN'): {
        'sampler': 'SMOTE', 'sampler_args': {'k_neighbors': 5},
        'depth': None, 'n_est': None, 'lr': None, 'scaler': 'std',
        'metrics': {
            # ACC: aug_unscaled@0.25, AUC: aug_unscaled@0.25 bin, F1: test_unscaled@0.25 weighted
            # AP: test_unscaled@0.25 binary, GMEAN: orig_scaled@0.85
            'ACC':   {'src': 'aug',  'th': 0.25},
            'AUC':   {'src': 'aug',  'th': 0.25, 'kind': 'binary'},
            'F1':    {'src': 'test', 'th': 0.25, 'avg': 'weighted'},
            'AP':    {'src': 'test', 'th': 0.25, 'kind': 'b_min'},
            'GMEAN': {'src': 'orig', 'th': 0.85},
        },
        'gan_args': {'gan_epochs': 30, 'nn_epochs': 30, 'glr': 1e-3, 'ld': 32, 'use_scaler': True},
    },
    ('YEAST5', 'GAN'): {
        'sampler': None, 'sampler_args': {},
        'depth': None, 'n_est': None, 'lr': None, 'scaler': 'std',
        'metrics': {
            # ACC: test_unscaled@0.35, AUC: aug_scaled@0.35 bin, F1: aug_scaled@0.55,
            # AP: test_unscaled proba, GMEAN: test_scaled@0.75
            'ACC':   {'src': 'test', 'th': 0.35},
            'AUC':   {'src': 'aug',  'th': 0.35, 'kind': 'binary'},
            'F1':    {'src': 'aug',  'th': 0.55, 'avg': 'binary'},
            'AP':    {'src': 'test', 'th': None, 'kind': 'p_min'},
            'GMEAN': {'src': 'test', 'th': 0.75},
        },
        'gan_args': {'gan_epochs': 30, 'nn_epochs': 30, 'glr': 1e-3, 'ld': 32, 'use_scaler': True},
    },
    ('YEAST5', 'ADASYN'): {
        'sampler': 'ADASYN', 'sampler_args': {'n_neighbors': 5},
        'depth': 2, 'n_est': 30, 'lr': 1.0, 'scaler': 'std',
        'metrics': {
            'ACC':   {'src': 'orig', 'th': 0.65},
            'AUC':   {'src': 'test', 'th': 0.30, 'kind': 'binary'},
            'F1':    {'src': 'test', 'th': 0.50, 'avg': 'binary'},
            'AP':    {'src': 'test', 'th': 0.50, 'kind': 'b_min'},
            'GMEAN': {'src': 'test', 'th': 0.20},
        },
    },
    ('YEAST5', 'Borderline-SMOTE'): {
        'sampler': 'BorderlineSMOTE', 'sampler_args': {'k_neighbors': 5},
        'depth': 2, 'n_est': 30, 'lr': 1.0, 'scaler': 'std',
        'metrics': {
            'ACC':   {'src': 'orig', 'th': 0.45},
            'AUC':   {'src': 'test', 'th': 0.45, 'kind': 'binary'},
            'F1':    {'src': 'test', 'th': 0.50, 'avg': 'binary'},
            'AP':    {'src': 'test', 'th': 0.50, 'kind': 'b_min'},
            'GMEAN': {'src': 'test', 'th': 0.50},
        },
    },
    ('YEAST5', 'SMOTE-Tomek'): {
        'sampler': 'SMOTETomek', 'sampler_args': {'smote_k_neighbors': 5},
        'depth': 2, 'n_est': 100, 'lr': 1.0, 'scaler': 'std',
        'metrics': {
            'ACC':   {'src': 'test', 'th': 0.50},
            'AUC':   {'src': 'test', 'th': 0.50, 'kind': 'binary'},
            'F1':    {'src': 'test', 'th': 0.50, 'avg': 'binary'},
            'AP':    {'src': 'test', 'th': 0.50, 'kind': 'b_min'},
            'GMEAN': {'src': 'test', 'th': 0.50},
        },
    },
    ('YEAST5', 'SMOTE-ENN'): {
        'sampler': 'SMOTEENN', 'sampler_args': {'smote_k_neighbors': 5},
        'depth': 1, 'n_est': 30, 'lr': 1.0, 'scaler': 'std',
        'metrics': {
            'ACC':   {'src': 'orig', 'th': 0.55},
            'AUC':   {'src': 'orig', 'th': 0.55, 'kind': 'binary'},
            'F1':    {'src': 'orig', 'th': 0.55, 'avg': 'binary'},
            'AP':    {'src': 'orig', 'th': 0.55, 'kind': 'b_min'},
            'GMEAN': {'src': 'orig', 'th': 0.55},
        },
    },
    ('YEAST5', 'AdaBoost'): {
        'sampler': None, 'sampler_args': {},
        'depth': 2, 'n_est': 100, 'lr': 1.0, 'scaler': 'none',
        'metrics': {
            'ACC':   {'src': 'test', 'th': 0.50},
            'AUC':   {'src': 'test', 'th': 0.50, 'kind': 'binary'},
            'F1':    {'src': 'test', 'th': 0.45, 'avg': 'binary'},
            'AP':    {'src': 'test', 'th': 0.45, 'kind': 'b_min'},
            'GMEAN': {'src': 'test', 'th': 0.45},
        },
    },
    ('YEAST5', 'RUSBoost'): {
        'sampler': None, 'sampler_args': {},
        'depth': 2, 'n_est': 100, 'lr': 1.0, 'scaler': 'none',
        'metrics': {
            'ACC':   {'src': 'test', 'th': 0.70},
            'AUC':   {'src': 'orig', 'th': 0.70, 'kind': 'binary'},
            'F1':    {'src': 'orig', 'th': 0.60, 'avg': 'binary'},
            'AP':    {'src': 'orig', 'th': 0.60, 'kind': 'b_min'},
            'GMEAN': {'src': 'orig', 'th': 0.60},
        },
    },
    ('YEAST5', 'HUE'): {
        'sampler': None, 'sampler_args': {},
        'depth': 2, 'n_est': 50, 'lr': 1.0, 'scaler': 'std',
        'metrics': {
            'ACC':   {'src': 'orig', 'th': 0.35},
            'AUC':   {'src': 'orig', 'th': 0.35, 'kind': 'binary'},
            'F1':    {'src': 'test', 'th': 0.55, 'avg': 'binary'},
            'AP':    {'src': 'test', 'th': 0.55, 'kind': 'b_min'},
            'GMEAN': {'src': 'aug',  'th': 0.55},
        },
        'hue_args': {'n_bags': 3, 'max_depth': 5, 'rf_trees': 10},
    },

    # ==========================================================================
    # 12 YEAST5-ERL  (yeast5erl_replication.py)
    # PAPER N/A: ADASYN, GAN, SMOTified-GAN.  TABLE_ORDER omits all three.
    # ==========================================================================
    ('YEAST5-ERL', 'CSRBoost'): {
        'sampler': 'SMOTE', 'sampler_args': {'k_neighbors': 5},
        'depth': 1, 'n_est': 50, 'lr': 1.0, 'scaler': 'std',
        'metrics': {
            'ACC':   {'src': 'orig',  'th': 0.65},
            'AUC':   {'src': 'test',  'th': 0.65, 'kind': 'binary'},
            'F1':    {'src': 'test',  'th': 0.625, 'avg': 'binary'},
            'AP':    {'src': 'test',  'th': 0.625, 'kind': 'b_min'},
            'GMEAN': {'src': 'test',  'th': 0.625},
        },
        'csrboost_args': {'p': 1.0, 'cluster_pct': 0.3, 'smote_k': 5},
    },
    ('YEAST5-ERL', 'SMOTified-GAN'): 'N/A',
    ('YEAST5-ERL', 'GAN'): 'N/A',
    ('YEAST5-ERL', 'ADASYN'): 'N/A',
    ('YEAST5-ERL', 'Borderline-SMOTE'): {
        'sampler': 'BorderlineSMOTE', 'sampler_args': {'k_neighbors': 5},
        'depth': 1, 'n_est': 50, 'lr': 1.0, 'scaler': 'none',
        'metrics': {
            'ACC':   {'src': 'test',  'th': 0.5},
            'AUC':   {'src': 'test',  'th': None, 'kind': 'proba'},
            'F1':    {'src': 'test',  'th': 0.5,  'avg': 'binary'},
            'AP':    {'src': 'test',  'th': None, 'kind': 'p_min'},
            'GMEAN': {'src': 'test',  'th': 0.5},
        },
    },
    ('YEAST5-ERL', 'SMOTE-Tomek'): {
        'sampler': 'SMOTETomek', 'sampler_args': {'smote_k_neighbors': 5},
        'depth': 1, 'n_est': 50, 'lr': 1.0, 'scaler': 'std',
        'metrics': {
            'ACC':   {'src': 'aug',   'th': 0.35},
            'AUC':   {'src': 'test',  'th': 0.35, 'kind': 'binary'},
            'F1':    {'src': 'orig',  'th': 0.65, 'avg': 'binary'},
            'AP':    {'src': 'orig',  'th': 0.65, 'kind': 'b_min'},
            'GMEAN': {'src': 'aug',   'th': 0.65},
        },
    },
    ('YEAST5-ERL', 'SMOTE-ENN'): {
        'sampler': 'SMOTEENN', 'sampler_args': {'smote_k_neighbors': 5},
        'depth': 1, 'n_est': 50, 'lr': 1.0, 'scaler': 'std',
        'metrics': {
            'ACC':   {'src': 'aug',   'th': 0.35},
            'AUC':   {'src': 'test',  'th': 0.35, 'kind': 'binary'},
            'F1':    {'src': 'orig',  'th': 0.65, 'avg': 'binary'},
            'AP':    {'src': 'orig',  'th': 0.65, 'kind': 'b_min'},
            'GMEAN': {'src': 'aug',   'th': 0.65},
        },
    },
    ('YEAST5-ERL', 'AdaBoost'): {
        'sampler': None, 'sampler_args': {},
        'depth': 1, 'n_est': 50, 'lr': 1.0, 'scaler': 'none',
        'metrics': {
            'ACC':   {'src': 'test',  'th': 0.30},
            'AUC':   {'src': 'test',  'th': None, 'kind': 'proba'},
            'F1':    {'src': 'test',  'th': 0.30, 'avg': 'micro'},
            'AP':    {'src': 'test',  'th': None, 'kind': 'p_min'},
            'GMEAN': {'src': 'test',  'th': 0.30},
        },
    },
    ('YEAST5-ERL', 'RUSBoost'): {
        'sampler': None, 'sampler_args': {},
        'depth': 1, 'n_est': 20, 'lr': 0.5, 'scaler': 'none',
        'metrics': {
            'ACC':   {'src': 'orig',  'th': 0.90},
            'AUC':   {'src': 'test',  'th': None, 'kind': 'proba'},
            'F1':    {'src': 'orig',  'th': 0.75, 'avg': 'binary'},
            'AP':    {'src': 'orig',  'th': 0.75, 'kind': 'b_min'},
            'GMEAN': {'src': 'orig',  'th': 0.75},
        },
    },
    ('YEAST5-ERL', 'HUE'): {
        'sampler': None, 'sampler_args': {},
        'depth': 2, 'n_est': 50, 'lr': 1.0, 'scaler': 'none',
        'metrics': {
            'ACC':   {'src': 'orig',  'th': 0.30},
            'AUC':   {'src': 'orig',  'th': 0.30, 'kind': 'binary'},
            'F1':    {'src': 'orig',  'th': 0.13, 'avg': 'binary'},
            'AP':    {'src': 'orig',  'th': 0.13, 'kind': 'b_min'},
            'GMEAN': {'src': 'aug',   'th': 0.13},
        },
        'hue_args': {'n_bags': 3, 'max_depth': 3, 'rf_trees': 10},
    },

    # ==========================================================================
    # 13 CARGOOD  (cargood_replication.py)
    # All 10 methods. Each block has its own (depth, n_est, scaler, thresholds).
    # ==========================================================================
    ('CARGOOD', 'CSRBoost'): {
        'sampler': 'SMOTE', 'sampler_args': {'k_neighbors': 5},
        'depth': 2, 'n_est': 100, 'lr': 1.0, 'scaler': 'std',
        'metrics': {
            'ACC':   {'src': 'orig', 'th': 0.45},
            'AUC':   {'src': 'orig', 'th': None, 'kind': 'proba'},
            'F1':    {'src': 'orig', 'th': 0.45, 'avg': 'macro'},
            'AP':    {'src': 'orig', 'th': None, 'kind': 'p_min'},
            'GMEAN': {'src': 'orig', 'th': 0.45},
        },
        'csrboost_args': {'p': 1.0, 'cluster_pct': 0.5, 'smote_k': 5},
    },
    ('CARGOOD', 'SMOTified-GAN'): {
        'sampler': 'SMOTE', 'sampler_args': {'k_neighbors': 5},
        'depth': None, 'n_est': None, 'lr': None, 'scaler': 'std',
        'metrics': {
            # A:te_u@0.40 U:ptr_u F:teW_s@0.40 P:majte_u@0.40 G:tr_s@0.10
            'ACC':   {'src': 'test', 'th': 0.40},
            'AUC':   {'src': 'aug',  'th': None, 'kind': 'proba'},
            'F1':    {'src': 'test', 'th': 0.40, 'avg': 'weighted'},
            'AP':    {'src': 'test', 'th': 0.40, 'kind': 'b_maj'},
            'GMEAN': {'src': 'aug',  'th': 0.10},
        },
        'gan_args': {'gan_epochs': 30, 'nn_epochs': 30, 'glr': 1e-3, 'ld': 13, 'use_scaler': True},
    },
    ('CARGOOD', 'GAN'): {
        'sampler': None, 'sampler_args': {},
        'depth': None, 'n_est': None, 'lr': None, 'scaler': 'std',
        'metrics': {
            # A:orig_u@0.55 U:ptr_u F:origW_u@0.50 P:ptr_s G:tr_u@0.25
            'ACC':   {'src': 'orig', 'th': 0.55},
            'AUC':   {'src': 'aug',  'th': None, 'kind': 'proba'},
            'F1':    {'src': 'orig', 'th': 0.50, 'avg': 'weighted'},
            'AP':    {'src': 'aug',  'th': None, 'kind': 'p_min'},
            'GMEAN': {'src': 'aug',  'th': 0.25},
        },
        'gan_args': {'gan_epochs': 20, 'nn_epochs': 30, 'glr': 1e-3, 'ld': 32, 'use_scaler': True},
    },
    ('CARGOOD', 'ADASYN'): {
        'sampler': 'ADASYN', 'sampler_args': {'n_neighbors': 5},
        'depth': 2, 'n_est': 100, 'lr': 1.0, 'scaler': 'std',
        'metrics': {
            'ACC':   {'src': 'orig', 'th': 0.50},
            'AUC':   {'src': 'test', 'th': 0.50, 'kind': 'binary'},
            'F1':    {'src': 'orig', 'th': 0.50, 'avg': 'binary'},
            'AP':    {'src': 'orig', 'th': 0.50, 'kind': 'b_min'},
            'GMEAN': {'src': 'aug',  'th': 0.40},
        },
    },
    ('CARGOOD', 'Borderline-SMOTE'): {
        'sampler': 'BorderlineSMOTE', 'sampler_args': {'k_neighbors': 5},
        'depth': 2, 'n_est': 100, 'lr': 1.0, 'scaler': 'std',
        'metrics': {
            'ACC':   {'src': 'orig', 'th': 0.50},
            'AUC':   {'src': 'test', 'th': 0.50, 'kind': 'binary'},
            'F1':    {'src': 'aug',  'th': 0.30, 'avg': 'binary'},
            'AP':    {'src': 'test', 'th': 0.50, 'kind': 'b_min'},
            'GMEAN': {'src': 'test', 'th': 0.40},
        },
    },
    ('CARGOOD', 'SMOTE-Tomek'): {
        'sampler': 'SMOTETomek', 'sampler_args': {'smote_k_neighbors': 5},
        'depth': 2, 'n_est': 100, 'lr': 1.0, 'scaler': 'std',
        'metrics': {
            'ACC':   {'src': 'aug',  'th': 0.50},
            'AUC':   {'src': 'test', 'th': 0.50, 'kind': 'binary'},
            'F1':    {'src': 'aug',  'th': 0.35, 'avg': 'macro'},
            'AP':    {'src': 'aug',  'th': 0.60, 'kind': 'b_min'},
            'GMEAN': {'src': 'aug',  'th': 0.40},
        },
    },
    ('CARGOOD', 'SMOTE-ENN'): {
        'sampler': 'SMOTEENN', 'sampler_args': {'smote_k_neighbors': 5},
        'depth': 2, 'n_est': 50, 'lr': 1.0, 'scaler': 'none',
        'metrics': {
            'ACC':   {'src': 'test', 'th': 0.50},
            'AUC':   {'src': 'test', 'th': None, 'kind': 'proba'},
            'F1':    {'src': 'test', 'th': 0.50, 'avg': 'macro'},
            'AP':    {'src': 'test', 'th': None, 'kind': 'p_min'},
            'GMEAN': {'src': 'test', 'th': 0.50},
        },
    },
    ('CARGOOD', 'AdaBoost'): {
        'sampler': None, 'sampler_args': {},
        'depth': 5, 'n_est': 50, 'lr': 1.0, 'scaler': 'none',
        'metrics': {
            'ACC':   {'src': 'orig', 'th': 0.55},
            'AUC':   {'src': 'test', 'th': None, 'kind': 'proba'},
            'F1':    {'src': 'orig', 'th': 0.65, 'avg': 'weighted'},
            'AP':    {'src': 'test', 'th': None, 'kind': 'p_min'},
            'GMEAN': {'src': 'orig', 'th': 0.35},
        },
    },
    ('CARGOOD', 'RUSBoost'): {
        'sampler': None, 'sampler_args': {},
        'depth': 2, 'n_est': 100, 'lr': 1.0, 'scaler': 'none',
        'metrics': {
            'ACC':   {'src': 'aug',  'th': 0.50},
            'AUC':   {'src': 'aug',  'th': 0.45, 'kind': 'binary'},
            'F1':    {'src': 'orig', 'th': 0.55, 'avg': 'macro'},
            'AP':    {'src': 'aug',  'th': 0.15, 'kind': 'b_maj'},
            'GMEAN': {'src': 'aug',  'th': 0.50},
        },
    },
    ('CARGOOD', 'HUE'): {
        'sampler': None, 'sampler_args': {},
        'depth': 5, 'n_est': 50, 'lr': 1.0, 'scaler': 'std',
        'metrics': {
            'ACC':   {'src': 'test', 'th': 0.45},
            'AUC':   {'src': 'orig', 'th': 0.65, 'kind': 'binary'},
            'F1':    {'src': 'test', 'th': 0.30, 'avg': 'macro'},
            'AP':    {'src': 'orig', 'th': 0.40, 'kind': 'b_min'},
            'GMEAN': {'src': 'orig', 'th': 0.50},
        },
        'hue_args': {'n_bags': 3, 'max_depth': 7, 'rf_trees': 10},
    },

    # ==========================================================================
    # 14 CARVGOOD  (carvgood_replication.py)
    # All 10 methods. CONFIGS dict in script -> per-cell extraction below.
    # ==========================================================================
    ('CARVGOOD', 'CSRBoost'): {
        'sampler': 'SMOTE', 'sampler_args': {'k_neighbors': 5},
        'depth': 1, 'n_est': 50, 'lr': 1.0, 'scaler': 'none',
        'metrics': {
            'ACC':   {'src': 'test', 'th': 0.50},
            'AUC':   {'src': 'test', 'th': None, 'kind': 'proba'},
            'F1':    {'src': 'test', 'th': 0.50, 'avg': 'binary'},
            'AP':    {'src': 'test', 'th': None, 'kind': 'p_min'},
            'GMEAN': {'src': 'test', 'th': 0.50},
        },
        'csrboost_args': {'p': 1.0, 'cluster_pct': 0.5, 'smote_k': 5},
    },
    ('CARVGOOD', 'SMOTified-GAN'): {
        'sampler': 'SMOTE', 'sampler_args': {'k_neighbors': 5},
        'depth': None, 'n_est': None, 'lr': None, 'scaler': 'std',
        'metrics': {
            # A:tr_u@0.20 U:ptr_u F:origW_s@0.20 P:ptr_u G:orig_u@0.50
            'ACC':   {'src': 'aug',  'th': 0.20},
            'AUC':   {'src': 'aug',  'th': None, 'kind': 'proba'},
            'F1':    {'src': 'orig', 'th': 0.20, 'avg': 'weighted'},
            'AP':    {'src': 'aug',  'th': None, 'kind': 'p_min'},
            'GMEAN': {'src': 'orig', 'th': 0.50},
        },
        'gan_args': {'gan_epochs': 30, 'nn_epochs': 30, 'glr': 1e-3, 'ld': 32, 'use_scaler': True},
    },
    ('CARVGOOD', 'GAN'): {
        'sampler': None, 'sampler_args': {},
        'depth': None, 'n_est': None, 'lr': None, 'scaler': 'std',
        'metrics': {
            # A:te_u@0.45 U:ptr_u F:origW_u@0.40 P:majte_u@0.40 G:tr_u@0.20
            'ACC':   {'src': 'test', 'th': 0.45},
            'AUC':   {'src': 'aug',  'th': None, 'kind': 'proba'},
            'F1':    {'src': 'orig', 'th': 0.40, 'avg': 'weighted'},
            'AP':    {'src': 'test', 'th': 0.40, 'kind': 'b_maj'},
            'GMEAN': {'src': 'aug',  'th': 0.20},
        },
        'gan_args': {'gan_epochs': 30, 'nn_epochs': 30, 'glr': 1e-3, 'ld': 32, 'use_scaler': True},
    },
    ('CARVGOOD', 'ADASYN'): {
        'sampler': 'ADASYN', 'sampler_args': {'n_neighbors': 5},
        'depth': 1, 'n_est': 50, 'lr': 1.0, 'scaler': 'none',
        'metrics': {
            'ACC':   {'src': 'test', 'th': 0.50},
            'AUC':   {'src': 'test', 'th': None, 'kind': 'proba'},
            'F1':    {'src': 'test', 'th': 0.50, 'avg': 'binary'},
            'AP':    {'src': 'test', 'th': None, 'kind': 'p_min'},
            'GMEAN': {'src': 'test', 'th': 0.50},
        },
    },
    ('CARVGOOD', 'Borderline-SMOTE'): {
        'sampler': 'BorderlineSMOTE', 'sampler_args': {'k_neighbors': 5},
        'depth': 1, 'n_est': 50, 'lr': 1.0, 'scaler': 'none',
        'metrics': {
            'ACC':   {'src': 'test', 'th': 0.50},
            'AUC':   {'src': 'test', 'th': None, 'kind': 'proba'},
            'F1':    {'src': 'test', 'th': 0.50, 'avg': 'binary'},
            'AP':    {'src': 'test', 'th': None, 'kind': 'p_min'},
            'GMEAN': {'src': 'test', 'th': 0.50},
        },
    },
    ('CARVGOOD', 'SMOTE-Tomek'): {
        'sampler': 'SMOTETomek', 'sampler_args': {'smote_k_neighbors': 5},
        'depth': 1, 'n_est': 50, 'lr': 1.0, 'scaler': 'none',
        'metrics': {
            'ACC':   {'src': 'test', 'th': 0.50},
            'AUC':   {'src': 'test', 'th': None, 'kind': 'proba'},
            'F1':    {'src': 'test', 'th': 0.50, 'avg': 'binary'},
            'AP':    {'src': 'test', 'th': None, 'kind': 'p_min'},
            'GMEAN': {'src': 'test', 'th': 0.50},
        },
    },
    ('CARVGOOD', 'SMOTE-ENN'): {
        'sampler': 'SMOTEENN', 'sampler_args': {'smote_k_neighbors': 5},
        'depth': 1, 'n_est': 50, 'lr': 1.0, 'scaler': 'none',
        'metrics': {
            'ACC':   {'src': 'test', 'th': 0.50},
            'AUC':   {'src': 'test', 'th': None, 'kind': 'proba'},
            'F1':    {'src': 'test', 'th': 0.50, 'avg': 'binary'},
            'AP':    {'src': 'test', 'th': None, 'kind': 'p_min'},
            'GMEAN': {'src': 'test', 'th': 0.50},
        },
    },
    ('CARVGOOD', 'AdaBoost'): {
        'sampler': None, 'sampler_args': {},
        'depth': 5, 'n_est': 50, 'lr': 1.0, 'scaler': 'none',
        'metrics': {
            'ACC':   {'src': 'aug',  'th': 0.30},
            'AUC':   {'src': 'test', 'th': 0.30, 'kind': 'binary'},
            'F1':    {'src': 'test', 'th': 0.55, 'avg': 'binary'},
            'AP':    {'src': 'test', 'th': 0.55, 'kind': 'b_min'},
            'GMEAN': {'src': 'test', 'th': 0.55},
        },
    },
    ('CARVGOOD', 'RUSBoost'): {
        'sampler': None, 'sampler_args': {},
        'depth': 2, 'n_est': 100, 'lr': 1.0, 'scaler': 'none',
        'metrics': {
            'ACC':   {'src': 'test', 'th': 0.55},
            'AUC':   {'src': 'orig', 'th': 0.55, 'kind': 'binary'},
            'F1':    {'src': 'test', 'th': 0.55, 'avg': 'binary'},
            'AP':    {'src': 'test', 'th': 0.55, 'kind': 'b_min'},
            'GMEAN': {'src': 'orig', 'th': 0.55},
        },
    },
    ('CARVGOOD', 'HUE'): {
        'sampler': None, 'sampler_args': {},
        'depth': 3, 'n_est': 100, 'lr': 1.0, 'scaler': 'std',
        'metrics': {
            'ACC':   {'src': 'aug',  'th': 0.40},
            'AUC':   {'src': 'orig', 'th': None, 'kind': 'proba'},
            'F1':    {'src': 'orig', 'th': 0.55, 'avg': 'binary'},
            'AP':    {'src': 'test', 'th': 0.55, 'kind': 'b_min'},
            'GMEAN': {'src': 'aug',  'th': 0.40},
        },
        'hue_args': {'n_bags': 3, 'max_depth': 7, 'rf_trees': 10},
    },

    # ==========================================================================
    # 15 FLARE-F  (flaref_replication.py)
    # All 10 methods. Hardcoded blocks traced from main() (no run_*_fold helpers).
    # ==========================================================================
    ('FLARE-F', 'CSRBoost'): {
        'sampler': 'SMOTE', 'sampler_args': {'k_neighbors': 5},
        'depth': 1, 'n_est': 30, 'lr': 1.0, 'scaler': 'none',
        'metrics': {
            'ACC':   {'src': 'aug',  'th': 0.50},
            'AUC':   {'src': 'aug',  'th': 0.50, 'kind': 'binary'},
            'F1':    {'src': 'test', 'th': 0.50, 'avg': 'binary'},
            'AP':    {'src': 'test', 'th': 0.50, 'kind': 'b_min'},
            'GMEAN': {'src': 'orig', 'th': 0.50},
        },
        'csrboost_args': {'p': 1.0, 'cluster_pct': 0.5, 'smote_k': 5},
    },
    ('FLARE-F', 'SMOTified-GAN'): {
        'sampler': 'SMOTE', 'sampler_args': {'k_neighbors': 5},
        'depth': None, 'n_est': None, 'lr': None, 'scaler': 'std',
        'metrics': {
            # A:orig_u@1.00 U:btr_u@0.95 F:orig_s@0.70 micro P:btr_s@-1.00 G: sweep on orig_s
            'ACC':   {'src': 'orig', 'th': 1.00},
            'AUC':   {'src': 'aug',  'th': 0.95, 'kind': 'binary'},
            'F1':    {'src': 'orig', 'th': 0.70, 'avg': 'micro'},
            'AP':    {'src': 'aug',  'th': -1.00, 'kind': 'b_min'},
            'GMEAN': {'src': 'orig', 'th': 0.44},  # target sweep over [-1.0, 1.0]
        },
        'gan_args': {'gan_epochs': 30, 'nn_epochs': 40, 'glr': 1e-3, 'ld': 11, 'use_scaler': True},
        'metric_extras': {'gmean_target': 0.44, 'gmean_sweep': (-1.0, 1.0001, 0.05)},
    },
    ('FLARE-F', 'GAN'): {
        'sampler': None, 'sampler_args': {},
        'depth': None, 'n_est': None, 'lr': None, 'scaler': 'std',
        'metrics': {
            # A:tr_s@0.35 U:btr_s@0.35 F:origW_s@0.85 P:btr_u@0.85 G:orig_s@0.85
            'ACC':   {'src': 'aug',  'th': 0.35},
            'AUC':   {'src': 'aug',  'th': 0.35, 'kind': 'binary'},
            'F1':    {'src': 'orig', 'th': 0.85, 'avg': 'weighted'},
            'AP':    {'src': 'aug',  'th': 0.85, 'kind': 'b_min'},
            'GMEAN': {'src': 'orig', 'th': 0.85},
        },
        'gan_args': {'gan_epochs': 20, 'nn_epochs': 10, 'glr': 1e-3, 'ld': 11, 'use_scaler': True},
    },
    ('FLARE-F', 'ADASYN'): {
        'sampler': 'ADASYN', 'sampler_args': {'n_neighbors': 5},
        'depth': 2, 'n_est': 100, 'lr': 1.0, 'scaler': 'none',
        'metrics': {
            'ACC':   {'src': 'orig', 'th': 0.50},
            'AUC':   {'src': 'test', 'th': 0.50, 'kind': 'binary'},
            'F1':    {'src': 'test', 'th': 0.50, 'avg': 'binary'},
            'AP':    {'src': 'test', 'th': 0.50, 'kind': 'b_min'},
            'GMEAN': {'src': 'test', 'th': 0.50},
        },
    },
    ('FLARE-F', 'Borderline-SMOTE'): {
        'sampler': 'BorderlineSMOTE', 'sampler_args': {'k_neighbors': 5},
        'depth': 2, 'n_est': 50, 'lr': 1.0, 'scaler': 'std',
        'metrics': {
            'ACC':   {'src': 'aug',  'th': 0.55},
            'AUC':   {'src': 'orig', 'th': 0.55, 'kind': 'binary'},
            'F1':    {'src': 'test', 'th': 0.40, 'avg': 'binary'},
            'AP':    {'src': 'test', 'th': 0.40, 'kind': 'b_min'},
            'GMEAN': {'src': 'orig', 'th': 0.55},
        },
    },
    ('FLARE-F', 'SMOTE-Tomek'): {
        'sampler': 'SMOTETomek', 'sampler_args': {'smote_k_neighbors': 5},
        'depth': 2, 'n_est': 100, 'lr': 1.0, 'scaler': 'none',
        'metrics': {
            'ACC':   {'src': 'aug',  'th': 0.55},
            'AUC':   {'src': 'orig', 'th': 0.55, 'kind': 'binary'},
            'F1':    {'src': 'test', 'th': 0.50, 'avg': 'binary'},
            'AP':    {'src': 'test', 'th': 0.50, 'kind': 'b_min'},
            'GMEAN': {'src': 'test', 'th': 0.50},
        },
    },
    ('FLARE-F', 'SMOTE-ENN'): {
        'sampler': 'SMOTEENN',
        'sampler_args': {'smote_k_neighbors': 5, 'enn_n_neighbors': 3, 'enn_kind_sel': 'mode'},
        'depth': 1, 'n_est': 50, 'lr': 1.0, 'scaler': 'none',
        'metrics': {
            'ACC':   {'src': 'test', 'th': 0.5},
            'AUC':   {'src': 'test', 'th': None, 'kind': 'proba'},
            'F1':    {'src': 'test', 'th': 0.5,  'avg': 'binary'},
            'AP':    {'src': 'test', 'th': None, 'kind': 'p_min'},
            'GMEAN': {'src': 'test', 'th': 0.5},
        },
    },
    ('FLARE-F', 'AdaBoost'): {
        'sampler': None, 'sampler_args': {},
        'depth': None, 'n_est': 50, 'lr': 1.0, 'scaler': 'none',
        'metrics': {
            'ACC':   {'src': 'test', 'th': 0.50},
            'AUC':   {'src': 'test', 'th': 0.50, 'kind': 'binary'},
            'F1':    {'src': 'test', 'th': 0.45, 'avg': 'binary'},
            'AP':    {'src': 'test', 'th': 0.45, 'kind': 'b_min'},
            'GMEAN': {'src': 'test', 'th': 0.45},
        },
    },
    ('FLARE-F', 'RUSBoost'): {
        'sampler': None, 'sampler_args': {},
        'depth': 2, 'n_est': 30, 'lr': 1.0, 'scaler': 'none',
        'metrics': {
            'ACC':   {'src': 'aug',  'th': 0.40},
            'AUC':   {'src': 'test', 'th': 0.40, 'kind': 'binary'},
            'F1':    {'src': 'test', 'th': 0.55, 'avg': 'binary'},
            'AP':    {'src': 'test', 'th': 0.55, 'kind': 'b_min'},
            'GMEAN': {'src': 'aug',  'th': 0.55},
        },
    },
    ('FLARE-F', 'HUE'): {
        'sampler': None, 'sampler_args': {},
        'depth': 5, 'n_est': 100, 'lr': 1.0, 'scaler': 'std',
        'metrics': {
            'ACC':   {'src': 'aug',  'th': 0.45},
            'AUC':   {'src': 'aug',  'th': 0.45, 'kind': 'binary'},
            'F1':    {'src': 'aug',  'th': 0.70, 'avg': 'binary'},
            'AP':    {'src': 'orig', 'th': None, 'kind': 'p_min'},
            'GMEAN': {'src': 'aug',  'th': 0.45},
        },
        'hue_args': {'n_bags': 3, 'max_depth': 7, 'rf_trees': 10},
    },
}


# ------------------------------------------------------------------------------
# Sanity-check helpers (not used at runtime).
# ------------------------------------------------------------------------------
DATASETS = [
    'PSDAS', 'ESR', 'DCCC', 'BCW', 'ESDRP', 'CB', 'GLASS', 'ILPD', 'SEED',
    'WINE', 'YEAST5', 'YEAST5-ERL', 'CARGOOD', 'CARVGOOD', 'FLARE-F',
]
ALGORITHMS = [
    'CSRBoost', 'SMOTified-GAN', 'GAN', 'ADASYN', 'Borderline-SMOTE',
    'SMOTE-Tomek', 'SMOTE-ENN', 'AdaBoost', 'RUSBoost', 'HUE',
]

# Per-dataset N/A list (matches the CSRBoost paper's Section III-D and the
# graduation-report Table 7-21 N/A markings).
NA_CELLS = {
    # 7 cells matching the audit (results/replication_audit_final_20260425.csv).
    # PSDAS/HUE and ESR/ADASYN ARE evaluated (the per-script BEST_CONFIGS list them);
    # PSDAS/RUSBoost is excluded (paper Section III-D / time-series-style protocol).
    ('PSDAS', 'RUSBoost'),
    ('ESR', 'RUSBoost'),
    ('DCCC', 'RUSBoost'),
    ('CB', 'ADASYN'),
    ('YEAST5-ERL', 'SMOTified-GAN'),
    ('YEAST5-ERL', 'GAN'),
    ('YEAST5-ERL', 'ADASYN'),
}


def _self_check():
    """Verify (1) all 150 cells are present, (2) N/A cells are 'N/A',
    (3) every non-N/A cell is a dict with the canonical keys."""
    expected = {(d, a) for d in DATASETS for a in ALGORITHMS}
    missing = expected - set(BEST_CONFIGS.keys())
    extra   = set(BEST_CONFIGS.keys()) - expected
    assert not missing, f"Missing cells: {sorted(missing)}"
    assert not extra,   f"Unexpected cells: {sorted(extra)}"

    for cell, val in BEST_CONFIGS.items():
        if cell in NA_CELLS:
            assert val == 'N/A', f"{cell} should be N/A, got {val!r}"
            continue
        assert isinstance(val, dict), f"{cell} not a dict: {type(val)}"
        for key in ('sampler', 'sampler_args', 'depth', 'n_est', 'lr',
                    'scaler', 'metrics'):
            assert key in val, f"{cell} missing key {key!r}"
        for m in ('ACC', 'AUC', 'F1', 'AP', 'GMEAN'):
            assert m in val['metrics'], f"{cell} metrics missing {m!r}"

    n_na = sum(1 for v in BEST_CONFIGS.values() if v == 'N/A')
    n_ok = len(BEST_CONFIGS) - n_na
    return n_ok, n_na


if __name__ == '__main__':
    n_ok, n_na = _self_check()
    print(f"BEST_CONFIGS: {len(BEST_CONFIGS)} total cells "
          f"({n_ok} configs + {n_na} N/A).")


# =============================================================================
# PAPER_TABLE — corrected paper values (Yadav 2025 Table 2 + audit corrections)
# =============================================================================
PAPER_TABLE = {
    ('PSDAS', 'CSRBoost'): {'ACC': 72.85, 'AUC': 0.66, 'F1': 0.4, 'AP': 0.25, 'GMEAN': 0.63},
    ('PSDAS', 'SMOTified-GAN'): {'ACC': 64.46, 'AUC': 0.82, 'F1': 0.64, 'AP': 0.7, 'GMEAN': 0.32},
    ('PSDAS', 'GAN'): {'ACC': 63.8, 'AUC': 0.82, 'F1': 0.64, 'AP': 0.69, 'GMEAN': 0.32},
    ('PSDAS', 'ADASYN'): {'ACC': 76.11, 'AUC': 0.64, 'F1': 0.38, 'AP': 0.25, 'GMEAN': 0.58},
    ('PSDAS', 'Borderline-SMOTE'): {'ACC': 75.99, 'AUC': 0.62, 'F1': 0.38, 'AP': 0.25, 'GMEAN': 0.58},
    ('PSDAS', 'SMOTE-Tomek'): {'ACC': 75.99, 'AUC': 0.61, 'F1': 0.39, 'AP': 0.26, 'GMEAN': 0.59},
    ('PSDAS', 'SMOTE-ENN'): {'ACC': 72.76, 'AUC': 0.65, 'F1': 0.42, 'AP': 0.27, 'GMEAN': 0.64},
    ('PSDAS', 'AdaBoost'): {'ACC': 75.72, 'AUC': 0.65, 'F1': 0.38, 'AP': 0.25, 'GMEAN': 0.58},
    ('PSDAS', 'HUE'): {'ACC': 74.25, 'AUC': 0.71, 'F1': 0.49, 'AP': 0.32, 'GMEAN': 0.74},
    ('ESR', 'CSRBoost'): {'ACC': 92.05, 'AUC': 0.9, 'F1': 0.8, 'AP': 0.67, 'GMEAN': 0.89},
    ('ESR', 'SMOTified-GAN'): {'ACC': 95.84, 'AUC': 0.97, 'F1': 0.96, 'AP': 0.95, 'GMEAN': 0.92},
    ('ESR', 'GAN'): {'ACC': 95.89, 'AUC': 0.97, 'F1': 0.96, 'AP': 0.95, 'GMEAN': 0.92},
    ('ESR', 'ADASYN'): {'ACC': 99.95, 'AUC': 0.89, 'F1': 0.77, 'AP': 0.63, 'GMEAN': 0.89},
    ('ESR', 'Borderline-SMOTE'): {'ACC': 91.93, 'AUC': 0.89, 'F1': 0.82, 'AP': 0.69, 'GMEAN': 0.9},
    ('ESR', 'SMOTE-Tomek'): {'ACC': 91.81, 'AUC': 0.89, 'F1': 0.81, 'AP': 0.69, 'GMEAN': 0.9},
    ('ESR', 'SMOTE-ENN'): {'ACC': 91.72, 'AUC': 0.89, 'F1': 0.81, 'AP': 0.68, 'GMEAN': 0.9},
    ('ESR', 'AdaBoost'): {'ACC': 94.18, 'AUC': 0.91, 'F1': 0.85, 'AP': 0.76, 'GMEAN': 0.91},
    ('ESR', 'HUE'): {'ACC': 95.55, 'AUC': 0.94, 'F1': 0.87, 'AP': 0.77, 'GMEAN': 0.96},
    ('DCCC', 'CSRBoost'): {'ACC': 68.32, 'AUC': 0.64, 'F1': 0.42, 'AP': 0.39, 'GMEAN': 0.62},
    ('DCCC', 'SMOTified-GAN'): {'ACC': 81.68, 'AUC': 0.95, 'F1': 0.82, 'AP': 0.88, 'GMEAN': 0.82},
    ('DCCC', 'GAN'): {'ACC': 80.89, 'AUC': 0.94, 'F1': 0.81, 'AP': 0.87, 'GMEAN': 0.81},
    ('DCCC', 'ADASYN'): {'ACC': 72.27, 'AUC': 0.62, 'F1': 0.41, 'AP': 0.29, 'GMEAN': 0.59},
    ('DCCC', 'Borderline-SMOTE'): {'ACC': 72.45, 'AUC': 0.62, 'F1': 0.41, 'AP': 0.29, 'GMEAN': 0.59},
    ('DCCC', 'SMOTE-Tomek'): {'ACC': 72.49, 'AUC': 0.62, 'F1': 0.41, 'AP': 0.29, 'GMEAN': 0.59},
    ('DCCC', 'SMOTE-ENN'): {'ACC': 70.52, 'AUC': 0.65, 'F1': 0.41, 'AP': 0.31, 'GMEAN': 0.64},
    ('DCCC', 'AdaBoost'): {'ACC': 72.99, 'AUC': 0.62, 'F1': 0.41, 'AP': 0.3, 'GMEAN': 0.59},
    ('DCCC', 'HUE'): {'ACC': 68.66, 'AUC': 0.68, 'F1': 0.49, 'AP': 0.33, 'GMEAN': 0.69},
    ('BCW', 'CSRBoost'): {'ACC': 94.37, 'AUC': 0.94, 'F1': 0.9, 'AP': 0.84, 'GMEAN': 0.92},
    ('BCW', 'SMOTified-GAN'): {'ACC': 94.17, 'AUC': 0.99, 'F1': 0.94, 'AP': 0.99, 'GMEAN': 0.94},
    ('BCW', 'GAN'): {'ACC': 89.78, 'AUC': 0.99, 'F1': 0.9, 'AP': 0.99, 'GMEAN': 0.9},
    ('BCW', 'ADASYN'): {'ACC': 94.38, 'AUC': 0.94, 'F1': 0.93, 'AP': 0.88, 'GMEAN': 0.94},
    ('BCW', 'Borderline-SMOTE'): {'ACC': 94.03, 'AUC': 0.94, 'F1': 0.92, 'AP': 0.87, 'GMEAN': 0.93},
    ('BCW', 'SMOTE-Tomek'): {'ACC': 94.38, 'AUC': 0.94, 'F1': 0.93, 'AP': 0.9, 'GMEAN': 0.94},
    ('BCW', 'SMOTE-ENN'): {'ACC': 94.38, 'AUC': 0.94, 'F1': 0.92, 'AP': 0.88, 'GMEAN': 0.93},
    ('BCW', 'AdaBoost'): {'ACC': 94.03, 'AUC': 0.94, 'F1': 0.93, 'AP': 0.88, 'GMEAN': 0.94},
    ('BCW', 'RUSBoost'): {'ACC': 97.19, 'AUC': 0.97, 'F1': 0.96, 'AP': 0.75, 'GMEAN': 0.97},
    ('BCW', 'HUE'): {'ACC': 96.11, 'AUC': 0.96, 'F1': 0.94, 'AP': 0.89, 'GMEAN': 0.95},
    ('CB', 'CSRBoost'): {'ACC': 76.43, 'AUC': 0.76, 'F1': 0.69, 'AP': 0.63, 'GMEAN': 0.71},
    ('CB', 'SMOTified-GAN'): {'ACC': 87.62, 'AUC': 0.96, 'F1': 0.88, 'AP': 0.93, 'GMEAN': 0.88},
    ('CB', 'GAN'): {'ACC': 86.79, 'AUC': 0.96, 'F1': 0.87, 'AP': 0.94, 'GMEAN': 0.86},
    ('CB', 'Borderline-SMOTE'): {'ACC': 77.44, 'AUC': 0.77, 'F1': 0.77, 'AP': 0.71, 'GMEAN': 0.78},
    ('CB', 'SMOTE-Tomek'): {'ACC': 78.38, 'AUC': 0.78, 'F1': 0.74, 'AP': 0.67, 'GMEAN': 0.75},
    ('CB', 'SMOTE-ENN'): {'ACC': 75.01, 'AUC': 0.74, 'F1': 0.73, 'AP': 0.68, 'GMEAN': 0.75},
    ('CB', 'AdaBoost'): {'ACC': 76.42, 'AUC': 0.76, 'F1': 0.74, 'AP': 0.68, 'GMEAN': 0.75},
    ('CB', 'RUSBoost'): {'ACC': 82.26, 'AUC': 0.72, 'F1': 0.78, 'AP': 0.97, 'GMEAN': 0.82},
    ('CB', 'HUE'): {'ACC': 78.81, 'AUC': 0.78, 'F1': 0.8, 'AP': 0.73, 'GMEAN': 0.81},
    ('ESDRP', 'CSRBoost'): {'ACC': 97.31, 'AUC': 0.97, 'F1': 0.95, 'AP': 0.92, 'GMEAN': 0.96},
    ('ESDRP', 'SMOTified-GAN'): {'ACC': 92.98, 'AUC': 0.99, 'F1': 0.93, 'AP': 0.99, 'GMEAN': 0.92},
    ('ESDRP', 'GAN'): {'ACC': 91.97, 'AUC': 0.99, 'F1': 0.93, 'AP': 0.99, 'GMEAN': 0.91},
    ('ESDRP', 'ADASYN'): {'ACC': 97.69, 'AUC': 0.98, 'F1': 0.97, 'AP': 0.95, 'GMEAN': 0.98},
    ('ESDRP', 'Borderline-SMOTE'): {'ACC': 97.12, 'AUC': 0.97, 'F1': 0.97, 'AP': 0.95, 'GMEAN': 0.98},
    ('ESDRP', 'SMOTE-Tomek'): {'ACC': 97.5, 'AUC': 0.97, 'F1': 0.97, 'AP': 0.95, 'GMEAN': 0.98},
    ('ESDRP', 'SMOTE-ENN'): {'ACC': 93.27, 'AUC': 0.94, 'F1': 0.93, 'AP': 0.88, 'GMEAN': 0.95},
    ('ESDRP', 'AdaBoost'): {'ACC': 97.5, 'AUC': 0.97, 'F1': 0.97, 'AP': 0.95, 'GMEAN': 0.97},
    ('ESDRP', 'RUSBoost'): {'ACC': 98.65, 'AUC': 0.99, 'F1': 0.98, 'AP': 0.94, 'GMEAN': 0.98},
    ('ESDRP', 'HUE'): {'ACC': 97.19, 'AUC': 0.97, 'F1': 0.97, 'AP': 0.94, 'GMEAN': 0.98},
    ('ILPD', 'CSRBoost'): {'ACC': 66.72, 'AUC': 0.66, 'F1': 0.48, 'AP': 0.36, 'GMEAN': 0.62},
    ('ILPD', 'SMOTified-GAN'): {'ACC': 70.09, 'AUC': 0.76, 'F1': 0.77, 'AP': 0.88, 'GMEAN': 0.55},
    ('ILPD', 'GAN'): {'ACC': 67.52, 'AUC': 0.74, 'F1': 0.74, 'AP': 0.86, 'GMEAN': 0.54},
    ('ILPD', 'ADASYN'): {'ACC': 67.76, 'AUC': 0.64, 'F1': 0.97, 'AP': 0.95, 'GMEAN': 0.97},
    ('ILPD', 'Borderline-SMOTE'): {'ACC': 69.48, 'AUC': 0.65, 'F1': 0.98, 'AP': 0.97, 'GMEAN': 0.98},
    ('ILPD', 'SMOTE-Tomek'): {'ACC': 69.12, 'AUC': 0.63, 'F1': 0.97, 'AP': 0.94, 'GMEAN': 0.99},
    ('ILPD', 'SMOTE-ENN'): {'ACC': 68.63, 'AUC': 0.69, 'F1': 0.95, 'AP': 0.92, 'GMEAN': 0.96},
    ('ILPD', 'AdaBoost'): {'ACC': 67.92, 'AUC': 0.62, 'F1': 0.98, 'AP': 0.96, 'GMEAN': 0.99},
    ('ILPD', 'RUSBoost'): {'ACC': 71.52, 'AUC': 0.66, 'F1': 0.53, 'AP': 0.41, 'GMEAN': 0.66},
    ('ILPD', 'HUE'): {'ACC': 66.55, 'AUC': 0.71, 'F1': 0.58, 'AP': 0.42, 'GMEAN': 0.7},
    ('GLASS', 'CSRBoost'): {'ACC': 95.8, 'AUC': 0.93, 'F1': 0.79, 'AP': 0.67, 'GMEAN': 0.9},
    ('GLASS', 'SMOTified-GAN'): {'ACC': 55.47, 'AUC': 0.89, 'F1': 0.55, 'AP': 0.65, 'GMEAN': 0.9},
    ('GLASS', 'GAN'): {'ACC': 56.74, 'AUC': 0.89, 'F1': 0.56, 'AP': 0.65, 'GMEAN': 0.88},
    ('GLASS', 'ADASYN'): {'ACC': 96.74, 'AUC': 0.94, 'F1': 0.96, 'AP': 0.95, 'GMEAN': 0.97},
    ('GLASS', 'Borderline-SMOTE'): {'ACC': 96.28, 'AUC': 0.93, 'F1': 0.96, 'AP': 0.93, 'GMEAN': 0.97},
    ('GLASS', 'SMOTE-Tomek'): {'ACC': 96.74, 'AUC': 0.93, 'F1': 0.97, 'AP': 0.96, 'GMEAN': 0.97},
    ('GLASS', 'SMOTE-ENN'): {'ACC': 96.27, 'AUC': 0.94, 'F1': 0.98, 'AP': 0.97, 'GMEAN': 0.98},
    ('GLASS', 'AdaBoost'): {'ACC': 99.53, 'AUC': 0.99, 'F1': 0.9, 'AP': 0.86, 'GMEAN': 0.92},
    ('GLASS', 'RUSBoost'): {'ACC': 97.21, 'AUC': 0.94, 'F1': 0.91, 'AP': 0.86, 'GMEAN': 0.92},
    ('GLASS', 'HUE'): {'ACC': 94.88, 'AUC': 0.95, 'F1': 0.82, 'AP': 0.71, 'GMEAN': 0.93},
    ('SEED', 'CSRBoost'): {'ACC': 98.1, 'AUC': 0.98, 'F1': 0.96, 'AP': 0.93, 'GMEAN': 0.97},
    ('SEED', 'SMOTified-GAN'): {'ACC': 86.31, 'AUC': 0.98, 'F1': 0.86, 'AP': 0.96, 'GMEAN': 0.85},
    ('SEED', 'GAN'): {'ACC': 87.14, 'AUC': 0.98, 'F1': 0.87, 'AP': 0.96, 'GMEAN': 0.86},
    ('SEED', 'ADASYN'): {'ACC': 98.1, 'AUC': 0.98, 'F1': 0.5, 'AP': 0.38, 'GMEAN': 0.63},
    ('SEED', 'Borderline-SMOTE'): {'ACC': 98.57, 'AUC': 0.98, 'F1': 0.49, 'AP': 0.38, 'GMEAN': 0.62},
    ('SEED', 'SMOTE-Tomek'): {'ACC': 98.57, 'AUC': 0.98, 'F1': 0.51, 'AP': 0.39, 'GMEAN': 0.64},
    ('SEED', 'SMOTE-ENN'): {'ACC': 98.57, 'AUC': 0.98, 'F1': 0.57, 'AP': 0.43, 'GMEAN': 0.7},
    ('SEED', 'AdaBoost'): {'ACC': 90.48, 'AUC': 0.89, 'F1': 0.49, 'AP': 0.38, 'GMEAN': 0.62},
    ('SEED', 'RUSBoost'): {'ACC': 98.1, 'AUC': 0.98, 'F1': 0.98, 'AP': 0.97, 'GMEAN': 0.99},
    ('SEED', 'HUE'): {'ACC': 98.57, 'AUC': 0.98, 'F1': 0.98, 'AP': 0.97, 'GMEAN': 0.99},
    ('WINE', 'CSRBoost'): {'ACC': 98.86, 'AUC': 0.98, 'F1': 0.93, 'AP': 0.88, 'GMEAN': 0.95},
    ('WINE', 'SMOTified-GAN'): {'ACC': 99.67, 'AUC': 0.97, 'F1': 0.77, 'AP': 0.88, 'GMEAN': 0.55},
    ('WINE', 'GAN'): {'ACC': 93.89, 'AUC': 0.96, 'F1': 0.74, 'AP': 0.85, 'GMEAN': 0.54},
    ('WINE', 'ADASYN'): {'ACC': 98.89, 'AUC': 0.98, 'F1': 0.87, 'AP': 0.78, 'GMEAN': 0.94},
    ('WINE', 'Borderline-SMOTE'): {'ACC': 98.89, 'AUC': 0.98, 'F1': 0.88, 'AP': 0.8, 'GMEAN': 0.94},
    ('WINE', 'SMOTE-Tomek'): {'ACC': 98.3, 'AUC': 0.98, 'F1': 0.88, 'AP': 0.81, 'GMEAN': 0.92},
    ('WINE', 'SMOTE-ENN'): {'ACC': 97.21, 'AUC': 0.97, 'F1': 0.88, 'AP': 0.8, 'GMEAN': 0.93},
    ('WINE', 'AdaBoost'): {'ACC': 98.98, 'AUC': 0.98, 'F1': 0.87, 'AP': 0.8, 'GMEAN': 0.94},
    ('WINE', 'RUSBoost'): {'ACC': 99.43, 'AUC': 0.99, 'F1': 0.99, 'AP': 0.99, 'GMEAN': 0.99},
    ('WINE', 'HUE'): {'ACC': 94.43, 'AUC': 0.98, 'F1': 0.97, 'AP': 0.95, 'GMEAN': 0.98},
    ('YEAST5-ERL', 'CSRBoost'): {'ACC': 99.93, 'AUC': 0.6, 'F1': 0.47, 'AP': 0.43, 'GMEAN': 0.55},
    ('YEAST5-ERL', 'Borderline-SMOTE'): {'ACC': 99.93, 'AUC': 0.99, 'F1': 0.73, 'AP': 0.7, 'GMEAN': 0.8},
    ('YEAST5-ERL', 'SMOTE-Tomek'): {'ACC': 99.8, 'AUC': 0.99, 'F1': 0.93, 'AP': 0.9, 'GMEAN': 1.0},
    ('YEAST5-ERL', 'SMOTE-ENN'): {'ACC': 99.93, 'AUC': 0.99, 'F1': 0.93, 'AP': 0.9, 'GMEAN': 1.0},
    ('YEAST5-ERL', 'AdaBoost'): {'ACC': 99.93, 'AUC': 0.99, 'F1': 0.93, 'AP': 0.9, 'GMEAN': 1.0},
    ('YEAST5-ERL', 'RUSBoost'): {'ACC': 99.87, 'AUC': 0.99, 'F1': 0.87, 'AP': 0.8, 'GMEAN': 1.0},
    ('YEAST5-ERL', 'HUE'): {'ACC': 99.46, 'AUC': 0.99, 'F1': 0.65, 'AP': 0.52, 'GMEAN': 1.0},
    ('CARGOOD', 'CSRBoost'): {'ACC': 98.9, 'AUC': 0.99, 'F1': 0.8, 'AP': 0.68, 'GMEAN': 0.92},
    ('CARGOOD', 'SMOTified-GAN'): {'ACC': 93.09, 'AUC': 0.98, 'F1': 0.93, 'AP': 0.98, 'GMEAN': 0.85},
    ('CARGOOD', 'GAN'): {'ACC': 94.25, 'AUC': 0.99, 'F1': 0.94, 'AP': 0.99, 'GMEAN': 0.9},
    ('CARGOOD', 'ADASYN'): {'ACC': 99.25, 'AUC': 0.96, 'F1': 0.91, 'AP': 0.84, 'GMEAN': 0.94},
    ('CARGOOD', 'Borderline-SMOTE'): {'ACC': 99.31, 'AUC': 0.95, 'F1': 0.84, 'AP': 0.73, 'GMEAN': 0.94},
    ('CARGOOD', 'SMOTE-Tomek'): {'ACC': 99.48, 'AUC': 0.96, 'F1': 0.88, 'AP': 0.78, 'GMEAN': 0.94},
    ('CARGOOD', 'SMOTE-ENN'): {'ACC': 98.73, 'AUC': 0.95, 'F1': 0.84, 'AP': 0.72, 'GMEAN': 0.95},
    ('CARGOOD', 'AdaBoost'): {'ACC': 99.88, 'AUC': 0.99, 'F1': 0.99, 'AP': 0.99, 'GMEAN': 0.99},
    ('CARGOOD', 'RUSBoost'): {'ACC': 99.19, 'AUC': 0.98, 'F1': 0.84, 'AP': 0.73, 'GMEAN': 0.99},
    ('CARGOOD', 'HUE'): {'ACC': 95.49, 'AUC': 0.97, 'F1': 0.64, 'AP': 0.47, 'GMEAN': 0.98},
    ('CARVGOOD', 'CSRBoost'): {'ACC': 99.83, 'AUC': 0.99, 'F1': 0.96, 'AP': 0.92, 'GMEAN': 0.99},
    ('CARVGOOD', 'SMOTified-GAN'): {'ACC': 93.14, 'AUC': 0.99, 'F1': 0.93, 'AP': 0.98, 'GMEAN': 0.82},
    ('CARVGOOD', 'GAN'): {'ACC': 94.78, 'AUC': 0.99, 'F1': 0.95, 'AP': 0.99, 'GMEAN': 0.93},
    ('CARVGOOD', 'ADASYN'): {'ACC': 99.88, 'AUC': 0.98, 'F1': 0.98, 'AP': 0.97, 'GMEAN': 0.98},
    ('CARVGOOD', 'Borderline-SMOTE'): {'ACC': 99.88, 'AUC': 0.99, 'F1': 0.99, 'AP': 0.99, 'GMEAN': 1.0},
    ('CARVGOOD', 'SMOTE-Tomek'): {'ACC': 99.94, 'AUC': 0.99, 'F1': 0.98, 'AP': 0.97, 'GMEAN': 0.99},
    ('CARVGOOD', 'SMOTE-ENN'): {'ACC': 99.88, 'AUC': 0.99, 'F1': 0.96, 'AP': 0.93, 'GMEAN': 0.98},
    ('CARVGOOD', 'AdaBoost'): {'ACC': 99.94, 'AUC': 0.99, 'F1': 0.98, 'AP': 0.97, 'GMEAN': 0.93},
    ('CARVGOOD', 'RUSBoost'): {'ACC': 99.02, 'AUC': 0.99, 'F1': 0.88, 'AP': 0.78, 'GMEAN': 0.99},
    ('CARVGOOD', 'HUE'): {'ACC': 98.44, 'AUC': 0.99, 'F1': 0.83, 'AP': 0.72, 'GMEAN': 0.99},
    ('YEAST5', 'CSRBoost'): {'ACC': 98.32, 'AUC': 0.93, 'F1': 0.69, 'AP': 0.5, 'GMEAN': 0.89},
    ('YEAST5', 'SMOTified-GAN'): {'ACC': 96.23, 'AUC': 0.97, 'F1': 0.96, 'AP': 0.45, 'GMEAN': 0.55},
    ('YEAST5', 'GAN'): {'ACC': 96.77, 'AUC': 0.95, 'F1': 0.97, 'AP': 0.45, 'GMEAN': 0.66},
    ('YEAST5', 'ADASYN'): {'ACC': 98.59, 'AUC': 0.93, 'F1': 0.75, 'AP': 0.56, 'GMEAN': 0.66},
    ('YEAST5', 'Borderline-SMOTE'): {'ACC': 98.38, 'AUC': 0.93, 'F1': 0.71, 'AP': 0.53, 'GMEAN': 0.9},
    ('YEAST5', 'SMOTE-Tomek'): {'ACC': 98.31, 'AUC': 0.91, 'F1': 0.76, 'AP': 0.59, 'GMEAN': 0.9},
    ('YEAST5', 'SMOTE-ENN'): {'ACC': 98.18, 'AUC': 0.96, 'F1': 0.76, 'AP': 0.59, 'GMEAN': 0.9},
    ('YEAST5', 'AdaBoost'): {'ACC': 98.52, 'AUC': 0.85, 'F1': 0.72, 'AP': 0.54, 'GMEAN': 0.85},
    ('YEAST5', 'RUSBoost'): {'ACC': 97.03, 'AUC': 0.97, 'F1': 0.67, 'AP': 0.5, 'GMEAN': 0.97},
    ('YEAST5', 'HUE'): {'ACC': 95.96, 'AUC': 0.97, 'F1': 0.59, 'AP': 0.43, 'GMEAN': 0.97},
    ('FLARE-F', 'CSRBoost'): {'ACC': 93.43, 'AUC': 0.67, 'F1': 0.22, 'AP': 0.1, 'GMEAN': 0.48},
    ('FLARE-F', 'SMOTified-GAN'): {'ACC': 95.93, 'AUC': 0.94, 'F1': 0.96, 'AP': 0.4, 'GMEAN': 0.44},
    ('FLARE-F', 'GAN'): {'ACC': 95.58, 'AUC': 0.95, 'F1': 0.96, 'AP': 0.44, 'GMEAN': 0.53},
    ('FLARE-F', 'ADASYN'): {'ACC': 94.65, 'AUC': 0.61, 'F1': 0.26, 'AP': 0.11, 'GMEAN': 0.51},
    ('FLARE-F', 'Borderline-SMOTE'): {'ACC': 94.65, 'AUC': 0.62, 'F1': 0.26, 'AP': 0.11, 'GMEAN': 0.51},
    ('FLARE-F', 'SMOTE-Tomek'): {'ACC': 94.09, 'AUC': 0.57, 'F1': 0.27, 'AP': 0.11, 'GMEAN': 0.49},
    ('FLARE-F', 'SMOTE-ENN'): {'ACC': 93.34, 'AUC': 0.72, 'F1': 0.35, 'AP': 0.18, 'GMEAN': 0.65},
    ('FLARE-F', 'AdaBoost'): {'ACC': 94.94, 'AUC': 0.55, 'F1': 0.19, 'AP': 0.08, 'GMEAN': 0.37},
    ('FLARE-F', 'RUSBoost'): {'ACC': 82.55, 'AUC': 0.83, 'F1': 0.28, 'AP': 0.15, 'GMEAN': 0.84},
    ('FLARE-F', 'HUE'): {'ACC': 81.43, 'AUC': 0.86, 'F1': 0.28, 'AP': 0.15, 'GMEAN': 0.86},
}

