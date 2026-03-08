"""
model.py
Ensemble model (XGBoost + LightGBM + CatBoost) for mean reversion prediction.
Includes temporal split, Optuna optimization, and training pipeline.
"""

import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

warnings.filterwarnings("ignore", message="X does not have valid feature names")

from features import FEATURE_COLS
from config import TRAIN_RATIO, VAL_RATIO, OPTUNA_TRIALS, OPTUNA_CV_SPLITS


# ── Temporal Split ───────────────────────────────────────────


def temporal_split_3way(df: pd.DataFrame,
                        train_ratio: float = TRAIN_RATIO,
                        val_ratio: float = VAL_RATIO) -> tuple:
    """
    Temporal split into train / validation / test.
    No shuffling — preserves time order.
    """
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train = df.iloc[:train_end].copy()
    val = df.iloc[train_end:val_end].copy()
    test = df.iloc[val_end:].copy()

    print(f"  Train: {len(train)} rows "
          f"({train['timestamp'].iloc[0]} → {train['timestamp'].iloc[-1]})")
    print(f"  Val:   {len(val)} rows "
          f"({val['timestamp'].iloc[0]} → {val['timestamp'].iloc[-1]})")
    print(f"  Test:  {len(test)} rows "
          f"({test['timestamp'].iloc[0]} → {test['timestamp'].iloc[-1]})")

    return train, val, test


# ── Ensemble Model ───────────────────────────────────────────


class EnsembleModel:
    """
    Ensemble of 3 classifiers with voting.
    Predicts probability of mean reversion given extreme Z-score.
    """

    def __init__(self):
        self.models = {}
        self.scaler = None
        self.model_names = ["xgboost", "lightgbm", "catboost"]

    def _build_models(self, spw: float, params: dict = None) -> dict:
        """Build the 3 models with high regularization defaults."""
        p = params or {}

        xgb_params = p.get("xgboost", {
            "n_estimators": 100,
            "max_depth": 3,
            "learning_rate": 0.05,
            "subsample": 0.6,
            "colsample_bytree": 0.5,
            "min_child_weight": 15,
            "gamma": 5.0,
            "reg_alpha": 3.0,
            "reg_lambda": 10.0,
            "scale_pos_weight": spw,
            "eval_metric": "logloss",
            "random_state": 42,
        })

        lgbm_params = p.get("lightgbm", {
            "n_estimators": 100,
            "max_depth": 3,
            "learning_rate": 0.05,
            "subsample": 0.6,
            "colsample_bytree": 0.5,
            "min_child_weight": 10.0,
            "num_leaves": 15,
            "reg_alpha": 3.0,
            "reg_lambda": 10.0,
            "min_child_samples": 30,
            "scale_pos_weight": spw,
            "random_state": 42,
            "verbosity": -1,
        })

        cat_params = p.get("catboost", {
            "iterations": 100,
            "depth": 3,
            "learning_rate": 0.05,
            "subsample": 0.6,
            "l2_leaf_reg": 15.0,
            "min_data_in_leaf": 30,
            "auto_class_weights": "Balanced",
            "random_seed": 42,
            "verbose": 0,
        })

        return {
            "xgboost": XGBClassifier(**xgb_params),
            "lightgbm": LGBMClassifier(**lgbm_params),
            "catboost": CatBoostClassifier(**cat_params),
        }

    def fit(self, train_df: pd.DataFrame,
            feature_cols: list = FEATURE_COLS,
            params: dict = None):
        """Train all 3 models on the training data."""
        X = train_df[feature_cols].values
        y = train_df["target"].values.astype(int)

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        n_neg = (y == 0).sum()
        n_pos = (y == 1).sum()
        spw = n_neg / n_pos if n_pos > 0 else 1.0

        self.models = self._build_models(spw, params)

        print(f"  Training ensemble ({len(self.models)} models)...")
        for name, model in self.models.items():
            model.fit(X_scaled, y)
            proba = model.predict_proba(X_scaled)[:, 1]
            auc = roc_auc_score(y, proba)
            print(f"    {name:<12} train AUC: {auc:.4f}")

        print(f"  Ensemble trained — {len(y)} samples, "
              f"revert=1: {n_pos} ({y.mean():.1%})")

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Average probability from all 3 models."""
        X_scaled = self.scaler.transform(X)
        probas = []
        for model in self.models.values():
            probas.append(model.predict_proba(X_scaled)[:, 1])
        return np.mean(probas, axis=0)

    def predict_proba_individual(self, X: np.ndarray) -> dict:
        """Per-model probabilities."""
        X_scaled = self.scaler.transform(X)
        return {
            name: model.predict_proba(X_scaled)[:, 1]
            for name, model in self.models.items()
        }

    def predict_with_voting(self, X: np.ndarray,
                            threshold: float = 0.60,
                            min_agree: int = 2) -> tuple:
        """
        Voting prediction.
        Returns (avg_proba, n_agree, signals).
        """
        X_scaled = self.scaler.transform(X)
        individual = np.array([
            model.predict_proba(X_scaled)[:, 1]
            for model in self.models.values()
        ])

        avg_proba = individual.mean(axis=0)
        n_agree = (individual > threshold).sum(axis=0)
        signals = n_agree >= min_agree

        return avg_proba, n_agree, signals


# ── Optuna Optimization ──────────────────────────────────────


def optimize_ensemble(train_df: pd.DataFrame,
                      feature_cols: list = FEATURE_COLS,
                      n_trials: int = OPTUNA_TRIALS,
                      n_splits: int = OPTUNA_CV_SPLITS) -> dict:
    """
    Optimize hyperparameters for each model in the ensemble
    using Optuna + TimeSeriesSplit. Objective: maximize ROC AUC.
    """
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    X = train_df[feature_cols].values
    y = train_df["target"].values.astype(int)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    n_neg = (y == 0).sum()
    n_pos = (y == 1).sum()
    spw = n_neg / n_pos if n_pos > 0 else 1.0

    tscv = TimeSeriesSplit(n_splits=n_splits)
    optimized = {}

    def _penalized_score(clf, X_tr, y_tr, X_vl, y_vl):
        """Val AUC minus penalty for overfitting gap > 10%."""
        train_auc = roc_auc_score(y_tr, clf.predict_proba(X_tr)[:, 1])
        val_auc = roc_auc_score(y_vl, clf.predict_proba(X_vl)[:, 1])
        overfit_gap = max(0.0, train_auc - val_auc - 0.10)
        return val_auc - 0.5 * overfit_gap

    # ── XGBoost ──
    print("  Optimizing XGBoost...")

    def obj_xgb(trial):
        p = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 150),
            "max_depth": 3,  # fixed — deeper trees overfit on ~700 samples
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 0.8),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 0.7),
            "min_child_weight": trial.suggest_int("min_child_weight", 10, 30),
            "gamma": trial.suggest_float("gamma", 2.0, 8.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1.0, 10.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 5.0, 20.0),
            "scale_pos_weight": spw,
            "eval_metric": "logloss",
            "random_state": 42,
        }
        scores = []
        for tr_idx, vl_idx in tscv.split(X_scaled):
            clf = XGBClassifier(**p)
            clf.fit(X_scaled[tr_idx], y[tr_idx], verbose=False)
            scores.append(_penalized_score(
                clf, X_scaled[tr_idx], y[tr_idx], X_scaled[vl_idx], y[vl_idx]
            ))
        return np.mean(scores)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(obj_xgb, n_trials=n_trials, show_progress_bar=True)
    best_xgb = study.best_params
    best_xgb.update({
        "max_depth": 3, "scale_pos_weight": spw,
        "eval_metric": "logloss", "random_state": 42,
    })
    optimized["xgboost"] = best_xgb
    print(f"    XGBoost best AUC: {study.best_value:.4f}")

    # ── LightGBM ──
    print("  Optimizing LightGBM...")

    def obj_lgbm(trial):
        p = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 150),
            "max_depth": 3,  # fixed
            "num_leaves": 15,  # fixed — prevents exponential splits
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 0.8),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 0.7),
            "min_child_weight": trial.suggest_float("min_child_weight", 5.0, 20.0),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 50),
            "reg_alpha": trial.suggest_float("reg_alpha", 1.0, 10.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 5.0, 20.0),
            "scale_pos_weight": spw,
            "random_state": 42,
            "verbosity": -1,
        }
        scores = []
        for tr_idx, vl_idx in tscv.split(X_scaled):
            clf = LGBMClassifier(**p)
            clf.fit(X_scaled[tr_idx], y[tr_idx])
            scores.append(_penalized_score(
                clf, X_scaled[tr_idx], y[tr_idx], X_scaled[vl_idx], y[vl_idx]
            ))
        return np.mean(scores)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=43))
    study.optimize(obj_lgbm, n_trials=n_trials, show_progress_bar=True)
    best_lgbm = study.best_params
    best_lgbm.update({
        "max_depth": 3, "num_leaves": 15,
        "scale_pos_weight": spw, "random_state": 42, "verbosity": -1,
    })
    optimized["lightgbm"] = best_lgbm
    print(f"    LightGBM best AUC: {study.best_value:.4f}")

    # ── CatBoost ──
    print("  Optimizing CatBoost...")

    def obj_cat(trial):
        p = {
            "iterations": trial.suggest_int("iterations", 50, 150),
            "depth": 3,  # fixed
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 0.8),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 5.0, 25.0),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 20, 60),
            "auto_class_weights": "Balanced",
            "random_seed": 42,
            "verbose": 0,
        }
        scores = []
        for tr_idx, vl_idx in tscv.split(X_scaled):
            clf = CatBoostClassifier(**p)
            clf.fit(X_scaled[tr_idx], y[tr_idx])
            scores.append(_penalized_score(
                clf, X_scaled[tr_idx], y[tr_idx], X_scaled[vl_idx], y[vl_idx]
            ))
        return np.mean(scores)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=44))
    study.optimize(obj_cat, n_trials=n_trials, show_progress_bar=True)
    best_cat = study.best_params
    best_cat.update({
        "depth": 3, "auto_class_weights": "Balanced", "random_seed": 42, "verbose": 0,
    })
    optimized["catboost"] = best_cat
    print(f"    CatBoost best AUC: {study.best_value:.4f}")

    return optimized
