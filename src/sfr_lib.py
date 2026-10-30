"""Shared experiment code for the galaxy SFR regression task.

Importable from scripts (``scripts/``) and from the report notebook::

    import sys; sys.path.insert(0, "<project>/src")
    import sfr_lib as L

Design notes (see results/PREREGISTRATION.md and results/EXPERIMENT_LOG.md):
- The fixed template split (cell 15) is reproduced exactly; the 20 % validation
  set is never used for model selection.
- CV lives inside the 80 % training split only: StratifiedKFold(5, shuffle,
  seed 42) on an 11-class label = {SFR == 0} + 10 deciles of log10 SFR (> 0).
- Every fitted quantity (smearing factor, spline knots, scalers, early-stopping
  split) is computed from the training part of a fold only.
- Neighbour/environment features use positions and halo masses (inputs) of the
  full catalogue, never the target.
"""
from __future__ import annotations

import json
import os
import platform
import socket
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (ExtraTreesRegressor, HistGradientBoostingClassifier,
                              HistGradientBoostingRegressor)
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, StratifiedKFold, train_test_split
from sklearn.preprocessing import SplineTransformer, StandardScaler

# --------------------------------------------------------------------------
# Paths and constants
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "Galaxy Features with Position and Velocity for Star Formation Rate Prediction.csv"
RESULTS_DIR = PROJECT_ROOT / "results"
OOF_DIR = RESULTS_DIR / "oof"
META_DIR = RESULTS_DIR / "exp_meta"
VAL_DIR = RESULTS_DIR / "val"
TUNING_DIR = RESULTS_DIR / "tuning"
SPLIT_PATH = RESULTS_DIR / "split_indices.parquet"
ENV_CACHE = PROJECT_ROOT / "data" / "env_features.parquet"

RANDOM_SEED = 42
TARGET = "sfr_linear"
ID_COLUMN = "subhalo_id"
BOX_SIZE = 200.0  # ASSUMPTION: periodic box of side 200 (same units as pos_*)
N_THREADS = int(os.environ.get("SFR_N_THREADS", "12"))
N_FOLDS = 5
LOG_PRED_FLOOR = 1e-3  # predictions clipped to >= this for log10-RMSE

BASE_FEATURES = [
    "log10_halo_mass", "log10_stellar_mass", "log10_BH_mass",
    "log10_main_prog_mass", "log10_main_prog_stellar_mass", "log10_mass_ratio",
    "log10_overdensity_plus_1", "pos_x", "pos_y", "pos_z",
    "vel_x", "vel_y", "vel_z",
]
MASS_FEATURES = BASE_FEATURES[:6]
POS_COLS = ["pos_x", "pos_y", "pos_z"]
VEL_COLS = ["vel_x", "vel_y", "vel_z"]

PHYS_FEATURES = [
    "d_logMs_prog",          # logMs - logMs_prog
    "dMs_lin_1e9",           # (10^logMs - 10^logMs_prog) / 1e9
    "d_logMh_prog",          # logMh - logMh_prog
    "logMs_minus_logMh",
    "logMBH_minus_logMs",
    "is_merger",             # log10_mass_ratio > -6
    "is_BH_floor",           # log10_BH_mass < 5.05 (seed-mass floor)
    "is_prog_floor",         # progenitor stellar or halo mass at the 6.0 floor
    "v_mag",                 # |v|
]
ENV_FEATURES = [
    "env_d1", "env_d5", "env_d10",          # distance to k-th nearest galaxy
    "env_n_r1", "env_n_r2",                 # neighbours within r = 1, 2
    "env_d_heavier",                        # distance to nearest galaxy with larger halo mass
    "env_logMh_heavier",                    # its log halo mass
    "env_is_localmax_r1",                   # no heavier galaxy within r = 1 (central proxy)
    "env_dv_heavier",                       # |v_i - v_heavier|
]

FEATURE_SETS = {
    "F0": list(BASE_FEATURES),
    "F1": list(MASS_FEATURES),
    "F2": MASS_FEATURES + ["log10_overdensity_plus_1"],
    "F3": BASE_FEATURES + PHYS_FEATURES,
    "F4": BASE_FEATURES + PHYS_FEATURES + ENV_FEATURES,
}
FEATURE_SETS["F5"] = [c for c in FEATURE_SETS["F4"] if c not in POS_COLS + VEL_COLS]
FEATURE_SETS["F6"] = FEATURE_SETS["F5"] + ["mssm_n08_r1", "mssm_n08_r2", "mssm_n08_r5"]  # Amendment 2 (E6a)

# Groups for permutation importance
FEATURE_GROUPS = {
    "mass": ["log10_halo_mass", "log10_stellar_mass", "log10_BH_mass",
             "logMs_minus_logMh", "logMBH_minus_logMs", "is_BH_floor"],
    "growth": ["log10_main_prog_mass", "log10_main_prog_stellar_mass", "log10_mass_ratio",
               "d_logMs_prog", "dMs_lin_1e9", "d_logMh_prog", "is_merger", "is_prog_floor"],
    "environment": ["log10_overdensity_plus_1", "env_d1", "env_d5", "env_d10", "env_n_r1",
                    "env_n_r2", "env_d_heavier", "env_logMh_heavier", "env_is_localmax_r1",
                    "mssm_n08_r1", "mssm_n08_r2", "mssm_n08_r5"],
    "position": POS_COLS,
    "velocity": VEL_COLS + ["v_mag", "env_dv_heavier"],
}


# --------------------------------------------------------------------------
# Template metric functions (cell 16, verbatim)
# --------------------------------------------------------------------------
def regression_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": mean_squared_error(y_true, y_pred) ** 0.5,
        "R2": r2_score(y_true, y_pred),
    }


def evaluate_sfr(y_true, y_pred, label="Model"):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    rows = []
    masks = {
        "All": np.ones(len(y_true), dtype=bool),
        "SFR = 0": y_true == 0,
        "SFR > 0": y_true > 0,
        "Top 10% SFR": y_true >= np.quantile(y_true, 0.90),
    }
    for regime_name, mask in masks.items():
        if mask.sum() == 0:
            continue
        m = regression_metrics(y_true[mask], y_pred[mask])
        rows.append({"Model": label, "Regime": regime_name, "N": int(mask.sum()), **m})
    return pd.DataFrame(rows)


_REGIME_KEY = {"All": "all", "SFR = 0": "zero", "SFR > 0": "pos", "Top 10% SFR": "top10"}


def flat_metrics(y_true, y_pred):
    """Flat dict: template regime metrics + log10-RMSE on SFR > 0."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    out = {}
    for _, r in evaluate_sfr(y_true, y_pred).iterrows():
        k = _REGIME_KEY[r["Regime"]]
        out[f"rmse_{k}"] = float(r["RMSE"])
        out[f"mae_{k}"] = float(r["MAE"])
        out[f"r2_{k}"] = float(r["R2"])
    pos = y_true > 0
    lp = np.log10(np.clip(y_pred[pos], LOG_PRED_FLOOR, None))
    out["log10rmse_pos"] = float(np.sqrt(np.mean((lp - np.log10(y_true[pos])) ** 2)))
    out["mean_pred_over_mean_true"] = float(y_pred.mean() / y_true.mean())
    out["frac_neg_pred"] = float((y_pred < 0).mean())
    return out


# --------------------------------------------------------------------------
# Data, split, folds
# --------------------------------------------------------------------------
def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    n_missing_target = int(df[TARGET].isna().sum())
    if n_missing_target:
        df = df.dropna(subset=[TARGET]).copy()
    return df


def template_split(df: pd.DataFrame):
    """Exact reproduction of template cells 12 + 15."""
    allowed = [c for c in df.columns if c not in [ID_COLUMN, TARGET]]
    X = df[allowed].copy()
    y = df[TARGET].copy()
    regime = (y > 0).astype(int)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_SEED, stratify=regime)
    return X_train, X_val, y_train, y_val


def save_split(df: pd.DataFrame, path: Path = SPLIT_PATH):
    X_train, X_val, y_train, y_val = template_split(df)
    out = pd.concat([
        pd.DataFrame({"row": X_train.index.values, "subset": "train", "order": np.arange(len(X_train))}),
        pd.DataFrame({"row": X_val.index.values, "subset": "val", "order": np.arange(len(X_val))}),
    ], ignore_index=True)
    out["subhalo_id"] = df.loc[out["row"], ID_COLUMN].values
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False)
    return out


def load_split(path: Path = SPLIT_PATH):
    s = pd.read_parquet(path)
    tr = s[s.subset == "train"].sort_values("order")["row"].to_numpy()
    va = s[s.subset == "val"].sort_values("order")["row"].to_numpy()
    return tr, va


def strata_labels(y) -> np.ndarray:
    """0 for SFR == 0; 1..10 = decile of log10(SFR) among positive rows."""
    y = np.asarray(y, dtype=float)
    lab = np.zeros(len(y), dtype=int)
    pos = y > 0
    if pos.sum() >= 10:
        ly = np.log10(y[pos])
        edges = np.quantile(ly, np.linspace(0, 1, 11)[1:-1])
        lab[pos] = 1 + np.searchsorted(edges, ly, side="right")
    return lab


def make_random_folds(y_train) -> np.ndarray:
    """Fold id (0..4) per training row; StratifiedKFold(5, shuffle, 42) on strata."""
    lab = strata_labels(y_train)
    folds = np.full(len(lab), -1, dtype=int)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    for k, (_, te) in enumerate(skf.split(np.zeros(len(lab)), lab)):
        folds[te] = k
    return folds


def spatial_block_ids(pos: np.ndarray, n_per_side: int = 5) -> np.ndarray:
    p = np.mod(np.asarray(pos, dtype=float), BOX_SIZE)
    b = np.minimum((p / (BOX_SIZE / n_per_side)).astype(int), n_per_side - 1)
    return b[:, 0] * n_per_side ** 2 + b[:, 1] * n_per_side + b[:, 2]


def make_spatial_folds(pos_train: np.ndarray) -> np.ndarray:
    """GroupKFold(5) over 5x5x5 spatial blocks (no shuffle; size-balanced)."""
    g = spatial_block_ids(pos_train)
    folds = np.full(len(g), -1, dtype=int)
    for k, (_, te) in enumerate(GroupKFold(n_splits=N_FOLDS).split(np.zeros(len(g)), groups=g)):
        folds[te] = k
    return folds


# --------------------------------------------------------------------------
# Features
# --------------------------------------------------------------------------
def add_physics_features(df: pd.DataFrame) -> pd.DataFrame:
    """Row-wise derived features (no fitted quantities -> no leakage)."""
    out = pd.DataFrame(index=df.index)
    lms, lmsp = df["log10_stellar_mass"], df["log10_main_prog_stellar_mass"]
    lmh, lmhp = df["log10_halo_mass"], df["log10_main_prog_mass"]
    out["d_logMs_prog"] = lms - lmsp
    out["dMs_lin_1e9"] = (10.0 ** lms - 10.0 ** lmsp) / 1e9
    out["d_logMh_prog"] = lmh - lmhp
    out["logMs_minus_logMh"] = lms - lmh
    out["logMBH_minus_logMs"] = df["log10_BH_mass"] - lms
    out["is_merger"] = (df["log10_mass_ratio"] > -6.0).astype(float)
    out["is_BH_floor"] = (df["log10_BH_mass"] < 5.05).astype(float)
    out["is_prog_floor"] = ((lmsp <= 6.0) | (lmhp <= 6.0)).astype(float)
    out["v_mag"] = np.sqrt((df[VEL_COLS] ** 2).sum(axis=1))
    return out


def compute_env_features(df: pd.DataFrame, cache: Path = ENV_CACHE, workers: int = N_THREADS) -> pd.DataFrame:
    """Neighbour features from positions + halo masses of ALL galaxies (no target).

    Periodic KD-tree with boxsize = 200 (assumption). Indexed like ``df``.
    """
    if cache is not None and Path(cache).exists():
        env = pd.read_parquet(cache)
        if len(env) == len(df) and (env.index.values == df.index.values).all():
            return env
    pos = np.mod(df[POS_COLS].to_numpy(dtype=float), BOX_SIZE)
    vel = df[VEL_COLS].to_numpy(dtype=float)
    mh = df["log10_halo_mass"].to_numpy(dtype=float)
    n = len(df)
    tree = cKDTree(pos, boxsize=BOX_SIZE)
    d, _ = tree.query(pos, k=11, workers=workers)
    out = pd.DataFrame(index=df.index)
    out["env_d1"], out["env_d5"], out["env_d10"] = d[:, 1], d[:, 5], d[:, 10]
    for r in (1.0, 2.0):
        out[f"env_n_r{int(r)}"] = tree.query_ball_point(pos, r, return_length=True, workers=workers) - 1

    # nearest galaxy with strictly larger halo mass
    hv_idx = np.full(n, -1, dtype=np.int64)
    hv_d = np.full(n, np.nan)
    todo = np.arange(n)
    for k in (64, 1024):
        if len(todo) == 0:
            break
        dd, ii = tree.query(pos[todo], k=min(k, n), workers=workers)
        heavier = mh[ii] > mh[todo][:, None]
        has = heavier.any(axis=1)
        first = heavier.argmax(axis=1)
        sel = todo[has]
        hv_idx[sel] = ii[has, first[has]]
        hv_d[sel] = dd[has, first[has]]
        todo = todo[~has]
    for i in todo:  # brute force for the few most massive objects
        cand = np.where(mh > mh[i])[0]
        if len(cand) == 0:
            continue
        delta = np.abs(pos[cand] - pos[i])
        delta = np.minimum(delta, BOX_SIZE - delta)
        dist = np.sqrt((delta ** 2).sum(axis=1))
        j = dist.argmin()
        hv_idx[i], hv_d[i] = cand[j], dist[j]
    none = hv_idx < 0  # the global maximum: no heavier galaxy exists
    max_dist = np.sqrt(3) * BOX_SIZE / 2
    hv_d[none] = max_dist
    safe = np.where(none, np.arange(n), hv_idx)
    out["env_d_heavier"] = hv_d
    out["env_logMh_heavier"] = np.where(none, mh, mh[safe])
    out["env_is_localmax_r1"] = (hv_d > 1.0).astype(float)
    out["env_dv_heavier"] = np.where(none, 0.0, np.linalg.norm(vel - vel[safe], axis=1))
    out.attrs["n_no_heavier"] = int(none.sum())
    if cache is not None:
        out.to_parquet(cache)
    return out


MSSM_RADII = (1.0, 2.0, 5.0)
MSSM_FEATURES = [f"mssm_n08_r{int(r)}" for r in MSSM_RADII]
MSSM_CACHE = PROJECT_ROOT / "data" / "env_mssm.parquet"


def compute_mssm_counts(df: pd.DataFrame, cache: Path = MSSM_CACHE, workers: int = N_THREADS) -> pd.DataFrame:
    """Jo & Kim (2019)-style counts: other galaxies within R (periodic, boxsize 200) whose
    log halo mass >= own log halo mass + log10(0.8). Inputs only (positions, halo masses)."""
    if cache is not None and Path(cache).exists():
        out = pd.read_parquet(cache)
        if len(out) == len(df) and (out.index.values == df.index.values).all():
            return out
    pos = np.mod(df[POS_COLS].to_numpy(dtype=float), BOX_SIZE)
    mh = df["log10_halo_mass"].to_numpy(dtype=float)
    tree = cKDTree(pos, boxsize=BOX_SIZE)
    rmax = max(MSSM_RADII)
    pairs = tree.query_pairs(rmax, output_type="ndarray")  # i < j, periodic
    i, j = pairs[:, 0], pairs[:, 1]
    delta = np.abs(pos[i] - pos[j])
    delta = np.minimum(delta, BOX_SIZE - delta)
    d = np.sqrt((delta ** 2).sum(axis=1))
    thr = np.log10(0.8)
    j_counts_for_i = mh[j] >= mh[i] + thr
    i_counts_for_j = mh[i] >= mh[j] + thr
    out = pd.DataFrame(index=df.index)
    n = len(df)
    for r, name in zip(MSSM_RADII, MSSM_FEATURES):
        w = d <= r
        out[name] = (np.bincount(i[w & j_counts_for_i], minlength=n)
                     + np.bincount(j[w & i_counts_for_j], minlength=n)).astype(float)
    out.attrs["n_pairs_rmax"] = int(len(pairs))
    if cache is not None:
        out.to_parquet(cache)
    return out


def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """All candidate columns (base + physics + environment + MSSM counts), indexed like df."""
    return pd.concat([df[BASE_FEATURES], add_physics_features(df), compute_env_features(df),
                      compute_mssm_counts(df)], axis=1)


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------
E1_HGB = dict(learning_rate=0.05, max_iter=1000, max_leaf_nodes=31)
ES_FRACTION = 0.10
ES_ROUNDS = 10


def _es_split(y, seed):
    lab = strata_labels(y)
    idx = np.arange(len(y))
    try:
        tr, es = train_test_split(idx, test_size=ES_FRACTION, random_state=seed, stratify=lab)
    except ValueError:
        tr, es = train_test_split(idx, test_size=ES_FRACTION, random_state=seed)
    return tr, es


class _SplineGLM:
    """Poisson GLM on cubic B-spline expansion of continuous columns (+ binary passthrough)."""

    def __init__(self, n_knots=6, alpha=1e-4, max_iter=2000):
        self.n_knots, self.alpha, self.max_iter = n_knots, alpha, max_iter

    def _design(self, X, fit):
        Xc = X[:, self.cont_]
        parts = [self.spl_.fit_transform(Xc) if fit else self.spl_.transform(Xc)]
        if len(self.bin_):
            parts.append(X[:, self.bin_])
        Z = np.hstack(parts)
        return self.sc_.fit_transform(Z) if fit else self.sc_.transform(Z)

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        nun = np.array([len(np.unique(X[:, j])) > 2 for j in range(X.shape[1])])
        self.cont_, self.bin_ = np.where(nun)[0], np.where(~nun)[0]
        self.spl_ = SplineTransformer(n_knots=self.n_knots, degree=3, knots="quantile",
                                      extrapolation="constant")
        self.sc_ = StandardScaler()
        Z = self._design(X, fit=True)
        self.glm_ = PoissonRegressor(alpha=self.alpha, max_iter=self.max_iter, solver="newton-cholesky")
        self.glm_.fit(Z, y)
        return self

    def predict(self, X):
        return self.glm_.predict(self._design(np.asarray(X, dtype=float), fit=False))


def _make_hgb(loss, params, seed):
    p = dict(E1_HGB)
    p.update(params or {})
    return HistGradientBoostingRegressor(loss=loss, early_stopping=True, n_iter_no_change=ES_ROUNDS,
                                         random_state=seed, **p)


def _fit_gbm(family, objective, params, X, y, seed):
    """Fit HGB or LightGBM with an explicit stratified 10 % early-stopping split.

    Returns (model, info, es_index) where es_index are the rows held out for ES.
    """
    tr, es = _es_split(y, seed)
    if family == "hgb":
        loss = {"regression": "squared_error", "poisson": "poisson"}[objective]
        m = _make_hgb(loss, params, seed)
        m.fit(X[tr], y[tr], X_val=X[es], y_val=y[es])
        info = {"n_iter": int(m.n_iter_)}
    elif family == "lgbm":
        import lightgbm as lgb
        p = dict(learning_rate=E1_HGB["learning_rate"], n_estimators=E1_HGB["max_iter"],
                 num_leaves=E1_HGB["max_leaf_nodes"], min_child_samples=20)
        p.update(params or {})
        es_rounds = p.pop("es_rounds", ES_ROUNDS)
        if objective.startswith("tweedie"):
            p["tweedie_variance_power"] = float(objective.split(":")[1])
            obj = "tweedie"
        else:
            obj = objective
        if p.get("subsample", 1.0) < 1.0 and "subsample_freq" not in p:
            p["subsample_freq"] = 1
        m = lgb.LGBMRegressor(objective=obj, random_state=seed, n_jobs=N_THREADS, verbose=-1, **p)
        m.fit(X[tr], y[tr], eval_set=[(X[es], y[es])],
              callbacks=[lgb.early_stopping(es_rounds, verbose=False)])
        info = {"n_iter": int(m.best_iteration_ or p["n_estimators"])}
    else:
        raise ValueError(family)
    return m, info, es


@dataclass
class ModelSpec:
    """A complete model recipe: family x target treatment x hyper-parameters.

    family: 'hgb' | 'lgbm' | 'glm' | 'et' | 'dummy' | 'hgb_template'
    treatment: 'raw' | 'log1p' | 'log1p_smear' | 'log1p_smear_es' | 'poisson' |
               'tweedie:<p>' | 'hurdle'
    For 'hurdle', ``pos_spec`` is the ModelSpec of the positive-part regressor.
    """
    family: str = "hgb"
    treatment: str = "raw"
    params: dict = field(default_factory=dict)
    pos_spec: "ModelSpec | None" = None
    clf_params: dict = field(default_factory=dict)

    def to_dict(self):
        d = {"family": self.family, "treatment": self.treatment, "params": self.params}
        if self.pos_spec is not None:
            d["pos_spec"] = self.pos_spec.to_dict()
            d["clf_params"] = self.clf_params
        return d

    @staticmethod
    def from_dict(d):
        ps = ModelSpec.from_dict(d["pos_spec"]) if d.get("pos_spec") else None
        return ModelSpec(d["family"], d["treatment"], d.get("params", {}) or {}, ps, d.get("clf_params", {}) or {})


class SFRModel:
    """fit(X, y) / predict(X) wrapper implementing all target treatments."""

    def __init__(self, spec: ModelSpec, seed: int = RANDOM_SEED):
        self.spec, self.seed = spec, seed
        self.info = {}

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        s = self.spec
        t0 = time.time()
        if s.family == "dummy":
            self.m_ = DummyRegressor(strategy="median").fit(X, y)
        elif s.family == "hgb_template":
            self.m_ = HistGradientBoostingRegressor(learning_rate=0.08, max_iter=150, max_leaf_nodes=31,
                                                    random_state=RANDOM_SEED).fit(X, y)
            self.info["n_iter"] = int(self.m_.n_iter_)
        elif s.treatment == "hurdle":
            self._fit_hurdle(X, y)
        elif s.family == "glm":
            assert s.treatment == "poisson"
            self.m_ = _SplineGLM(**(s.params or {})).fit(X, y)
        elif s.family == "et":
            assert s.treatment == "raw"
            p = dict(n_estimators=300, min_samples_leaf=3, max_features=0.5)
            p.update(s.params or {})
            self.m_ = ExtraTreesRegressor(n_jobs=N_THREADS, random_state=self.seed, **p).fit(X, y)
        elif s.treatment in ("raw", "poisson") or s.treatment.startswith("tweedie"):
            obj = "regression" if s.treatment == "raw" else s.treatment
            self.m_, inf, _ = _fit_gbm(s.family, obj, s.params, X, y, self.seed)
            self.info.update(inf)
        elif s.treatment.startswith("log1p"):
            z = np.log1p(y)
            self.m_, inf, es = _fit_gbm(s.family, "regression", s.params, X, z, self.seed)
            self.info.update(inf)
            f = self.m_.predict(X)
            res = z - f
            self.smear_train_ = float(np.mean(np.exp(res)))
            self.smear_es_ = float(np.mean(np.exp(res[es])))
            self.info.update(smear_train=self.smear_train_, smear_es=self.smear_es_)
        else:
            raise ValueError(s.treatment)
        self.info["fit_seconds"] = time.time() - t0
        return self

    def _fit_hurdle(self, X, y):
        s = self.spec
        pos = y > 0
        cp = dict(learning_rate=0.05, max_iter=1000, max_leaf_nodes=31)
        cp.update(s.clf_params or {})
        tr, es = _es_split(pos.astype(float), self.seed)
        self.clf_ = HistGradientBoostingClassifier(early_stopping=True, n_iter_no_change=ES_ROUNDS,
                                                   random_state=self.seed, **cp)
        self.clf_.fit(X[tr], pos[tr].astype(int), X_val=X[es], y_val=pos[es].astype(int))
        self.reg_ = SFRModel(s.pos_spec, seed=self.seed).fit(X[pos], y[pos])
        self.info.update(n_iter_clf=int(self.clf_.n_iter_),
                         **{f"pos_{k}": v for k, v in self.reg_.info.items()})

    def predict_parts(self, X):
        X = np.asarray(X, dtype=float)
        return self.clf_.predict_proba(X)[:, 1], self.reg_.predict(X)

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        s = self.spec
        if s.treatment == "hurdle":
            p, mu = self.predict_parts(X)
            return p * mu
        f = self.m_.predict(X)
        if s.treatment == "log1p":
            return np.expm1(f)
        if s.treatment == "log1p_smear":
            return self.smear_train_ * np.exp(f) - 1.0
        if s.treatment == "log1p_smear_es":
            return self.smear_es_ * np.exp(f) - 1.0
        return f


# --------------------------------------------------------------------------
# Experiment bookkeeping
# --------------------------------------------------------------------------
def env_info() -> dict:
    import lightgbm, scipy, sklearn
    load = os.getloadavg()
    return {
        "python": platform.python_version(), "platform": platform.platform(),
        "machine": platform.machine(), "host": socket.gethostname(),
        "cpu_count": os.cpu_count(), "n_threads": N_THREADS,
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
        "OMP_WAIT_POLICY": os.environ.get("OMP_WAIT_POLICY"),
        "loadavg_1_5_15": [round(x, 2) for x in load],
        "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__,
        "sklearn": sklearn.__version__, "lightgbm": lightgbm.__version__,
    }


def set_status(msg: str):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {msg}\n"
    with open(RESULTS_DIR / "STATUS", "a") as fh:
        fh.write(line)
    print(line, end="", flush=True)


def exp_done(exp_id: str) -> bool:
    return (META_DIR / f"{exp_id}.json").exists() and (OOF_DIR / f"{exp_id}.parquet").exists()


def load_meta(exp_id: str) -> dict:
    return json.loads((META_DIR / f"{exp_id}.json").read_text())


def load_oof(exp_id: str) -> pd.DataFrame:
    return pd.read_parquet(OOF_DIR / f"{exp_id}.parquet")


def run_cv(exp_id: str, question: str, spec: ModelSpec, feature_set: str, Xall: pd.DataFrame,
           y_train: pd.Series, folds: np.ndarray, cv_scheme: str = "random", stage: str = "",
           features: list | None = None, seed: int = RANDOM_SEED, force: bool = False,
           keep_models: bool = False, note: str = ""):
    """Run CV for one configuration; write OOF parquet + meta json. Skips if done.

    Xall: feature frame indexed by original row id; y_train: target indexed by
    training row ids (the template's X_train index order); folds aligned with y_train.
    """
    if exp_done(exp_id) and not force and not keep_models:
        return load_meta(exp_id)
    OOF_DIR.mkdir(parents=True, exist_ok=True)
    META_DIR.mkdir(parents=True, exist_ok=True)
    feats = features if features is not None else FEATURE_SETS[feature_set]
    rows = y_train.index.to_numpy()
    X = Xall.loc[rows, feats].to_numpy(dtype=float)
    y = y_train.to_numpy(dtype=float)
    oof = np.full(len(y), np.nan)
    fold_rows, models = [], []
    t_wall = time.time()
    for k in range(N_FOLDS):
        te = folds == k
        tr = ~te
        m = SFRModel(spec, seed=seed).fit(X[tr], y[tr])
        t1 = time.time()
        oof[te] = m.predict(X[te])
        pred_s = time.time() - t1
        fm = flat_metrics(y[te], oof[te])
        fold_rows.append({"fold": k, "n_test": int(te.sum()), **fm,
                          "pred_seconds": pred_s, **{f"info_{a}": b for a, b in m.info.items()}})
        if keep_models:
            models.append(m)
    wall = time.time() - t_wall
    assert np.isfinite(oof).all()
    fdf = pd.DataFrame(fold_rows)
    pd.DataFrame({"row": rows, "fold": folds, "y": y, "pred": oof}).to_parquet(
        OOF_DIR / f"{exp_id}.parquet", index=False)
    meta = {
        "exp_id": exp_id, "stage": stage, "question": question, "note": note,
        "cv_scheme": cv_scheme, "feature_set": feature_set, "features": feats,
        "n_features": len(feats), "spec": spec.to_dict(), "seed": seed,
        "folds": fdf.to_dict(orient="list"), "wall_seconds": wall,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "env": env_info(),
        "oof_metrics": flat_metrics(y, oof),
    }
    (META_DIR / f"{exp_id}.json").write_text(json.dumps(meta, indent=1, default=float))
    if keep_models:
        return meta, models, X, y
    return meta


SUMMARY_METRICS = ["rmse_all", "mae_all", "r2_all", "rmse_zero", "mae_zero", "rmse_pos", "mae_pos",
                   "r2_pos", "rmse_top10", "mae_top10", "r2_top10", "log10rmse_pos"]


def rebuild_summary(path: Path = RESULTS_DIR / "cv_experiments.csv") -> pd.DataFrame:
    """Assemble one row per experiment from exp_meta/*.json (+ long fold file)."""
    rows, long = [], []
    for f in sorted(META_DIR.glob("*.json")):
        m = json.loads(f.read_text())
        fd = pd.DataFrame(m["folds"])
        r = {"exp_id": m["exp_id"], "stage": m["stage"], "question": m["question"],
             "cv_scheme": m["cv_scheme"], "feature_set": m["feature_set"], "n_features": m["n_features"],
             "family": m["spec"].get("family", m["spec"].get("ensemble", "")), "treatment": m["spec"].get("treatment", ""),
             "config": json.dumps(m["spec"], sort_keys=True), "note": m.get("note", "")}
        for k in range(len(fd)):
            for met in ("rmse_all", "mae_all", "r2_all"):
                r[f"{met}_f{k}"] = fd[met].iloc[k]
        for met in SUMMARY_METRICS:
            r[f"{met}_mean"] = fd[met].mean()
            r[f"{met}_sd"] = fd[met].std(ddof=1)
        r["oof_rmse_all_pooled"] = m["oof_metrics"]["rmse_all"]
        if "info_n_iter" in fd:
            r["n_iter_mean"] = fd["info_n_iter"].mean()
        if "info_fit_seconds" in fd:
            r["fit_seconds_total"] = fd["info_fit_seconds"].sum()
        r["wall_seconds"] = m["wall_seconds"]
        r["timestamp"] = m["timestamp"]
        r["n_threads"] = m["env"]["n_threads"]
        r["loadavg_1m"] = m["env"]["loadavg_1_5_15"][0]
        r["sklearn"], r["lightgbm"] = m["env"]["sklearn"], m["env"]["lightgbm"]
        rows.append(r)
        fd.insert(0, "exp_id", m["exp_id"])
        long.append(fd)
    out = pd.DataFrame(rows)
    out.to_csv(path, index=False)
    if long:
        pd.concat(long, ignore_index=True).to_csv(RESULTS_DIR / "cv_fold_metrics.csv", index=False)
    return out


def paired(exp_a: str, exp_b: str, metric: str = "rmse_all") -> dict:
    """Paired per-fold difference a - b: mean, SE = sd/sqrt(K)."""
    a = np.asarray(load_meta(exp_a)["folds"][metric], dtype=float)
    b = np.asarray(load_meta(exp_b)["folds"][metric], dtype=float)
    d = a - b
    se = d.std(ddof=1) / np.sqrt(len(d))
    return {"a": exp_a, "b": exp_b, "metric": metric, "mean_diff": float(d.mean()), "se": float(se),
            "t": float(d.mean() / se) if se > 0 else np.nan, "n_a_better": int((d < 0).sum()),
            "diffs": [round(float(x), 5) for x in d]}


def paired_table(pairs, metric="rmse_all") -> pd.DataFrame:
    return pd.DataFrame([paired(a, b, metric) for a, b in pairs])


def select_one_se(candidates: list[str], complexity: dict, metric: str = "rmse_all") -> dict:
    """Pre-registered rule: lowest CV mean; then the simplest candidate whose paired
    difference to the best is within 1 SE (mean_diff <= SE). Ties in complexity -> lower mean."""
    means = {c: float(np.mean(load_meta(c)["folds"][metric])) for c in candidates}
    best = min(candidates, key=lambda c: means[c])
    eligible = [best]
    details = []
    for c in candidates:
        if c == best:
            continue
        p = paired(c, best, metric)
        ok = complexity[c] < complexity[best] and p["mean_diff"] <= p["se"]
        details.append({**p, "complexity": complexity[c], "eligible_simpler": ok})
        if ok:
            eligible.append(c)
    chosen = min(eligible, key=lambda c: (complexity[c], means[c]))
    return {"best_by_mean": best, "chosen": chosen, "means": means, "comparisons": details}


def bootstrap_paired_rmse(y, pred_a, pred_b, n_boot: int = 2000, seed: int = RANDOM_SEED) -> dict:
    """Paired bootstrap of RMSE(a) - RMSE(b) on the same resampled rows."""
    y, pa, pb = (np.asarray(v, dtype=float) for v in (y, pred_a, pred_b))
    rng = np.random.default_rng(seed)
    n = len(y)
    ea, eb = (pa - y) ** 2, (pb - y) ** 2
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        diffs[i] = np.sqrt(ea[idx].mean()) - np.sqrt(eb[idx].mean())
    return {"point": float(np.sqrt(ea.mean()) - np.sqrt(eb.mean())),
            "ci_low": float(np.quantile(diffs, 0.025)), "ci_high": float(np.quantile(diffs, 0.975)),
            "boot_sd": float(diffs.std(ddof=1)), "p_a_not_better": float((diffs >= 0).mean()),
            "n_boot": n_boot, "seed": seed}


def register_oof_experiment(exp_id: str, question: str, oof: np.ndarray, y_train: pd.Series,
                            folds: np.ndarray, stage: str, feature_set: str, spec_dict: dict,
                            note: str = "", cv_scheme: str = "random", extra_fold_info=None):
    """Store an experiment whose OOF predictions were assembled elsewhere (e.g. ensembles)."""
    OOF_DIR.mkdir(parents=True, exist_ok=True)
    META_DIR.mkdir(parents=True, exist_ok=True)
    y = y_train.to_numpy(dtype=float)
    rows = []
    for k in range(N_FOLDS):
        te = folds == k
        r = {"fold": k, "n_test": int(te.sum()), **flat_metrics(y[te], oof[te])}
        if extra_fold_info is not None:
            r.update(extra_fold_info[k])
        rows.append(r)
    fdf = pd.DataFrame(rows)
    pd.DataFrame({"row": y_train.index.to_numpy(), "fold": folds, "y": y, "pred": oof}).to_parquet(
        OOF_DIR / f"{exp_id}.parquet", index=False)
    feats = FEATURE_SETS.get(feature_set, [])
    meta = {"exp_id": exp_id, "stage": stage, "question": question, "note": note,
            "cv_scheme": cv_scheme, "feature_set": feature_set, "features": feats,
            "n_features": len(feats), "spec": spec_dict, "seed": None,
            "folds": fdf.to_dict(orient="list"), "wall_seconds": 0.0,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "env": env_info(),
            "oof_metrics": flat_metrics(y, oof)}
    (META_DIR / f"{exp_id}.json").write_text(json.dumps(meta, indent=1, default=float))
    return meta


# --------------------------------------------------------------------------
# Recipes: a final model = weighted list of (spec, seed) on one feature set
# --------------------------------------------------------------------------
def recipe_fit(recipe: dict, X, y):
    """recipe = {"feature_set": str, "components": [{"spec": dict, "seed": int, "weight": float}]}"""
    models = []
    for c in recipe["components"]:
        models.append((SFRModel(ModelSpec.from_dict(c["spec"]), seed=c["seed"]).fit(X, y), c["weight"]))
    return models


def recipe_predict(models, X):
    return sum(w * m.predict(X) for m, w in models)
