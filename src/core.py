"""Shared code for the AutoML sweep (galaxy SFR, DataHub).

Reuses the original project's library (src/sfr_lib.py) for the fixed template split, the 11-strata 5-fold CV,
metrics and the physics / neighbour features. Every experiment writes
  results/oof/<name>.parquet   out-of-fold predictions on the training split (selection uses only these)
  results/val/<name>.npy       predictions for the 41,688 validation rows (read only in the final step)
  results/meta/<name>.json     per-fold metrics, settings, timing
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import sfr_lib as L  # noqa: E402

RES = ROOT / "results"
OOF_DIR, VAL_DIR, META_DIR = RES / "oof", RES / "val", RES / "meta"
for _d in (OOF_DIR, VAL_DIR, META_DIR):
    _d.mkdir(parents=True, exist_ok=True)
FEAT_PATH = ROOT / "data" / "features_all.parquet"
STATUS = RES / "status.json"
N_THREADS = int(os.environ.get("SFR_N_THREADS", "14"))
L.N_THREADS = N_THREADS
SEED = 42

# ------------------------------------------------------------------ extra environment features (E8 "A")
RADII_A1 = (0.5, 1.0, 2.0, 5.0)


def _rn(r):
    return str(r).replace(".", "p")


A1_COLS = [f"a1_{s}_r{_rn(r)}" for r in RADII_A1 for s in
           ("n", "logMs_sum", "logMh_sum", "rank", "vdisp", "hn_dlogMs", "hn_logMBH")]
A2_COLS = ["a2_has_host", "a2_r_rvir", "a2_dv_sigloc", "a2_dv_vrel", "a2_ps", "a2_dlogMh_host"]


def compute_A(df):
    """Multi-scale environment (A1) and host phase-space (A2) features; inputs only, periodic box 200.
    Same definitions as the original project's E8 (Amendment 4)."""
    pos = np.mod(df[L.POS_COLS].to_numpy(float), L.BOX_SIZE)
    vel = df[L.VEL_COLS].to_numpy(float)
    lms = df.log10_stellar_mass.to_numpy(float)
    lmh = df.log10_halo_mass.to_numpy(float)
    lbh = df.log10_BH_mass.to_numpy(float)
    grow = lms - df.log10_main_prog_stellar_mass.to_numpy(float)
    n = len(df)
    tree = cKDTree(pos, boxsize=L.BOX_SIZE)
    pr = tree.query_pairs(max(RADII_A1), output_type="ndarray")
    dl = np.abs(pos[pr[:, 0]] - pos[pr[:, 1]]); dl = np.minimum(dl, L.BOX_SIZE - dl)
    dd = np.sqrt((dl ** 2).sum(1))
    I = np.concatenate([pr[:, 0], pr[:, 1]]); J = np.concatenate([pr[:, 1], pr[:, 0]]); D = np.concatenate([dd, dd])
    del pr, dl, dd
    out = pd.DataFrame(index=df.index)

    def argmax_by(Iw, Jw, val):
        o = np.lexsort((val[Jw], Iw))
        Is, Js = Iw[o], Jw[o]
        last = np.r_[Is[1:] != Is[:-1], True]
        best = np.full(n, -1, dtype=np.int64)
        best[Is[last]] = Js[last]
        return best

    sig_r1 = cnt_r1 = None
    for r in RADII_A1:
        w = D <= r
        Iw, Jw = I[w], J[w]
        rn = _rn(r)
        cnt = np.bincount(Iw, minlength=n).astype(float)
        out[f"a1_n_r{rn}"] = cnt
        out[f"a1_logMs_sum_r{rn}"] = np.log10(1.0 + np.bincount(Iw, weights=10.0 ** lms[Jw], minlength=n))
        out[f"a1_logMh_sum_r{rn}"] = np.log10(1.0 + np.bincount(Iw, weights=10.0 ** lmh[Jw], minlength=n))
        hb = argmax_by(Iw, Jw, lms)
        has = hb >= 0
        mx = np.where(has, np.maximum(lms, lms[np.where(has, hb, 0)]), lms)
        out[f"a1_rank_r{rn}"] = lms - mx
        S1 = np.stack([np.bincount(Iw, weights=vel[Jw, a], minlength=n) + vel[:, a] for a in range(3)], 1)
        S2 = np.stack([np.bincount(Iw, weights=vel[Jw, a] ** 2, minlength=n) + vel[:, a] ** 2 for a in range(3)], 1)
        nn = (cnt + 1.0)[:, None]
        var = np.clip(S2 / nn - (S1 / nn) ** 2, 0, None)
        sig = np.where(cnt > 0, np.sqrt(var.sum(1) / 3.0), 0.0)
        out[f"a1_vdisp_r{rn}"] = sig
        out[f"a1_hn_dlogMs_r{rn}"] = np.where(has, grow[np.where(has, hb, 0)], np.nan)
        out[f"a1_hn_logMBH_r{rn}"] = np.where(has, lbh[np.where(has, hb, 0)], np.nan)
        if r == 1.0:
            sig_r1, cnt_r1 = sig, cnt
    w = D <= 2.0
    Iw, Jw, Dw = I[w], J[w], D[w]
    o = np.lexsort((lmh[Jw], Iw))
    Is, Js, Ds = Iw[o], Jw[o], Dw[o]
    last = np.r_[Is[1:] != Is[:-1], True]
    host = np.full(n, -1, dtype=np.int64); hd = np.full(n, np.nan)
    host[Is[last]] = Js[last]; hd[Is[last]] = Ds[last]
    hs = host >= 0
    hs[hs] = lmh[host[hs]] > lmh[hs]
    h = np.where(hs, host, 0)
    R_rel = 10.0 ** ((lmh[h] - 12.0) / 3.0)
    V_rel = R_rel.copy()
    dv = np.linalg.norm(vel - vel[h], axis=1)
    ok_sig = cnt_r1[h] + 1 >= 3
    ratio_med = float(np.median((sig_r1[h] / V_rel)[hs & ok_sig]))
    sig_loc = np.where(ok_sig, sig_r1[h], ratio_med * V_rel)
    BIG = 99.0
    out["a2_has_host"] = hs.astype(float)
    out["a2_r_rvir"] = np.where(hs, hd / R_rel, BIG)
    out["a2_dv_sigloc"] = np.where(hs, dv / np.maximum(sig_loc, 1e-6), BIG)
    out["a2_dv_vrel"] = np.where(hs, dv / V_rel, BIG)
    out["a2_ps"] = np.where(hs, (hd / R_rel) * (dv / V_rel), BIG)
    out["a2_dlogMh_host"] = np.where(hs, lmh[h] - lmh, 0.0)
    return out


# ------------------------------------------------------------------ feature sets
FS = dict(L.FEATURE_SETS)                     # F0, F1, F2, F3, F4, F5, F6 from the original project
FS["F5A"] = FS["F5"] + A1_COLS + A2_COLS      # F5 + multi-scale environment + phase space (59)
FS["ALL"] = FS["F4"] + L.MSSM_FEATURES + A1_COLS + A2_COLS   # everything incl. raw coordinates (68)


def build_features(df):
    if FEAT_PATH.exists():
        F = pd.read_parquet(FEAT_PATH)
        if len(F) == len(df):
            return F
    L.ENV_CACHE = ROOT / "data" / "env_features.parquet"
    L.MSSM_CACHE = ROOT / "data" / "env_mssm.parquet"
    base = pd.concat([df[L.BASE_FEATURES], L.add_physics_features(df),
                      L.compute_env_features(df, cache=L.ENV_CACHE),
                      L.compute_mssm_counts(df, cache=L.MSSM_CACHE)], axis=1)
    F = pd.concat([base, compute_A(df)], axis=1)
    F.to_parquet(FEAT_PATH)
    return F


_G: dict = {}


def setup():
    """df, F (all features), y_train, y_val (template split order), folds aligned with y_train."""
    if not _G:
        df = L.load_data(ROOT / "data" / "Galaxy Features with Position and Velocity for Star Formation Rate Prediction.csv")
        F = build_features(df)
        _, _, y_train, y_val = L.template_split(df)
        folds = L.make_random_folds(y_train)
        _G.update(df=df, F=F, y_train=y_train, y_val=y_val, folds=folds)
    return _G


# ------------------------------------------------------------------ bookkeeping
def fold_metrics(y, oof, folds):
    return pd.DataFrame([{"fold": k, **L.flat_metrics(y[folds == k], oof[folds == k])} for k in range(5)])


def save_exp(name, oof, val_pred, info):
    G = setup()
    y = G["y_train"].to_numpy(float)
    assert np.isfinite(oof).all() and np.isfinite(val_pred).all()
    fm = fold_metrics(y, oof, G["folds"])
    pd.DataFrame({"row": G["y_train"].index, "fold": G["folds"], "y": y, "oof": oof}).to_parquet(OOF_DIR / f"{name}.parquet")
    np.save(VAL_DIR / f"{name}.npy", np.asarray(val_pred, float))
    meta = {"name": name, **info, "cv_rmse_mean": float(fm.rmse_all.mean()), "cv_rmse_sd": float(fm.rmse_all.std(ddof=1)),
            "folds": fm.to_dict(orient="list"), "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}
    (META_DIR / f"{name}.json").write_text(json.dumps(meta, indent=1, default=float))
    return meta


def done(name):
    return (META_DIR / f"{name}.json").exists()


def load_oof(name):
    return pd.read_parquet(OOF_DIR / f"{name}.parquet")["oof"].to_numpy()


def fold_rmse(name):
    return np.asarray(json.loads((META_DIR / f"{name}.json").read_text())["folds"]["rmse_all"], float)


def paired(a, b):
    d = fold_rmse(a) - fold_rmse(b)
    return {"diff": float(d.mean()), "se": float(d.std(ddof=1) / np.sqrt(5)), "n_better": int((d < 0).sum())}


def summary_table(ref="M01_lgbm_tw13_F5"):
    rows = []
    for f in sorted(META_DIR.glob("*.json")):
        m = json.loads(f.read_text())
        fd = pd.DataFrame(m["folds"])
        r = {"name": m["name"], "stage": m.get("stage", ""), "features": m.get("features", ""), "desc": m.get("desc", ""),
             "cv_rmse": fd.rmse_all.mean(), "cv_sd": fd.rmse_all.std(ddof=1), "rmse_zero": fd.rmse_zero.mean(),
             "rmse_top10": fd.rmse_top10.mean(), "log10rmse": fd.log10rmse_pos.mean(), "minutes": m.get("seconds", 0) / 60}
        if done(ref) and m["name"] != ref:
            p = paired(m["name"], ref)
            r.update(diff_vs_R0=p["diff"], se=p["se"], folds_better=p["n_better"])
        rows.append(r)
    return pd.DataFrame(rows).sort_values("cv_rmse")


def set_status(stage, msg, **kw):
    s = json.loads(STATUS.read_text()) if STATUS.exists() else {"log": []}
    s.update(stage=stage, message=msg, updated=time.strftime("%Y-%m-%d %H:%M:%S"), **kw)
    s["log"].append({"t": s["updated"], "stage": stage, "msg": msg})
    STATUS.write_text(json.dumps(s, indent=1, ensure_ascii=False))
