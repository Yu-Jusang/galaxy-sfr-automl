"""Predict SFR for NEW galaxies with the submitted final model (ENS_top3), without retraining.

The feature code is an exact copy of notebook cells 4.0 (row-wise physics features) and 9.2a (neighbour-environment
and literature features). The three AutoGluon members are loaded from ../ag_runs.

Main entry points
    model = load_final_model()
    pred  = predict_sfr(new_df, reference_df=train_df, model=model)   # same box: neighbours = reference + new
    pred  = predict_sfr(new_df, reference_df=None, model=model)       # other box given in full: neighbours = new only
    same_box_check(new_df, reference_df)                              # do the new galaxies sit in the reference box?
    fb = train_fallback(train_df); predict_fallback(fb, new_df)        # no positions available
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

N_THREADS = 12
HERE = Path(__file__).resolve().parent
AG_DIR = HERE.parent / "ag_runs"

TARGET, ID_COLUMN = "sfr_linear", "subhalo_id"
INPUT_COLUMNS = ["log10_halo_mass", "log10_stellar_mass", "log10_BH_mass", "log10_main_prog_mass",
                 "log10_main_prog_stellar_mass", "log10_mass_ratio", "log10_overdensity_plus_1",
                 "pos_x", "pos_y", "pos_z", "vel_x", "vel_y", "vel_z"]
POS, VEL = ["pos_x", "pos_y", "pos_z"], ["vel_x", "vel_y", "vel_z"]
NO_POSITION_INPUTS = [c for c in INPUT_COLUMNS if c not in POS + VEL]

# ---------------------------------------------------------------- notebook cell 4.0 (row-wise physics features)
PHYS_FEATURES = ["d_logMs_prog", "dMs_lin_1e9", "d_logMh_prog", "logMs_minus_logMh", "logMBH_minus_logMs",
                 "is_merger", "is_BH_floor", "is_prog_floor", "v_mag"]


def add_physics_features(d):
    out = pd.DataFrame(index=d.index)
    lms_, lmsp_ = d["log10_stellar_mass"], d["log10_main_prog_stellar_mass"]
    lmh_, lmhp_ = d["log10_halo_mass"], d["log10_main_prog_mass"]
    out["d_logMs_prog"] = lms_ - lmsp_
    out["dMs_lin_1e9"] = (10.0 ** lms_ - 10.0 ** lmsp_) / 1e9
    out["d_logMh_prog"] = lmh_ - lmhp_
    out["logMs_minus_logMh"] = lms_ - lmh_
    out["logMBH_minus_logMs"] = d["log10_BH_mass"] - lms_
    out["is_merger"] = (d["log10_mass_ratio"] > -6.0).astype(float)
    out["is_BH_floor"] = (d["log10_BH_mass"] < 5.05).astype(float)
    out["is_prog_floor"] = ((lmsp_ <= 6.0) | (lmhp_ <= 6.0)).astype(float)
    out["v_mag"] = np.sqrt((d[VEL] ** 2).sum(axis=1)) if all(c in d for c in VEL) else np.nan
    return out


# ---------------------------------------------------------------- notebook cell 9.2a (copied unchanged)
BOX = 200.0
RHO_C = 2.775e11
G_GRAV = 4.30091e-9
NFW_C = 6.0
M_SEED = 1e5
R_PAIR = 8.0
ENV_FEATURES = ["env_d1", "env_d5", "env_d10", "env_n_r1", "env_n_r2", "env_d_heavier", "env_logMh_heavier",
                "env_is_localmax_r1", "env_dv_heavier"]


def compute_env_features(d):
    pos = np.mod(d[POS].to_numpy(float), BOX); vel = d[VEL].to_numpy(float)
    mh = d["log10_halo_mass"].to_numpy(float); n = len(d)
    tree = cKDTree(pos, boxsize=BOX)
    dd, _ = tree.query(pos, k=11, workers=N_THREADS)
    out = pd.DataFrame(index=d.index)
    out["env_d1"], out["env_d5"], out["env_d10"] = dd[:, 1], dd[:, 5], dd[:, 10]
    for r in (1.0, 2.0):
        out[f"env_n_r{int(r)}"] = tree.query_ball_point(pos, r, return_length=True, workers=N_THREADS) - 1
    hv_idx = np.full(n, -1, dtype=np.int64); hv_d = np.full(n, np.nan); todo = np.arange(n)
    for k in (64, 1024):
        if len(todo) == 0:
            break
        dk, ik = tree.query(pos[todo], k=min(k, n), workers=N_THREADS)
        heavier = mh[ik] > mh[todo][:, None]; has = heavier.any(axis=1); first = heavier.argmax(axis=1)
        sel = todo[has]; hv_idx[sel] = ik[has, first[has]]; hv_d[sel] = dk[has, first[has]]; todo = todo[~has]
    for i in todo:
        cand = np.where(mh > mh[i])[0]
        if len(cand) == 0:
            continue
        delta = np.abs(pos[cand] - pos[i]); delta = np.minimum(delta, BOX - delta)
        dist = np.sqrt((delta ** 2).sum(axis=1)); j = dist.argmin(); hv_idx[i], hv_d[i] = cand[j], dist[j]
    none = hv_idx < 0
    hv_d[none] = np.sqrt(3) * BOX / 2
    safe = np.where(none, np.arange(n), hv_idx)
    out["env_d_heavier"] = hv_d
    out["env_logMh_heavier"] = np.where(none, mh, mh[safe])
    out["env_is_localmax_r1"] = (hv_d > 1.0).astype(float)
    out["env_dv_heavier"] = np.where(none, 0.0, np.linalg.norm(vel - vel[safe], axis=1))
    return out[ENV_FEATURES]


LIT_GROUPS = {
    "L1": ["l1_logMpk", "l1_logMs_minus_logMpk", "l1_strip", "l1_acc_pct_Mpk", "l1_acc_pct_Mprog"],
    "L3": ["l3_log_BH_acc", "l3_dBH_at_Ms", "l3_dBH_at_Mpk", "l3_bind"],
    "L4": ["l4_s", "l4_ur", "l4_ut", "l4_ur_H", "l4_E", "l4_bound", "l4_vesc_ratio", "l4_backsplash", "l4_infall", "l4_j"],
    "L7": ["l7_log_Mbound", "l7_N_bound", "l7_bound_to_heavier"],
}
LIT_FEATURES = [c for g in LIT_GROUPS.values() for c in g]


def _pct_within(x, key, width):
    return pd.Series(x).groupby(np.floor(key / width).astype(int)).rank(pct=True).to_numpy()


def _median_resid(x, key, width):
    b = np.floor(key / width).astype(int)
    return x - pd.Series(x).groupby(b).transform("median").to_numpy()


def compute_lit_features(d):
    pos = np.mod(d[POS].to_numpy(float), BOX); vel = d[VEL].to_numpy(float)
    s = d.log10_stellar_mass.to_numpy(float); m = d.log10_halo_mass.to_numpy(float)
    mp = d.log10_main_prog_mass.to_numpy(float); b = d.log10_BH_mass.to_numpy(float)
    n = len(d); Mh = 10.0 ** m
    R200 = (3.0 * Mh / (4.0 * np.pi * 200.0 * RHO_C)) ** (1.0 / 3.0)
    V200 = np.sqrt(G_GRAV * Mh / R200)
    mc = np.log(1 + NFW_C) - NFW_C / (1 + NFW_C)
    vesc = lambda V, x: V * np.sqrt(2.0 * np.log1p(NFW_C * x) / (x * mc))
    out = pd.DataFrame(index=d.index)
    mpk = np.maximum(m, mp); dm = m - mp
    out["l1_logMpk"], out["l1_logMs_minus_logMpk"], out["l1_strip"] = mpk, s - mpk, m - mpk
    out["l1_acc_pct_Mpk"] = _pct_within(dm, mpk, 0.1)
    out["l1_acc_pct_Mprog"] = _pct_within(dm, mp, 0.1)
    acc = np.log10(np.maximum(10.0 ** b - M_SEED, 1.0))
    out["l3_log_BH_acc"], out["l3_bind"] = acc, acc - (5.0 / 3.0) * mpk
    out["l3_dBH_at_Ms"] = _median_resid(b, s, 0.1)
    out["l3_dBH_at_Mpk"] = _median_resid(b, mpk, 0.1)
    tree = cKDTree(pos, boxsize=BOX)
    pr = tree.query_pairs(R_PAIR, output_type="ndarray")
    I = np.concatenate([pr[:, 0], pr[:, 1]]); J = np.concatenate([pr[:, 1], pr[:, 0]]); del pr
    rv = pos[J] - pos[I]; rv -= BOX * np.round(rv / BOX); D = np.sqrt((rv ** 2).sum(1))
    heavier = m[J] > m[I]
    cI, cJ, cR = [I[heavier]], [J[heavier]], [D[heavier] / R200[J[heavier]]]
    idx = np.where(m >= 12.5)[0]
    dd, ii = cKDTree(pos[idx], boxsize=BOX).query(pos, k=min(16, len(idx)), workers=N_THREADS)
    if ii.ndim == 1:
        dd, ii = dd[:, None], ii[:, None]
    jj = idx[ii]; ratio = np.where(jj == np.arange(n)[:, None], np.inf, dd / R200[jj])
    ok = (m[jj] > m[:, None]) & np.isfinite(ratio); r_, c_ = np.nonzero(ok)
    cI.append(r_); cJ.append(jj[r_, c_]); cR.append(ratio[r_, c_])
    cI, cJ, cR = np.concatenate(cI), np.concatenate(cJ), np.concatenate(cR)
    o = np.lexsort((cR, cI)); first = np.r_[True, cI[o][1:] != cI[o][:-1]]
    host = np.full(n, -1, dtype=np.int64); host[cI[o][first]] = cJ[o][first]
    hs = host >= 0; h = np.where(hs, host, 0)
    r = pos - pos[h]; r -= BOX * np.round(r / BOX); dvel = vel - vel[h]
    rr = np.maximum(np.sqrt((r ** 2).sum(1)), 1e-6); x = rr / R200[h]
    ur = (dvel * r).sum(1) / rr / V200[h]
    u2 = (dvel ** 2).sum(1) / V200[h] ** 2
    E = 0.5 * u2 - np.log1p(NFW_C * x) / (x * mc)
    nan = lambda a: np.where(hs, a, np.nan)
    out["l4_s"], out["l4_ur"] = nan(x), nan(ur)
    out["l4_ut"] = nan(np.sqrt(np.maximum(u2 - ur ** 2, 0.0)))
    out["l4_ur_H"] = nan(((dvel + 100.0 * r) * r).sum(1) / rr / V200[h])
    out["l4_E"] = nan(E)
    out["l4_bound"] = (hs & (x < 3.0) & (E < 0)).astype(float)
    out["l4_vesc_ratio"] = nan(np.sqrt(u2) * V200[h] / vesc(V200[h], x))
    out["l4_backsplash"] = (hs & (x > 1) & (x < 2.5) & (ur > 0) & (E < 0)).astype(float)
    out["l4_infall"] = (hs & (x < 2.5) & (ur < 0)).astype(float)
    out["l4_j"] = nan(np.linalg.norm(np.cross(r, dvel), axis=1) / (R200[h] * V200[h]))
    dvp = np.linalg.norm(vel[J] - vel[I], axis=1); x_i = D / R200[I]
    bnd = (x_i < 3.0) & (dvp < vesc(V200[I], x_i))
    out["l7_log_Mbound"] = np.log10(1.0 + np.bincount(I[bnd], weights=Mh[J[bnd]], minlength=n))
    out["l7_N_bound"] = np.bincount(I[bnd], minlength=n).astype(float)
    x_j = D / R200[J]; to_h = heavier & (x_j < 3.0) & (dvp < vesc(V200[J], x_j))
    flag = np.zeros(n); flag[np.unique(I[to_h])] = 1.0
    out["l7_bound_to_heavier"] = flag
    return out[LIT_FEATURES]


F5 = NO_POSITION_INPUTS + PHYS_FEATURES + ENV_FEATURES
F5C = F5 + LIT_FEATURES
FALLBACK_FEATURES = NO_POSITION_INPUTS + [c for c in PHYS_FEATURES if c != "v_mag"]   # 15 columns, no pos/vel at all
MEMBERS = {"AG26_F5C_wmassive": F5C, "AG3_F5": F5, "AG23_F5C_tweedie": F5C}


# ---------------------------------------------------------------- input checks
def check_inputs(d, need_positions=True):
    """Raise a clear error for missing or non-finite inputs; return the input columns as floats."""
    need = INPUT_COLUMNS if need_positions else NO_POSITION_INPUTS
    miss = [c for c in need if c not in d.columns]
    if miss:
        hint = (" Positions/velocities are missing: use train_fallback() / predict_fallback()."
                if set(miss) <= set(POS + VEL) else "")
        raise ValueError(f"missing input columns: {miss}.{hint}")
    x = d[need].apply(pd.to_numeric, errors="coerce").astype(float)
    bad = ~np.isfinite(x.to_numpy())
    if bad.any():
        rows, cols = np.nonzero(bad)
        raise ValueError(f"{bad.sum()} non-finite input value(s), e.g. row {d.index[rows[0]]!r} column {need[cols[0]]!r}")
    return x


def build_features(catalogue):
    """All F5C features for every row of `catalogue` (neighbours = the catalogue itself)."""
    c = catalogue.reset_index(drop=True)
    return pd.concat([c[NO_POSITION_INPUTS], add_physics_features(c), compute_env_features(c),
                      compute_lit_features(c)], axis=1)


def _drop_overlap(new_x, ref_df):
    """Remove reference rows that are the same galaxy as a new row (same id, or identical 13 inputs)."""
    ref_x = check_inputs(ref_df)
    keep = np.ones(len(ref_x), bool)
    if ID_COLUMN in ref_df.columns and ID_COLUMN in new_x.attrs.get("cols", []):
        keep &= ~ref_df[ID_COLUMN].isin(new_x.attrs["ids"]).to_numpy()
    key = lambda f: pd.util.hash_pandas_object(f[INPUT_COLUMNS].round(9), index=False)
    keep &= ~key(ref_x).isin(set(key(new_x))).to_numpy()
    return ref_x[keep], int((~keep).sum())


# ---------------------------------------------------------------- final model
def load_final_model(ag_dir=AG_DIR):
    from autogluon.tabular import TabularPredictor
    return {m: TabularPredictor.load(str(Path(ag_dir) / m), require_py_version_match=False) for m in MEMBERS}


def predict_from_features(model, feat, members=None):
    preds = {}
    for m in (members or MEMBERS):
        p = model[m].predict(feat[MEMBERS[m]].reset_index(drop=True))
        preds[m] = np.clip(p.to_numpy(float), 0, None)
    return np.mean(list(preds.values()), axis=0), preds


def predict_sfr(new_df, reference_df=None, model=None, return_features=False):
    """Final-model (ENS_top3) predictions for the rows of new_df, in new_df's row order.

    reference_df: other galaxies of the SAME box and coordinate frame (e.g. the competition CSV). Features are then
    computed on reference + new, exactly as the notebook did for the validation split. Rows of reference_df that are
    the same galaxy as a new row (same subhalo_id, or identical inputs) are dropped first, so a galaxy never counts
    as its own neighbour. reference_df=None: the new galaxies are a complete catalogue of their own box.
    `sfr_linear`, `subhalo_id` and any other extra columns of new_df are ignored.
    """
    new_x = check_inputs(new_df)
    new_x.attrs["cols"] = list(new_df.columns)
    new_x.attrs["ids"] = set(new_df[ID_COLUMN]) if ID_COLUMN in new_df.columns else set()
    n_drop = 0
    if reference_df is not None:
        ref_x, n_drop = _drop_overlap(new_x, reference_df)
        cat = pd.concat([ref_x, new_x], ignore_index=True)
    else:
        if len(new_x) < 1000:
            raise ValueError(f"only {len(new_x)} galaxies and no reference_df: too few to measure neighbours")
        cat = new_x.reset_index(drop=True)
    feat = build_features(cat).iloc[len(cat) - len(new_x):].reset_index(drop=True)
    model = model if model is not None else load_final_model()
    pred, members = predict_from_features(model, feat)
    info = {"n_new": len(new_x), "n_reference_used": len(cat) - len(new_x), "n_reference_dropped_as_same_galaxy": n_drop}
    if return_features:
        return pred, info, feat, members
    return pred, info


# ---------------------------------------------------------------- does a new set live in the reference box?
def nearest_distance(points, ref_points, exclude_self=False):
    tree = cKDTree(np.mod(ref_points, BOX), boxsize=BOX)
    d, _ = tree.query(np.mod(points, BOX), k=2 if exclude_self else 1, workers=N_THREADS)
    return d[:, 1] if exclude_self else d


def same_box_check(new_df, reference_df, n_control=20000, seed=0):
    """Distance from each new galaxy to its nearest reference galaxy, against the same statistic for reference
    galaxies (leave-one-out). Galaxies cluster, so new galaxies from the same box and frame sit as close to the
    reference galaxies as the reference galaxies sit to each other; galaxies from another box or frame do not."""
    from scipy.stats import ks_2samp
    new_x = check_inputs(new_df); ref_x = check_inputs(reference_df)
    ref_p = ref_x[POS].to_numpy(float)
    d_new = nearest_distance(new_x[POS].to_numpy(float), ref_p)
    rng = np.random.default_rng(seed)
    ctrl = rng.choice(len(ref_p), size=min(n_control, len(ref_p)), replace=False)
    d_ctl = nearest_distance(ref_p[ctrl], ref_p, exclude_self=True)
    exact = float((d_new < 1e-9).mean())
    d_new_nz = d_new[d_new >= 1e-9] if exact < 1 else d_new
    ratio = float(np.median(d_new_nz) / np.median(d_ctl))
    ks = float(ks_2samp(d_new_nz, d_ctl).statistic)
    verdict = ("same galaxies as the reference" if exact > 0.5 else
               "consistent with the same box and frame" if (ratio < 1.25 and ks < 0.1) else
               "NOT consistent with the reference box/frame")
    return {"median_nearest_new": float(np.median(d_new_nz)), "median_nearest_control": float(np.median(d_ctl)),
            "ratio": ratio, "ks": ks, "share_exact_duplicates": exact, "verdict": verdict}


# ---------------------------------------------------------------- fallback without positions
def train_fallback(train_df, seed=42):
    """LightGBM Tweedie 1.3 on the 15 inputs that need no positions or velocities (F3 without pos/vel and |v|): the R0 recipe
    of the notebook (lr 0.05, <= 1000 trees, early stopping on a stratified 10 % hold-out, 31 leaves)."""
    import lightgbm as lgb
    from sklearn.model_selection import train_test_split
    x = check_inputs(train_df, need_positions=False)
    X = pd.concat([x, add_physics_features(x)], axis=1)[FALLBACK_FEATURES].to_numpy(float)
    y = train_df[TARGET].to_numpy(float)
    lab = np.zeros(len(y), int); p = y > 0
    edges = np.quantile(np.log10(y[p]), np.linspace(0, 1, 11)[1:-1]); lab[p] = 1 + np.searchsorted(edges, np.log10(y[p]), "right")
    tr, es = train_test_split(np.arange(len(y)), test_size=0.10, random_state=seed, stratify=lab)
    m = lgb.LGBMRegressor(objective="tweedie", tweedie_variance_power=1.3, learning_rate=0.05, n_estimators=1000,
                          num_leaves=31, min_child_samples=20, random_state=seed, n_jobs=N_THREADS, verbose=-1)
    m.fit(X[tr], y[tr], eval_set=[(X[es], y[es])], callbacks=[lgb.early_stopping(10, verbose=False)])
    return m


def predict_fallback(model, new_df):
    x = check_inputs(new_df, need_positions=False)
    return model.predict(pd.concat([x, add_physics_features(x)], axis=1)[FALLBACK_FEATURES].to_numpy(float))
