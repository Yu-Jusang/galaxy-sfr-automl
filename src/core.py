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


# ------------------------------------------------------------------ G: environment grid (radius / k / threshold variants)
G_K = (1, 2, 3, 5, 10, 20, 50)
G_R = (0.5, 1.5, 3.0)
G_COLS = ([f"g_d{k}" for k in G_K] + [f"g_n_r{_rn(r)}" for r in G_R]
          + ["g_central_r0p5", "g_central_r2p0", "g_dense_1p5", "g_dense_1p6", "g_dense_1p7"])


def compute_G(df, env):
    """More radii / k for the neighbour features, alternative central-proxy radii and the Part I overdensity step
    (zero-SFR fraction jumps at log10(1+delta) = 1.5-1.7) as explicit flags. Inputs only."""
    pos = np.mod(df[L.POS_COLS].to_numpy(float), L.BOX_SIZE)
    tree = cKDTree(pos, boxsize=L.BOX_SIZE)
    out = pd.DataFrame(index=df.index)
    d, _ = tree.query(pos, k=max(G_K) + 1, workers=N_THREADS)
    for k in G_K:
        out[f"g_d{k}"] = d[:, k]
    for r in G_R:
        out[f"g_n_r{_rn(r)}"] = tree.query_ball_point(pos, r, return_length=True, workers=N_THREADS) - 1
    out["g_central_r0p5"] = (env["env_d_heavier"] > 0.5).astype(float)
    out["g_central_r2p0"] = (env["env_d_heavier"] > 2.0).astype(float)
    od = df["log10_overdensity_plus_1"]
    for t in (1.5, 1.6, 1.7):
        out[f"g_dense_{_rn(t)}"] = (od > t).astype(float)
    return out


# ------------------------------------------------------------------ N: new physics / relative / neighbour-average features
N_COLS = ["n_dMh_lin_1e12", "n_logMBH_minus_logMh", "n_od_pct_in_mass", "n_growth_pct_in_mass", "n_ms_pct_in_halo",
          "n_bh_pct_in_mass", "n_nb_mean_dlogMs_r1", "n_nb_mean_dlogMs_r2", "n_nb_frac_merger_r2",
          "n_nb_frac_BHfloor_r2", "n_central_x_od", "n_is_massive"]


def _pct_within(x, key, width):
    """Percentile rank of x among galaxies in the same bin of `key` (bin width `width`); inputs only."""
    b = np.floor(key / width).astype(int)
    s = pd.Series(x).groupby(b).rank(pct=True)
    return s.to_numpy()


def compute_N(df, env):
    """Part I-motivated additions:
    - relative quantities at fixed mass (the EDA showed the signal appears once mass is fixed):
      overdensity / growth / BH-mass percentile inside a 0.25-dex stellar-mass bin, stellar-mass percentile inside a
      0.25-dex halo-mass bin
    - linear halo growth and BH-to-halo mass ratio
    - neighbour averages of inputs (mean growth, merger and BH-floor fractions of neighbours): galactic-conformity
      proxies that use no labels
    - explicit central x overdensity interaction and a massive-galaxy flag (log M* > 11.5 holds 82 % of the variance)
    """
    lms = df.log10_stellar_mass.to_numpy(float); lmh = df.log10_halo_mass.to_numpy(float)
    lbh = df.log10_BH_mass.to_numpy(float); od = df.log10_overdensity_plus_1.to_numpy(float)
    grow = lms - df.log10_main_prog_stellar_mass.to_numpy(float)
    mer = (df.log10_mass_ratio.to_numpy(float) > -6).astype(float); bhf = (lbh < 5.05).astype(float)
    out = pd.DataFrame(index=df.index)
    out["n_dMh_lin_1e12"] = (10.0 ** lmh - 10.0 ** df.log10_main_prog_mass.to_numpy(float)) / 1e12
    out["n_logMBH_minus_logMh"] = lbh - lmh
    out["n_od_pct_in_mass"] = _pct_within(od, lms, 0.25)
    out["n_growth_pct_in_mass"] = _pct_within(grow, lms, 0.25)
    out["n_ms_pct_in_halo"] = _pct_within(lms, lmh, 0.25)
    out["n_bh_pct_in_mass"] = _pct_within(lbh, lms, 0.25)
    pos = np.mod(df[L.POS_COLS].to_numpy(float), L.BOX_SIZE)
    tree = cKDTree(pos, boxsize=L.BOX_SIZE)
    pr = tree.query_pairs(2.0, output_type="ndarray")
    dl = np.abs(pos[pr[:, 0]] - pos[pr[:, 1]]); dl = np.minimum(dl, L.BOX_SIZE - dl)
    dd = np.sqrt((dl ** 2).sum(1))
    I = np.concatenate([pr[:, 0], pr[:, 1]]); J = np.concatenate([pr[:, 1], pr[:, 0]]); D = np.concatenate([dd, dd])
    n = len(df)
    for r in (1.0, 2.0):
        w = D <= r
        c = np.bincount(I[w], minlength=n).astype(float)
        with np.errstate(invalid="ignore", divide="ignore"):
            out[f"n_nb_mean_dlogMs_r{int(r)}"] = np.bincount(I[w], weights=grow[J[w]], minlength=n) / c
            if r == 2.0:
                out["n_nb_frac_merger_r2"] = np.bincount(I[w], weights=mer[J[w]], minlength=n) / c
                out["n_nb_frac_BHfloor_r2"] = np.bincount(I[w], weights=bhf[J[w]], minlength=n) / c
    out["n_central_x_od"] = env["env_is_localmax_r1"].to_numpy(float) * od
    out["n_is_massive"] = (lms > 11.5).astype(float)
    return out[N_COLS]


# ------------------------------------------------------------------ L: literature-based features (7 groups)
# Units are not documented; we assume z = 0, masses in h^-1 Msun and lengths in h^-1 cMpc (then h cancels in V200).
RHO_C = 2.775e11            # critical density, h^2 Msun Mpc^-3
G_GRAV = 4.30091e-9         # Mpc km^2 s^-2 Msun^-1
F_B = 0.156                 # cosmic baryon fraction (Moster et al. 2018)
NFW_C = 6.0                 # fixed concentration for escape speeds (Rhee et al. 2017)
M_SEED = 1e5                # CROCODILE BH seed mass (Oku & Nagamine 2024)
EMERGE = dict(logM1=11.339, epsN=0.005, beta=3.344, gamma=0.966)   # Moster et al. 2018, z = 0
R_PAIR = 8.0                # neighbour search radius for the pair-based groups
CLUSTER_T = (12.5, 13.0, 13.5, 14.0)

L_GROUPS = {
    "L1": ["l1_logMpk", "l1_logMs_minus_logMpk", "l1_strip", "l1_acc_pct_Mpk", "l1_acc_pct_Mprog"],
    "L2": ["l2_log_sfr_emerge", "l2_no_halo_growth", "l2_slog_eff"],
    "L3": ["l3_log_BH_acc", "l3_dBH_at_Ms", "l3_dBH_at_Mpk", "l3_bind"],
    "L4": ["l4_s", "l4_ur", "l4_ut", "l4_ur_H", "l4_E", "l4_bound", "l4_vesc_ratio", "l4_backsplash", "l4_infall", "l4_j"],
    "L5": ["l5_Dvir", "l5_logMh_host"] + [f"l5_d_{_rn(t)}" for t in CLUSTER_T] + ["l5_preproc"],
    "L6": ["l6_logtau_near", "l6_logtau_far", "l6_aniso_near", "l6_aniso_far", "l6_rHill"],
    "L7": ["l7_log_Mbound", "l7_N_bound", "l7_bound_to_heavier"],
}
L_COLS = [c for g in L_GROUPS.values() for c in g]


def _median_resid(x, key, width):
    """x minus the median of x among galaxies in the same `width` bin of `key`."""
    b = np.floor(key / width).astype(int)
    return x - pd.Series(x).groupby(b).transform("median").to_numpy()


def _slog(x):
    return np.sign(x) * np.log10(1.0 + np.abs(x))


def compute_L(df):
    """Seven literature-motivated feature groups (inputs only, periodic box 200). See report/feature_formulas.pdf."""
    pos = np.mod(df[L.POS_COLS].to_numpy(float), L.BOX_SIZE); vel = df[L.VEL_COLS].to_numpy(float)
    s = df.log10_stellar_mass.to_numpy(float); sp = df.log10_main_prog_stellar_mass.to_numpy(float)
    m = df.log10_halo_mass.to_numpy(float); mp = df.log10_main_prog_mass.to_numpy(float)
    b = df.log10_BH_mass.to_numpy(float)
    n = len(df); Mh = 10.0 ** m
    R200 = (3.0 * Mh / (4.0 * np.pi * 200.0 * RHO_C)) ** (1.0 / 3.0)
    V200 = np.sqrt(G_GRAV * Mh / R200)
    mc = np.log(1 + NFW_C) - NFW_C / (1 + NFW_C)

    def vesc(V, x):  # NFW escape speed at x = r / R200
        return V * np.sqrt(2.0 * np.log1p(NFW_C * x) / (x * mc))

    out = pd.DataFrame(index=df.index)
    # ---- L1 accretion at fixed peak mass (UniverseMachine)
    mpk = np.maximum(m, mp); dm = m - mp
    out["l1_logMpk"] = mpk
    out["l1_logMs_minus_logMpk"] = s - mpk
    out["l1_strip"] = m - mpk
    out["l1_acc_pct_Mpk"] = _pct_within(dm, mpk, 0.1)
    out["l1_acc_pct_Mprog"] = _pct_within(dm, mp, 0.1)
    # ---- L2 Emerge SFR law
    M1 = 10 ** EMERGE["logM1"]; Mpk = 10.0 ** mpk
    eps = 2 * EMERGE["epsN"] / ((Mpk / M1) ** (-EMERGE["beta"]) + (Mpk / M1) ** EMERGE["gamma"])
    dMh = np.maximum(Mh - 10.0 ** mp, 0.0)
    out["l2_log_sfr_emerge"] = np.log10(1.0 + F_B * eps * dMh)
    out["l2_no_halo_growth"] = (dMh <= 0).astype(float)
    out["l2_slog_eff"] = _slog((10.0 ** s - 10.0 ** sp) / (F_B * np.maximum(dMh, 1e8)))
    # ---- L3 black hole
    acc = np.log10(np.maximum(10.0 ** b - M_SEED, 1.0))
    out["l3_log_BH_acc"] = acc
    out["l3_dBH_at_Ms"] = _median_resid(b, s, 0.1)
    out["l3_dBH_at_Mpk"] = _median_resid(b, mpk, 0.1)
    out["l3_bind"] = acc - (5.0 / 3.0) * mpk
    # ---- pairs within R_PAIR (both directions); rv = x_J - x_I (minimum image)
    tree = cKDTree(pos, boxsize=L.BOX_SIZE)
    pr = tree.query_pairs(R_PAIR, output_type="ndarray")
    I = np.concatenate([pr[:, 0], pr[:, 1]]); J = np.concatenate([pr[:, 1], pr[:, 0]]); del pr
    rv = pos[J] - pos[I]; rv -= L.BOX_SIZE * np.round(rv / L.BOX_SIZE)
    D = np.sqrt((rv ** 2).sum(1))
    heavier = m[J] > m[I]
    # ---- L5 D_vir host (Mansfield & Kravtsov 2020) + cluster distances
    cI, cJ, cR = [I[heavier]], [J[heavier]], [D[heavier] / R200[J[heavier]]]
    for t in CLUSTER_T:
        idx = np.where(m >= t)[0]
        ct = cKDTree(pos[idx], boxsize=L.BOX_SIZE)
        k = min(16, len(idx))
        dd, ii = ct.query(pos, k=k, workers=N_THREADS)
        jj = idx[ii]; ratio = dd / R200[jj]
        ratio = np.where(jj == np.arange(n)[:, None], np.inf, ratio)
        out[f"l5_d_{_rn(t)}"] = ratio.min(1)
        if t == CLUSTER_T[0]:   # far, heavy candidates for the D_vir host (beyond R_PAIR)
            ok = (m[jj] > m[:, None]) & np.isfinite(ratio)
            r_, c_ = np.nonzero(ok)
            cI.append(r_); cJ.append(jj[r_, c_]); cR.append(ratio[r_, c_])
    cI, cJ, cR = np.concatenate(cI), np.concatenate(cJ), np.concatenate(cR)
    o = np.lexsort((cR, cI)); first = np.r_[True, cI[o][1:] != cI[o][:-1]]
    host = np.full(n, -1, dtype=np.int64); dvir = np.full(n, np.nan)
    host[cI[o][first]] = cJ[o][first]; dvir[cI[o][first]] = cR[o][first]
    hs = host >= 0; h = np.where(hs, host, 0)
    out["l5_Dvir"] = np.where(hs, dvir, 99.0)
    out["l5_logMh_host"] = np.where(hs, m[h], m)
    grp = (m >= 12.5) & (m < 14.0)
    near_grp = np.zeros(n, bool)
    w = grp[J] & (D < 1.5 * R200[J]); near_grp[np.unique(I[w])] = True
    out["l5_preproc"] = ((out["l5_d_14p0"].to_numpy() < 5.0) & near_grp).astype(float)
    # ---- L4 signed phase space relative to the D_vir host (Rhee et al. 2017; Wetzel et al. 2014)
    r = pos - pos[h]; r -= L.BOX_SIZE * np.round(r / L.BOX_SIZE)
    dvel = vel - vel[h]
    rr = np.maximum(np.sqrt((r ** 2).sum(1)), 1e-6); x = rr / R200[h]
    ur = (dvel * r).sum(1) / rr / V200[h]
    u2 = (dvel ** 2).sum(1) / V200[h] ** 2
    ut = np.sqrt(np.maximum(u2 - ur ** 2, 0.0))
    urH = ((dvel + 100.0 * r) * r).sum(1) / rr / V200[h]
    E = 0.5 * u2 - np.log1p(NFW_C * x) / (x * mc)
    jnorm = np.linalg.norm(np.cross(r, dvel), axis=1) / (R200[h] * V200[h])
    vr = np.sqrt(u2) * V200[h] / vesc(V200[h], x)
    nan = lambda a: np.where(hs, a, np.nan)
    out["l4_s"] = nan(x); out["l4_ur"] = nan(ur); out["l4_ut"] = nan(ut); out["l4_ur_H"] = nan(urH)
    # an NFW potential never reaches zero, so "bound" is only meaningful near the host: require r < 3 R200
    out["l4_E"] = nan(E); out["l4_bound"] = (hs & (x < 3.0) & (E < 0)).astype(float); out["l4_vesc_ratio"] = nan(vr)
    out["l4_backsplash"] = (hs & (x > 1) & (x < 2.5) & (ur > 0) & (E < 0)).astype(float)
    out["l4_infall"] = (hs & (x < 2.5) & (ur < 0)).astype(float)
    out["l4_j"] = nan(jnorm)
    # ---- L6 tidal tensor from point masses (Mansfield & Kravtsov 2020; Hahn et al. 2009)
    coef = G_GRAV * Mh[J] / D ** 5
    near = D < 10.0 * R200[I]
    for tag, sel in (("near", near), ("far", ~near)):
        T = np.zeros((n, 3, 3))
        Ii, ci, rvi, Di = I[sel], coef[sel], rv[sel], D[sel]
        for a in range(3):
            for c in range(a, 3):
                val = ci * ((Di ** 2 if a == c else 0.0) - 3.0 * rvi[:, a] * rvi[:, c])
                acc_ = np.bincount(Ii, weights=val, minlength=n)
                T[:, a, c] = acc_; T[:, c, a] = acc_
        lam = np.linalg.eigvalsh(T)                        # ascending
        has = np.bincount(Ii, minlength=n) > 0
        tau = np.abs(lam[:, 0]) * R200 ** 3 / (G_GRAV * Mh)
        out[f"l6_logtau_{tag}"] = np.where(has, np.log10(np.maximum(tau, 1e-12)), np.nan)
        out[f"l6_aniso_{tag}"] = np.where(has, (lam[:, 2] - lam[:, 0]) / np.maximum(np.abs(lam).sum(1), 1e-30), np.nan)
    hill = np.full(n, np.inf)
    np.minimum.at(hill, I[heavier], D[heavier] * (Mh[I[heavier]] / (3.0 * Mh[J[heavier]])) ** (1.0 / 3.0))
    out["l6_rHill"] = np.where(np.isfinite(hill), hill / R200, R_PAIR / R200)
    # ---- L7 bound companions (Mansfield & Kravtsov 2020 bound mass)
    dvp = np.linalg.norm(vel[J] - vel[I], axis=1)
    x_i = D / R200[I]
    bnd = (x_i < 3.0) & (dvp < vesc(V200[I], x_i))
    out["l7_log_Mbound"] = np.log10(1.0 + np.bincount(I[bnd], weights=Mh[J[bnd]], minlength=n))
    out["l7_N_bound"] = np.bincount(I[bnd], minlength=n).astype(float)
    x_j = D / R200[J]
    to_h = heavier & (x_j < 3.0) & (dvp < vesc(V200[J], x_j))
    flag = np.zeros(n); flag[np.unique(I[to_h])] = 1.0
    out["l7_bound_to_heavier"] = flag
    return out[L_COLS]


# ------------------------------------------------------------------ S: teammate's neighbour features (27)
# Definitions follow sfr_neighbor_feature_definitions.pdf (teammate, 2026-10-31), with two project conventions:
# periodic minimum-image distances, and the neighbour catalogue = all galaxies (inputs only, no SFR).
S_K = (8, 32, 128)
S_R = (0.1, 0.3, 1.0, 3.0)
S_SEL = ("nearest", "Amax", "Tmax")
S_COLS = ([f"logA_sum{k}" for k in S_K] + [f"logT_sum{k}" for k in S_K] + ["logA_max128", "logT_max128"]
          + [f"count_r{r}" for r in S_R]
          + [f"{p}_{v}" for p in S_SEL for v in ("logr", "logmassratio", "vrel", "vr", "vt")])


def compute_S(df):
    pos = np.mod(df[L.POS_COLS].to_numpy(float), L.BOX_SIZE); vel = df[L.VEL_COLS].to_numpy(float)
    Ms = 10.0 ** df.log10_stellar_mass.to_numpy(float)
    n = len(df)
    tree = cKDTree(pos, boxsize=L.BOX_SIZE)
    d, idx = tree.query(pos, k=max(S_K) + 1, workers=N_THREADS)
    self_ = idx == np.arange(n)[:, None]                       # drop the galaxy itself (not just column 0)
    order = np.argsort(np.where(self_, np.inf, d), axis=1, kind="stable")[:, :max(S_K)]
    d = np.maximum(np.take_along_axis(d, order, 1), 1e-6)      # tiny floor only to avoid division by zero
    idx = np.take_along_axis(idx, order, 1)
    A = Ms[idx] / d ** 2; T = Ms[idx] / d ** 3
    out = pd.DataFrame(index=df.index)
    for k in S_K:
        out[f"logA_sum{k}"] = np.log10(A[:, :k].sum(1))
    for k in S_K:
        out[f"logT_sum{k}"] = np.log10(T[:, :k].sum(1))
    out["logA_max128"] = np.log10(A.max(1)); out["logT_max128"] = np.log10(T.max(1))
    for r in S_R:
        out[f"count_r{r}"] = tree.query_ball_point(pos, r, return_length=True, workers=N_THREADS) - 1
    rows = np.arange(n)
    for p, col in (("nearest", np.zeros(n, int)), ("Amax", A.argmax(1)), ("Tmax", T.argmax(1))):
        j = idx[rows, col]
        dx = pos[j] - pos; dx -= L.BOX_SIZE * np.round(dx / L.BOX_SIZE)
        r = np.maximum(np.sqrt((dx ** 2).sum(1)), 1e-6)
        dv = vel[j] - vel
        vrel = np.sqrt((dv ** 2).sum(1)); vr = (dv * dx).sum(1) / r
        out[f"{p}_logr"] = np.log10(r)
        out[f"{p}_logmassratio"] = np.log10(Ms[j] / Ms)
        out[f"{p}_vrel"] = vrel; out[f"{p}_vr"] = vr
        out[f"{p}_vt"] = np.sqrt(np.maximum(vrel ** 2 - vr ** 2, 0.0))
    return out[S_COLS]


# ------------------------------------------------------------------ feature sets
FS = dict(L.FEATURE_SETS)                     # F0, F1, F2, F3, F4, F5, F6 from the original project
FS["F5A"] = FS["F5"] + A1_COLS + A2_COLS      # F5 + multi-scale environment + phase space (59)
FS["ALL"] = FS["F4"] + L.MSSM_FEATURES + A1_COLS + A2_COLS   # everything incl. raw coordinates (68)
FS["F5G"] = FS["F5"] + G_COLS                 # F5 + environment grid and overdensity-step flags (40)
FS["F5N"] = FS["F5"] + N_COLS                 # F5 + new relative / neighbour-average features (37)
FS["F5GN"] = FS["F5"] + G_COLS + N_COLS       # both (52)
FS["F5lean"] = [c for c in FS["F5"] if c not in ("v_mag", "env_dv_heavier", "is_prog_floor")]  # drop weakest (22)
FS["MAX"] = FS["F5"] + G_COLS + N_COLS + A1_COLS + A2_COLS + L.MSSM_FEATURES   # every engineered feature (89)
for _g, _c in L_GROUPS.items():
    FS[f"F5+{_g}"] = FS["F5"] + _c             # one literature group at a time (ablation)
FS["F5L"] = FS["F5"] + L_COLS                 # all seven literature groups
FS["F5+S"] = FS["F5"] + S_COLS                # teammate's 27 neighbour features
FS["F5LS"] = FS["F5"] + L_COLS + S_COLS       # literature + teammate
FS["F5X"] = FS["F5"] + L_COLS + S_COLS + A1_COLS + A2_COLS + G_COLS + N_COLS + L.MSSM_FEATURES   # everything engineered


def build_features(df):
    if FEAT_PATH.exists():
        F = pd.read_parquet(FEAT_PATH)
        if len(F) == len(df) and set(G_COLS + N_COLS + L_COLS + S_COLS) <= set(F.columns):
            return F
    L.ENV_CACHE = ROOT / "data" / "env_features.parquet"
    L.MSSM_CACHE = ROOT / "data" / "env_mssm.parquet"
    base = pd.concat([df[L.BASE_FEATURES], L.add_physics_features(df),
                      L.compute_env_features(df, cache=L.ENV_CACHE),
                      L.compute_mssm_counts(df, cache=L.MSSM_CACHE)], axis=1)
    F = pd.concat([base, compute_A(df), compute_G(df, base), compute_N(df, base), compute_L(df), compute_S(df)], axis=1)
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
