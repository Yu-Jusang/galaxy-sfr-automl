"""
sfr_features.py
===============
CROCODILE 은하 SFR 예측 대회용 피처 생성기 (AutoGluon 등 tabular AutoML 입력용).

핵심 원칙
---------
* **타깃(sfr_linear)을 전혀 쓰지 않는다.** 모든 피처는 13개 허용 입력 열만으로 만든다.
  `subhalo_id`도 쓰지 않는다.
* 이웃·격자·잔차·비지도 피처는 **전체 카탈로그(train + val)의 입력값**으로 계산한다
  (transductive, label-free). 보고서 4.1절에 이 사실을 명시할 것.
* 결과는 원본 df와 **같은 index**를 갖는 DataFrame이다. 템플릿의 locked split 이후
  `FE.loc[X_train.index]`, `FE.loc[X_val.index]`로 정렬해서 쓰면 된다.
  (템플릿의 `X`에 열을 추가하면 locked HGB baseline까지 바뀌므로 그렇게 하지 말 것.)

사용 예
-------
>>> from sfr_features import build_features
>>> FE, FE_DOCS = build_features(df)                 # 전체 그룹
>>> FE, FE_DOCS = build_features(df, groups=("raw", "flags", "mass", "growth"))
>>> FE_DOCS.to_csv("feature_dictionary.csv", index=False)

그룹 목록 (ALL_GROUPS)
---------------------
raw        원본 13열 (keep_raw_positions=False면 pos_* 제외)
flags      sentinel 값 분리: 병합 여부, progenitor 없음, BH seed 더미, 박리 등
mass       질량비: M*/Mh, MBH/M*, MBH/Mh, progenitor M*/Mh와 그 변화
growth     progenitor 대비 성장: 비율 성장, 절대 성장, 병합 기여 추정, in-situ 성장 추정
kinematic  속력, 국소 흐름 대비 특이속도, 이웃 속도분산, 최근접 이웃 상대속도
knn        k번째 이웃 거리와 밀도 (k=1..100), 밀도 대비(shell contrast)
counts     반경 0.25–8 h^-1 Mpc 안의 은하 수 (정확한 개수)
aggregates 이웃 k=10/30개의 물리량 집계 (GNN 메시지 전달의 고정 근사, 타깃 미사용)
massive    Mh ≥ 10^12..10^14 은하까지의 거리, R200 단위 거리, 상대 시선/반경 속도
host       숨은 숙주 복원: 더 무거운 이웃 중 r/R200 최소 → 위상공간 좌표, 위성 수
grid       주기 FFT 격자에서 Gaussian 평활한 수·질량 밀도장 (σ = 1–8 h^-1 Mpc)
tidal      평활 밀도장의 tidal tensor 고유값, T-web 분류, 이방성
residual   고정 M*(또는 고정 Mh)에서의 조건부 잔차와 조건부 백분위
peng       Peng+2010 분리형 퀜칭 위험률 (SDSS 계수를 사전(prior)으로 사용)
unsup      PCA 성분, MiniBatchKMeans 군집 라벨(범주형)과 중심까지 거리
--- v2 추가 ---
heavier    Mh가 더 무거운 가장 가까운 이웃 (거리, 질량, 상대속도, 반경 0.5/1/2 중심 판정)
mssm       반경 1/2/5 안 Mh ≥ 0.8×, 2× 자기인 이웃 수 (Jo & Kim 2019)
radius     반경 0.5/1/2/5 정확 집계: 개수, 질량 합, 질량 서열, 속도분산, Mh 가중 흐름,
           Σ Mh/r, Σ Mh/r³, 최대 이웃의 성장·BH·SHMR, 병합 이웃 수, SDSS식 고립 판정
host2      숙주 = 반경 2 안 최대 Mh 은하: r/R_rel, Δv/σ_loc, Δv/V_rel, 위상공간 곱, 질량 차
hostphys   숙주에 대한 조석 반경 비, 동역학적 마찰 시간, 탈출속도 비, 묶임 여부
influence  A = M/r², T = M/r³ (M*와 Mh 두 가지)의 합(k=8/32/128)과 최대, A·T 최대 이웃의 거리·질량비·상대속도
fof        FoF 무리 찾기 (연결 길이 0.3/0.5/1.0): 무리 크기·질량, 무리 안 서열, 중심 거리, 속도분산
shape      이웃 10/30개 분포의 관성 텐서 축비, 삼축성, 중심 이탈, 흐름–장축 정렬
phase6d    위치+속도 6차원 이웃 밀도 (속도 척도 100, 1000 km/s per h^-1 Mpc)
thresh     overdensity·M* 문턱 플래그 (신경망·선형 모델용)

물리 상수/가정
-------------
* 상자 크기 200 (h^-1 cMpc로 가정), 주기 경계 가정.
* R200, V200 근사: 200ρ_c 정의, M=10^12 h^-1 M☉에서 z=0 기준 R200≈0.163 h^-1 Mpc,
  V200≈163 km/s. `redshift` 인자로 E(z) 보정 (스냅샷 z 미상 → 기본 0).
  숙주 선택(argmin)과 트리 모델은 이 상수에 불변이고, x<1 같은 문턱 플래그만 영향을 받는다.
* log10_BH_mass: 10^5 seed 더미(58%)와 log10(2×10^5)=5.301의 이중 seed 스파이크가 있다.
"""

from __future__ import annotations

import time
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

__all__ = ["build_features", "ALL_GROUPS", "FEATURE_SETS", "SELECTED_FEATURES", "RAW_COLS", "align_to_split",
           "feature_group_map", "prune_correlated"]

RAW_COLS = [
    "log10_halo_mass", "log10_stellar_mass", "log10_BH_mass",
    "log10_main_prog_mass", "log10_main_prog_stellar_mass", "log10_mass_ratio",
    "log10_overdensity_plus_1", "pos_x", "pos_y", "pos_z", "vel_x", "vel_y", "vel_z",
]
ALL_GROUPS = (
    "raw", "flags", "mass", "growth", "kinematic", "knn", "counts", "aggregates",
    "massive", "host", "grid", "tidal", "residual", "peng", "unsup",
    # v2 추가 그룹
    "heavier", "mssm", "radius", "host2", "hostphys", "influence", "fof", "shape", "phase6d", "thresh",
)
# 권장 피처 세트 (train split 5-fold LightGBM ablation 근거)
#   core : 원본 + 내부 물리 피처 41개 — 전체 RMSE가 가장 낮았음 (꼬리 은하 적합에 유리)
#   env  : core + 환경·숙주 — 본체(상위 0.1% 제외) RMSE와 SFR=0 판별 개선
#   full : 229개 전부 — 본체 RMSE·SFR=0 MAE 최저, 꼬리는 core보다 약간 불리
FEATURE_SETS = {
    "core": ("raw", "flags", "mass", "growth"),
    "env": ("raw", "flags", "mass", "growth", "kinematic", "knn", "counts", "aggregates", "massive", "host"),
    "full_v1": ("raw", "flags", "mass", "growth", "kinematic", "knn", "counts", "aggregates", "massive", "host",
                "grid", "tidal", "residual", "peng", "unsup"),
    "full": ALL_GROUPS,
}

# SHAP 기반 선별 세트 (select_features.py 결과로 채움: train split의 절반 A에서 선택, 다른 절반 B에서 검증)
# 2026-10-31 결과: 89개 = core 41 + SHAP 선별 48 (회귀 23, 게이트 29, 겹침 4). influence 그룹 제외, 잡음 피처 기준 통과
# 독립 절반 B 검증: 전체 424개 대비 RMSE 6.54→6.21, 상위 0.1% 제외 3.92→3.90, SFR=0 MAE 0.501→0.487
SELECTED_FEATURES: list[str] = [
    "log10_halo_mass",
    "log10_stellar_mass",
    "log10_BH_mass",
    "log10_main_prog_mass",
    "log10_main_prog_stellar_mass",
    "log10_mass_ratio",
    "log10_overdensity_plus_1",
    "pos_x",
    "pos_y",
    "pos_z",
    "vel_x",
    "vel_y",
    "vel_z",
    "has_merger",
    "merger_ratio_nan",
    "prog_missing",
    "bh_seed",
    "bh_double_seed",
    "bh_grown",
    "bh_excess_dseed",
    "halo_floor",
    "stripped",
    "halo_lost",
    "star_lost",
    "shm",
    "bh_ms",
    "bh_mh",
    "prog_shm",
    "d_shm",
    "log_mtot",
    "ms_minus_mprog_total",
    "dlog_ms",
    "dlog_mh",
    "log_dms_abs",
    "dms_lin_1e9",
    "dmh_lin_1e11",
    "log_merger_mass_est",
    "log_merger_star_est",
    "log_insitu_est",
    "insitu_frac",
    "spec_insitu",
    "kmeans32",
    "kmeans16",
    "kmeans8",
    "peng_env_share",
    "dlog_ms_res_ms",
    "bh_res_mh",
    "peng_fq",
    "ms_res_mh",
    "log_n_sat_x2p5",
    "nn1_dms",
    "log_n_sat_x1",
    "nb10_max_ms_minus_self",
    "bh_res_ms",
    "dlog_ms_res_od",
    "lrho_contrast_5_50",
    "r2p0_mean_od",
    "bh_pct_ms",
    "pca_2",
    "pca_5",
    "fof0p3_log_ms_tot",
    "pca_3",
    "r0p5_log_vrel_mw",
    "dv_over_v200_mh14p0",
    "r5p0_log_pot",
    "fof0p5_log_mh_tot",
    "host_od",
    "fof1p0_log_vdisp",
    "nb10_mean_od",
    "r2p0_log_sum_ms",
    "lrho_5nn",
    "h2_host_sig_loc",
    "grid_mh_s1p0",
    "lrho_3nn",
    "log_vrel_flow10",
    "fof1p0_log_ms_tot",
    "lrho_2nn",
    "fof1p0_mh_frac",
    "nb10_mean_dlog_ms",
    "r1p0_hn_merger",
    "r1p0_log_sum_ms",
    "host_vt",
    "tid_l1_over_trace_s4p0",
    "fof1p0_log_n",
    "r2p0_hn_merger",
    "r5p0_n_merger",
    "tid_prolate_s4p0",
    "r1p0_hn_dlog_ms",
    "nb30_mean_dlog_ms",
]

BOX = 200.0
LOG_SEED = 5.0                      # 10^5 seed (clip floor)
LOG_DSEED = float(np.log10(2e5))    # 5.30103: two merged seeds
FLOOR_PROG = 6.0
FLOOR_RATIO = -6.0
FLOOR_HALO = 8.0
EPS = 1e-12


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
class _Bucket:
    """Collects feature columns + a documentation table."""

    def __init__(self, index: pd.Index, groups: set, verbose: bool):
        self.index = index
        self.groups = groups
        self.verbose = verbose
        self.cols: dict[str, object] = {}
        self.docs: list[tuple[str, str, str]] = []
        self._t0 = time.time()

    def want(self, group: str) -> bool:
        return group in self.groups

    def add(self, name: str, values, group: str, desc: str):
        if group not in self.groups:
            return
        if isinstance(values, pd.Categorical):
            self.cols[name] = values
        else:
            v = np.asarray(values)
            if v.dtype == bool:
                v = v.astype(np.int8)
            elif v.dtype.kind == "f":
                v = v.astype(np.float32)
                v[~np.isfinite(v)] = np.nan
            self.cols[name] = v
        self.docs.append((name, group, desc))

    def log(self, msg: str):
        if self.verbose:
            print(f"[sfr_features] {time.time() - self._t0:7.1f}s  {msg}", flush=True)

    def frame(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        fe = pd.DataFrame(self.cols, index=self.index)
        docs = pd.DataFrame(self.docs, columns=["feature", "group", "description"])
        return fe, docs


def _min_image(d: np.ndarray, L: float) -> np.ndarray:
    return (d + 0.5 * L) % L - 0.5 * L


def _r200_v200(log_m: np.ndarray, redshift: float = 0.0, om: float = 0.31):
    """Comoving R200 [h^-1 Mpc] and V200 [km/s] for 200*rho_crit(z) haloes."""
    ez2 = om * (1.0 + redshift) ** 3 + (1.0 - om)
    m3 = np.power(10.0, (np.asarray(log_m, dtype=np.float64) - 12.0) / 3.0)
    r200 = 0.1627 * m3 * ez2 ** (-1.0 / 3.0) * (1.0 + redshift)
    v200 = 162.6 * m3 * ez2 ** (1.0 / 6.0)
    return r200, v200


def _safe_log10(x, floor=1e-30):
    x = np.asarray(x, dtype=np.float64)
    out = np.full(x.shape, np.nan)
    ok = np.isfinite(x) & (x > floor)
    out[ok] = np.log10(x[ok])
    return out


def _knn_without_self(tree: cKDTree, P: np.ndarray, k: int, workers: int):
    """k nearest neighbours of every point, excluding the point itself."""
    D, I = tree.query(P, k=k + 1, workers=workers)
    n = len(P)
    is_self = I == np.arange(n)[:, None]
    order = np.argsort(is_self, axis=1, kind="stable")[:, :k]   # self → last, then dropped
    D = np.take_along_axis(D, order, axis=1).astype(np.float32)
    I = np.take_along_axis(I, order, axis=1).astype(np.int32)
    return D, I


def _cic_weights(P: np.ndarray, N: int, L: float):
    u = P / L * N - 0.5
    i0 = np.floor(u).astype(np.int64)
    f = u - i0
    out = []
    for dx in (0, 1):
        wx = f[:, 0] if dx else 1.0 - f[:, 0]
        for dy in (0, 1):
            wy = f[:, 1] if dy else 1.0 - f[:, 1]
            for dz in (0, 1):
                wz = f[:, 2] if dz else 1.0 - f[:, 2]
                flat = (((i0[:, 0] + dx) % N) * N + ((i0[:, 1] + dy) % N)) * N + ((i0[:, 2] + dz) % N)
                out.append((flat, wx * wy * wz))
    return out


def _cic_deposit(cic, w: np.ndarray, N: int) -> np.ndarray:
    grid = np.zeros(N ** 3, dtype=np.float64)
    for flat, ww in cic:
        grid += np.bincount(flat, weights=w * ww, minlength=N ** 3)
    return grid.reshape(N, N, N)


def _cic_read(cic, grid: np.ndarray) -> np.ndarray:
    g = grid.ravel()
    val = np.zeros(len(cic[0][0]), dtype=np.float64)
    for flat, ww in cic:
        val += g[flat] * ww
    return val


def _kgrid(N: int, L: float):
    kx = 2.0 * np.pi * np.fft.fftfreq(N, d=L / N)
    kz = 2.0 * np.pi * np.fft.rfftfreq(N, d=L / N)
    KX, KY, KZ = np.meshgrid(kx, kx, kz, indexing="ij")
    K2 = KX ** 2 + KY ** 2 + KZ ** 2
    return KX, KY, KZ, K2


def _ball_pairs(tree: cKDTree, P: np.ndarray, r: float, L: float, workers: int):
    """All (i, j≠i) pairs with periodic distance ≤ r → (rows, neighbours, distances)."""
    lists = tree.query_ball_point(P, r, workers=workers)
    lens = np.fromiter((len(x) for x in lists), dtype=np.int64, count=len(lists))
    nb = np.concatenate([np.asarray(x, dtype=np.int64) for x in lists])
    rows = np.repeat(np.arange(len(lists), dtype=np.int64), lens)
    keep = nb != rows
    rows, nb = rows[keep], nb[keep]
    dist = np.linalg.norm(_min_image(P[rows] - P[nb], L), axis=1)
    return rows, nb, dist


def _seg_argmax(rows: np.ndarray, vals: np.ndarray, n: int) -> np.ndarray:
    """For each row id, the pair index whose value is maximal (−1 if the row has no pairs)."""
    order = np.lexsort((vals, rows))
    rs = rows[order]
    last = np.r_[rs[1:] != rs[:-1], True] if len(rs) else np.zeros(0, bool)
    best = np.full(n, -1, dtype=np.int64)
    best[rs[last]] = order[last]
    return best


def _binned_median_residual(v: np.ndarray, cond: np.ndarray, n_bins: int = 40):
    """v − median(v | cond-bin) and percentile of v within the cond-bin (equal-count bins)."""
    s = pd.Series(v)
    b = pd.qcut(pd.Series(cond), q=n_bins, labels=False, duplicates="drop")
    med = s.groupby(b).transform("median")
    pct = s.groupby(b).rank(pct=True)
    return (s - med).to_numpy(), pct.to_numpy()


# ----------------------------------------------------------------------------
# main builder
# ----------------------------------------------------------------------------
def build_features(
    df: pd.DataFrame,
    groups: Iterable[str] = ALL_GROUPS,
    box: float = BOX,
    redshift: float = 0.0,
    k_max: int = 128,
    count_radii: Sequence[float] = (0.1, 0.25, 0.5, 1.0, 2.0, 3.0, 4.0, 8.0),
    ball_radii: Sequence[float] = (0.5, 1.0, 2.0, 5.0),
    fof_links: Sequence[float] = (0.3, 0.5, 1.0),
    knn_ks: Sequence[int] = (1, 2, 3, 5, 10, 20, 50, 100),
    agg_ks: Sequence[int] = (10, 30),
    massive_thresholds: Sequence[float] = (12.0, 12.5, 13.0, 13.5, 14.0),
    grid_n: int = 256,
    grid_sigmas: Sequence[float] = (1.0, 2.0, 4.0, 8.0),
    tidal_sigmas: Sequence[float] = (2.0, 4.0),
    keep_raw_positions: bool = True,
    n_jobs: int = -1,
    random_state: int = 42,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build a wide, target-free feature table.

    Returns
    -------
    FE : DataFrame (same index as df)
    FE_DOCS : DataFrame with columns [feature, group, description]
    """
    groups = set(groups)
    unknown = groups - set(ALL_GROUPS)
    if unknown:
        raise ValueError(f"unknown groups: {sorted(unknown)}; choose from {ALL_GROUPS}")
    missing = [c for c in RAW_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"missing input columns: {missing}")
    if "sfr_linear" in groups:
        raise ValueError("target must never be a feature")

    B = _Bucket(df.index, groups, verbose)
    n = len(df)
    L = float(box)
    ar = np.arange(n)

    # ---------------- base arrays (always computed; only added if group requested)
    ms = df["log10_stellar_mass"].to_numpy(np.float64)
    mh = df["log10_halo_mass"].to_numpy(np.float64)
    bh = df["log10_BH_mass"].to_numpy(np.float64)
    mp = df["log10_main_prog_mass"].to_numpy(np.float64)
    msp = df["log10_main_prog_stellar_mass"].to_numpy(np.float64)
    mr = df["log10_mass_ratio"].to_numpy(np.float64)
    od = df["log10_overdensity_plus_1"].to_numpy(np.float64)
    P = np.mod(df[["pos_x", "pos_y", "pos_z"]].to_numpy(np.float64), L)
    P[P >= L] = 0.0                          # guard: cKDTree requires [0, L)
    V = df[["vel_x", "vel_y", "vel_z"]].to_numpy(np.float64)

    has_merger = mr > FLOOR_RATIO + 1e-9
    prog_missing = (mp <= FLOOR_PROG + 1e-6) | (msp <= FLOOR_PROG + 1e-6)
    bh_seed = bh < LOG_SEED + 0.01
    bh_dseed = np.abs(bh - LOG_DSEED) < 0.003
    stripped = mh < ms
    halo_lost = (mh < mp) & ~prog_missing

    shm = ms - mh
    bh_ms = bh - ms
    bh_mh = bh - mh
    dlog_ms = np.where(prog_missing, np.nan, ms - msp)
    dlog_mh = np.where(prog_missing, np.nan, mh - mp)
    speed = np.linalg.norm(V, axis=1)
    r200, v200 = _r200_v200(mh, redshift)

    # ---------------- raw
    for c in RAW_COLS:
        if not keep_raw_positions and c.startswith("pos_"):
            continue
        B.add(c, df[c].to_numpy(np.float64), "raw", "원본 입력 열")

    # ---------------- flags
    B.add("has_merger", has_merger, "flags", "log10_mass_ratio > -6 (최근 병합 있음)")
    B.add("merger_ratio_nan", np.where(has_merger, mr, np.nan), "flags",
          "병합 질량비, 병합 없으면 NaN (−6 sentinel 제거)")
    B.add("prog_missing", prog_missing, "flags", "main progenitor 질량이 floor(10^6) — 선조 추적 실패 (68% 퀜칭)")
    B.add("bh_seed", bh_seed, "flags", "BH가 10^5 seed 더미 안 (log < 5.01)")
    B.add("bh_double_seed", bh_dseed, "flags", "BH = 2×10^5 (seed 두 개 병합 스파이크)")
    B.add("bh_grown", bh > 5.4, "flags", "BH가 seed 더미를 확실히 벗어남 (log > 5.4)")
    B.add("bh_excess_dseed", np.clip(bh - LOG_DSEED, 0, None), "flags", "log MBH − log(2e5), 0 이하 절단")
    B.add("halo_floor", mh <= FLOOR_HALO + 1e-6, "flags", "halo mass가 floor(10^8)")
    B.add("stripped", stripped, "flags", "Mh < M* (헤일로가 별보다 가벼움 — 박리된 위성)")
    B.add("halo_lost", halo_lost, "flags", "현재 Mh < 선조 Mh (헤일로 질량 손실 — 낙하 후 조석 박리)")
    B.add("star_lost", (ms < msp) & ~prog_missing, "flags", "현재 M* < 선조 M* (별 질량 손실)")
    B.log("flags")

    # ---------------- mass ratios
    B.add("shm", shm, "mass", "log M* − log Mh (stellar-to-halo mass ratio)")
    B.add("bh_ms", bh_ms, "mass", "log MBH − log M*")
    B.add("bh_mh", bh_mh, "mass", "log MBH − log Mh")
    prog_shm = np.where(prog_missing, np.nan, msp - mp)
    B.add("prog_shm", prog_shm, "mass", "선조의 log M*_prog − log Mh_prog")
    B.add("d_shm", shm - prog_shm, "mass", "SHMR 변화 (현재 − 선조)")
    B.add("log_mtot", np.log10(10 ** ms + 10 ** mh), "mass", "log(M* + Mh)")
    B.add("ms_minus_mprog_total", np.where(prog_missing, np.nan, ms - mp), "mass", "log M* − log Mh_prog")
    B.log("mass")

    # ---------------- growth
    if B.want("growth") or B.want("aggregates") or B.want("residual") or B.want("unsup"):
        lin_ms, lin_msp = 10 ** ms, np.where(prog_missing, np.nan, 10 ** msp)
        dms_abs = lin_ms - lin_msp
        merger_star = np.where(has_merger & ~prog_missing, 10 ** (mr + msp), 0.0)
        insitu = dms_abs - merger_star
    B.add("dlog_ms", dlog_ms, "growth", "log M* − log M*_prog (비율 성장, sSFR과 ρ≈0.58)")
    B.add("dlog_mh", dlog_mh, "growth", "log Mh − log Mh_prog (헤일로 성장/손실)")
    if B.want("growth"):
        B.add("log_dms_abs", _safe_log10(dms_abs), "growth", "log10(M* − M*_prog), 양수일 때만 (SFR과 ρ≈0.81)")
        B.add("dms_lin_1e9", dms_abs / 1e9, "growth", "(M* − M*_prog) / 10^9, 부호 유지 (별 질량 감소도 표현)")
        B.add("dmh_lin_1e11", np.where(prog_missing, np.nan, 10 ** mh - 10 ** mp) / 1e11, "growth",
              "(Mh − Mh_prog) / 10^11, 부호 유지 (헤일로 질량 손실량)")
        B.add("log_merger_mass_est", np.where(has_merger & ~prog_missing, mr + mp, np.nan), "growth",
              "병합 상대 총질량 추정 log(ratio × Mh_prog)")
        B.add("log_merger_star_est", np.where(has_merger & ~prog_missing, mr + msp, np.nan), "growth",
              "병합으로 들어온 별 질량 추정 log(ratio × M*_prog)")
        B.add("log_insitu_est", _safe_log10(insitu), "growth", "in-situ 성장 추정 log(ΔM* − 병합 별 질량)")
        B.add("insitu_frac", np.where(dms_abs > 0, insitu / np.where(dms_abs > 0, dms_abs, 1), np.nan), "growth",
              "in-situ 성장 / 전체 성장")
        B.add("spec_insitu", _safe_log10(insitu) - ms, "growth", "log(in-situ 성장 / M*) (장기 sSFR 대리)")
    B.log("growth")

    # ---------------- kinematic (global part)
    B.add("speed", speed, "kinematic", "|v| (km/s)")
    B.add("speed_over_v200", speed / v200, "kinematic", "|v| / V200(자기 헤일로)")

    # ---------------- neighbour search
    need_tree = any(B.want(g) for g in ("kinematic", "knn", "counts", "aggregates", "massive", "host",
                                         "residual", "unsup", "hostphys", "radius", "mssm", "host2",
                                         "fof", "heavier", "shape", "influence"))
    D = I = None
    if need_tree:
        tree = cKDTree(P, boxsize=L)
        kk = max(k_max, max(knn_ks), max(agg_ks))
        D, I = _knn_without_self(tree, P, kk, n_jobs)
        B.log(f"kNN query k={kk}")

    # ---------------- knn densities
    lrho = {}
    if D is not None:
        for k in knn_ks:
            dk = np.maximum(D[:, k - 1].astype(np.float64), 1e-4)
            lrho[k] = np.log10(k / (4.0 / 3.0 * np.pi * dk ** 3))
            # log d_k는 lrho_k의 선형변환이라 넣지 않는다 (중복)
            B.add(f"lrho_{k}nn", lrho[k], "knn", f"k={k} 이웃 수밀도 log(k / (4/3 π d_k^3)) — k번째 이웃 거리와 등가")
        for a, b in ((1, 10), (5, 50), (10, 100), (3, 20)):
            if a in lrho and b in lrho:
                B.add(f"lrho_contrast_{a}_{b}", lrho[a] - lrho[b], "knn", f"밀도 대비 log ρ_{a} − log ρ_{b} (작은 규모 대 큰 규모)")
        B.add("log_d1nn_over_r200", np.log10(np.maximum(D[:, 0], 1e-4) / r200), "knn", "최근접 이웃 거리 / 자기 R200")
        B.log("knn")

    # ---------------- kinematic (local flow)
    if D is not None and B.want("kinematic"):
        for k in (10, 30):
            Vn = V[I[:, :k]]
            vmean = Vn.mean(axis=1)
            vrel = np.linalg.norm(V - vmean, axis=1)
            sig = np.sqrt(((Vn - vmean[:, None, :]) ** 2).sum(axis=2).mean(axis=1) / 3.0)
            B.add(f"log_vrel_flow{k}", _safe_log10(vrel), "kinematic", f"k={k} 이웃 평균 속도(국소 흐름) 대비 특이속도")
            B.add(f"log_sigv{k}", _safe_log10(sig), "kinematic", f"k={k} 이웃 1D 속도분산")
            B.add(f"mach{k}", vrel / np.maximum(sig, 1e-3), "kinematic", f"특이속도 / 속도분산 (k={k})")
            del Vn
        j1 = I[:, 0]
        d1 = _min_image(P - P[j1], L)
        r1 = np.maximum(np.linalg.norm(d1, axis=1), 1e-6)
        dv1 = V - V[j1]
        B.add("log_dv_nn1", _safe_log10(np.linalg.norm(dv1, axis=1)), "kinematic", "최근접 이웃과의 상대속도 크기")
        vr1 = (dv1 * d1).sum(1) / r1
        B.add("vr_nn1", vr1, "kinematic", "최근접 이웃 대비 반경 속도 (+ 멀어짐)")
        B.add("vt_nn1", np.sqrt(np.maximum((dv1 ** 2).sum(1) - vr1 ** 2, 0.0)), "kinematic", "최근접 이웃 대비 접선 속도")
        B.log("kinematic")

    # ---------------- exact counts
    if B.want("counts") or B.want("residual"):
        cnt = {}
        for r in count_radii:
            c = tree.query_ball_point(P, r, return_length=True, workers=n_jobs).astype(np.float64) - 1.0
            cnt[r] = c
            rn = str(r).replace(".", "p")
            B.add(f"log_cnt_r{rn}", np.log1p(c), "counts", f"반경 {r} h^-1 Mpc 안 은하 수 (log1p)")
        for a, b in ((0.5, 2.0), (1.0, 4.0), (2.0, 8.0)):
            if a in cnt and b in cnt:
                an, bn = str(a).replace(".", "p"), str(b).replace(".", "p")
                B.add(f"cnt_shell_ratio_{an}_{bn}", (cnt[a] + 1) / (cnt[b] + 1), "counts",
                      f"N(<{a}) / N(<{b}) — 중심 집중도")
        B.log("counts")

    # ---------------- neighbour aggregates (label-free message passing)
    if D is not None and B.want("aggregates"):
        f_strip = stripped.astype(np.float64)
        f_lost = halo_lost.astype(np.float64)
        f_merg = has_merger.astype(np.float64)
        f_seed = bh_seed.astype(np.float64)
        dms_f = np.where(np.isfinite(dlog_ms), dlog_ms, np.nan)
        for k in agg_ks:
            J = I[:, :k]
            Dk = np.maximum(D[:, :k].astype(np.float64), 1e-2)
            msn, mhn = ms[J], mh[J]
            B.add(f"nb{k}_mean_shm", shm[J].mean(1), "aggregates", f"이웃 {k}개 평균 log(M*/Mh)")
            B.add(f"nb{k}_frac_stripped", f_strip[J].mean(1), "aggregates", f"이웃 {k}개 중 박리(Mh<M*) 비율")
            B.add(f"nb{k}_frac_halo_lost", f_lost[J].mean(1), "aggregates", f"이웃 {k}개 중 헤일로 질량 손실 비율")
            B.add(f"nb{k}_frac_merger", f_merg[J].mean(1), "aggregates", f"이웃 {k}개 중 병합 비율")
            B.add(f"nb{k}_frac_bh_seed", f_seed[J].mean(1), "aggregates", f"이웃 {k}개 중 BH seed 비율")
            B.add(f"nb{k}_mean_bh_ms", bh_ms[J].mean(1), "aggregates", f"이웃 {k}개 평균 log(MBH/M*)")
            B.add(f"nb{k}_mean_od", od[J].mean(1), "aggregates", f"이웃 {k}개 평균 overdensity 열")
            B.add(f"nb{k}_mean_dlog_ms", np.nanmean(dms_f[J], axis=1), "aggregates", f"이웃 {k}개 평균 비율 성장")
            B.add(f"nb{k}_max_mh", mhn.max(1), "aggregates", f"이웃 {k}개 중 최대 log Mh")
            B.add(f"nb{k}_max_ms_minus_self", msn.max(1) - ms, "aggregates", f"이웃 {k}개 최대 log M* − 자기 log M*")
            B.add(f"nb{k}_log_sum_ms", np.log10((10 ** msn).sum(1)), "aggregates", f"이웃 {k}개 M* 합 (log)")
            B.add(f"nb{k}_log_sum_mh", np.log10((10 ** mhn).sum(1)), "aggregates", f"이웃 {k}개 Mh 합 (log)")
            B.add(f"nb{k}_frac_more_massive", (msn > ms[:, None]).mean(1), "aggregates",
                  f"이웃 {k}개 중 자기보다 M*가 큰 비율 (1 − 국소 질량 순위)")
            B.add(f"nb{k}_log_sum_mh_over_r", np.log10((10 ** mhn / Dk).sum(1)), "aggregates",
                  f"Σ Mh/r (Jo & Kim 2019식 중력 영향, 이웃 {k}개)")
            B.add(f"nb{k}_log_sum_ms_over_r2", np.log10((10 ** msn / Dk ** 2).sum(1)), "aggregates",
                  f"Σ M*/r² (조석 영향 근사, 이웃 {k}개)")
            del msn, mhn, Dk
        j1 = I[:, 0]
        B.add("nn1_ms", ms[j1], "aggregates", "최근접 이웃의 log M*")
        B.add("nn1_mh", mh[j1], "aggregates", "최근접 이웃의 log Mh")
        B.add("nn1_dms", ms[j1] - ms, "aggregates", "최근접 이웃 log M* − 자기")
        B.add("nn1_shm", shm[j1], "aggregates", "최근접 이웃의 log(M*/Mh)")
        B.add("nn1_merger", has_merger[j1], "aggregates", "최근접 이웃의 병합 여부")
        B.log("aggregates")

    # ---------------- nearest massive haloes
    massive_nn = {}
    if B.want("massive") or B.want("host") or B.want("hostphys"):
        for T in massive_thresholds:
            idx = np.where(mh >= T)[0]
            if len(idx) < 3:
                continue
            t = cKDTree(P[idx], boxsize=L)
            dd, ii = t.query(P, k=2, workers=n_jobs)
            j0 = idx[ii[:, 0]]
            is_self = j0 == ar
            j = np.where(is_self, idx[ii[:, 1]], j0)
            d = np.where(is_self, dd[:, 1], dd[:, 0])
            massive_nn[T] = (j, d)
            tn = str(T).replace(".", "p")
            dvec = _min_image(P - P[j], L)
            rr = np.maximum(np.linalg.norm(dvec, axis=1), 1e-6)
            dv = V - V[j]
            B.add(f"log_dist_mh{tn}", np.log10(np.maximum(d, 1e-4)), "massive", f"Mh ≥ 10^{T}인 가장 가까운 은하까지 거리 (log)")
            B.add(f"log_x_mh{tn}", np.log10(np.maximum(d, 1e-4) / r200[j]), "massive", f"그 거리 / 그 은하의 R200 (log)")
            B.add(f"vr_mh{tn}", (dv * dvec).sum(1) / rr / v200[j], "massive", f"그 은하 대비 반경 속도 / V200 (− 낙하, + 이탈)")
            if T in (12.0, 13.0, 14.0):
                B.add(f"mh_of_nearest_mh{tn}", mh[j], "massive", f"그 은하의 log Mh")
                B.add(f"dv_over_v200_mh{tn}", np.linalg.norm(dv, axis=1) / v200[j], "massive", f"상대속도 크기 / V200")
        B.log("massive")

    # ---------------- hidden host (phase space)
    if (B.want("host") or B.want("hostphys")) and D is not None:
        # candidates: kNN list with larger M*, plus nearest massive galaxies (cover rich clusters)
        x = D.astype(np.float64) / r200[I]
        x[ms[I] <= ms[:, None]] = np.inf
        kb = np.argmin(x, axis=1)
        host = I[ar, kb].astype(np.int64)
        hx = x[ar, kb]
        for T, (j, d) in massive_nn.items():
            xt = d / r200[j]
            ok = (ms[j] > ms) & (xt < hx)
            host[ok], hx[ok] = j[ok], xt[ok]
        has_host = np.isfinite(hx)
        host[~has_host] = -1
        hj = np.where(has_host, host, 0)
        dvec = _min_image(P - P[hj], L)
        rr = np.maximum(np.linalg.norm(dvec, axis=1), 1e-6)
        dv = V - V[hj]
        vh = v200[hj]
        y = np.linalg.norm(dv, axis=1) / vh
        vr = (dv * dvec).sum(1) / rr / vh
        vt = np.sqrt(np.maximum(y ** 2 - vr ** 2, 0.0))
        hxf = np.where(has_host, hx, np.nan)
        nanify = lambda a: np.where(has_host, a, np.nan)  # noqa: E731
        B.add("host_found", has_host, "host", "자기보다 M*가 큰 숙주 후보가 있음")
        B.add("host_log_x", _safe_log10(hxf), "host", "log(r / R200_host) — 3D 위상공간 반경 좌표")
        B.add("host_y", nanify(y), "host", "|Δv| / V200_host — 위상공간 속도 좌표")
        B.add("host_vr", nanify(vr), "host", "반경 속도 / V200_host (− 낙하, + backsplash)")
        B.add("host_vt", nanify(vt), "host", "접선 속도 / V200_host")
        B.add("host_energy", nanify(0.5 * y ** 2 - 1.0 / np.maximum(hxf, 0.05)), "host",
              "궤도 에너지 근사 ½y² − 1/x (점질량 퍼텐셜, V200² 단위)")
        B.add("host_xy", nanify(hxf * y), "host", "x·y (Rhee+2017 위상공간 낙하 시간 대리)")
        B.add("host_dms", nanify(ms[hj] - ms), "host", "숙주 log M* − 자기 log M*")
        B.add("host_mh", nanify(mh[hj]), "host", "숙주 log Mh (중심 은하면 ≈ 숙주 헤일로)")
        B.add("host_dmh", nanify(mh[hj] - mh), "host", "숙주 log Mh − 자기 log Mh")
        B.add("host_shm", nanify(shm[hj]), "host", "숙주의 log(M*/Mh)")
        B.add("host_bh_ms", nanify(bh_ms[hj]), "host", "숙주의 log(MBH/M*)")
        B.add("host_od", nanify(od[hj]), "host", "숙주의 overdensity 열")
        B.add("host_merger", np.where(has_host, has_merger[hj], 0), "host", "숙주의 병합 여부")
        B.add("host_log_dist", nanify(np.log10(rr)), "host", "숙주까지 공동 거리 (log)")
        sat1 = has_host & (hx < 1.0)
        sat25 = has_host & (hx < 2.5)
        B.add("sat_x1", sat1, "host", "r < R200_host (위성 후보)")
        B.add("zone_x2p5", sat25, "host", "r < 2.5 R200_host (backsplash 영역 포함)")
        B.add("backsplash_cand", has_host & (hx >= 1.0) & (hx < 2.5) & (vr > 0), "host", "1–2.5 R200 바깥으로 이동 중")
        B.add("infall_cand", sat25 & (vr < 0), "host", "2.5 R200 안에서 낙하 중")
        B.add("host_is_sat", np.where(has_host, sat1[hj], 0), "host", "숙주 자신도 위성 후보")
        lin = 10 ** ms
        n_sat1 = np.bincount(host[sat1], minlength=n).astype(np.float64)
        n_sat25 = np.bincount(host[sat25], minlength=n).astype(np.float64)
        m_sat1 = np.bincount(host[sat1], weights=lin[sat1], minlength=n)
        B.add("log_n_sat_x1", np.log1p(n_sat1), "host", "나를 숙주로 고른 위성 수 (r<R200, log1p)")
        B.add("log_n_sat_x2p5", np.log1p(n_sat25), "host", "나를 숙주로 고른 은하 수 (r<2.5 R200, log1p)")
        B.add("log_m_sat_x1", np.log10(1.0 + m_sat1), "host", "위성 M* 합 (log)")
        B.log("host")

        # ---- hostphys: 숙주에 대한 동역학 물리량 (점질량 근사)
        mratio = np.power(10.0, mh[hj] - mh)                       # M_host / M_self (헤일로)
        # (조석 반경 비 R200/r_t는 점질량 근사에서 1/x와 같아 넣지 않는다)
        B.add("hp_log_df_time", nanify(np.log10(mratio / np.log1p(mratio) * np.maximum(hxf, 1e-3))), "hostphys",
              "동역학적 마찰 시간 근사 log[(M_h/M_s)/ln(1+M_h/M_s) · r/R200] (작을수록 빨리 가라앉음)")
        B.add("hp_v_over_vesc", nanify(y / np.sqrt(2.0 / np.maximum(hxf, 1e-3))), "hostphys",
              "|Δv| / 탈출속도(점질량 √(2/x)·V200) — 1 미만이면 묶여 있음")
        B.add("hp_bound", np.where(has_host, (0.5 * y ** 2 - 1.0 / np.maximum(hxf, 0.05)) < 0, 0), "hostphys",
              "궤도 에너지 < 0 (숙주에 중력적으로 묶임)")
        B.add("hp_x_over_mratio", nanify(hxf / np.power(mratio, 1.0 / 3.0)), "hostphys",
              "r/R200_host ÷ (M_h/M_s)^(1/3) — 자기 크기 대비 숙주 중심까지 거리")
        B.log("hostphys")

    # ---------------- radius-based (exact ball) features: heavier / mssm / radius / host2 / fof
    need_balls = any(B.want(g) for g in ("radius", "mssm", "host2", "fof"))
    balls = {}
    if need_balls:
        radii_needed = sorted(set(ball_radii) | ({2.0} if B.want("host2") else set())
                              | ({max(fof_links)} if B.want("fof") else set()))
        for r in radii_needed:
            balls[r] = _ball_pairs(tree, P, r, L, n_jobs)
        B.log(f"ball pairs r={radii_needed}")

    if B.want("radius"):
        lin_ms, lin_mh = 10 ** ms, 10 ** mh
        f_strip = stripped.astype(np.float64)
        f_merg = has_merger.astype(np.float64)
        for r in ball_radii:
            rows, nb, dist = balls[r]
            rn = str(r).replace(".", "p")
            cntr = np.bincount(rows, minlength=n).astype(np.float64)
            has_nb = cntr > 0
            nz = lambda a: np.where(has_nb, a, np.nan)  # noqa: E731
            if r not in count_radii:  # counts 그룹에 같은 반경이 있으면 중복이라 생략
                B.add(f"r{rn}_n", cntr, "radius", f"반경 {r} 안 이웃 수")
            B.add(f"r{rn}_log_sum_ms", np.log10(1.0 + np.bincount(rows, lin_ms[nb], n)), "radius", f"반경 {r} 안 이웃 M* 합 log(1+Σ)")
            B.add(f"r{rn}_log_sum_mh", np.log10(1.0 + np.bincount(rows, lin_mh[nb], n)), "radius", f"반경 {r} 안 이웃 Mh 합 log(1+Σ)")
            mx = np.full(n, -np.inf)
            np.maximum.at(mx, rows, ms[nb])
            B.add(f"r{rn}_rank_ms", ms - np.maximum(mx, ms), "radius", f"자기 M* − 반경 {r} 안 최대 M* (0이면 그 반경의 최고)")
            # 1D velocity dispersion of self + neighbours
            N1 = cntr + 1.0
            s1 = np.column_stack([np.bincount(rows, V[nb, a], n) for a in range(3)]) + V
            s2 = np.column_stack([np.bincount(rows, V[nb, a] ** 2, n) for a in range(3)]) + V ** 2
            var = np.maximum(s2 / N1[:, None] - (s1 / N1[:, None]) ** 2, 0.0).sum(1) / 3.0
            sig = np.sqrt(var)
            B.add(f"r{rn}_log_vdisp", nz(_safe_log10(sig)), "radius", f"자기+이웃 1D 속도분산 (반경 {r})")
            # mass-weighted local flow (neighbours only)
            wsum = np.bincount(rows, lin_mh[nb], n)
            vmw = np.column_stack([np.bincount(rows, lin_mh[nb] * V[nb, a], n) for a in range(3)]) / np.maximum(wsum, 1e-30)[:, None]
            B.add(f"r{rn}_log_vrel_mw", nz(_safe_log10(np.linalg.norm(V - vmw, axis=1))), "radius",
                  f"Mh 가중 국소 흐름 대비 특이속도 (반경 {r})")
            B.add(f"r{rn}_frac_stripped", nz(np.bincount(rows, f_strip[nb], n) / np.maximum(cntr, 1)), "radius", f"반경 {r} 안 박리 은하 비율")
            B.add(f"r{rn}_n_merger", np.bincount(rows, f_merg[nb], n), "radius", f"반경 {r} 안 병합 중인 이웃 수")
            B.add(f"r{rn}_mean_od", nz(np.bincount(rows, od[nb], n) / np.maximum(cntr, 1)), "radius", f"반경 {r} 안 이웃 평균 overdensity 열")
            dd = np.maximum(dist, 0.02)
            B.add(f"r{rn}_log_pot", np.log10(1e-30 + np.bincount(rows, lin_mh[nb] / dd, n)), "radius", f"Σ Mh/r (중력 퍼텐셜 근사, 반경 {r})")
            B.add(f"r{rn}_log_tidal", np.log10(1e-30 + np.bincount(rows, lin_mh[nb] / dd ** 3, n)), "radius", f"Σ Mh/r³ (조석 세기 근사, 반경 {r})")
            # heaviest neighbour (by Mh) inside r
            best = _seg_argmax(rows, mh[nb], n)
            hb = best >= 0
            jh = np.where(hb, nb[np.maximum(best, 0)], 0)
            hn = lambda a: np.where(hb, a, np.nan)  # noqa: E731
            B.add(f"r{rn}_hn_dmh", hn(mh[jh] - mh), "radius", f"반경 {r} 안 최대 Mh 이웃 − 자기 Mh")
            B.add(f"r{rn}_hn_dlog_ms", hn(dlog_ms[jh]), "radius", f"그 최대 이웃의 비율 성장 (무리 중심이 지금도 자라나)")
            B.add(f"r{rn}_hn_bh", hn(bh[jh]), "radius", f"그 최대 이웃의 log MBH")
            B.add(f"r{rn}_hn_shm", hn(shm[jh]), "radius", f"그 최대 이웃의 log(M*/Mh)")
            B.add(f"r{rn}_hn_merger", np.where(hb, has_merger[jh], 0), "radius", f"그 최대 이웃의 병합 여부")
            # isolation (SDSS-like): no neighbour with M* ≥ M*/2 and |Δv| < 500 km/s
            dvn = np.linalg.norm(V[rows] - V[nb], axis=1)
            comp = (ms[nb] >= ms[rows] - np.log10(2.0)) & (dvn < 500.0)
            B.add(f"r{rn}_isolated", np.bincount(rows[comp], minlength=n) == 0, "radius",
                  f"반경 {r} 안에 M* ≥ 자기/2, |Δv|<500 km/s인 이웃이 없음 (고립 은하)")
            del dvn, comp
        B.log("radius")

    if B.want("mssm"):
        for r in ball_radii:
            if r < 1.0:
                continue
            rows, nb, _ = balls[r]
            rn = str(r).replace(".", "p")
            for fac in (0.8, 2.0):
                fn = str(fac).replace(".", "p")
                c = np.bincount(rows[mh[nb] >= mh[rows] + np.log10(fac)], minlength=n)
                B.add(f"mssm_n{fn}_r{rn}", c, "mssm", f"반경 {r} 안 Mh ≥ {fac}×자기 Mh인 이웃 수 (Jo & Kim 2019식)")
        B.log("mssm")

    if B.want("host2"):
        rows, nb, dist = balls[2.0]
        best = _seg_argmax(rows, mh[nb], n)
        hb = best >= 0
        jh = np.where(hb, nb[np.maximum(best, 0)], 0)
        has2 = hb & (mh[jh] > mh)
        # local dispersion around the host (within r=2 of the host)
        cnt2 = np.bincount(rows, minlength=n).astype(np.float64) + 1.0
        s1 = np.column_stack([np.bincount(rows, V[nb, a], n) for a in range(3)]) + V
        s2 = np.column_stack([np.bincount(rows, V[nb, a] ** 2, n) for a in range(3)]) + V ** 2
        sig2 = np.sqrt(np.maximum(s2 / cnt2[:, None] - (s1 / cnt2[:, None]) ** 2, 0.0).sum(1) / 3.0)
        dvec = _min_image(P - P[jh], L)
        rr2 = np.linalg.norm(dvec, axis=1)
        dvh = np.linalg.norm(V - V[jh], axis=1)
        rrel = np.power(10.0, (mh[jh] - 12.0) / 3.0)
        h2 = lambda a: np.where(has2, a, np.nan)  # noqa: E731
        B.add("h2_r_rrel", h2(rr2 / rrel), "host2", "숙주까지 거리 / R_rel (R_rel = 10^((logMh_host−12)/3))")
        B.add("h2_dv_sigloc", h2(dvh / np.maximum(sig2[jh], 1.0)), "host2", "숙주와의 속도 차 / 숙주 주변(반경 2) 속도분산")
        B.add("h2_dv_vrel", h2(dvh / (100.0 * rrel)), "host2", "숙주와의 속도 차 / V_rel (100 km/s × 10^((logMh_host−12)/3))")
        B.add("h2_ps", h2((rr2 / rrel) * (dvh / (100.0 * rrel))), "host2", "(거리 비) × (속도 비) — 작으면 오래전에 가라앉은 위성")
        B.add("h2_host_sig_loc", h2(_safe_log10(sig2[jh])), "host2", "숙주 주변 속도분산 (log) — 무리 질량 대리")
        B.log("host2")

    if B.want("fof"):
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components
        rows_all, nb_all, dist_all = balls[max(fof_links)]
        lin_ms, lin_mh = 10 ** ms, 10 ** mh
        for ell in fof_links:
            keep = dist_all <= ell
            A = coo_matrix((np.ones(keep.sum(), dtype=np.int8), (rows_all[keep], nb_all[keep])), shape=(n, n))
            ncomp, lab = connected_components(A, directed=False)
            en = str(ell).replace(".", "p")
            size = np.bincount(lab, minlength=ncomp).astype(np.float64)
            gms = np.bincount(lab, lin_ms, ncomp)
            gmh = np.bincount(lab, lin_mh, ncomp)
            # brightest (max M*) member of each group
            order = np.lexsort((ms, lab))
            ls = lab[order]
            last = np.r_[ls[1:] != ls[:-1], True]
            bcg = np.empty(ncomp, dtype=np.int64)
            bcg[ls[last]] = order[last]
            jb = bcg[lab]
            rel = _min_image(P - P[jb], L)                        # 무리 최대 은하 기준 상대 위치
            cen = np.column_stack([np.bincount(lab, lin_mh * rel[:, a], ncomp) for a in range(3)]) / gmh[:, None]
            dcen = np.linalg.norm(rel - cen[lab], axis=1)
            rad = np.sqrt(np.maximum(np.bincount(lab, (rel ** 2).sum(1), ncomp) / size - (cen ** 2).sum(1), 0.0))
            vg = np.column_stack([np.bincount(lab, V[:, a], ncomp) for a in range(3)]) / size[:, None]
            vdisp = np.sqrt(np.maximum(np.column_stack([np.bincount(lab, V[:, a] ** 2, ncomp) for a in range(3)]) / size[:, None] - vg ** 2, 0).sum(1) / 3.0)
            # rank of M* within group (0 = brightest)
            rank = np.empty(n)
            first_pos = np.r_[0, np.flatnonzero(ls[1:] != ls[:-1]) + 1]
            seg_len = np.diff(np.r_[first_pos, n])
            last_pos = np.repeat(first_pos + seg_len - 1, seg_len)     # 정렬 위치별 그 무리의 마지막 위치
            rank[order] = last_pos - np.arange(n)
            multi = size[lab] > 1
            gm = lambda a: np.where(multi, a, np.nan)  # noqa: E731
            B.add(f"fof{en}_log_n", np.log10(size[lab]), "fof", f"FoF(연결 {ell}) 무리 은하 수 (log)")
            B.add(f"fof{en}_log_ms_tot", np.log10(gms[lab]), "fof", f"무리 총 M* (log)")
            B.add(f"fof{en}_log_mh_tot", np.log10(gmh[lab]), "fof", f"무리 총 Mh (log) — 무리 질량")
            B.add(f"fof{en}_mh_frac", lin_mh / gmh[lab], "fof", f"자기 Mh / 무리 총 Mh")
            B.add(f"fof{en}_rank", rank, "fof", f"무리 안 M* 순위 (0 = 가장 무거움 = 중심 은하)")
            B.add(f"fof{en}_is_bcg", (rank == 0) & multi, "fof", f"다수 은하 무리의 최대 은하")
            B.add(f"fof{en}_dcen_over_rad", gm(dcen / np.maximum(rad[lab], 1e-3)), "fof", f"무리 질량중심까지 거리 / 무리 rms 반경")
            B.add(f"fof{en}_log_rad", gm(_safe_log10(rad[lab])), "fof", f"무리 rms 반경 (log)")
            B.add(f"fof{en}_log_vdisp", gm(_safe_log10(vdisp[lab])), "fof", f"무리 1D 속도분산 (log)")
            B.add(f"fof{en}_vrel_over_sig", gm(np.linalg.norm(V - vg[lab], axis=1) / np.maximum(vdisp[lab], 1.0)), "fof",
                  f"무리 평균 속도 대비 특이속도 / 무리 속도분산")
            B.add(f"fof{en}_dms_bcg", ms[jb] - ms, "fof", f"무리 최대 은하 log M* − 자기")
            del A, lab, order, ls, rel, rank
        B.log("fof")
    del balls

    # ---------------- nearest heavier-halo neighbour (Mh 기준, kNN + 0.1 dex 문턱 트리 보완)
    if B.want("heavier") and D is not None:
        mask = mh[I] > mh[:, None]
        has = mask.any(axis=1)
        first = mask.argmax(axis=1)
        jv = I[ar, first].astype(np.int64)
        dvv = D[ar, first].astype(np.float64)
        fb = np.flatnonzero(~has)
        exact = has.copy()
        if fb.size:
            thr = np.ceil((mh[fb] + 1e-6) * 10.0) / 10.0
            for T in np.unique(thr):
                sel = fb[thr == T]
                idx = np.flatnonzero(mh >= T)
                if idx.size == 0:
                    jv[sel], dvv[sel] = -1, np.nan
                    continue
                tt = cKDTree(P[idx], boxsize=L)
                ddd, iii = tt.query(P[sel], k=1, workers=n_jobs)
                jv[sel], dvv[sel] = idx[iii], ddd
        found = jv >= 0
        jj = np.where(found, jv, 0)
        dvec = _min_image(P - P[jj], L)
        rrh = np.maximum(np.linalg.norm(dvec, axis=1), 1e-6)
        dvh = V - V[jj]
        fv = lambda a: np.where(found, a, np.nan)  # noqa: E731
        B.add("hv_log_d", fv(np.log10(np.maximum(dvv, 1e-4))), "heavier", "자기보다 Mh가 무거운 가장 가까운 은하까지 거리 (log)")
        B.add("hv_logmh", fv(mh[jj]), "heavier", "그 무거운 이웃의 log Mh (어떤 규모의 무리인가)")
        B.add("hv_dmh", fv(mh[jj] - mh), "heavier", "그 이웃 log Mh − 자기 log Mh")
        B.add("hv_dms", fv(ms[jj] - ms), "heavier", "그 이웃 log M* − 자기 log M*")
        B.add("hv_log_dv", fv(_safe_log10(np.linalg.norm(dvh, axis=1))), "heavier", "그 이웃과의 속도 차 크기 (log)")
        B.add("hv_vr", fv((dvh * dvec).sum(1) / rrh), "heavier", "그 이웃 대비 반경 속도 (km/s, − 접근)")
        B.add("hv_log_x", fv(np.log10(np.maximum(dvv, 1e-4) / r200[jj])), "heavier", "거리 / 그 이웃의 R200 (log)")
        B.add("hv_exact", exact, "heavier", "128 최근접 안에서 찾음(1) / 0.1 dex 문턱 트리 근사(0)")
        for r in (0.5, 1.0, 2.0):
            rn = str(r).replace(".", "p")
            B.add(f"hv_localmax_r{rn}", ~found | (dvv > r), "heavier", f"반경 {r} 안에 더 무거운 은하 없음 (중심 은하 판정)")
        B.log("heavier")

    # ---------------- gravitational influence of neighbours: A = M/r², T = M/r³ (kNN, softening 0.01)
    if B.want("influence") and D is not None:
        kk = min(128, I.shape[1])
        Dd = np.maximum(D[:, :kk].astype(np.float64), 0.01)
        for mname, mlog in (("ms", ms), ("mh", mh)):
            Mn = np.power(10.0, mlog[I[:, :kk]])
            A = Mn / Dd ** 2
            T = Mn / Dd ** 3
            for k in (8, 32, 128):
                if k <= kk:
                    B.add(f"inf_{mname}_logA_sum{k}", np.log10(A[:, :k].sum(1)), "influence",
                          f"Σ {mname}/r² (가까운 {k}개, 가속도 대리지표)")
                    B.add(f"inf_{mname}_logT_sum{k}", np.log10(T[:, :k].sum(1)), "influence",
                          f"Σ {mname}/r³ (가까운 {k}개, 조석 세기 대리지표)")
            for sel_name, W in (("Amax", A), ("Tmax", T)):
                kb = np.argmax(W, axis=1)
                j = I[ar, kb].astype(np.int64)
                r = D[ar, kb].astype(np.float64)
                dvec = _min_image(P[j] - P, L)                     # i → j
                rr = np.maximum(np.linalg.norm(dvec, axis=1), 1e-6)
                dv = V[j] - V
                vrel = np.linalg.norm(dv, axis=1)
                vr_ = (dv * dvec).sum(1) / rr
                pre = f"inf_{mname}_{sel_name}"
                B.add(f"{pre}_logW", np.log10(W[ar, kb]), "influence", f"가까운 {kk}개 중 최대 {sel_name[0]} ({mname} 기준)")
                B.add(f"{pre}_logr", np.log10(np.maximum(r, 1e-4)), "influence", f"{sel_name} 이웃까지 거리 (log)")
                B.add(f"{pre}_dms", ms[j] - ms, "influence", f"{sel_name} 이웃 log M* − 자기 (질량비)")
                B.add(f"{pre}_dmh", mh[j] - mh, "influence", f"{sel_name} 이웃 log Mh − 자기")
                B.add(f"{pre}_vrel", vrel, "influence", f"{sel_name} 이웃과의 상대속도 크기 ‖Δv‖")
                B.add(f"{pre}_vr", vr_, "influence", f"{sel_name} 이웃 방향 상대속도 (− 접근)")
                B.add(f"{pre}_vt", np.sqrt(np.maximum(vrel ** 2 - vr_ ** 2, 0.0)), "influence", f"{sel_name} 이웃 수직 상대속도")
                B.add(f"{pre}_log_x", np.log10(np.maximum(r, 1e-4) / r200[j]), "influence", f"{sel_name} 이웃까지 거리 / 그 이웃의 R200")
                B.add(f"{pre}_shm", shm[j], "influence", f"{sel_name} 이웃의 log(M*/Mh)")
            del Mn, A, T
        del Dd
        B.log("influence")

    # ---------------- local shape of the neighbour distribution (inertia tensor)
    if B.want("shape") and D is not None:
        for k in (10, 30):
            rel = _min_image(P[I[:, :k]] - P[:, None, :], L)          # n×k×3
            S = np.einsum("nki,nkj->nij", rel, rel) / k
            w, vec = np.linalg.eigh(S)                                 # ascending
            l3, l2, l1 = np.maximum(w[:, 0], 1e-12), np.maximum(w[:, 1], 1e-12), np.maximum(w[:, 2], 1e-12)
            B.add(f"sh{k}_c_over_a", np.sqrt(l3 / l1), "shape", f"이웃 {k}개 분포 축비 c/a (작으면 판·필라멘트)")
            B.add(f"sh{k}_b_over_a", np.sqrt(l2 / l1), "shape", f"이웃 {k}개 분포 축비 b/a (작으면 필라멘트)")
            B.add(f"sh{k}_triax", (l1 - l2) / np.maximum(l1 - l3, 1e-12), "shape", f"삼축성 T (1에 가까우면 길쭉)")
            centroid = rel.mean(1)
            B.add(f"sh{k}_offset", np.linalg.norm(centroid, axis=1) / np.sqrt(l1 + l2 + l3), "shape",
                  f"이웃 무게중심이 자기에서 벗어난 정도 (가장자리에 있으면 큼)")
            vflow = V - V[I[:, :k]].mean(1)
            major = vec[:, :, 2]
            B.add(f"sh{k}_v_align", np.abs((vflow * major).sum(1)) / np.maximum(np.linalg.norm(vflow, axis=1), 1e-6), "shape",
                  f"국소 흐름 대비 속도와 이웃 분포 장축의 정렬 |cos| (필라멘트 따라 흐르나)")
            del rel, S, w, vec
        B.log("shape")

    # ---------------- 6D phase-space density (position + scaled velocity)
    if B.want("phase6d"):
        for H in (100.0, 1000.0):
            vs = V / H
            off = -vs.min(axis=0) + 1.0
            Q = np.column_stack([P, vs + off])
            big = float(np.ceil((vs + off).max() * 4 + 10))
            box6 = np.array([L, L, L, big, big, big])
            t6 = cKDTree(Q, boxsize=box6)
            d6, _ = t6.query(Q, k=21, workers=n_jobs)
            hn_ = str(int(H))
            for k in (5, 20):
                B.add(f"ps6_h{hn_}_lrho{k}", -6.0 * np.log10(np.maximum(d6[:, k], 1e-6)), "phase6d",
                      f"위치+속도/{int(H)} 6차원 공간 k={k} 이웃 밀도 (−6 log d_k) — 묶인 무리 탐지")
            del t6, d6, Q
        B.log("phase6d")

    # ---------------- explicit threshold flags (trees don't need them; NN/linear models do)
    if B.want("thresh"):
        for t in (1.4, 1.5, 1.6, 1.7, 1.8):
            B.add(f"dense_od_gt{str(t).replace('.', 'p')}", od > t, "thresh", f"overdensity 열 > {t} (SFR=0 비율이 계단처럼 뛰는 구간)")
        for t in (11.0, 11.5):
            B.add(f"massive_ms_gt{str(t).replace('.', 'p')}", ms > t, "thresh", f"log M* > {t} (분산 대부분을 차지하는 거대 은하)")
        B.log("thresh")

    # ---------------- FFT grid fields
    need_grid = B.want("grid") or B.want("tidal")
    grid_feats = {}
    if need_grid:
        N = int(grid_n)
        cic = _cic_weights(P, N, L)
        KX, KY, KZ, K2 = _kgrid(N, L)
        fields = {
            "n": np.ones(n),
            "ms": 10 ** ms,
            "mh": 10 ** mh,
        }
        dk_cache = {}
        for fname, w in fields.items():
            g = _cic_deposit(cic, w, N)
            delta = g / g.mean() - 1.0
            dk = np.fft.rfftn(delta)
            dk_cache[fname] = dk
            sigmas = grid_sigmas if fname != "ms" else tuple(s for s in grid_sigmas if s >= 2.0)
            if B.want("grid"):
                for s in sigmas:
                    sm = np.fft.irfftn(dk * np.exp(-0.5 * K2 * s * s), s=(N, N, N))
                    val = _cic_read(cic, sm)
                    sn = str(s).replace(".", "p")
                    grid_feats[(fname, s)] = val
                    B.add(f"grid_{fname}_s{sn}", np.log10(np.maximum(1.0 + val, 1e-3)), "grid",
                          f"{'수' if fname == 'n' else fname} 밀도장, Gaussian σ={s} h^-1 Mpc, log(1+δ)")
            del g, delta
        if B.want("grid"):
            for a, b in ((1.0, 4.0), (2.0, 8.0)):
                if ("mh", a) in grid_feats and ("mh", b) in grid_feats:
                    an, bn = str(a).replace(".", "p"), str(b).replace(".", "p")
                    B.add(f"grid_mh_contrast_{an}_{bn}",
                          np.log10(np.maximum(1 + grid_feats[("mh", a)], 1e-3)) - np.log10(np.maximum(1 + grid_feats[("mh", b)], 1e-3)),
                          "grid", f"질량 밀도 대비 σ={a} 대 σ={b}")
        B.log("grid")

        if B.want("tidal"):
            K2s = K2.copy()
            K2s[0, 0, 0] = 1.0
            comps = ((KX, KX), (KY, KY), (KZ, KZ), (KX, KY), (KX, KZ), (KY, KZ))
            for s in tidal_sigmas:
                dks = dk_cache["mh"] * np.exp(-0.5 * K2 * s * s)
                T = np.empty((n, 6))
                for c, (ka, kb_) in enumerate(comps):
                    tk = ka * kb_ / K2s * dks
                    tk[0, 0, 0] = 0.0
                    T[:, c] = _cic_read(cic, np.fft.irfftn(tk, s=(N, N, N)))
                M = np.empty((n, 3, 3))
                M[:, 0, 0], M[:, 1, 1], M[:, 2, 2] = T[:, 0], T[:, 1], T[:, 2]
                M[:, 0, 1] = M[:, 1, 0] = T[:, 3]
                M[:, 0, 2] = M[:, 2, 0] = T[:, 4]
                M[:, 1, 2] = M[:, 2, 1] = T[:, 5]
                lam = np.linalg.eigvalsh(M)          # ascending: λ3 ≤ λ2 ≤ λ1
                l3, l2, l1 = lam[:, 0], lam[:, 1], lam[:, 2]
                tr = l1 + l2 + l3
                absum = np.abs(l1) + np.abs(l2) + np.abs(l3) + 1e-6
                sn = str(s).replace(".", "p")
                B.add(f"tid_l1_s{sn}", l1, "tidal", f"tidal tensor 최대 고유값 (σ={s})")
                B.add(f"tid_l2_s{sn}", l2, "tidal", f"tidal tensor 중간 고유값 (σ={s})")
                B.add(f"tid_l3_s{sn}", l3, "tidal", f"tidal tensor 최소 고유값 (σ={s})")
                B.add(f"tid_shear_s{sn}", np.sqrt(0.5 * ((l1 - l2) ** 2 + (l1 - l3) ** 2 + (l2 - l3) ** 2)), "tidal",
                      f"전단 크기 (traceless 부분, σ={s})")
                B.add(f"tid_ellip_s{sn}", (l1 - l3) / (2 * absum), "tidal", f"타원율 e (σ={s})")
                B.add(f"tid_prolate_s{sn}", (l1 + l3 - 2 * l2) / (2 * absum), "tidal", f"prolateness p (σ={s})")
                for th in (0.0, 0.5):
                    tn = str(th).replace(".", "p")
                    B.add(f"tweb_s{sn}_th{tn}", ((l1 > th).astype(int) + (l2 > th) + (l3 > th)).astype(np.int8), "tidal",
                          f"T-web 분류 (λ>{th}인 고유값 수: 0 void, 1 sheet, 2 filament, 3 knot; σ={s})")
                B.add(f"tid_l1_over_trace_s{sn}", l1 / (np.abs(tr) + 1e-3), "tidal", f"λ1 / |trace| (σ={s})")
                del M, T, lam
            B.log("tidal")
        del dk_cache, cic

    # ---------------- conditional residuals (fixed M* / fixed Mh)
    if B.want("residual"):
        cond_ms = [
            ("mh", mh), ("bh", bh), ("mprog", np.where(prog_missing, np.nan, mp)),
            ("dlog_ms", dlog_ms), ("dlog_mh", dlog_mh), ("od", od), ("log_speed", _safe_log10(speed)),
        ]
        if lrho:
            cond_ms += [("lrho_5nn", lrho.get(5)), ("lrho_20nn", lrho.get(20))]
        if B.want("counts") and 1.0 in cnt:
            cond_ms.append(("log_cnt_r1", np.log1p(cnt[1.0])))
        if ("mh", 2.0) in grid_feats:
            cond_ms.append(("grid_mh_s2", np.log10(np.maximum(1 + grid_feats[("mh", 2.0)], 1e-3))))
        if "host_log_x" in B.cols:
            cond_ms.append(("host_log_x", B.cols["host_log_x"].astype(np.float64)))
        for name, v in cond_ms:
            if v is None:
                continue
            res, pct = _binned_median_residual(v, ms)
            B.add(f"{name}_res_ms", res, "residual", f"{name} − median({name} | M* bin) (고정 M* 잔차)")
            B.add(f"{name}_pct_ms", pct, "residual", f"고정 M* bin 안에서 {name}의 백분위")
        for name, v in (("ms", ms), ("bh", bh), ("od", od)):
            res, pct = _binned_median_residual(v, mh)
            B.add(f"{name}_res_mh", res, "residual", f"{name} − median({name} | Mh bin) (고정 Mh 잔차)")
            B.add(f"{name}_pct_mh", pct, "residual", f"고정 Mh bin 안에서 {name}의 백분위")
        res, _ = _binned_median_residual(dlog_ms, od)
        B.add("dlog_ms_res_od", res, "residual", "비율 성장 − median(· | overdensity bin)")
        B.log("residual")

    # ---------------- Peng+2010 separable quenching hazard (SDSS coefficients as prior)
    if B.want("peng"):
        h_env = np.power(10 ** od / 10 ** 1.84, 0.60)
        h_mass = np.power(10 ** (ms - 10.56), 0.80)
        B.add("peng_h_env", h_env, "peng", "환경 위험률 ((1+δ)/10^1.84)^0.60 (Peng+2010 SDSS 계수)")
        B.add("peng_h_mass", h_mass, "peng", "질량 위험률 (M*/10^10.56)^0.80")
        B.add("peng_fq", 1 - np.exp(-(h_env + h_mass)), "peng", "f_Q = 1 − exp(−h_env − h_mass)")
        B.add("peng_env_share", h_env / (h_env + h_mass), "peng", "환경 위험률 비중")
        B.log("peng")

    # ---------------- unsupervised structure
    if B.want("unsup"):
        from sklearn.cluster import MiniBatchKMeans
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler

        core = [ms, mh, bh, np.where(prog_missing, np.nan, mp), np.where(prog_missing, np.nan, msp),
                mr, od, shm, bh_ms, dlog_ms, dlog_mh, _safe_log10(speed)]
        if lrho:
            core += [lrho[k] for k in (5, 20) if k in lrho]
        Z = np.column_stack(core)
        med = np.nanmedian(Z, axis=0)
        Z = np.where(np.isfinite(Z), Z, med)
        Z = StandardScaler().fit_transform(Z)
        pcs = PCA(n_components=6, random_state=random_state).fit_transform(Z)
        for c in range(pcs.shape[1]):
            B.add(f"pca_{c + 1}", pcs[:, c], "unsup", f"물리 피처 표준화 후 PCA 성분 {c + 1}")
        for k in (8, 16, 32):
            km = MiniBatchKMeans(n_clusters=k, random_state=random_state, batch_size=4096, n_init=3).fit(Z)
            lab = km.labels_
            B.add(f"kmeans{k}", pd.Categorical(lab.astype(str)), "unsup", f"MiniBatchKMeans k={k} 군집 라벨 (범주형)")
            if k == 16:
                dist = np.linalg.norm(Z - km.cluster_centers_[lab], axis=1)
                B.add("kmeans16_dist", dist, "unsup", "k=16 군집 중심까지 거리 (전형성)")
        B.log("unsup")

    FE, DOCS = B.frame()
    B.log(f"done: {FE.shape[1]} features")
    return FE, DOCS


def align_to_split(FE: pd.DataFrame, X_train: pd.DataFrame, X_val: pd.DataFrame):
    """템플릿 locked split의 index에 맞춰 피처 표를 나눈다."""
    return FE.loc[X_train.index], FE.loc[X_val.index]


def feature_group_map(DOCS: pd.DataFrame) -> dict[str, list[str]]:
    """{group: [feature, ...]} — 그룹 단위 ablation용."""
    return DOCS.groupby("group")["feature"].apply(list).to_dict()


def prune_correlated(FE: pd.DataFrame, threshold: float = 0.98, sample: int = 30000,
                     random_state: int = 0) -> list[str]:
    """순위 상관 |Spearman| ≥ threshold인 열을 앞쪽 열 기준으로 하나만 남긴다 (타깃 미사용).

    열 순서가 우선순위다: build_features는 raw → flags → mass → growth → … 순서로 만들므로
    해석하기 쉬운 기본 피처가 남고 뒤쪽 다중 척도 변형이 빠진다. 범주형 열은 그대로 둔다.
    threshold=0.9999 → 트리 모델에 완전히 같은 정보만 제거, 0.98 → 약 350개, 0.95 → 약 310개.
    """
    num = FE.select_dtypes(include=[np.number])
    S = num.sample(min(sample, len(num)), random_state=random_state).rank(pct=True)
    C = S.corr().abs().to_numpy()
    C = np.nan_to_num(C, nan=0.0)
    kept: list[int] = []
    for i in range(C.shape[0]):
        if not any(C[i, k] >= threshold for k in kept):
            kept.append(i)
    keep_num = [num.columns[i] for i in kept]
    return [c for c in FE.columns if c in set(keep_num) or c not in num.columns]
