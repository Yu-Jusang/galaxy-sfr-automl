"""Apply docs/log_metric_selection_rule.md: B3 recalibration, D ensembles, 1-SE choice, nested estimate, TT bias.

CV only (training split, the notebook's 5 folds). The validation split is not read.
Inputs: results/oof/*.parquet (A, 139 stored candidates), results/log_metric/oof_B/*.parquet (B1, B2).
Outputs: results/log_metric/selection_<metric>.csv, results/log_metric/selection_summary.json
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "log_metric"
EPS = 1e-3
K = 5

# ---------------------------------------------------------------- candidates
ref = pd.read_parquet(ROOT / "results" / "oof" / "M01_lgbm_tw13_F5.parquet")
order = ref["row"].to_numpy(); y = ref["y"].to_numpy(float); fold = ref["fold"].to_numpy()


def load(path):
    d = pd.read_parquet(path).set_index("row").loc[order]
    assert np.allclose(d["y"].to_numpy(float), y) and np.array_equal(d["fold"].to_numpy(), fold), path.name
    return np.clip(np.nan_to_num(d["oof"].to_numpy(float), nan=0.0, posinf=1e12), 0, None)


A = {p.stem: load(p) for p in sorted((ROOT / "results" / "oof").glob("*.parquet"))}
B = {p.stem: load(p) for p in sorted((OUT / "oof_B").glob("*.parquet"))}
stage = {n: json.loads((ROOT / "results" / "meta" / f"{n}.json").read_text(encoding="utf-8"))["stage"] for n in A}
assert set(A) == set(stage), set(A) ^ set(stage)
TIER_A = {"single": 1, "tuned": 1, "selected": 1, "autogluon": 4, "ensemble": 5}
tier = {n: TIER_A[stage[n]] for n in A}
tier.update({n: (1 if n.startswith("B1") else 2) for n in B})

G = {"D1": (lambda v: np.log10(np.clip(v, EPS, None)), lambda z: 10.0 ** z, lambda v: v > 0),
     "D2": (lambda v: np.log10(v + EPS), lambda z: np.maximum(10.0 ** z - EPS, 0.0), lambda v: np.ones(len(v), bool)),
     "D3": (lambda v: np.log10(1.0 + v), lambda z: np.maximum(10.0 ** z - 1.0, 0.0), lambda v: np.ones(len(v), bool)),
     "DS": (lambda v: np.log10(np.maximum(np.clip(v, 0, None), 1e-13)), lambda z: 10.0 ** z,
            lambda v: np.ones(len(v), bool))}   # organisers' metric (primary)


def fold_scores(pred, dn):
    g, _, sel = G[dn]
    return np.array([np.sqrt(np.mean((g(pred[(fold == k) & sel(y)]) - g(y[(fold == k) & sel(y)])) ** 2))
                     for k in range(K)])


def recalibrate(pred, dn):
    """B3: isotonic regression of g(SFR) on g(prediction), fitted on the other folds (scored galaxies only)."""
    g, ginv, sel = G[dn]
    out = np.zeros(len(y)); s = sel(y)
    for k in range(K):
        tr = (fold != k) & s
        iso = IsotonicRegression(out_of_bounds="clip").fit(g(pred[tr]), g(y[tr]))
        out[fold == k] = ginv(iso.predict(g(pred[fold == k])))
    return out


def ensembles(pool, scores, dn):
    """D: equal-weight average in g-space of the top-k candidates ranked on the other folds (k = 2, 3)."""
    g, ginv, _ = G[dn]
    names = list(pool); S = np.array([scores[n] for n in names])
    out = {f"D_top{kk}_{dn}": np.zeros(len(y)) for kk in (2, 3)}
    members = {kk: [] for kk in (2, 3)}
    for k in range(K):
        rank = np.argsort(S[:, [j for j in range(K) if j != k]].mean(1))
        for kk in (2, 3):
            top = [names[i] for i in rank[:kk]]; members[kk].append(top)
            out[f"D_top{kk}_{dn}"][fold == k] = ginv(np.mean([g(pool[n][fold == k]) for n in top], axis=0))
    return out, members


def choose(scores, folds_used):
    """1-SE rule on the given folds: lowest CV, then the simplest within one SE of it."""
    names = list(scores); M = np.array([scores[n][folds_used].mean() for n in names])
    best = names[int(M.argmin())]
    ok = []
    for n, m in zip(names, M):
        d = scores[n][folds_used] - scores[best][folds_used]
        if d.mean() <= d.std(ddof=1) / np.sqrt(len(folds_used)):
            ok.append((tier[n], m, n))
    return best, min(ok)[2], len(ok)


summary = {}
for dn in G:
    pool = {**A, **B}
    for n in A:
        pool[f"B3_{n}_{dn}"] = recalibrate(A[n], dn); tier[f"B3_{n}_{dn}"] = 3
    scores = {n: fold_scores(p, dn) for n, p in pool.items()}
    ens, members = ensembles(pool, scores, dn)
    for n, p in ens.items():
        pool[n] = p; scores[n] = fold_scores(p, dn); tier[n] = 5
    best, chosen, n_within = choose(scores, np.arange(K))
    nested = []
    for k in range(K):
        _, ch_k, _ = choose(scores, np.array([j for j in range(K) if j != k]))
        nested.append({"held_out_fold": k, "chosen": ch_k, "score": float(scores[ch_k][k])})
    best_by_fold = np.array([min(s[k] for s in scores.values()) for k in range(K)])
    tt_bias = float(np.mean(scores[best] - best_by_fold))
    tab = pd.DataFrame([{"name": n, "tier": tier[n], "cv": s.mean(), "d_vs_best": (s - scores[best]).mean(),
                         "se_vs_best": (s - scores[best]).std(ddof=1) / np.sqrt(K),
                         "d_vs_R0": (s - scores["M01_lgbm_tw13_F5"]).mean(),
                         "se_vs_R0": (s - scores["M01_lgbm_tw13_F5"]).std(ddof=1) / np.sqrt(K)}
                        for n, s in scores.items()]).sort_values("cv")
    tab.to_csv(OUT / f"selection_{dn}.csv", index=False)
    bestA = tab[tab.name.isin(A)].iloc[0]; bestB = tab[tab.name.str.startswith(("B1_", "B2_", "B3_"))].iloc[0]
    summary[dn] = {"n_candidates": len(scores), "best": best, "best_cv": float(scores[best].mean()),
                   "chosen_1se": chosen, "chosen_cv": float(scores[chosen].mean()), "n_within_1se": n_within,
                   "nested": nested, "nested_mean": float(np.mean([r["score"] for r in nested])),
                   "tt_bias": tt_bias, "tt_corrected": float(scores[best].mean() + tt_bias),
                   "best_A": [bestA["name"], float(bestA.cv)], "best_B": [bestB["name"], float(bestB.cv)],
                   "R0_cv": float(scores["M01_lgbm_tw13_F5"].mean()),
                   "D_members_top3": members[3]}
    pd.set_option("display.width", 220)
    print(f"\n=== {dn}: {len(scores)} candidates | best {best} {scores[best].mean():.4f} | 1-SE choice {chosen} "
          f"{scores[chosen].mean():.4f} ({n_within} within 1 SE) | nested {summary[dn]['nested_mean']:.4f} | "
          f"TT-corrected {summary[dn]['tt_corrected']:.4f} | R0 {summary[dn]['R0_cv']:.4f}")
    print(tab.head(15).round(4).to_string(index=False))
    print("nested choices:", [r["chosen"] for r in nested])
    print("best A:", summary[dn]["best_A"], "| best B:", summary[dn]["best_B"])
(OUT / "selection_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
