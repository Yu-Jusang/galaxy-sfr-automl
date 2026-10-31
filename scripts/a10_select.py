"""Stage 5: nested, label-based feature selection from a large pool (guards against over-fitting many features).

For each outer fold k (and once more on the whole training split for the validation predictions):
  1. inside the outer training part, a stratified 3-fold inner CV fits the R0 LightGBM (Tweedie 1.3) on
     pool + a shuffled "shadow" copy of every pool column (Boruta idea, Kursa & Rudnicki 2010) and measures
     mean |SHAP| on each inner hold-out;
  2. a feature is kept if its |SHAP| beats the most important shadow column in all 3 inner folds
     (the 25 F5 columns are always kept, so selection can only add to the original model);
  3. near-duplicates are removed (Spearman > 0.98 on a 20k-row sample; the more important one stays);
  4. at most MAX_KEEP features by importance;
  5. models are refitted on the outer training part with the selected columns and scored on fold k.
The outer fold never influences which features its own model uses, so the CV score of the selected set is honest.
The per-fold selections are saved, so how often each feature is chosen (stability) can be reported.
usage: python scripts/a10_select.py [pool ...]       pools: F5X (ours), FULL (ours + teammate 2, if present)
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sklearn.model_selection import StratifiedKFold  # noqa: E402

from core import FS, L, N_THREADS, RES, SEED, done, np, pd, save_exp, set_status, setup  # noqa: E402

MAX_KEEP = 120


def lgbm_fit(X, y, seed=SEED, **kw):
    import lightgbm as lgb
    p = dict(objective="tweedie", tweedie_variance_power=1.3, learning_rate=0.05, n_estimators=1000, num_leaves=31,
             min_child_samples=20, random_state=seed, n_jobs=N_THREADS, verbose=-1)
    p.update(kw)
    es_rounds = p.pop("es_rounds", 10)
    tr, es = L._es_split(y, seed)
    m = lgb.LGBMRegressor(**p)
    m.fit(X[tr], y[tr], eval_set=[(X[es], y[es])], callbacks=[lgb.early_stopping(es_rounds, verbose=False)])
    return m


def select(F, rows, y, pool, seed=SEED):
    """Return the selected column list using only `rows` (training galaxies) and their labels."""
    rng = np.random.default_rng(seed)
    X = F.loc[rows, pool].to_numpy(float)
    shadow = np.column_stack([rng.permutation(X[:, j]) for j in range(X.shape[1])])   # same marginals, no signal
    Xn = np.hstack([X, shadow])
    inner = StratifiedKFold(3, shuffle=True, random_state=seed).split(np.zeros(len(y)), L.strata_labels(y))
    wins = np.zeros(len(pool)); imp = np.zeros(len(pool))
    for a, b in inner:
        m = lgbm_fit(Xn[a], y[a], seed)
        bb = b if len(b) <= 20000 else rng.choice(b, 20000, replace=False)
        contrib = m.predict(Xn[bb], pred_contrib=True)[:, :-1]          # TreeSHAP in raw-score (log) units
        s = np.abs(contrib).mean(0)
        wins += s[:len(pool)] > s[len(pool):].max()
        imp += s[:len(pool)]
    keep = [c for c, w in zip(pool, wins) if w >= 3]
    order = {c: v for c, v in zip(pool, imp)}
    keep = sorted(set(keep) | set(FS["F5"]), key=lambda c: -order[c])
    # remove near-duplicates (rank correlation > 0.98), keeping the more important column
    sub = F.loc[rng.choice(rows, min(20000, len(rows)), replace=False), keep].rank().to_numpy()
    C = np.corrcoef(np.nan_to_num(sub, nan=0.0), rowvar=False)
    dropped = set()
    for i, c in enumerate(keep):
        if c in dropped:
            continue
        for j in range(i + 1, len(keep)):
            if keep[j] not in dropped and keep[j] not in FS["F5"] and abs(C[i, j]) > 0.98:
                dropped.add(keep[j])
    keep = [c for c in keep if c not in dropped]
    extra = [c for c in keep if c not in FS["F5"]][:max(0, MAX_KEEP - len(FS["F5"]))]
    return FS["F5"] + extra, {c: float(order[c]) for c in keep}


MODELS = {   # model fitted on the selected columns
    "lgbm": lambda X, y, Xq: lgbm_fit(X, y).predict(Xq),
    "lgbm_reg": lambda X, y, Xq: lgbm_fit(X, y, num_leaves=15, min_child_samples=100, colsample_bytree=0.5,
                                          subsample=0.8, subsample_freq=1, reg_lambda=5.0, learning_rate=0.03,
                                          n_estimators=3000, es_rounds=50).predict(Xq),
}


def xgb_pred(X, y, Xq):
    import xgboost as xgb
    tr, es = L._es_split(y, SEED)
    m = xgb.XGBRegressor(objective="reg:tweedie", tweedie_variance_power=1.3, tree_method="hist", grow_policy="lossguide",
                         max_leaves=31, max_depth=0, learning_rate=0.05, n_estimators=2000, early_stopping_rounds=20,
                         random_state=SEED, n_jobs=N_THREADS, verbosity=0)
    m.fit(X[tr], y[tr], eval_set=[(X[es], y[es])], verbose=False)
    return m.predict(Xq, iteration_range=(0, m.best_iteration + 1))


def cat_pred(X, y, Xq):
    from catboost import CatBoostRegressor
    tr, es = L._es_split(y, SEED)
    m = CatBoostRegressor(loss_function="Tweedie:variance_power=1.3", depth=8, learning_rate=0.05, iterations=5000,
                          od_type="Iter", od_wait=30, use_best_model=True, random_seed=SEED, thread_count=N_THREADS,
                          verbose=False, allow_writing_files=False)
    m.fit(X[tr], y[tr], eval_set=(X[es], y[es]))
    return m.predict(Xq, prediction_type="Exponent")


MODELS.update(xgb=xgb_pred, cat=cat_pred)


def run(pool_name):
    names = {k: f"SEL_{pool_name}_{k}" for k in MODELS}
    if all(done(n) for n in names.values()):
        return
    G = setup()
    if pool_name not in FS:
        print(f"pool {pool_name} not available (teammate module missing?) - skipped", flush=True)
        return
    F, ytr, yva, folds = G["F"], G["y_train"], G["y_val"], G["folds"]
    pool = FS[pool_name]
    rows = ytr.index.to_numpy(); y = ytr.to_numpy(float)
    t0 = time.time()
    oof = {k: np.zeros(len(y)) for k in MODELS}; chosen = []
    for k in range(5):
        set_status("selection", f"SEL_{pool_name}: outer fold {k + 1}/5 selecting from {len(pool)}")
        tr = folds != k; te = folds == k
        cols, imp = select(F, rows[tr], y[tr], pool)
        chosen.append({"fold": k, "n": len(cols), "cols": cols})
        X = F.loc[rows[tr], cols].to_numpy(float); Xq = F.loc[rows[te], cols].to_numpy(float)
        for mk, fn in MODELS.items():
            oof[mk][te] = fn(X, y[tr], Xq)
        print(f"  fold {k}: {len(cols)} columns kept ({time.time() - t0:.0f} s)", flush=True)
    cols, imp = select(F, rows, y, pool)
    chosen.append({"fold": "full", "n": len(cols), "cols": cols, "importance": imp})
    X = F.loc[rows, cols].to_numpy(float); Xv = F.loc[yva.index, cols].to_numpy(float)
    freq = pd.Series([c for ch in chosen[:5] for c in ch["cols"]]).value_counts()
    (RES / "meta" / f"SEL_{pool_name}_selection.json").write_text(json.dumps(
        {"pool": pool_name, "pool_size": len(pool), "per_fold": chosen, "times_selected_of_5": freq.to_dict()},
        indent=1, default=float))
    for mk, fn in MODELS.items():
        vp = fn(X, y, Xv)
        meta = save_exp(names[mk], np.clip(oof[mk], 0, None), np.clip(vp, 0, None),
                        {"stage": "selected", "features": f"SEL({pool_name})", "n_features": len(cols),
                         "desc": f"{mk} on columns selected inside each fold from {pool_name} ({len(pool)})",
                         "seconds": time.time() - t0})
        print(f"{names[mk]:24s} CV {meta['cv_rmse_mean']:.4f} ± {meta['cv_rmse_sd']:.3f}", flush=True)
    set_status("selection", f"SEL_{pool_name}: done, {len(cols)} columns on the full split")


if __name__ == "__main__":
    for p in [a for a in sys.argv[1:] if not a.startswith("--")] or ["F5X", "FULL"]:
        run(p)
