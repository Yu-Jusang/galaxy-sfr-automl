"""Stage 4: nested Optuna tuning of model AND domain (feature-construction) hyper-parameters.

Like tuning a decay half-life in the KCB project, the physical choices made in Part I become tunable:
  - Tweedie power p                                  (E1: zero mass + heavy tail)
  - neighbour-count radii, k of the k-th neighbour distance, central-proxy radius   (E-5: environment, satellites)
  - an overdensity step flag  log10(1+delta) > t     (E-5: zero fraction jumps at 1.5-1.7)
  - optional extra environment blocks (E8 host phase space / multi-scale)
  - extra weight on massive galaxies                 (E-2: log M* > 11.5 holds 82 % of the variance)
plus the usual model hyper-parameters. Nested: for each outer fold, Optuna (TPE, seed 42) searches on a stratified
3-fold CV inside the outer training part only; the best setting is refitted there and scored on the outer fold.
usage: python scripts/a07_tune.py <lgbm|xgb|cat> [n_trials]
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import optuna  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402

from core import A1_COLS, A2_COLS, FS, L, N_THREADS, ROOT, SEED, done, np, pd, save_exp, set_status, setup  # noqa: E402

optuna.logging.set_verbosity(optuna.logging.WARNING)
GRID_PATH = ROOT / "data" / "env_grid.parquet"
R_COUNT = (0.5, 1.0, 1.5, 2.0, 3.0)
K_DIST = (1, 2, 3, 5, 10, 20, 50)
R_CENTRAL = (0.5, 1.0, 2.0)
K_SETS = {"1-5-10": (1, 5, 10), "1-3-10": (1, 3, 10), "2-10-20": (2, 10, 20), "1-10-50": (1, 10, 50)}
CORE = [c for c in FS["F5"] if not c.startswith("env_")]               # 16 catalogue + physics columns
ENV_FIXED = ["env_d_heavier", "env_logMh_heavier", "env_dv_heavier"]


def env_grid(df):
    """Environment features for every candidate radius / k (inputs only, periodic box 200), cached."""
    if GRID_PATH.exists():
        return pd.read_parquet(GRID_PATH)
    pos = np.mod(df[L.POS_COLS].to_numpy(float), L.BOX_SIZE)
    tree = cKDTree(pos, boxsize=L.BOX_SIZE)
    out = pd.DataFrame(index=df.index)
    d, _ = tree.query(pos, k=max(K_DIST) + 1, workers=N_THREADS)
    for k in K_DIST:
        out[f"g_d{k}"] = d[:, k]
    for r in R_COUNT:
        out[f"g_n_r{r}"] = tree.query_ball_point(pos, r, return_length=True, workers=N_THREADS) - 1
    out.to_parquet(GRID_PATH)
    return out


def make_frame(G, p, rows):
    """Feature matrix for the given rows under domain parameters p (dict from a trial)."""
    F, E = G["F"], G["E"]
    cols = [F.loc[rows, CORE + ENV_FIXED]]
    ks = K_SETS[p["k_set"]]
    cols.append(E.loc[rows, [f"g_d{k}" for k in ks]])
    cols.append(E.loc[rows, [f"g_n_r{p['r_small']}", f"g_n_r{p['r_large']}"]])
    extra = pd.DataFrame(index=rows)
    extra["central"] = (F.loc[rows, "env_d_heavier"] > p["r_central"]).astype(float)
    if p["dense_t"] > 0:
        extra["dense"] = (F.loc[rows, "log10_overdensity_plus_1"] > p["dense_t"]).astype(float)
    cols.append(extra)
    if p["extra_env"] == "A2":
        cols.append(F.loc[rows, A2_COLS])
    elif p["extra_env"] == "A1r1+A2":
        cols.append(F.loc[rows, [c for c in A1_COLS if c.endswith("r1p0")] + A2_COLS])
    return pd.concat(cols, axis=1).to_numpy(float)


def weights(G, p, rows):
    lms = G["F"].loc[rows, "log10_stellar_mass"].to_numpy()
    return np.where(lms > p["w_cut"], p["w_massive"], 1.0)


def suggest(trial, family):
    p = {"power": trial.suggest_float("power", 1.1, 1.7),
         "k_set": trial.suggest_categorical("k_set", list(K_SETS)),
         "r_small": trial.suggest_categorical("r_small", [0.5, 1.0]),
         "r_large": trial.suggest_categorical("r_large", [1.5, 2.0, 3.0]),
         "r_central": trial.suggest_categorical("r_central", list(R_CENTRAL)),
         "dense_t": trial.suggest_categorical("dense_t", [0.0, 1.4, 1.5, 1.6, 1.7, 1.8]),
         "extra_env": trial.suggest_categorical("extra_env", ["none", "A2", "A1r1+A2"]),
         "w_cut": trial.suggest_categorical("w_cut", [11.0, 11.5]),
         "w_massive": trial.suggest_float("w_massive", 1.0, 3.0)}
    if family == "lgbm":
        p.update(num_leaves=trial.suggest_int("num_leaves", 15, 255, log=True),
                 min_child_samples=trial.suggest_int("min_child_samples", 5, 200, log=True),
                 learning_rate=trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
                 colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
                 subsample=trial.suggest_float("subsample", 0.5, 1.0),
                 reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 30.0, log=True))
    elif family == "xgb":
        p.update(max_leaves=trial.suggest_int("max_leaves", 15, 255, log=True),
                 min_child_weight=trial.suggest_float("min_child_weight", 0.5, 50.0, log=True),
                 learning_rate=trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
                 colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
                 subsample=trial.suggest_float("subsample", 0.5, 1.0),
                 reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 30.0, log=True))
    else:
        p.update(depth=trial.suggest_int("depth", 4, 9),
                 learning_rate=trial.suggest_float("learning_rate", 0.03, 0.15, log=True),
                 l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 0.5, 30.0, log=True),
                 rsm=trial.suggest_float("rsm", 0.5, 1.0))
    return p


def fit_predict(family, p, X, y, w, Xq, seed=SEED):
    tr, es = L._es_split(y, seed)
    if family == "lgbm":
        import lightgbm as lgb
        m = lgb.LGBMRegressor(objective="tweedie", tweedie_variance_power=p["power"], n_estimators=3000,
                              num_leaves=p["num_leaves"], min_child_samples=p["min_child_samples"],
                              learning_rate=p["learning_rate"], colsample_bytree=p["colsample_bytree"],
                              subsample=p["subsample"], subsample_freq=1, reg_lambda=p["reg_lambda"],
                              random_state=seed, n_jobs=N_THREADS, verbose=-1)
        m.fit(X[tr], y[tr], sample_weight=w[tr], eval_set=[(X[es], y[es])],
              callbacks=[lgb.early_stopping(30, verbose=False)])
        return m.predict(Xq)
    if family == "xgb":
        import xgboost as xgb
        m = xgb.XGBRegressor(objective="reg:tweedie", tweedie_variance_power=p["power"], tree_method="hist",
                             grow_policy="lossguide", max_depth=0, max_leaves=p["max_leaves"],
                             min_child_weight=p["min_child_weight"], learning_rate=p["learning_rate"],
                             colsample_bytree=p["colsample_bytree"], subsample=p["subsample"], reg_lambda=p["reg_lambda"],
                             n_estimators=3000, early_stopping_rounds=30, random_state=seed, n_jobs=N_THREADS, verbosity=0)
        m.fit(X[tr], y[tr], sample_weight=w[tr], eval_set=[(X[es], y[es])], verbose=False)
        return m.predict(Xq, iteration_range=(0, m.best_iteration + 1))
    from catboost import CatBoostRegressor
    m = CatBoostRegressor(loss_function=f"Tweedie:variance_power={p['power']}", depth=p["depth"],
                          learning_rate=p["learning_rate"], l2_leaf_reg=p["l2_leaf_reg"], rsm=p["rsm"], iterations=4000,
                          od_type="Iter", od_wait=30, use_best_model=True, random_seed=seed, thread_count=N_THREADS,
                          verbose=False, allow_writing_files=False)
    m.fit(X[tr], y[tr], sample_weight=w[tr], eval_set=(X[es], y[es]))
    return m.predict(Xq, prediction_type="Exponent")


def tune(G, family, rows, n_trials, tag):
    """Optuna search on a stratified 3-fold CV over `rows` (index labels of training galaxies)."""
    y = G["y"].loc[rows].to_numpy(float)
    lab = L.strata_labels(y)
    inner = list(StratifiedKFold(3, shuffle=True, random_state=SEED).split(np.zeros(len(y)), lab))

    def objective(trial):
        p = suggest(trial, family)
        X = make_frame(G, p, rows); w = weights(G, p, rows)
        errs = []
        for j, (a, b) in enumerate(inner):
            pred = fit_predict(family, p, X[a], y[a], w[a], X[b])
            errs.append(np.sqrt(np.mean((pred - y[b]) ** 2)))
            trial.report(float(np.mean(errs)), j)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return float(np.mean(errs))

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED),
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=1))
    # start from the original final model's choices so the search can only add information
    study.enqueue_trial({"power": 1.3, "k_set": "1-5-10", "r_small": 1.0, "r_large": 2.0, "r_central": 1.0,
                         "dense_t": 0.0, "extra_env": "none", "w_cut": 11.5, "w_massive": 1.0,
                         **({"num_leaves": 31, "min_child_samples": 20, "learning_rate": 0.05, "colsample_bytree": 1.0,
                             "subsample": 1.0, "reg_lambda": 0.001} if family == "lgbm" else {})})
    study.optimize(objective, n_trials=n_trials)
    best = {**suggest(optuna.trial.FixedTrial(study.best_params), family)}
    print(f"  [{tag}] best inner CV {study.best_value:.4f} after {len(study.trials)} trials: {study.best_params}", flush=True)
    return best, float(study.best_value)


def main(family, n_trials):
    name = {"lgbm": "T01_lgbm_tuned", "xgb": "T02_xgb_tuned", "cat": "T03_cat_tuned"}[family]
    if done(name):
        return
    G = setup()
    G["E"] = env_grid(G["df"])
    G["y"] = pd.concat([G["y_train"], G["y_val"]])
    ytr, yva, folds = G["y_train"], G["y_val"], G["folds"]
    rows_tr = ytr.index.to_numpy(); y = ytr.to_numpy(float)
    t0 = time.time()
    oof = np.zeros(len(y)); chosen = []
    for k in range(5):
        set_status("tuning", f"{name}: outer fold {k + 1}/5 ({n_trials} trials)")
        fit_rows, te = rows_tr[folds != k], folds == k
        p, inner = tune(G, family, fit_rows, n_trials, f"{name} fold {k}")
        X = make_frame(G, p, fit_rows)
        oof[te] = fit_predict(family, p, X, ytr.loc[fit_rows].to_numpy(float), weights(G, p, fit_rows),
                              make_frame(G, p, rows_tr[te]))
        chosen.append({"fold": k, "inner_cv": inner, **p})
    set_status("tuning", f"{name}: final search on the full training split")
    p, inner = tune(G, family, rows_tr, n_trials, f"{name} full")
    vp = fit_predict(family, p, make_frame(G, p, rows_tr), y, weights(G, p, rows_tr), make_frame(G, p, yva.index.to_numpy()))
    chosen.append({"fold": "full", "inner_cv": inner, **p})
    meta = save_exp(name, oof, vp, {"stage": "tuned", "features": "F5 + tuned env", "desc":
                                    f"{family} nested Optuna ({n_trials} trials/fold): model + domain parameters",
                                    "seconds": time.time() - t0, "chosen": chosen})
    pd.DataFrame(chosen).to_csv(ROOT / "results" / "meta" / f"{name}_chosen.csv", index=False)
    set_status("tuning", f"{name}: nested CV {meta['cv_rmse_mean']:.3f} ({(time.time() - t0) / 60:.0f} min)")
    print(f"{name}: nested CV {meta['cv_rmse_mean']:.4f} ± {meta['cv_rmse_sd']:.3f}  {(time.time() - t0) / 60:.0f} min", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 40)
