"""Stage 1: many single models under the same 5-fold CV (training split only).

For each config: 5 fold fits -> OOF; one refit on the full training split -> validation predictions (stored, unread).
usage: python scripts/a02_models.py [name ...]
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from core import FS, L, N_THREADS, SEED, done, np, save_exp, set_status, setup  # noqa: E402

TW = 1.3


def lgbm(obj="tweedie", **p):
    def fit(X, y, Xq, seed):
        import lightgbm as lgb
        params = dict(learning_rate=0.05, n_estimators=1000, num_leaves=31, min_child_samples=20)
        params.update(p)
        es_rounds = params.pop("es_rounds", 10)
        if obj == "tweedie":
            params["tweedie_variance_power"] = params.pop("power", TW)
        tr, es = L._es_split(y, seed)
        m = lgb.LGBMRegressor(objective=obj, random_state=seed, n_jobs=N_THREADS, verbose=-1, **params)
        m.fit(X[tr], y[tr], eval_set=[(X[es], y[es])], callbacks=[lgb.early_stopping(es_rounds, verbose=False)])
        return m.predict(Xq)
    return fit


def xgb_tw(**p):
    def fit(X, y, Xq, seed):
        import xgboost as xgb
        params = dict(objective="reg:tweedie", tweedie_variance_power=TW, tree_method="hist", grow_policy="lossguide",
                      max_leaves=31, max_depth=0, learning_rate=0.05, n_estimators=2000, early_stopping_rounds=20)
        params.update(p)
        tr, es = L._es_split(y, seed)
        m = xgb.XGBRegressor(random_state=seed, n_jobs=N_THREADS, verbosity=0, **params)
        m.fit(X[tr], y[tr], eval_set=[(X[es], y[es])], verbose=False)
        return m.predict(Xq, iteration_range=(0, m.best_iteration + 1))
    return fit


def cat(loss=f"Tweedie:variance_power={TW}", **p):
    def fit(X, y, Xq, seed):
        from catboost import CatBoostRegressor
        params = dict(depth=6, learning_rate=0.05, iterations=5000, od_type="Iter", od_wait=30)
        params.update(p)
        tr, es = L._es_split(y, seed)
        m = CatBoostRegressor(loss_function=loss, use_best_model=True, random_seed=seed, thread_count=N_THREADS,
                              verbose=False, allow_writing_files=False, **params)
        m.fit(X[tr], y[tr], eval_set=(X[es], y[es]))
        if loss.startswith("Tweedie") or loss.startswith("Poisson"):
            return m.predict(Xq, prediction_type="Exponent")
        return m.predict(Xq)
    return fit


def sk(make):
    def fit(X, y, Xq, seed):
        return make(seed).fit(np.nan_to_num(X), y).predict(np.nan_to_num(Xq))
    return fit


def hgb_poisson(X, y, Xq, seed):
    return L.SFRModel(L.ModelSpec("hgb", "poisson"), seed=seed).fit(X, y).predict(Xq)


def lgbm_log_smear(X, y, Xq, seed):
    """Log target + smearing (hold-out residuals): the log-scale view the literature uses."""
    return L.SFRModel(L.ModelSpec("lgbm", "log1p_smear_es"), seed=seed).fit(X, y).predict(Xq)


def mlp(X, y, Xq, seed):
    """MLP on quantile-normalised inputs, log1p target with smearing (inner 10 % hold-out)."""
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import QuantileTransformer
    qt = QuantileTransformer(output_distribution="normal", n_quantiles=1000, random_state=seed)
    Z = qt.fit_transform(np.nan_to_num(X)); Zq = qt.transform(np.nan_to_num(Xq))
    tr, es = L._es_split(y, seed)
    m = MLPRegressor(hidden_layer_sizes=(256, 128, 64), alpha=1e-4, batch_size=512, learning_rate_init=1e-3,
                     max_iter=60, early_stopping=True, n_iter_no_change=6, random_state=seed)
    m.fit(Z[tr], np.log1p(y[tr]))
    smear = float(np.mean(np.exp(np.log1p(y[es]) - m.predict(Z[es]))))
    return np.clip(smear * np.exp(m.predict(Zq)) - 1.0, 0, None)


def _rf(seed):
    from sklearn.ensemble import RandomForestRegressor
    return RandomForestRegressor(n_estimators=300, min_samples_leaf=3, max_features=0.5, n_jobs=N_THREADS, random_state=seed)


def _et(seed):
    from sklearn.ensemble import ExtraTreesRegressor
    return ExtraTreesRegressor(n_estimators=300, min_samples_leaf=3, max_features=0.5, n_jobs=N_THREADS, random_state=seed)


# name -> (fit function, feature set, description)
CONFIGS = {
    "M01_lgbm_tw13_F5": (lgbm(), "F5", "LightGBM Tweedie 1.3 (original final model, R0)"),
    "M02_lgbm_tw13_F0": (lgbm(), "F0", "LightGBM Tweedie 1.3, raw 13 columns"),
    "M03_lgbm_tw13_F3": (lgbm(), "F3", "LightGBM Tweedie 1.3, + physics features"),
    "M04_lgbm_tw13_F5A": (lgbm(), "F5A", "LightGBM Tweedie 1.3, F5 + multi-scale env + phase space"),
    "M05_lgbm_tw13_ALL": (lgbm(), "ALL", "LightGBM Tweedie 1.3, every feature incl. raw coordinates"),
    "M06_lgbm_tw12_F5": (lgbm(power=1.2), "F5", "LightGBM Tweedie 1.2"),
    "M07_lgbm_tw14_F5": (lgbm(power=1.4), "F5", "LightGBM Tweedie 1.4"),
    "M08_lgbm_tw13_slow_F5": (lgbm(learning_rate=0.02, n_estimators=5000, num_leaves=63, subsample=0.8, subsample_freq=1,
                                   colsample_bytree=0.8, es_rounds=50), "F5", "LightGBM Tweedie 1.3, lr 0.02, 63 leaves, bagging"),
    "M09_lgbm_tw13_reg_F5": (lgbm(num_leaves=15, min_child_samples=50, reg_lambda=5.0, es_rounds=30), "F5",
                             "LightGBM Tweedie 1.3, smaller trees + L2"),
    "M10_lgbm_sq_F5": (lgbm(obj="regression", es_rounds=30), "F5", "LightGBM squared error"),
    "M11_lgbm_poisson_F5": (lgbm(obj="poisson"), "F5", "LightGBM Poisson"),
    "M12_lgbm_logsmear_F5": (lgbm_log_smear, "F5", "LightGBM log1p target + smearing"),
    "M13_xgb_tw13_F5": (xgb_tw(), "F5", "XGBoost Tweedie 1.3"),
    "M14_xgb_tw13_deep_F5": (xgb_tw(max_leaves=0, max_depth=8, grow_policy="depthwise", learning_rate=0.03,
                                    subsample=0.8, colsample_bytree=0.8), "F5", "XGBoost Tweedie 1.3, depth 8"),
    "M15_cat_tw13_F5": (cat(), "F5", "CatBoost Tweedie 1.3, depth 6"),
    "M16_cat_tw13_d8_F5": (cat(depth=8), "F5", "CatBoost Tweedie 1.3, depth 8"),
    "M17_cat_rmse_F5": (cat(loss="RMSE", depth=8), "F5", "CatBoost RMSE, depth 8"),
    "M18_hgb_poisson_F5": (hgb_poisson, "F5", "HistGradientBoosting Poisson"),
    "M19_et_F5": (sk(_et), "F5", "ExtraTrees (squared error)"),
    "M20_rf_F5": (sk(_rf), "F5", "RandomForest (squared error)"),
    "M21_mlp_F5": (mlp, "F5", "MLP 256-128-64, quantile inputs, log target + smearing"),
    "M22_cat_tw13_F5A": (cat(depth=8), "F5A", "CatBoost Tweedie 1.3, depth 8, F5A"),
    "M23_xgb_tw13_F5A": (xgb_tw(), "F5A", "XGBoost Tweedie 1.3, F5A"),
    # feature-construction variants (same R0 recipe, only the inputs change)
    "M50_lgbm_tw13_F5G": (lgbm(), "F5G", "LightGBM Tweedie 1.3, F5 + environment grid / overdensity-step flags"),
    "M51_lgbm_tw13_F5N": (lgbm(), "F5N", "LightGBM Tweedie 1.3, F5 + relative-at-fixed-mass / neighbour-average features"),
    "M52_lgbm_tw13_F5GN": (lgbm(), "F5GN", "LightGBM Tweedie 1.3, F5 + G + N"),
    "M53_lgbm_tw13_F5lean": (lgbm(), "F5lean", "LightGBM Tweedie 1.3, F5 without the three weakest columns"),
    "M54_lgbm_tw13_MAX": (lgbm(), "MAX", "LightGBM Tweedie 1.3, every engineered feature (89)"),
    "M55_cat_tw13_F5N": (cat(depth=8), "F5N", "CatBoost Tweedie 1.3, depth 8, F5N"),
    "M56_xgb_tw13_F5N": (xgb_tw(), "F5N", "XGBoost Tweedie 1.3, F5N"),
    # literature groups one at a time, all of them, and the teammate's neighbour features (ablation, R0 recipe)
    **{f"M6{i}_lgbm_tw13_F5+L{i}": (lgbm(), f"F5+L{i}", f"LightGBM Tweedie 1.3, F5 + literature group L{i}")
       for i in range(1, 8)},
    "M68_lgbm_tw13_F5L": (lgbm(), "F5L", "LightGBM Tweedie 1.3, F5 + all literature groups"),
    "M69_lgbm_tw13_F5+S": (lgbm(), "F5+S", "LightGBM Tweedie 1.3, F5 + teammate neighbour features"),
    "M70_lgbm_tw13_F5LS": (lgbm(), "F5LS", "LightGBM Tweedie 1.3, F5 + literature + teammate features"),
    # model zoo on the enlarged feature set (more columns -> also a regularised LightGBM)
    "M71_lgbm_tw13_reg_F5LS": (lgbm(num_leaves=15, min_child_samples=100, colsample_bytree=0.5, subsample=0.8,
                                    subsample_freq=1, reg_lambda=5.0, learning_rate=0.03, n_estimators=3000,
                                    es_rounds=50), "F5LS", "LightGBM Tweedie 1.3, regularised, F5LS"),
    "M72_xgb_tw13_F5LS": (xgb_tw(), "F5LS", "XGBoost Tweedie 1.3, F5LS"),
    "M73_cat_tw13_d8_F5LS": (cat(depth=8), "F5LS", "CatBoost Tweedie 1.3, depth 8, F5LS"),
    "M74_et_F5LS": (sk(_et), "F5LS", "ExtraTrees (squared error), F5LS"),
    "M75_rf_F5LS": (sk(_rf), "F5LS", "RandomForest (squared error), F5LS"),
    "M76_mlp_F5LS": (mlp, "F5LS", "MLP 256-128-64, quantile inputs, log target + smearing, F5LS"),
    "M77_lgbm_logsmear_F5LS": (lgbm_log_smear, "F5LS", "LightGBM log1p target + smearing, F5LS"),
}


def run(name):
    fit, fs, desc = CONFIGS[name]
    if done(name):
        return
    G = setup()
    F, ytr, yva, folds = G["F"], G["y_train"], G["y_val"], G["folds"]
    cols = FS[fs]
    X = F.loc[ytr.index, cols].to_numpy(float); y = ytr.to_numpy(float)
    Xv = F.loc[yva.index, cols].to_numpy(float)
    set_status("models", f"{name}: CV")
    t0 = time.time()
    oof = np.zeros(len(y))
    for k in range(5):
        te = folds == k
        oof[te] = fit(X[~te], y[~te], X[te], SEED)
    val_pred = fit(X, y, Xv, SEED)
    meta = save_exp(name, oof, val_pred, {"stage": "single", "features": fs, "n_features": len(cols), "desc": desc,
                                          "seconds": time.time() - t0})
    set_status("models", f"{name}: CV {meta['cv_rmse_mean']:.3f} ({(time.time() - t0) / 60:.1f} min)")
    print(f"{name:28s} CV {meta['cv_rmse_mean']:.4f} ± {meta['cv_rmse_sd']:.3f}  {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    for n in (sys.argv[1:] or list(CONFIGS)):
        run(n)
