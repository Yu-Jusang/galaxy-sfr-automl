"""Stage 1d: linear / GLM / kernel / nearest-neighbour models on F5 and F5LS, untuned and nested-tuned.

Every model sees quantile-normalised inputs (fitted on the training part of each fold only; NaNs -> column median).
Tuned versions: for each outer fold Optuna (TPE, seed 42) searches on a stratified 3-fold CV inside a 40,000-galaxy
stratified subsample of the outer training part only; the best setting is refitted there and scored on the outer fold (same design as a07_tune.py).
usage: python scripts/a08_tune_linear.py [name ...]
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import optuna  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.kernel_approximation import Nystroem  # noqa: E402
from sklearn.linear_model import Lasso, Ridge, TweedieRegressor  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402
from sklearn.neighbors import KNeighborsRegressor  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import QuantileTransformer, SplineTransformer, StandardScaler  # noqa: E402

from core import FS, L, N_THREADS, SEED, done, np, save_exp, set_status, setup  # noqa: E402

optuna.logging.set_verbosity(optuna.logging.WARNING)


def prep(seed):
    return [SimpleImputer(strategy="median"),
            QuantileTransformer(output_distribution="normal", n_quantiles=1000, subsample=200_000, random_state=seed)]


# family -> (pipeline builder from params, target ("log" or "raw"), Optuna search space, default params)
def b_ridge(p, seed):
    return make_pipeline(*prep(seed), SplineTransformer(n_knots=p["knots"], degree=3), StandardScaler(),
                         Ridge(alpha=p["alpha"]))


def b_lasso(p, seed):
    return make_pipeline(*prep(seed), SplineTransformer(n_knots=p["knots"], degree=3), StandardScaler(),
                         Lasso(alpha=p["alpha"], max_iter=5000, random_state=seed))


def b_tweedie(p, seed):
    return make_pipeline(*prep(seed), SplineTransformer(n_knots=p["knots"], degree=3), StandardScaler(),
                         TweedieRegressor(power=p["power"], link="log", alpha=p["alpha"], max_iter=3000))


def b_kernel(p, seed):
    return make_pipeline(*prep(seed), StandardScaler(),
                         Nystroem(kernel="rbf", gamma=p["gamma"], n_components=p["ncomp"], random_state=seed),
                         Ridge(alpha=p["alpha"]))


def b_knn(p, seed):
    return make_pipeline(*prep(seed), KNeighborsRegressor(n_neighbors=p["k"], weights=p["weights"], n_jobs=N_THREADS))


FAMILIES = {
    "ridge": (b_ridge, "log", lambda t: {"knots": t.suggest_int("knots", 3, 10),
                                          "alpha": t.suggest_float("alpha", 1e-3, 1e3, log=True)},
              {"knots": 6, "alpha": 1.0}),
    "lasso": (b_lasso, "log", lambda t: {"knots": t.suggest_int("knots", 3, 10),
                                          "alpha": t.suggest_float("alpha", 1e-5, 1e-1, log=True)},
              {"knots": 6, "alpha": 1e-3}),
    "tweedie_glm": (b_tweedie, "raw", lambda t: {"knots": t.suggest_int("knots", 3, 10),
                                                  "power": t.suggest_float("power", 1.1, 1.7),
                                                  "alpha": t.suggest_float("alpha", 1e-5, 1e-1, log=True)},
                    {"knots": 6, "power": 1.3, "alpha": 1e-3}),
    "kernel_ridge": (b_kernel, "log", lambda t: {"gamma": t.suggest_float("gamma", 1e-3, 0.3, log=True),
                                                  "ncomp": t.suggest_categorical("ncomp", [500, 1000, 2000]),
                                                  "alpha": t.suggest_float("alpha", 1e-3, 1e2, log=True)},
                     {"gamma": 0.02, "ncomp": 1000, "alpha": 1.0}),
    "knn": (b_knn, "raw", lambda t: {"k": t.suggest_int("k", 5, 200, log=True),
                                      "weights": t.suggest_categorical("weights", ["uniform", "distance"])},
            {"k": 30, "weights": "distance"}),
}
N_TRIALS = {"ridge": 12, "lasso": 12, "tweedie_glm": 12, "kernel_ridge": 12, "knn": 10}
TUNE_SUB = 40_000   # the search uses a stratified subsample of the outer training part; the final refit uses all of it


def fit_predict(fam, p, X, y, Xq, seed=SEED):
    build, target, _, _ = FAMILIES[fam]
    if target == "raw":
        return np.clip(build(p, seed).fit(X, y).predict(Xq), 0, None)
    # a spline model in log space can extrapolate to absurd values for a few unusual galaxies (e.g. progenitor-floor
    # rows); clip the log prediction to the range of its own fitted values on the training rows before exponentiating
    tr, es = L._es_split(y, seed)                       # smearing factor from a 10 % hold-out
    m = build(p, seed).fit(X[tr], np.log1p(y[tr]))
    f_tr = m.predict(X[tr])
    smear = float(np.mean(np.exp(np.log1p(y[es]) - np.clip(m.predict(X[es]), f_tr.min(), f_tr.max()))))
    m = build(p, seed).fit(X, np.log1p(y))
    f_all = m.predict(X)
    return np.clip(smear * np.exp(np.clip(m.predict(Xq), f_all.min(), f_all.max())) - 1.0, 0, None)


def tune(fam, X, y, n_trials):
    _, _, space, default = FAMILIES[fam]
    if len(y) > TUNE_SUB:
        from sklearn.model_selection import train_test_split
        sub, _ = train_test_split(np.arange(len(y)), train_size=TUNE_SUB, random_state=SEED, stratify=L.strata_labels(y))
        X, y = X[sub], y[sub]
    inner = list(StratifiedKFold(3, shuffle=True, random_state=SEED).split(np.zeros(len(y)), L.strata_labels(y)))

    def objective(trial):
        p = space(trial)
        return float(np.mean([np.sqrt(np.mean((fit_predict(fam, p, X[a], y[a], X[b]) - y[b]) ** 2)) for a, b in inner]))

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.enqueue_trial(default)
    study.optimize(objective, n_trials=n_trials)
    return space(optuna.trial.FixedTrial(study.best_params)), float(study.best_value)


CONFIGS = {f"LK_{fam}_{fs}{'_tuned' if tuned else ''}": (fam, fs, tuned)
           for fs in ("F5", "F5LS") for fam in FAMILIES for tuned in (False, True)}


def run(name):
    if done(name):
        return
    fam, fs, tuned = CONFIGS[name]
    G = setup()
    F, ytr, yva, folds = G["F"], G["y_train"], G["y_val"], G["folds"]
    cols = FS[fs]
    X = F.loc[ytr.index, cols].to_numpy(float); y = ytr.to_numpy(float)
    Xv = F.loc[yva.index, cols].to_numpy(float)
    t0 = time.time(); oof = np.zeros(len(y)); chosen = []
    default = FAMILIES[fam][3]
    for k in range(5):
        set_status("models", f"{name}: fold {k + 1}/5")
        te = folds == k
        p = tune(fam, X[~te], y[~te], N_TRIALS[fam])[0] if tuned else default
        chosen.append({"fold": k, **p})
        oof[te] = fit_predict(fam, p, X[~te], y[~te], X[te])
    p = tune(fam, X, y, N_TRIALS[fam])[0] if tuned else default
    chosen.append({"fold": "full", **p})
    vp = fit_predict(fam, p, X, y, Xv)
    meta = save_exp(name, oof, vp, {"stage": "tuned" if tuned else "single", "family": "linear/kernel", "features": fs,
                                    "n_features": len(cols), "chosen": chosen, "seconds": time.time() - t0,
                                    "desc": f"{fam} on quantile inputs{' (nested Optuna)' if tuned else ''}"})
    set_status("models", f"{name}: CV {meta['cv_rmse_mean']:.3f} ({(time.time() - t0) / 60:.0f} min)")
    print(f"{name:32s} CV {meta['cv_rmse_mean']:.4f} ± {meta['cv_rmse_sd']:.3f}  {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    for n in [a for a in sys.argv[1:] if not a.startswith("--")] or list(CONFIGS):
        try:
            run(n)
        except Exception as e:
            print(f"{n} FAILED: {e!r}", flush=True)
            set_status("models", f"{n}: failed ({type(e).__name__})")
