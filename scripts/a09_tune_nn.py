"""Stage 4b: nested tuning of the neural tabular models TabM and RealMLP (GPU / Apple GPU recommended).

Neural fits are expensive, so the inner search is lighter than for the trees: for each outer fold, Optuna (TPE,
seed 42, 6 trials) scores each setting on one stratified 15 % hold-out of the outer training part; the best setting
is refitted on the whole outer training part and scored on the outer fold. Every fit is capped (DEEP_FIT_SECONDS).
usage: python scripts/a09_tune_nn.py [name ...]
"""
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import optuna  # noqa: E402
import torch  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

from core import FS, L, N_THREADS, SEED, done, epoch_time_cap, impute_missing, np, save_exp, set_status, setup  # noqa: E402

optuna.logging.set_verbosity(optuna.logging.WARNING)
torch.set_num_threads(N_THREADS)
DEVICE = os.environ.get("SFR_DEVICE") or ("cuda" if torch.cuda.is_available() else   # SFR_DEVICE=cpu if MPS fails
                                         "mps" if torch.backends.mps.is_available() else "cpu")
FIT_SECONDS = int(os.environ.get("DEEP_FIT_SECONDS", "300"))
N_TRIALS = int(os.environ.get("NN_TRIALS", "6"))


def space_tabm(t):
    return {"lr": t.suggest_float("lr", 1e-4, 3e-3, log=True), "d_block": t.suggest_categorical("d_block", [256, 512, 1024]),
            "n_blocks": t.suggest_int("n_blocks", 1, 4), "dropout": t.suggest_float("dropout", 0.0, 0.5),
            "weight_decay": t.suggest_float("weight_decay", 1e-5, 1e-1, log=True)}


def space_realmlp(t):
    return {"lr": t.suggest_float("lr", 0.01, 0.15, log=True),
            "hidden_sizes": t.suggest_categorical("hidden_sizes", ["256x3", "512x3", "256x2", "512x2"]),
            "p_drop": t.suggest_float("p_drop", 0.0, 0.3), "wd": t.suggest_float("wd", 1e-4, 5e-2, log=True)}


def make(fam, p, seed):
    from pytabkit import RealMLP_TD_Regressor, TabM_D_Regressor
    if fam == "tabm":
        return TabM_D_Regressor(device=DEVICE, random_state=seed, n_threads=N_THREADS, batch_size=1024, verbosity=1, **p)
    q = dict(p)
    w, n = q.pop("hidden_sizes").split("x")
    return RealMLP_TD_Regressor(device=DEVICE, random_state=seed, n_threads=N_THREADS, verbosity=0,
                                hidden_sizes=[int(w)] * int(n), **q)


def fit_predict(fam, p, X, y, Xq, seed=SEED):
    X, Xq = impute_missing(X, Xq)   # medians from the fitting rows only
    tr, es = L._es_split(y, seed)
    m = make(fam, p, seed)
    with epoch_time_cap(FIT_SECONDS):   # TabM ignores time_to_fit_in_seconds (verbosity=1 routes it through the cap)
        m.fit(X[tr], y[tr], X_val=X[es], y_val=y[es], time_to_fit_in_seconds=FIT_SECONDS)
    return np.clip(m.predict(Xq), 0, None)


DEFAULTS = {"tabm": {}, "realmlp": {}}
SPACES = {"tabm": space_tabm, "realmlp": space_realmlp}


def tune(fam, X, y):
    a, b = train_test_split(np.arange(len(y)), test_size=0.15, random_state=SEED, stratify=L.strata_labels(y))

    def objective(trial):
        p = SPACES[fam](trial)
        return float(np.sqrt(np.mean((fit_predict(fam, p, X[a], y[a], X[b]) - y[b]) ** 2)))

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=N_TRIALS)
    return SPACES[fam](optuna.trial.FixedTrial(study.best_params)), float(study.best_value)


CONFIGS = {f"NN_{fam}_F5C_tuned": (fam, "F5C") for fam in ("tabm", "realmlp")}   # curated set only (cost)


def run(name):
    if done(name):
        return
    fam, fs = CONFIGS[name]
    G = setup()
    F, ytr, yva, folds = G["F"], G["y_train"], G["y_val"], G["folds"]
    cols = FS[fs]
    X = F.loc[ytr.index, cols].to_numpy(float); y = ytr.to_numpy(float)
    Xv = F.loc[yva.index, cols].to_numpy(float)
    t0 = time.time(); oof = np.zeros(len(y)); chosen = []
    for k in range(5):
        set_status("tuning", f"{name}: outer fold {k + 1}/5")
        te = folds == k
        p, inner = tune(fam, X[~te], y[~te])
        chosen.append({"fold": k, "inner_rmse": inner, **p})
        oof[te] = fit_predict(fam, p, X[~te], y[~te], X[te])
    p, inner = tune(fam, X, y)
    chosen.append({"fold": "full", "inner_rmse": inner, **p})
    vp = fit_predict(fam, p, X, y, Xv)
    meta = save_exp(name, oof, vp, {"stage": "tuned", "family": "neural", "features": fs, "n_features": len(cols),
                                    "chosen": chosen, "seconds": time.time() - t0, "device": DEVICE,
                                    "desc": f"{fam} nested Optuna ({N_TRIALS} trials/fold, one inner hold-out)"})
    set_status("tuning", f"{name}: CV {meta['cv_rmse_mean']:.3f} ({(time.time() - t0) / 60:.0f} min)")
    print(f"{name:26s} CV {meta['cv_rmse_mean']:.4f} ± {meta['cv_rmse_sd']:.3f}  {(time.time() - t0) / 60:.0f} min", flush=True)


if __name__ == "__main__":
    for n in [a for a in sys.argv[1:] if not a.startswith("--")] or list(CONFIGS):
        try:
            run(n)
        except Exception as e:
            print(f"{n} FAILED: {e!r}", flush=True)
            set_status("tuning", f"{n}: failed ({type(e).__name__})")
