"""Stage 2: AutoGluon (best_quality) under different preprocessing / feature sets.

The bagging folds are forced to be the project's 5 CV folds (``groups``), so AutoGluon's out-of-fold predictions are
directly comparable, fold by fold, with every other experiment. The validation rows are never given to fit().
usage: python scripts/a03_autogluon.py <config> [<config> ...]
"""
import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from core import FS, RES, SEED, done, np, pd, save_exp, set_status, setup  # noqa: E402

TIME_LIMIT = 5400  # seconds per configuration
TW_HP = {
    "GBM": [{"objective": "tweedie", "tweedie_variance_power": 1.3, "ag_args": {"name_suffix": "Tweedie13"}},
            {"objective": "tweedie", "tweedie_variance_power": 1.5, "ag_args": {"name_suffix": "Tweedie15"}},
            {}, {"extra_trees": True, "ag_args": {"name_suffix": "XT"}}],
    "XGB": [{"objective": "reg:tweedie", "tweedie_variance_power": 1.3, "ag_args": {"name_suffix": "Tweedie13"}}, {}],
    "CAT": {},
    "NN_TORCH": {},
    "FASTAI": {},
    "RF": [{"criterion": "squared_error", "ag_args": {"name_suffix": "MSE"}}],
    "XT": [{"criterion": "squared_error", "ag_args": {"name_suffix": "MSE"}}],
}

# name -> (feature set, target treatment, hyperparameters or None (= preset default), description)
CONFIGS = {
    "AG1_F0": ("F0", "raw", None, "AutoGluon best_quality, raw 13 columns"),
    "AG2_F3": ("F3", "raw", None, "AutoGluon best_quality, + physics features (22)"),
    "AG3_F5": ("F5", "raw", None, "AutoGluon best_quality, final feature set F5 (25)"),
    "AG4_F5A": ("F5A", "raw", None, "AutoGluon best_quality, F5 + multi-scale environment + phase space (59)"),
    "AG5_ALL": ("ALL", "raw", None, "AutoGluon best_quality, every feature incl. raw coordinates (68)"),
    "AG6_F5_tweedie": ("F5", "raw", TW_HP, "AutoGluon best_quality, F5, Tweedie GBM/XGB added to the model zoo"),
    "AG7_F5_log": ("F5", "log1p", None, "AutoGluon best_quality, F5, log1p target + smearing"),
}


def run(name):
    if done(name):
        return
    from autogluon.tabular import TabularPredictor
    fs, treat, hp, desc = CONFIGS[name]
    G = setup()
    F, ytr, yva, folds = G["F"], G["y_train"], G["y_val"], G["folds"]
    cols = FS[fs]
    tr = F.loc[ytr.index, cols].reset_index(drop=True)
    y = ytr.to_numpy(float)
    tr["target"] = np.log1p(y) if treat == "log1p" else y
    tr["fold"] = folds
    va = F.loc[yva.index, cols].reset_index(drop=True)
    path = RES / "ag_models" / name
    if path.exists():
        shutil.rmtree(path)
    set_status("autogluon", f"{name}: fitting (limit {TIME_LIMIT / 3600:.1f} h)")
    t0 = time.time()
    pred = TabularPredictor(label="target", problem_type="regression", eval_metric="root_mean_squared_error",
                            groups="fold", path=str(path), verbosity=2)
    kw = dict(presets="best_quality", time_limit=TIME_LIMIT, num_stack_levels=1, dynamic_stacking=False,
              num_bag_sets=1, ag_args_fit={"num_cpus": 14})
    if hp is not None:
        kw["hyperparameters"] = hp
    pred.fit(tr, **kw)
    oof = pred.predict_oof().to_numpy(float)
    vp = pred.predict(va).to_numpy(float)
    if treat == "log1p":
        smear = float(np.mean(np.exp(np.log1p(y) - oof)))   # smearing from out-of-fold residuals
        oof, vp = smear * np.exp(oof) - 1.0, smear * np.exp(vp) - 1.0
    oof, vp = np.clip(oof, 0, None), np.clip(vp, 0, None)
    lb = pred.leaderboard(silent=True)
    lb.to_csv(RES / "meta" / f"{name}_leaderboard.csv", index=False)
    meta = save_exp(name, oof, vp, {"stage": "autogluon", "features": fs, "n_features": len(cols), "desc": desc,
                                    "seconds": time.time() - t0, "best_model": pred.model_best,
                                    "n_models": int(len(lb)), "target": treat})
    set_status("autogluon", f"{name}: CV {meta['cv_rmse_mean']:.3f} ({(time.time() - t0) / 60:.0f} min, best {pred.model_best})")
    print(f"{name}: CV {meta['cv_rmse_mean']:.4f} ± {meta['cv_rmse_sd']:.3f}  best={pred.model_best}  "
          f"{(time.time() - t0) / 60:.0f} min", flush=True)
    # keep disk use bounded: only the leaderboard and predictions are needed later
    shutil.rmtree(path, ignore_errors=True)


if __name__ == "__main__":
    for n in (sys.argv[1:] or list(CONFIGS)):
        run(n)
