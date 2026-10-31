"""Stage 2: AutoGluon under different feature sets and preprocessing choices.

The bagging folds are forced to be the project's 5 CV folds (``groups``), so AutoGluon's out-of-fold predictions are
directly comparable, fold by fold, with every other experiment. The validation rows are never given to fit().
usage: python scripts/a03_autogluon.py <config> [<config> ...] [--force]
"""
import json
import os
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from core import FS, N_THREADS, RES, done, np, save_exp, set_status, setup  # noqa: E402

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


def cfg(fs, desc, target="raw", hp=None, preset="best_quality", hours=1.5, weight=None, stack=1):
    return dict(fs=fs, desc=desc, target=target, hp=hp, preset=preset, hours=hours, weight=weight, stack=stack)


# Axis 1: feature construction (preprocessing = which engineered inputs); axis 2: target / loss / weights / preset
CONFIGS = {
    # --- feature axis (raw target, best_quality, 1.5 h)
    "AG1_F0": cfg("F0", "raw 13 columns"),
    "AG2_F3": cfg("F3", "+ physics features (22)"),
    "AG3_F5": cfg("F5", "final feature set of the original project (25)"),
    "AG4_F5A": cfg("F5A", "F5 + multi-scale environment A1 + host phase space A2 (59)"),
    "AG5_ALL": cfg("ALL", "F5A + raw coordinates + MSSM counts (68)"),
    "AG8_F5L": cfg("F5L", "F5 + literature groups L1-L7 (62)"),
    "AG9_F5S": cfg("F5+S", "F5 + teammate neighbour features (52)"),
    "AG10_F5LS": cfg("F5LS", "F5 + literature + teammate features (89)"),
    "AG11_F5X": cfg("F5X", "every engineered feature (F5 + L + S + A + G + N + MSSM)"),
    # --- preprocessing / training axis on F5LS
    "AG6_F5_tweedie": cfg("F5", "F5, Tweedie LightGBM / XGBoost added to the model zoo", hp=TW_HP),
    "AG7_F5_log": cfg("F5", "F5, log1p target + smearing", target="log1p"),
    "AG12_F5LS_tweedie": cfg("F5LS", "F5LS, Tweedie LightGBM / XGBoost added to the model zoo", hp=TW_HP),
    "AG13_F5LS_log": cfg("F5LS", "F5LS, log1p target + smearing", target="log1p"),
    "AG14_F5LS_sqrt": cfg("F5LS", "F5LS, square-root target (variance stabilising)", target="sqrt"),
    "AG15_F5LS_wmassive": cfg("F5LS", "F5LS, weight 2 on galaxies with log M* > 11.5", weight=(11.5, 2.0)),
    "AG16_F5LS_stack2": cfg("F5LS", "F5LS, two extra stack levels", stack=2),
    "AG17_F5LS_long": cfg("F5LS", "F5LS, 4-hour budget (more of the model portfolio)", hours=4.0),
    "AG18_F5LS_extreme": cfg("F5LS", "F5LS, extreme_quality (TabPFNv2 / TabM / RealMLP / Mitra; CUDA GPU)",
                             preset="extreme_quality", hours=2.0),
    # --- revised plan: curated set F5C carries the preprocessing axis (fewer columns, less over-fitting)
    "AG20_F5C": cfg("F5C", "F5 + literature groups L1, L3, L4, L7 (curated a priori, 47)"),
    "AG21_FULL": cfg("FULL", "every feature: ours + teammate 2 module"),
    "AG22_T2CORE": cfg("T2CORE", "teammate 2 core set"),
    "AG23_F5C_tweedie": cfg("F5C", "F5C, Tweedie LightGBM / XGBoost added to the model zoo", hp=TW_HP),
    "AG24_F5C_log": cfg("F5C", "F5C, log1p target + smearing", target="log1p"),
    "AG25_F5C_sqrt": cfg("F5C", "F5C, square-root target", target="sqrt"),
    "AG26_F5C_wmassive": cfg("F5C", "F5C, weight 2 on galaxies with log M* > 11.5", weight=(11.5, 2.0)),
    "AG27_F5C_extreme": cfg("F5C", "F5C, extreme_quality (TabPFNv2 / TabM / RealMLP / Mitra; CUDA GPU)",
                            preset="extreme_quality", hours=2.0),
}


def run(name):
    if done(name):
        return
    assigned = RES / "assigned_elsewhere.json"   # configs another machine is running; skipped here
    if assigned.exists() and name in json.loads(assigned.read_text()) and "--force" not in sys.argv:
        print(f"{name}: assigned to another machine, skipped", flush=True)
        return
    from autogluon.tabular import TabularPredictor
    c = CONFIGS[name]
    G = setup()
    F, ytr, yva, folds = G["F"], G["y_train"], G["y_val"], G["folds"]
    if c["fs"] not in FS:
        print(f"{name}: feature set {c['fs']} not available (teammate 2 module missing) - skipped", flush=True)
        return
    cols = FS[c["fs"]]
    tr = F.loc[ytr.index, cols].reset_index(drop=True)
    y = ytr.to_numpy(float)
    fwd = {"raw": lambda a: a, "log1p": np.log1p, "sqrt": np.sqrt}[c["target"]]
    tr["target"] = fwd(y)
    tr["fold"] = folds
    pkw = {}
    if c["weight"]:
        cut, w = c["weight"]
        tr["w"] = np.where(F.loc[ytr.index, "log10_stellar_mass"].to_numpy() > cut, w, 1.0)
        pkw = dict(sample_weight="w", weight_evaluation=False)
    va = F.loc[yva.index, cols].reset_index(drop=True)
    path = RES / "ag_models" / name
    if path.exists():
        shutil.rmtree(path)
    limit = int(c["hours"] * 3600)
    set_status("autogluon", f"{name}: fitting (limit {c['hours']:.1f} h)")
    t0 = time.time()
    pred = TabularPredictor(label="target", problem_type="regression", eval_metric="root_mean_squared_error",
                            groups="fold", path=str(path), verbosity=2, **pkw)
    kw = dict(presets=c["preset"], time_limit=limit, num_stack_levels=c["stack"], dynamic_stacking=False,
              num_bag_sets=1, ag_args_fit={"num_cpus": N_THREADS})
    if c["hp"] is not None:
        kw["hyperparameters"] = c["hp"]
    pred.fit(tr, **kw)
    oof = pred.predict_oof().to_numpy(float)
    vp = pred.predict(va).to_numpy(float)
    if c["target"] == "log1p":
        smear = float(np.mean(np.exp(np.log1p(y) - oof)))   # smearing from out-of-fold residuals
        oof, vp = smear * np.exp(oof) - 1.0, smear * np.exp(vp) - 1.0
    elif c["target"] == "sqrt":
        bias = float(np.mean(y) - np.mean(np.clip(oof, 0, None) ** 2))   # E[Y] = E[f]^2 + Var: additive correction
        oof, vp = np.clip(oof, 0, None) ** 2 + bias, np.clip(vp, 0, None) ** 2 + bias
    oof, vp = np.clip(oof, 0, None), np.clip(vp, 0, None)
    lb = pred.leaderboard(silent=True)
    lb.to_csv(RES / "meta" / f"{name}_leaderboard.csv", index=False)
    meta = save_exp(name, oof, vp, {"stage": "autogluon", "features": c["fs"], "n_features": len(cols),
                                    "desc": f"AutoGluon {c['preset']}: {c['desc']}", "seconds": time.time() - t0,
                                    "best_model": pred.model_best, "n_models": int(len(lb)), "target": c["target"],
                                    "machine": os.environ.get("SFR_MACHINE", "")})
    set_status("autogluon", f"{name}: CV {meta['cv_rmse_mean']:.3f} ({(time.time() - t0) / 60:.0f} min, best {pred.model_best})")
    print(f"{name}: CV {meta['cv_rmse_mean']:.4f} ± {meta['cv_rmse_sd']:.3f}  best={pred.model_best}  "
          f"{(time.time() - t0) / 60:.0f} min", flush=True)
    shutil.rmtree(path, ignore_errors=True)   # keep disk use bounded: only predictions and leaderboard are kept


if __name__ == "__main__":
    for n in [a for a in sys.argv[1:] if not a.startswith("--")] or list(CONFIGS):
        run(n)
