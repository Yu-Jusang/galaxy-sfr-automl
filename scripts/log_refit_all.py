"""Deployment refit of the chosen model D_top3_DS (docs/log_metric_selection_rule.md, change log entry 3b).

Same settings as the CV runs, on all 208,437 labelled galaxies (training + validation split), for the organisers'
private test set only. The validation score in the notebook comes from the training-split models, not from these.
Members: B2_DS_F5C (LightGBM hurdle), C_AG_DS_F5C, C_AG_DS_F5 (AutoGluon on g(SFR)); prediction = 10 ** mean(g).
Output: results/refit_all/B2_DS_F5C_all.joblib, results/ag_models/<name>_all/, results/refit_all/<name>_info.json
"""
import ctypes
import json
import sys
import time
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import notebook_features as nf  # noqa: E402

ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)   # keep the PC awake (ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
OUT = ROOT / "results" / "refit_all"; OUT.mkdir(parents=True, exist_ok=True)
T0 = time.time()
log = lambda *a: print(f"[{(time.time() - T0) / 60:6.1f} min]", *a, flush=True)

df = pd.read_csv(ROOT / "data" / "Galaxy Features with Position and Velocity for Star Formation Rate Prediction.csv")
y = df["sfr_linear"].to_numpy(float)


def strata(v):
    lab = np.zeros(len(v), int); p = v > 0; ly = np.log10(v[p])
    e = np.quantile(ly, np.linspace(0, 1, 11)[1:-1]); lab[p] = 1 + np.searchsorted(e, ly, "right")
    return lab


LAB = strata(y)
FOLDS = np.full(len(y), -1)
for k, (_, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(len(y)), LAB)):
    FOLDS[te] = k
FEAT = nf.build_features(df[nf.INPUT_COLUMNS]); FEAT.index = df.index
log("all labelled galaxies:", len(y), "| features", FEAT.shape)

g = lambda v: np.log10(np.maximum(v, 1e-13))
PARAMS = dict(learning_rate=0.05, n_estimators=1000, num_leaves=31, min_child_samples=20, random_state=42,
              n_jobs=12, verbose=-1)


def fit(Xa, target, kind, lab):
    tr, es = train_test_split(np.arange(len(target)), test_size=0.10, random_state=42, stratify=lab)
    m = (lgb.LGBMClassifier(objective="binary", **PARAMS) if kind == "clf"
         else lgb.LGBMRegressor(objective="regression", **PARAMS))
    m.fit(Xa[tr], target[tr], eval_set=[(Xa[es], target[es])], callbacks=[lgb.early_stopping(10, verbose=False)])
    return m


# B2 hurdle on F5C
X = FEAT[nf.F5C].to_numpy(float); pos = y > 0
clf = fit(X, pos.astype(int), "clf", LAB)
reg = fit(X[pos], g(y[pos]), "reg", LAB[pos])
joblib.dump({"clf": clf, "reg": reg, "features": list(nf.F5C)}, OUT / "B2_DS_F5C_all.joblib")
log("B2_DS_F5C refit: classifier", clf.best_iteration_, "trees, regression", reg.best_iteration_, "trees")

from autogluon.tabular import TabularPredictor  # noqa: E402

for name, cols in [("C_AG_DS_F5C_all", nf.F5C), ("C_AG_DS_F5_all", nf.F5)]:
    tr = FEAT[cols].reset_index(drop=True)
    tr["target"] = g(y); tr["fold"] = FOLDS
    t0 = time.time()
    pred = TabularPredictor(label="target", problem_type="regression", eval_metric="root_mean_squared_error",
                            groups="fold", path=str(ROOT / "results" / "ag_models" / name), verbosity=1)
    pred.fit(tr, presets="best_quality", time_limit=5400, num_stack_levels=1, dynamic_stacking=False, num_bag_sets=1,
             ag_args_fit={"num_cpus": 12})
    lb = pred.leaderboard(silent=True)
    info = {"seconds": time.time() - t0, "n_models": len(lb), "best_model": pred.model_best, "features": len(cols),
            "n_rows": len(tr)}
    (OUT / f"{name}_info.json").write_text(json.dumps(info, indent=1))
    log(name, info)
log("done")
