"""C candidates of docs/log_metric_selection_rule.md: AutoGluon best_quality regression on g(SFR) = log10(max(SFR, 1e-13)),
the organisers' metric DS. Same settings as the notebook (5-fold bagging on our folds, one stacking layer, 1.5 h).
Two runs: F5C, then F5. OOF predictions only; the validation split is not touched.
Output: results/log_metric/oof_C/<name>.parquet (row, fold, y, oof on the SFR scale) and <name>_info.json.
Models: results/ag_models/<name>/ (git-ignored).
"""
import ctypes
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import notebook_features as nf  # noqa: E402

ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)   # keep the PC awake (ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
OUT = ROOT / "results" / "log_metric" / "oof_C"; OUT.mkdir(parents=True, exist_ok=True)
T0 = time.time()
log = lambda *a: print(f"[{(time.time() - T0) / 60:6.1f} min]", *a, flush=True)

df = pd.read_csv(ROOT / "data" / "Galaxy Features with Position and Velocity for Star Formation Rate Prediction.csv")
X = df[[c for c in df.columns if c not in ("subhalo_id", "sfr_linear")]]
X_tr, _, y_tr, _ = train_test_split(X, df["sfr_linear"], test_size=0.20, random_state=42,
                                    stratify=(df["sfr_linear"] > 0).astype(int))
y = y_tr.to_numpy(float)


def strata(v):
    lab = np.zeros(len(v), int); p = v > 0; ly = np.log10(v[p])
    e = np.quantile(ly, np.linspace(0, 1, 11)[1:-1]); lab[p] = 1 + np.searchsorted(e, ly, "right")
    return lab


FOLDS = np.full(len(y), -1)
for k, (_, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(len(y)), strata(y))):
    FOLDS[te] = k
ref = pd.read_parquet(ROOT / "results" / "oof" / "M01_lgbm_tw13_F5.parquet")
assert np.array_equal(ref["row"].to_numpy(), X_tr.index.to_numpy()) and np.array_equal(ref["fold"].to_numpy(), FOLDS)
FEAT = nf.build_features(df[nf.INPUT_COLUMNS]); FEAT.index = df.index
log("split, folds and features ready", FEAT.shape)

from autogluon.tabular import TabularPredictor  # noqa: E402

g = lambda v: np.log10(np.maximum(v, 1e-13))
for name, cols in [("C_AG_DS_F5C", nf.F5C), ("C_AG_DS_F5", nf.F5)]:
    tr = FEAT.loc[X_tr.index, cols].reset_index(drop=True)
    tr["target"] = g(y); tr["fold"] = FOLDS
    t0 = time.time()
    pred = TabularPredictor(label="target", problem_type="regression", eval_metric="root_mean_squared_error",
                            groups="fold", path=str(ROOT / "results" / "ag_models" / name), verbosity=1)
    pred.fit(tr, presets="best_quality", time_limit=5400, num_stack_levels=1, dynamic_stacking=False, num_bag_sets=1,
             ag_args_fit={"num_cpus": 12})
    z = pred.predict_oof().to_numpy(float)
    pd.DataFrame({"row": X_tr.index.to_numpy(), "fold": FOLDS, "y": y, "oof": 10.0 ** z}).to_parquet(OUT / f"{name}.parquet")
    lb = pred.leaderboard(silent=True)
    info = {"seconds": time.time() - t0, "n_models": len(lb), "best_model": pred.model_best, "features": len(cols)}
    (OUT / f"{name}_info.json").write_text(json.dumps(info, indent=1))
    log(name, info)
log("done")
