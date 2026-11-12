"""B candidates of docs/log_metric_selection_rule.md: B1 direct log regression and B2 hurdle, on F5 and F5C.

OOF predictions on the notebook's 5 folds (training split only; the validation split is not touched).
Features: scripts/notebook_features.py = notebook cells 4.0 and 9.2a. LightGBM with the untuned R0 settings.
Output: results/log_metric/oof_B/<name>.parquet with columns row, fold, y, oof (prediction on the SFR scale).
"""
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import notebook_features as nf  # noqa: E402

OUT = ROOT / "results" / "log_metric" / "oof_B"; OUT.mkdir(parents=True, exist_ok=True)
EPS = 1e-3
T0 = time.time()
log = lambda *a: print(f"[{(time.time() - T0) / 60:5.1f} min]", *a, flush=True)

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
log("split and folds match the stored candidates")

FEAT = nf.build_features(df[nf.INPUT_COLUMNS]); FEAT.index = df.index
SETS = {"F5": nf.F5, "F5C": nf.F5C}
log("features built", FEAT.shape)

G = {"D1": (np.log10, lambda z: 10.0 ** z),
     "D2": (lambda v: np.log10(v + EPS), lambda z: np.maximum(10.0 ** z - EPS, 0.0)),
     "D3": (lambda v: np.log10(1.0 + v), lambda z: np.maximum(10.0 ** z - 1.0, 0.0)),
     "DS": (lambda v: np.log10(np.maximum(v, 1e-13)), lambda z: 10.0 ** z)}   # organisers' metric
PARAMS = dict(learning_rate=0.05, n_estimators=1000, num_leaves=31, min_child_samples=20, random_state=42,
              n_jobs=12, verbose=-1)


def fit(Xa, target, kind, lab):
    tr, es = train_test_split(np.arange(len(target)), test_size=0.10, random_state=42, stratify=lab)
    m = (lgb.LGBMClassifier(objective="binary", **PARAMS) if kind == "clf"
         else lgb.LGBMRegressor(objective="regression", **PARAMS))
    m.fit(Xa[tr], target[tr], eval_set=[(Xa[es], target[es])], callbacks=[lgb.early_stopping(10, verbose=False)])
    return m


def save(name, pred):
    pd.DataFrame({"row": X_tr.index.to_numpy(), "fold": FOLDS, "y": y, "oof": pred}).to_parquet(OUT / f"{name}.parquet")


for fs, cols in SETS.items():
    Xall = FEAT.loc[X_tr.index, cols].to_numpy(float)
    oof = {key: np.zeros(len(y)) for key in ["B1_D1", "B1_D2", "B1_D3", "B2_D2", "B2_D3", "B1_DS", "B2_DS"]}
    for k in range(5):
        trn, te = FOLDS != k, FOLDS == k
        Xk, yk = Xall[trn], y[trn]; labk = strata(yk); pos = yk > 0
        clf = fit(Xk, pos.astype(int), "clf", labk)
        p = clf.predict_proba(Xall[te])[:, 1]
        for dn, (g, ginv) in G.items():
            # regression on g(SFR) of the galaxies with SFR > 0 (B1 under D1 = B2 under D1, counted once)
            reg_pos = fit(Xk[pos], g(yk[pos]), "reg", labk[pos])
            m = reg_pos.predict(Xall[te])
            if dn == "D1":
                oof["B1_D1"][te] = ginv(m)
                continue
            oof[f"B2_{dn}"][te] = ginv((1.0 - p) * g(0.0) + p * m)                    # hurdle
            reg_all = fit(Xk, g(yk), "reg", labk)                                     # direct regression on g(SFR)
            oof[f"B1_{dn}"][te] = ginv(reg_all.predict(Xall[te]))
        log(fs, "fold", k, "done")
    for key, pred in oof.items():
        save(f"{key}_{fs}", pred)
log("all B candidates written to", OUT)
