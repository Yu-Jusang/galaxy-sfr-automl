"""Re-rank the stored project candidates under the log-SFR metrics D1-D3 (docs/log_metric_selection_rule.md).

CV only: reads results/oof/*.parquet (columns row, fold, y, oof). The validation predictions are not read.
Writes results/log_metric/rerank_139.csv.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OOF = ROOT / "results" / "oof"
OUT = ROOT / "results" / "log_metric"; OUT.mkdir(parents=True, exist_ok=True)
REF = "M01_lgbm_tw13_F5"          # R0
EPS = 1e-3

G = {
    "D1": lambda p, y: (np.log10(np.clip(p[y > 0], EPS, None)) - np.log10(y[y > 0])),
    "D2": lambda p, y: (np.log10(np.clip(p, 0, None) + EPS) - np.log10(y + EPS)),
    "D3": lambda p, y: (np.log10(1 + np.clip(p, 0, None)) - np.log10(1 + y)),
}

files = sorted(OOF.glob("*.parquet"))
base = pd.read_parquet(files[0]).sort_values("row")
y, fold = base["y"].to_numpy(float), base["fold"].to_numpy()
K = np.unique(fold)

per_fold = {}
for f in files:
    d = pd.read_parquet(f).sort_values("row")
    assert np.array_equal(d["y"].to_numpy(float), y) and np.array_equal(d["fold"].to_numpy(), fold), f.name
    p = d["oof"].to_numpy(float)
    ok = np.isfinite(p).all()
    p = np.nan_to_num(p, nan=0.0, posinf=1e12)
    per_fold[f.stem] = {dn: np.array([np.sqrt(np.mean(g(p[fold == k], y[fold == k]) ** 2)) for k in K]) for dn, g in G.items()}
    per_fold[f.stem]["finite"] = ok

rows = []
for name, v in per_fold.items():
    r = {"name": name, "finite_oof": v["finite"]}
    for dn in G:
        d = v[dn] - per_fold[REF][dn]
        r[f"{dn}_cv"] = v[dn].mean()
        r[f"{dn}_d_vs_R0"] = d.mean()
        r[f"{dn}_se"] = d.std(ddof=1) / np.sqrt(len(K))
        r[f"{dn}_folds_better"] = int((d < 0).sum())
    rows.append(r)
R = pd.DataFrame(rows)
for dn in G:
    R[f"{dn}_rank"] = R[f"{dn}_cv"].rank(method="min").astype(int)
R = R.sort_values("D2_cv")
R.to_csv(OUT / "rerank_139.csv", index=False)

pd.set_option("display.width", 220)
print(f"{len(R)} candidates, {len(y):,} training galaxies, folds {K.tolist()}; non-finite OOF: "
      f"{R.loc[~R.finite_oof, 'name'].tolist()}")
for dn in G:
    t = R.sort_values(f"{dn}_cv").head(12)[["name", f"{dn}_cv", f"{dn}_d_vs_R0", f"{dn}_se", f"{dn}_folds_better"]]
    print(f"\n=== {dn}: top 12 by CV (R0 = {per_fold[REF][dn].mean():.4f}, rank {int(R.loc[R.name == REF, f'{dn}_rank'].iloc[0])})")
    print(t.round(4).to_string(index=False))
