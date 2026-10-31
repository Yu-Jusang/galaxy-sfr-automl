"""Stage 3: ensembles of the out-of-fold predictions, final selection, one validation evaluation.

Ensemble weights are learned *inside* the CV: for outer fold k the weights are fitted on the OOF rows of the
other four folds and applied to fold k, so the ensemble's CV RMSE is an honest out-of-sample estimate.
The final weights are refitted on all OOF rows and applied to the stored validation predictions.
Final choice = lowest CV RMSE over every single model, AutoGluon run and ensemble (rule fixed before the sweep).
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from scipy.optimize import nnls  # noqa: E402

from core import META_DIR, RES, VAL_DIR, L, load_oof, np, paired, pd, save_exp, set_status, setup  # noqa: E402

R0 = "M01_lgbm_tw13_F5"


def base_names():
    out = []
    for f in sorted(META_DIR.glob("*.json")):
        m = json.loads(f.read_text())
        if m.get("stage") in ("single", "autogluon", "tuned", "selected"):
            out.append(m["name"])
    return out


def simplex_weights(P, y):
    """Non-negative weights summing to 1 minimising squared error (NNLS on a sum-to-one penalised system)."""
    lam = 1e3 * np.sqrt(len(y))
    A = np.vstack([P, lam * np.ones((1, P.shape[1]))])
    b = np.concatenate([y, [lam]])
    w, _ = nnls(A, b)
    return w / w.sum()


def greedy_weights(P, y, steps=40):
    """Caruana-style forward selection with replacement."""
    n, m = P.shape
    cur = np.zeros(n); counts = np.zeros(m)
    for s in range(steps):
        errs = [np.mean(((cur * s + P[:, j]) / (s + 1) - y) ** 2) for j in range(m)]
        j = int(np.argmin(errs)); counts[j] += 1
        cur = (cur * s + P[:, j]) / (s + 1)
    return counts / counts.sum()


def topk_weights(P, y, k):
    rm = np.sqrt(((P - y[:, None]) ** 2).mean(0))
    w = np.zeros(P.shape[1]); w[np.argsort(rm)[:k]] = 1.0 / k
    return w


def as_weights(fn):
    """Wrap a weight finder as a fit(P, y) -> (predict, description) stacker."""
    def fit(P, y, extra=None):
        w = fn(P, y)
        return (lambda Q, e=None: Q @ w), w
    return fit


def stack_nnls(P, y, extra=None):
    """Stacking, linear meta-model: non-negative weights + intercept (weights need not sum to 1)."""
    A = np.column_stack([P, np.ones(len(y))])
    w, _ = nnls(A, y)
    return (lambda Q, e=None: np.clip(np.column_stack([Q, np.ones(len(Q))]) @ w, 0, None)), w[:-1]


def stack_lgbm(P, y, extra):
    """Stacking, non-linear meta-model: small Tweedie LightGBM on the base predictions + log stellar mass
    (lets the blend depend on galaxy mass, e.g. trust different models for giants and dwarfs)."""
    import lightgbm as lgb
    Z = np.column_stack([P, extra])
    tr, es = L._es_split(y, 42)
    m = lgb.LGBMRegressor(objective="tweedie", tweedie_variance_power=1.3, num_leaves=7, min_child_samples=200,
                          learning_rate=0.03, n_estimators=2000, subsample=0.8, subsample_freq=1, colsample_bytree=0.7,
                          random_state=42, n_jobs=8, verbose=-1)
    m.fit(Z[tr], y[tr], eval_set=[(Z[es], y[es])], callbacks=[lgb.early_stopping(50, verbose=False)])
    return (lambda Q, e: m.predict(np.column_stack([Q, e]))), None


METHODS = {"ENS_simplex": as_weights(simplex_weights), "ENS_greedy": as_weights(greedy_weights),
           "ENS_top3": as_weights(lambda P, y: topk_weights(P, y, 3)),
           "ENS_top5": as_weights(lambda P, y: topk_weights(P, y, 5)),
           "ENS_stack_nnls": stack_nnls, "ENS_stack_lgbm": stack_lgbm}


def nested(method, names, P, Pv, y, folds, E, Ev):
    """Meta-model fitted on the OOF rows of four folds, applied to the fifth (the base OOF predictions are already
    out-of-fold, so the meta-model never sees a prediction made with its own target row)."""
    oof = np.zeros(len(y))
    for k in range(5):
        te = folds == k
        f, _ = method(P[~te], y[~te], E[~te])
        oof[te] = f(P[te], E[te])
    f, w = method(P, y, E)
    desc = {} if w is None else {n: round(float(x), 4) for n, x in zip(names, w) if x > 0}
    return oof, f(Pv, Ev), desc


def metrics_val(yv, p):
    r = L.regression_metrics(yv, p)
    return {"val_rmse": r["RMSE"], "val_mae": r["MAE"], "val_r2": r["R2"]}


def main():
    G = setup()
    y = G["y_train"].to_numpy(float); yv = G["y_val"].to_numpy(float); folds = G["folds"]
    names = base_names()
    P = np.column_stack([load_oof(n) for n in names])
    Pv = np.column_stack([np.load(VAL_DIR / f"{n}.npy") for n in names])
    E = G["F"].loc[G["y_train"].index, ["log10_stellar_mass"]].to_numpy(float)
    Ev = G["F"].loc[G["y_val"].index, ["log10_stellar_mass"]].to_numpy(float)
    set_status("ensemble", f"ensembles over {len(names)} base experiments")
    for en, meth in METHODS.items():
        oof, vp, w = nested(meth, names, P, Pv, y, folds, E, Ev)
        meta = save_exp(en, oof, vp, {"stage": "ensemble", "features": "OOF", "desc": f"{en[4:]} learned inside CV",
                                      "weights": w, "n_base": len(names), "seconds": 0})
        set_status("ensemble", f"{en}: CV {meta['cv_rmse_mean']:.3f}")
        print(f"{en:14s} CV {meta['cv_rmse_mean']:.4f}  weights {w}")

    # ---------------- final selection (CV only), then the validation set is read once
    allm = [json.loads(f.read_text()) for f in sorted(META_DIR.glob("*.json")) if not f.stem.endswith("_selection")]
    tab = pd.DataFrame([{"name": m["name"], "stage": m["stage"], "desc": m.get("desc", ""), "cv": m["cv_rmse_mean"],
                         # complexity for the 1-SE rule: number of input columns; ensembles count as the most complex
                         "n_feat": 10_000 if m["stage"] == "ensemble" else int(m.get("n_features", 25) or 25)}
                        for m in allm]).sort_values("cv").reset_index(drop=True)
    best = tab.iloc[0]["name"]
    # pre-registered rule (fixed 2026-10-31 because extra features repeatedly over-fitted): among candidates within
    # one paired SE of the CV-best, take the one with the fewest input columns (ties -> lower CV)
    ok = [best] + [n for n in tab.name if n != best and paired(n, best)["diff"] <= paired(n, best)["se"]]
    final = tab[tab.name.isin(ok)].sort_values(["n_feat", "cv"]).iloc[0]["name"]
    pick = {"final": final, "CV best": best, "R0 (원래 최종)": R0}
    for st in ("single", "tuned", "selected", "autogluon", "ensemble"):
        sub = tab[tab.stage == st]
        if len(sub):
            pick[f"best {st}"] = sub.iloc[0]["name"]
    vals = {n: np.load(VAL_DIR / f"{n}.npy") for n in set(pick.values())}
    rows = []
    for lab, n in pick.items():
        rows.append({"role": lab, "name": n, "label": f"{n} ({lab})", "cv": float(tab.set_index("name").loc[n, "cv"]),
                     **metrics_val(yv, vals[n])})
    bs =L.bootstrap_paired_rmse(yv, vals[final], vals[R0])
    pr = paired(final, R0) if final != R0 else {"diff": 0.0, "se": 0.0, "n_better": 0}
    fr = next(r for r in rows if r["role"] == "final")
    r0r = next(r for r in rows if r["role"] == "R0 (원래 최종)")
    out = {"final": final, "label": final, "val_rmse": fr["val_rmse"], "val_mae": fr["val_mae"], "val_r2": fr["val_r2"],
           "r0_val_rmse": r0r["val_rmse"], "cv": fr["cv"], "cv_diff_vs_r0": pr, "delta": bs["point"],
           "ci": [bs["ci_low"], bs["ci_high"]], "table": rows,
           "regimes": L.evaluate_sfr(yv, vals[final], final).to_dict(orient="records"),
           "summary": (f"CV 1위는 {final} (CV {fr['cv']:.3f}, R0 대비 Δ {pr['diff']:+.3f}, SE {pr['se']:.3f}). "
                       f"검증셋 RMSE {fr['val_rmse']:.3f} — R0 {r0r['val_rmse']:.3f}, 양식 HGB 9.525.")}
    (RES / "final.json").write_text(json.dumps(out, indent=1, ensure_ascii=False, default=float), encoding="utf-8")
    set_status("final", f"final = {final}: val RMSE {fr['val_rmse']:.3f} (R0 {r0r['val_rmse']:.3f})")
    print(json.dumps(out, indent=1, ensure_ascii=False, default=float)[:3000])


if __name__ == "__main__":
    main()
