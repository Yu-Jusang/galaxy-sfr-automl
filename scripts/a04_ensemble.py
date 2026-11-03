"""Stage 3: ensembles of the out-of-fold predictions, final selection, one test evaluation.

Ensemble weights are learned *inside* the CV: for outer fold k the weights are fitted on the OOF rows of the
other four folds and applied to fold k, so the ensemble's CV RMSE is an honest out-of-sample estimate.
The final weights are refitted on all OOF rows and applied to the stored test predictions (the template's 20% X_val).
Final choice (1-SE rule, fixed before the test split was read) = the lowest-CV candidate over every single model,
AutoGluon run and ensemble, unless it is an ensemble within one paired SE of a non-ensemble candidate. Afterwards every candidate's
test RMSE is written to results/test_table.csv for transparency; it is never used for the choice.
usage: python scripts/a04_ensemble.py [--cv-only]   (--cv-only: ensembles + selection check, no test scoring)
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from scipy.optimize import nnls  # noqa: E402

from core import META_DIR, RES, VAL_DIR, L, fold_rmse, load_oof, np, paired, pd, save_exp, set_status, setup  # noqa: E402

R0 = "M01_lgbm_tw13_F5"


def base_names(y):
    """Base experiments for the ensembles. A model whose OOF predictions are non-finite or exceed 10x the largest
    training SFR has diverged (e.g. a log-link GLM overflowing); it stays in the candidate table but is not blended.
    The check reads OOF predictions only, never the test set."""
    out, cap = [], 10 * y.max()
    for f in sorted(META_DIR.glob("*.json")):
        m = json.loads(f.read_text())
        if m.get("stage") in ("single", "autogluon", "tuned", "selected"):
            p = load_oof(m["name"])
            if np.isfinite(p).all() and p.max() <= cap:
                out.append(m["name"])
            else:
                print(f"excluded from ensembles (diverged OOF, max {np.nanmax(p):.3g}): {m['name']}")
    return out


def simplex_weights(P, y):
    """Non-negative weights summing to 1 minimising squared error (NNLS on a sum-to-one penalised system)."""
    lam = 1e3 * np.sqrt(len(y))
    A = np.vstack([P, lam * np.ones((1, P.shape[1]))])
    b = np.concatenate([y, [lam]])
    w, _ = nnls(A, b)
    return w / w.sum()


def greedy_weights(P, y, steps=40, init=0):
    """Caruana-style forward selection with replacement (Caruana et al. 2004). init > 0 starts from the init best
    models (sorted initialisation, Caruana et al. 2006)."""
    counts = np.zeros(P.shape[1])
    if init:
        counts[np.argsort(((P - y[:, None]) ** 2).mean(0))[:init]] = 1
    cur = P @ counts / max(counts.sum(), 1)
    for s in range(int(counts.sum()), steps):
        j = int(np.argmin(((((cur * s)[:, None] + P) / (s + 1) - y[:, None]) ** 2).mean(0)))
        counts[j] += 1
        cur = (cur * s + P[:, j]) / (s + 1)
    return counts / counts.sum()


def bagged_greedy(P, y, bags=20, frac=0.5, init=3, steps=40, seed=42):
    """Bagged ensemble selection (Caruana et al. 2006): sorted-init greedy selection on random halves of the
    library, weights averaged over the bags (curbs hill-climbing overfit on a large, correlated library)."""
    rng = np.random.default_rng(seed)
    m = P.shape[1]; w = np.zeros(m)
    for _ in range(bags):
        sub = rng.choice(m, max(init, int(frac * m)), replace=False)
        w[sub] += greedy_weights(P[:, sub], y, steps, init)
    return w / bags


def prune(P, y, r=0.99):
    """Drop near-duplicate base models: walking down the RMSE order, keep a model only if its predictions correlate
    below r with every model kept so far (Caruana et al. 2006; Salinas & Erickson 2024, TabRepo). Uses only the rows
    it is given, so inside the nested CV the held-out fold plays no part in the pruning."""
    order = np.argsort(((P - y[:, None]) ** 2).mean(0))
    C = np.nan_to_num(np.corrcoef(P.T), nan=1.0)
    keep = []
    for j in order:
        if all(C[j, i] < r for i in keep):
            keep.append(j)
    return np.array(keep)


def pruned(fn):
    def w(P, y):
        keep = prune(P, y)
        out = np.zeros(P.shape[1]); out[keep] = fn(P[:, keep], y)
        return out
    return w


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
           "ENS_stack_nnls": stack_nnls, "ENS_stack_lgbm": stack_lgbm,
           "ENS_greedy_bag": as_weights(bagged_greedy),
           "ENS_simplex_pruned": as_weights(pruned(simplex_weights)),
           "ENS_greedy_bag_pruned": as_weights(pruned(bagged_greedy))}


def family(name):
    """Model family, for the diversity report of the ensemble weights (TabArena: check cross-family blends)."""
    n = name.lower()
    if n.startswith("ag"):
        return "autogluon"
    if "tabpfn" in n:
        return "tabpfn"
    if any(k in n for k in ("mlp", "tabm", "nn_")):
        return "neural"
    if any(k in n for k in ("lgbm", "xgb", "cat", "hgb", "_et_", "_rf_", "t0", "sel_")):
        return "tree"
    return "linear/kernel"


def selection_check(names):
    """How optimistic is 'pick the CV-best'? Reads fold RMSEs only (before the test split is touched).
    nested: for each fold k the candidate with the lowest mean RMSE on the other four folds, scored on fold k
    (Varma & Simon 2006; Cawley & Talbot 2010). tt: Tibshirani & Tibshirani (2009) bias correction of the minimum.
    within_1se: candidates whose paired fold difference to the CV-best is smaller than its standard error."""
    E = np.array([fold_rmse(n) for n in names])
    best = int(np.argmin(E.mean(1)))
    picks = [int(np.argmin(np.delete(E, k, 1).mean(1))) for k in range(5)]
    D = E - E[best]
    se = D.std(1, ddof=1) / np.sqrt(5)
    ties = [names[i] for i in np.argsort(E.mean(1)) if i != best and D[i].mean() < se[i]]
    cv = float(E[best].mean()); bias = float((E[best] - E.min(0)).mean())
    return {"cv_best": names[best], "cv_min": cv,
            "nested_selection_cv": float(np.mean([E[p, k] for k, p in enumerate(picks)])),
            "nested_picks": [names[p] for p in picks], "tt_bias": bias, "tt_corrected_cv": cv + bias,
            "within_1se": ties[:10]}


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
    names = base_names(y)
    P = np.column_stack([load_oof(n) for n in names])
    Pv = np.column_stack([np.load(VAL_DIR / f"{n}.npy") for n in names])
    E = G["F"].loc[G["y_train"].index, ["log10_stellar_mass"]].to_numpy(float)
    Ev = G["F"].loc[G["y_val"].index, ["log10_stellar_mass"]].to_numpy(float)
    set_status("ensemble", f"ensembles over {len(names)} base experiments")
    for en, meth in METHODS.items():
        oof, vp, w = nested(meth, names, P, Pv, y, folds, E, Ev)
        fam = {}
        for n, x in w.items():
            fam[family(n)] = round(fam.get(family(n), 0) + x, 4)
        meta = save_exp(en, oof, vp, {"stage": "ensemble", "features": "OOF", "desc": f"{en[4:]} learned inside CV",
                                      "weights": w, "family_weights": fam, "n_base": len(names), "seconds": 0})
        set_status("ensemble", f"{en}: CV {meta['cv_rmse_mean']:.3f}")
        print(f"{en:22s} CV {meta['cv_rmse_mean']:.4f}  families {fam}  weights {w}", flush=True)

    # ---------------- final selection (CV only), then the validation set is read once
    allm = [json.loads(f.read_text()) for f in sorted(META_DIR.glob("*.json")) if not f.stem.endswith("_selection")]
    sel = selection_check([m["name"] for m in allm])
    print("selection check:", json.dumps(sel, ensure_ascii=False), flush=True)
    if "--cv-only" in sys.argv:     # dry run: ensembles + selection check, the test split is never scored
        return
    tab = pd.DataFrame([{"name": m["name"], "stage": m["stage"], "features": m.get("features", ""),
                         "n_features": m.get("n_features"), "desc": m.get("desc", ""), "cv": m["cv_rmse_mean"]}
                        for m in allm]).sort_values("cv").reset_index(drop=True)
    # rule (fixed 2026-11-03, before the test split was read; 80% training split only): 1-SE. The CV-best candidate,
    # unless it is an ensemble and a non-ensemble candidate is within one paired SE of it (fold differences,
    # SE = sd / sqrt 5); then the CV-best such non-ensemble candidate (Cawley & Talbot 2010: selection over many
    # candidates overfits, so a gain inside the noise does not justify the extra layer).
    cv_best = tab.iloc[0]["name"]
    final = cv_best
    if tab.iloc[0]["stage"] == "ensemble":
        def within(n):
            d = fold_rmse(n) - fold_rmse(cv_best)
            return d.mean() < d.std(ddof=1) / np.sqrt(5)
        simple = [n for n, st in zip(tab.name, tab.stage) if st != "ensemble" and within(n)]
        final = simple[0] if simple else cv_best
    # the 20% test split is read only after the choice; every candidate's test score is published for transparency
    vals = {n: np.load(VAL_DIR / f"{n}.npy") for n in tab.name}
    tab = tab.assign(**pd.DataFrame([metrics_val(yv, vals[n]) for n in tab.name]).rename(
        columns={"val_rmse": "test_rmse", "val_mae": "test_mae", "val_r2": "test_r2"}))
    tab["cv_rank"] = np.arange(1, len(tab) + 1)
    tab["test_rank"] = tab.test_rmse.rank(method="min").astype(int)
    tab.to_csv(RES / "test_table.csv", index=False, encoding="utf-8-sig")
    pick = {"final": final, "CV 1위": cv_best, "R0 (원래 최종)": R0}
    for st in ("single", "tuned", "selected", "autogluon", "ensemble"):
        sub = tab[tab.stage == st]
        if len(sub):
            pick[f"best {st}"] = sub.iloc[0]["name"]
    pick["test 1위 (참고, 선택에 안 씀)"] = tab.sort_values("test_rmse").iloc[0]["name"]
    rows = []
    for lab, n in pick.items():
        rows.append({"role": lab, "name": n, "label": f"{n} ({lab})", "cv": float(tab.set_index("name").loc[n, "cv"]),
                     **metrics_val(yv, vals[n])})
    bs =L.bootstrap_paired_rmse(yv, vals[final], vals[R0])
    pr = paired(final, R0) if final != R0 else {"diff": 0.0, "se": 0.0, "n_better": 0}
    fr = next(r for r in rows if r["role"] == "final")
    r0r = next(r for r in rows if r["role"] == "R0 (원래 최종)")
    out = {"final": final, "cv_best": cv_best, "rule": "1-SE: non-ensemble preferred within one paired SE of an ensemble CV-best",
           "label": final, "val_rmse": fr["val_rmse"], "val_mae": fr["val_mae"], "val_r2": fr["val_r2"],
           "r0_val_rmse": r0r["val_rmse"], "cv": fr["cv"], "cv_diff_vs_r0": pr, "delta": bs["point"],
           "ci": [bs["ci_low"], bs["ci_high"]], "table": rows,
           "final_test_rank": int(tab.set_index("name").loc[final, "test_rank"]), "n_candidates": len(tab),
           "selection_check": sel, "regimes": L.evaluate_sfr(yv, vals[final], final).to_dict(orient="records"),
           "summary": (f"최종(1-SE 규칙)은 {final}, CV 1위는 {cv_best}. 최종 (CV {fr['cv']:.3f}, R0 대비 Δ {pr['diff']:+.3f}, SE {pr['se']:.3f}). "
                       f"test RMSE {fr['val_rmse']:.3f} (후보 {len(tab)}개 중 test {int(tab.set_index('name').loc[final, 'test_rank'])}위)"
                       f" — R0 {r0r['val_rmse']:.3f}, 양식 HGB 9.525.")}
    (RES / "final.json").write_text(json.dumps(out, indent=1, ensure_ascii=False, default=float), encoding="utf-8")
    set_status("final", f"final = {final}: test RMSE {fr['val_rmse']:.3f} (R0 {r0r['val_rmse']:.3f})")
    sys.stdout.reconfigure(encoding="utf-8")     # Korean summary on a cp949 Windows console
    print(json.dumps(out, indent=1, ensure_ascii=False, default=float)[:3000])


if __name__ == "__main__":
    main()
