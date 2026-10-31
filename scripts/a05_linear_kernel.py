"""Stage 1b: non-tree single models (linear, GLM, kernel, nearest neighbours) under the same 5-fold CV.

Linear models need scaled, roughly Gaussian inputs: every config uses a QuantileTransformer (normal output) fitted
on the training part of each fold only; NaNs in F5 cannot occur (F5 has none), so no imputation is needed.
usage: python scripts/a05_linear_kernel.py [name ...]
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from core import FS, L, N_THREADS, SEED, done, np, save_exp, set_status, setup  # noqa: E402
from sklearn.kernel_approximation import Nystroem  # noqa: E402
from sklearn.linear_model import LassoCV, RidgeCV, TweedieRegressor  # noqa: E402
from sklearn.neighbors import KNeighborsRegressor  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import QuantileTransformer, SplineTransformer, StandardScaler  # noqa: E402


def qt(seed):
    return QuantileTransformer(output_distribution="normal", n_quantiles=1000, subsample=200_000, random_state=seed)


def log_target(make):
    """Fit on log1p(y), back-transform with smearing estimated on a 10 % hold-out of the training part."""
    def fit(X, y, Xq, seed):
        tr, es = L._es_split(y, seed)
        m = make(seed).fit(X[tr], np.log1p(y[tr]))
        smear = float(np.mean(np.exp(np.log1p(y[es]) - m.predict(X[es]))))
        m = make(seed).fit(X, np.log1p(y))
        return np.clip(smear * np.exp(m.predict(Xq)) - 1.0, 0, None)
    return fit


def raw_target(make):
    def fit(X, y, Xq, seed):
        return np.clip(make(seed).fit(X, y).predict(Xq), 0, None)
    return fit


def ridge(seed):
    return make_pipeline(qt(seed), RidgeCV(alphas=np.logspace(-3, 3, 13)))


def ridge_spline(seed):
    return make_pipeline(qt(seed), SplineTransformer(n_knots=6, degree=3), StandardScaler(), RidgeCV(alphas=np.logspace(-2, 4, 13)))


def lasso(seed):
    return make_pipeline(qt(seed), SplineTransformer(n_knots=6, degree=3), StandardScaler(),
                         LassoCV(n_alphas=20, cv=3, n_jobs=N_THREADS, random_state=seed, max_iter=3000))


def tweedie_glm(seed):
    return make_pipeline(qt(seed), SplineTransformer(n_knots=6, degree=3), StandardScaler(),
                         TweedieRegressor(power=1.3, link="log", alpha=1e-3, max_iter=3000))


def poisson_spline(X, y, Xq, seed):
    return L.SFRModel(L.ModelSpec("glm", "poisson"), seed=seed).fit(X, y).predict(Xq)


def nystroem_ridge(seed):
    return make_pipeline(qt(seed), StandardScaler(), Nystroem(kernel="rbf", gamma=0.02, n_components=1500, random_state=seed),
                         RidgeCV(alphas=np.logspace(-3, 2, 11)))


def knn(seed):
    return make_pipeline(qt(seed), KNeighborsRegressor(n_neighbors=30, weights="distance", n_jobs=N_THREADS))


CONFIGS = {
    "M30_ridge_log_F5": (log_target(ridge), "Ridge (linear), log1p target + smearing"),
    "M31_ridge_spline_log_F5": (log_target(ridge_spline), "Ridge on cubic splines (additive model), log1p target + smearing"),
    "M32_lasso_spline_log_F5": (log_target(lasso), "Lasso on cubic splines, log1p target + smearing"),
    "M33_tweedie_glm_F5": (raw_target(tweedie_glm), "Tweedie GLM (p = 1.3, log link) on cubic splines"),
    "M34_poisson_spline_glm_F5": (poisson_spline, "Poisson GLM on cubic splines (original project's GLM)"),
    "M35_nystroem_ridge_log_F5": (log_target(nystroem_ridge), "RBF kernel ridge (Nystroem 1500), log1p target + smearing"),
    "M36_knn_F5": (raw_target(knn), "k-nearest neighbours (k = 30, distance-weighted), raw target"),
    "M37_knn_log_F5": (log_target(knn), "k-nearest neighbours (k = 30), log1p target + smearing"),
}


def run(name):
    if done(name):
        return
    fit, desc = CONFIGS[name]
    G = setup()
    F, ytr, yva, folds = G["F"], G["y_train"], G["y_val"], G["folds"]
    cols = FS["F5"]
    X = F.loc[ytr.index, cols].to_numpy(float); y = ytr.to_numpy(float)
    Xv = F.loc[yva.index, cols].to_numpy(float)
    set_status("models", f"{name}: CV")
    t0 = time.time()
    oof = np.zeros(len(y))
    for k in range(5):
        te = folds == k
        oof[te] = fit(X[~te], y[~te], X[te], SEED)
    vp = fit(X, y, Xv, SEED)
    meta = save_exp(name, oof, vp, {"stage": "single", "family": "linear/kernel", "features": "F5", "n_features": len(cols),
                                    "desc": desc, "seconds": time.time() - t0})
    set_status("models", f"{name}: CV {meta['cv_rmse_mean']:.3f} ({(time.time() - t0) / 60:.1f} min)")
    print(f"{name:28s} CV {meta['cv_rmse_mean']:.4f} ± {meta['cv_rmse_sd']:.3f}  {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    for n in (sys.argv[1:] or list(CONFIGS)):
        try:
            run(n)
        except Exception as e:  # keep the queue going; a failed model is reported, not fatal
            print(f"{n} FAILED: {e!r}", flush=True)
            set_status("models", f"{n}: failed ({type(e).__name__})")
