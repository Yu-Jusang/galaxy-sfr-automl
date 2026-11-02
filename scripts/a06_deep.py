"""Stage 1c: neural, kernel and foundation tabular models under the same 5-fold CV (CPU only).

  TabM (Gorishniy et al. 2025)       pytabkit TabM_D_Regressor (tuned defaults), raw target
  RealMLP (Holzmüller et al. 2024)   pytabkit RealMLP_TD_Regressor (tuned defaults), raw and log1p target
  xRFM (Beaglehole et al. 2025)      tree-partitioned Recursive Feature Machines, log1p target + smearing
  TabPFN v2 (Hollmann et al. 2025)   pre-trained transformer, in-context learning; on CPU the context is a stratified
                                     sample of 10,000 training galaxies (the model's pre-training limit)
  MLP with Tweedie loss (PyTorch)    our own net trained directly on the Tweedie deviance, log link
usage: python scripts/a06_deep.py [name ...]
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from core import FS, L, N_THREADS, SEED, done, epoch_time_cap, impute_missing, np, save_exp, set_status, setup  # noqa: E402

import os  # noqa: E402
import torch  # noqa: E402

os.environ.setdefault("TABPFN_ALLOW_CPU_LARGE_DATASET", "1")
FIT_SECONDS = int(os.environ.get("DEEP_FIT_SECONDS", "900"))  # cap per fit for TabM / RealMLP
PFN_BATCH = 500   # TabPFN prediction batch: memory grows with batch x features (2,000 needed 34 GB on F5LS)

torch.set_num_threads(N_THREADS)
DEVICE = os.environ.get("SFR_DEVICE") or ("cuda" if torch.cuda.is_available() else   # SFR_DEVICE=cpu if MPS fails
                                         "mps" if torch.backends.mps.is_available() else "cpu")


def smear_log(fit_log):
    """Wrap a log1p-target fitter: smearing factor from a 10 % stratified hold-out of the training part."""
    def fit(X, y, Xq, seed):
        tr, es = L._es_split(y, seed)
        pe = fit_log(X[tr], np.log1p(y[tr]), X[es], seed)
        smear = float(np.mean(np.exp(np.log1p(y[es]) - pe)))
        return np.clip(smear * np.exp(fit_log(X, np.log1p(y), Xq, seed)) - 1.0, 0, None)
    return fit


def tabm(X, y, Xq, seed):
    from pytabkit import TabM_D_Regressor
    m = TabM_D_Regressor(device=DEVICE, random_state=seed, n_threads=N_THREADS, batch_size=1024, verbosity=1)
    tr, es = L._es_split(y, seed)
    with epoch_time_cap(FIT_SECONDS):   # verbosity=1 routes the epoch loop through the capped tqdm
        m.fit(X[tr], y[tr], X_val=X[es], y_val=y[es])
    return np.clip(m.predict(Xq), 0, None)


def realmlp(target):
    def fit_raw(X, y, Xq, seed):
        from pytabkit import RealMLP_TD_Regressor
        m = RealMLP_TD_Regressor(device=DEVICE, random_state=seed, n_threads=N_THREADS, verbosity=0)
        tr, es = L._es_split(y, seed)
        m.fit(X[tr], y[tr], X_val=X[es], y_val=y[es], time_to_fit_in_seconds=FIT_SECONDS)
        return np.clip(m.predict(Xq), 0, None)

    def fit_log(X, z, Xq, seed):
        from pytabkit import RealMLP_TD_Regressor
        m = RealMLP_TD_Regressor(device=DEVICE, random_state=seed, n_threads=N_THREADS, verbosity=0)
        idx = np.arange(len(z)); rng = np.random.default_rng(seed); rng.shuffle(idx)
        cut = int(0.9 * len(z))
        m.fit(X[idx[:cut]], z[idx[:cut]], X_val=X[idx[cut:]], y_val=z[idx[cut:]], time_to_fit_in_seconds=FIT_SECONDS)
        return m.predict(Xq)
    return fit_raw if target == "raw" else smear_log(fit_log)


def xrfm_log(X, z, Xq, seed):
    from sklearn.preprocessing import QuantileTransformer
    from xrfm import xRFM
    qt = QuantileTransformer(output_distribution="normal", n_quantiles=1000, random_state=seed)
    Z = qt.fit_transform(X).astype(np.float32); Zq = qt.transform(Xq).astype(np.float32)
    idx = np.arange(len(z)); rng = np.random.default_rng(seed); rng.shuffle(idx)
    cut = int(0.9 * len(z))
    m = xRFM(max_leaf_size=10_000, device="cpu", n_threads=N_THREADS, random_state=seed, verbose=False,
             tuning_metric="mse")
    m.fit(Z[idx[:cut]], z[idx[:cut]].astype(np.float32), Z[idx[cut:]], z[idx[cut:]].astype(np.float32))
    return np.asarray(m.predict(Zq), float).ravel()


def tabpfn(X, y, Xq, seed, n_ctx=10_000, batch=PFN_BATCH):
    from sklearn.model_selection import train_test_split
    from tabpfn import TabPFNRegressor
    idx, _ = train_test_split(np.arange(len(y)), train_size=n_ctx, random_state=seed, stratify=L.strata_labels(y))
    m = TabPFNRegressor(n_estimators=4, device=DEVICE, random_state=seed, n_jobs=N_THREADS, ignore_pretraining_limits=True)
    m.fit(X[idx], y[idx])
    out = np.concatenate([m.predict(Xq[i:i + batch]) for i in range(0, len(Xq), batch)])
    return np.clip(out, 0, None)


def mlp_tweedie(X, y, Xq, seed, p=1.3, epochs=60):
    """MLP whose output is log mu; trained on the Tweedie deviance (the same loss as the final LightGBM)."""
    from sklearn.preprocessing import QuantileTransformer
    torch.manual_seed(seed)
    qt = QuantileTransformer(output_distribution="normal", n_quantiles=1000, random_state=seed)
    tr, es = L._es_split(y, seed)
    Z = torch.tensor(qt.fit_transform(X[tr]), dtype=torch.float32); Ze = torch.tensor(qt.transform(X[es]), dtype=torch.float32)
    Zq = torch.tensor(qt.transform(Xq), dtype=torch.float32)
    yt = torch.tensor(y[tr], dtype=torch.float32); ye = y[es]
    d = Z.shape[1]
    net = torch.nn.Sequential(torch.nn.Linear(d, 256), torch.nn.SiLU(), torch.nn.Dropout(0.1), torch.nn.Linear(256, 256),
                              torch.nn.SiLU(), torch.nn.Dropout(0.1), torch.nn.Linear(256, 128), torch.nn.SiLU(),
                              torch.nn.Linear(128, 1))
    with torch.no_grad():
        net[-1].bias.fill_(float(np.log(y[tr].mean())))
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)

    def dev(f, yy):  # Tweedie deviance up to constants, f = log mu
        return (-yy * torch.exp((1 - p) * f) / (1 - p) + torch.exp((2 - p) * f) / (2 - p)).mean()

    best, best_state, bad = np.inf, None, 0
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(len(yt))
        for i in range(0, len(perm), 1024):
            b = perm[i:i + 1024]
            opt.zero_grad(); loss = dev(net(Z[b]).squeeze(1), yt[b]); loss.backward(); opt.step()
        sched.step(); net.eval()
        with torch.no_grad():
            pe = torch.exp(net(Ze).squeeze(1)).numpy()
        r = np.sqrt(np.mean((pe - ye) ** 2))
        if r < best:
            best, bad = r, 0
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= 8:
                break
    net.load_state_dict(best_state); net.eval()
    with torch.no_grad():
        return torch.exp(net(Zq).squeeze(1)).numpy().astype(float)


_BASE = {
    "M40_mlp_tweedie": (mlp_tweedie, "MLP 256-256-128 trained on the Tweedie deviance (p = 1.3, log link)"),
    "M41_tabm": (tabm, "TabM (tuned defaults), raw target"),
    "M42_realmlp": (realmlp("raw"), "RealMLP (tuned defaults), raw target"),
    "M43_realmlp_log": (realmlp("log"), "RealMLP (tuned defaults), log1p target + smearing"),
    "M44_xrfm_log": (smear_log(xrfm_log), "xRFM (tree-partitioned Recursive Feature Machine), log1p target + smearing"),
    "M45_tabpfn": (tabpfn, "TabPFN v2, 4 estimators, context = 10,000 stratified training galaxies"),
}
# every model on F5 (original), F5C (curated literature) and F5LS (F5 + literature + teammate 1)
CONFIGS = {f"{k}_{fs}": (fn, d, fs) for fs in ("F5", "F5C", "F5LS") for k, (fn, d) in _BASE.items()}


def run(name):
    if done(name):
        return
    fit, desc, fs = CONFIGS[name]
    G = setup()
    F, ytr, yva, folds = G["F"], G["y_train"], G["y_val"], G["folds"]
    cols = FS[fs]
    X = F.loc[ytr.index, cols].to_numpy(float); y = ytr.to_numpy(float)
    Xv = F.loc[yva.index, cols].to_numpy(float)
    t0 = time.time()
    oof = np.zeros(len(y))
    for k in range(5):
        set_status("models", f"{name}: fold {k + 1}/5")
        te = folds == k
        Xa, Xb = impute_missing(X[~te], X[te])   # medians from the training folds only
        oof[te] = fit(Xa, y[~te], Xb, SEED)
    set_status("models", f"{name}: refit for validation")
    Xf, Xvf = impute_missing(X, Xv)
    vp = fit(Xf, y, Xvf, SEED)
    meta = save_exp(name, oof, vp, {"stage": "single", "family": "neural/kernel/foundation", "features": fs,
                                    "n_features": len(cols), "desc": desc, "seconds": time.time() - t0})
    set_status("models", f"{name}: CV {meta['cv_rmse_mean']:.3f} ({(time.time() - t0) / 60:.0f} min)")
    print(f"{name:24s} CV {meta['cv_rmse_mean']:.4f} ± {meta['cv_rmse_sd']:.3f}  {(time.time() - t0) / 60:.0f} min", flush=True)


if __name__ == "__main__":
    for n in [a for a in sys.argv[1:] if not a.startswith("--")] or list(CONFIGS):
        try:
            run(n)
        except Exception as e:
            print(f"{n} FAILED: {e!r}", flush=True)
            set_status("models", f"{n}: failed ({type(e).__name__})")
