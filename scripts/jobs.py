"""Job list for running the sweep on several machines, and a one-command runner per machine.

usage:
  python scripts/jobs.py list                 # show every job, its machine and estimated hours
  python scripts/jobs.py run <A|B|C|D>        # run all jobs assigned to that machine (skips finished ones)
  python scripts/jobs.py pack <A|B|C|D>       # zip that machine's results into results/pack_<M>.zip
  python scripts/jobs.py unpack <zip> ...     # merge result zips from other machines (on machine A)

Machine roles (see report/distribution_plan.pdf):
  A  coordinator laptop (Windows, 16 threads, 16 GB, no GPU)      cheap single models + feature-axis AutoGluon + final ensemble
  B  Apple-silicon Mac (MPS)                                      neural / kernel / foundation models + NN tuning
  C  any CPU machine with >= 16 GB RAM                            nested Optuna tuning (trees) + two AutoGluon runs
  D  machine with a CUDA GPU if available (Colab, lab server)     AutoGluon preprocessing axis + extreme_quality
"""
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable

M = "scripts/a02_models.py"; LK = "scripts/a05_linear_kernel.py"; DP = "scripts/a06_deep.py"
AG = "scripts/a03_autogluon.py"; TU = "scripts/a07_tune.py"
LT = "scripts/a08_tune_linear.py"; NT = "scripts/a09_tune_nn.py"

# (job id, machine, script, args, result names, estimated hours on that machine)
JOBS = [
    ("J01", "A", M, ["M71_lgbm_tw13_reg_F5LS", "M72_xgb_tw13_F5LS", "M73_cat_tw13_d8_F5LS", "M74_et_F5LS",
                     "M75_rf_F5LS", "M76_mlp_F5LS", "M77_lgbm_logsmear_F5LS"], None, 1.0),
    ("J02", "A", LT, [f"LK_{f}_{fs}{t}" for fs in ("F5", "F5LS") for f in
                      ("ridge", "lasso", "tweedie_glm", "kernel_ridge", "knn") for t in ("", "_tuned")], None, 3.5),
    ("J03", "A", AG, ["AG8_F5L"], None, 1.5),
    ("J04", "A", AG, ["AG9_F5S"], None, 1.5),
    ("J05", "A", AG, ["AG2_F3"], None, 1.5),
    ("J06", "A", AG, ["AG1_F0"], None, 1.5),
    ("J07", "B", DP, [f"{k}_{fs}" for fs in ("F5", "F5LS") for k in
                      ("M40_mlp_tweedie", "M41_tabm", "M42_realmlp", "M43_realmlp_log", "M44_xrfm_log", "M45_tabpfn")],
     None, 4.0),
    ("J19", "B", NT, ["NN_tabm_F5LS_tuned", "NN_realmlp_F5LS_tuned"], None, 7.0),
    ("J09", "C", TU, ["lgbm", "50"], ["T01_lgbm_tuned"], 2.5),
    ("J10", "C", TU, ["xgb", "40"], ["T02_xgb_tuned"], 3.0),
    ("J11", "C", TU, ["cat", "25"], ["T03_cat_tuned"], 3.5),
    ("J12", "D", AG, ["AG10_F5LS"], None, 1.5),
    ("J13", "D", AG, ["AG11_F5X"], None, 1.5),
    ("J14", "D", AG, ["AG12_F5LS_tweedie"], None, 1.5),
    ("J15", "D", AG, ["AG13_F5LS_log"], None, 1.5),
    ("J16", "D", AG, ["AG15_F5LS_wmassive"], None, 1.5),
    ("J17", "D", AG, ["AG18_F5LS_extreme"], None, 2.0),
    ("J18", "C", AG, ["AG14_F5LS_sqrt", "AG16_F5LS_stack2"], None, 3.0),
    ("J08", "D", AG, ["AG17_F5LS_long"], None, 4.0),   # optional: run last, if time allows
]


def names(job):
    jid, mach, script, args, res, hrs = job
    return res if res is not None else [a for a in args if not a.startswith("--")]


def is_done(name):
    return (ROOT / "results" / "meta" / f"{name}.json").exists()


def cmd_list():
    tot = {}
    for j in JOBS:
        st = "done" if all(is_done(n) for n in names(j)) else "todo"
        print(f"{j[0]}  {j[1]}  {j[5]:4.1f} h  {st:4s}  {Path(j[2]).name:22s} {' '.join(j[3])}")
        tot[j[1]] = tot.get(j[1], 0) + j[5]
    print("hours per machine:", {k: round(v, 1) for k, v in sorted(tot.items())})


def cmd_run(mach):
    env = dict(os.environ, SFR_MACHINE=mach, PYTHONWARNINGS="ignore")
    if mach == "C":
        env.setdefault("SFR_N_THREADS", str(os.cpu_count() or 8))
    for j in [j for j in JOBS if j[1] == mach]:
        if all(is_done(n) for n in names(j)):
            print(f"{j[0]} already done", flush=True)
            continue
        print(f"=== {j[0]}: {j[2]} {' '.join(j[3])}", flush=True)
        log = ROOT / "results" / "logs" / f"{j[0]}_{mach}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        with open(log, "a", encoding="utf-8") as fh:
            extra = ["--force"] if j[2] == AG else []
            subprocess.run([PY, "-X", "utf8", j[2], *j[3], *extra], cwd=ROOT, env=env, stdout=fh,
                           stderr=subprocess.STDOUT)
        print(f"{j[0]} {'finished' if all(is_done(n) for n in names(j)) else 'FAILED - see ' + str(log)}", flush=True)


def cmd_pack(mach):
    out = ROOT / "results" / f"pack_{mach}.zip"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for j in [j for j in JOBS if j[1] == mach]:
            for n in names(j):
                for sub, ext in (("oof", ".parquet"), ("val", ".npy"), ("meta", ".json"), ("meta", "_leaderboard.csv"),
                                 ("meta", "_chosen.csv")):
                    f = ROOT / "results" / sub / f"{n}{ext}"
                    if f.exists():
                        z.write(f, f"results/{sub}/{f.name}")
            log = ROOT / "results" / "logs" / f"{j[0]}_{mach}.log"
            if log.exists():
                z.write(log, f"results/logs/{log.name}")
    print("wrote", out)


def cmd_unpack(zips):
    for zp in zips:
        with zipfile.ZipFile(zp) as z:
            z.extractall(ROOT)
            print(zp, "->", len(z.namelist()), "files")


if __name__ == "__main__":
    a = sys.argv[1:]
    {"list": lambda: cmd_list(), "run": lambda: cmd_run(a[1]), "pack": lambda: cmd_pack(a[1]),
     "unpack": lambda: cmd_unpack(a[1:])}[a[0]]()
