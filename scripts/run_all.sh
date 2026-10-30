#!/usr/bin/env bash
# Overnight sweep: single models, then AutoGluon variants, then ensembles. Logs to results/logs/.
set -u
cd "$(dirname "$0")/.."
PY=${PY:-/c/Users/ggnat/anaconda3/envs/sfr-ag/python.exe}
export SFR_N_THREADS=12 PYTHONWARNINGS=ignore
mkdir -p results/logs
"$PY" -X utf8 scripts/a02_models.py 2>&1 | grep -v -i "warn" | tee -a results/logs/a02_models.log
for c in AG3_F5 AG6_F5_tweedie AG4_F5A AG2_F3 AG5_ALL AG1_F0 AG7_F5_log; do
  "$PY" -X utf8 scripts/a03_autogluon.py "$c" > "results/logs/$c.log" 2>&1
  tail -1 "results/logs/$c.log" | tee -a results/logs/a03_autogluon.log
done
echo "SWEEP DONE $(date)" | tee -a results/logs/a03_autogluon.log
