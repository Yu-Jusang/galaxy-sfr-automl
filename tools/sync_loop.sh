#!/usr/bin/env bash
# Every 30 min: rebuild the status page and push results/meta + docs to GitHub, until the sweep finishes.
cd "$(dirname "$0")/.."
PY=/c/Users/ggnat/anaconda3/envs/sfr-ag/python.exe
while true; do
  "$PY" -X utf8 tools/build_site.py >/dev/null 2>&1
  git add -A docs results/meta results/status.json >/dev/null 2>&1
  if ! git diff --cached --quiet; then
    git commit -q -m "Progress update $(date '+%Y-%m-%d %H:%M')

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" && git push -q origin main
  fi
  grep -q "SWEEP DONE" results/logs/a03_autogluon.log 2>/dev/null && break
  sleep 1800
done
echo "sync loop finished $(date)"
