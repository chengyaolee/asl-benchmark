#!/usr/bin/env bash
# Matched-input skeleton-vs-image rungs (UPDATED PLAN.md) -> results/matched_ablations.json.
# Waits for run_pixel_queue.sh so GPU timings stay comparable. Restart-safe: finished rungs are skipped.
# Needs the caches from:  venv/bin/python -m asl_benchmark.models.matched --prepare
cd "$(dirname "$0")/.."
PY=venv/bin/python
OUT=results/matched_ablations.json
ORDER="S1 C1 C5 S2 S4 C3 C4 C2a S5 C2b S3"
stamp() { echo "[$(date '+%F %T')] $*"; }
while pgrep -f "scripts/run_pixel_queue" > /dev/null; do sleep 60; done
stamp "matched-input queue"
for key in $ORDER; do
    if [ -f $OUT ] && $PY -c "import json,sys; sys.exit(0 if any(r.get('key')=='$key' for r in json.load(open('$OUT'))) else 1)"; then
        stamp "   $key done, skipping"; continue; fi
    stamp "   $key"
    $PY -m asl_benchmark.models.matched --rungs $key --out $OUT \
        2>&1 | grep --line-buffered -v -i "warning\|W0000\|I0000" >> logs/matched.log || stamp "   $key FAILED"
done
stamp "matched-input queue done"
