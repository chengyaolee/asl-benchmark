#!/usr/bin/env bash
# Rerun the ResNet frame-encoder rungs on the shared-frame inputs (frames chosen from the MediaPipe
# keypoint rows, so pixels and skeletons are the same frames). Waits for run_pixel_queue.sh so GPU
# timings stay comparable. Restart-safe: finished rungs are skipped.   -> results/frame_encoder_shared.json
cd "$(dirname "$0")/.."
PY=venv/bin/python
OUT=results/frame_encoder_shared.json
stamp() { echo "[$(date '+%F %T')] $*"; }
while pgrep -f "scripts/run_pixel_queue" > /dev/null; do sleep 60; done
stamp "shared-frame frame encoder"
for rung in 1 2 3 4 5; do
    if [ -f $OUT ] && $PY -c "import json,sys; sys.exit(0 if any(r['rung'].startswith('$rung. ') for r in json.load(open('$OUT'))) else 1)"; then
        stamp "   rung $rung done, skipping"; continue; fi
    stamp "   rung $rung"
    $PY -m asl_benchmark.models.frame_encoder --rungs $rung --out $OUT \
        2>&1 | grep --line-buffered -v Warning >> logs/frame_encoder_shared.log || stamp "   rung $rung FAILED"
done
stamp "shared-frame queue done"
