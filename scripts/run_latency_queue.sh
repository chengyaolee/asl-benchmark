#!/usr/bin/env bash
# End-to-end latency (video file -> prediction, nothing cached). Runs last, with the GPU and CPU otherwise idle,
# so the timings are clean.   -> results/end_to_end_latency.json, logs/latency.log
cd "$(dirname "$0")/.."
while pgrep -f "scripts/run_matched_queue|scripts/run_curves_queue|scripts/run_baselines_queue|scripts/run_pixel_queue|models\.tgcn" > /dev/null; do sleep 60; done
echo "[$(date '+%F %T')] end-to-end latency"
venv/bin/python scripts/end_to_end_latency.py 2>&1 | grep --line-buffered -v -i "warning\|W0000\|I0000\|INFO" >> logs/latency.log \
    || echo "[$(date '+%F %T')] FAILED"
echo "[$(date '+%F %T')] latency done"
