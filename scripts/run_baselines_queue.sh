#!/usr/bin/env bash
# I3D baseline (Kinetics, the WLASL authors' implementation) on the matched frames, after the other GPU queues.
# Restart-safe: skipped if results/i3d.json exists.   -> results/i3d.json, logs/i3d.log
cd "$(dirname "$0")/.."
while pgrep -f "scripts/run_matched_queue|scripts/run_curves_queue|scripts/run_pixel_queue" > /dev/null; do sleep 60; done
if [ -f results/i3d.json ]; then echo "[$(date '+%F %T')] i3d done, skipping"; exit 0; fi
echo "[$(date '+%F %T')] i3d"
venv/bin/python -m asl_benchmark.models.i3d 2>&1 | grep --line-buffered -v -i "warning\|W0000\|I0000" >> logs/i3d.log \
    || echo "[$(date '+%F %T')] FAILED"
echo "[$(date '+%F %T')] i3d done"
