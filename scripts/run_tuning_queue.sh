#!/usr/bin/env bash
# Tuning budget for the better picture model (ResNet on hand crops or I3D, chosen by validation accuracy).
# Runs after every other queue, including the latency measurement, so their timings stay clean.
# Restart-safe (scripts/tune_picture.py skips finished runs).   -> results/tuning.json, logs/tuning.log
cd "$(dirname "$0")/.."
while pgrep -f "scripts/run_matched_queue|scripts/run_curves_queue|scripts/run_baselines_queue|scripts/run_latency_queue|scripts/run_pixel_queue|models\.tgcn" > /dev/null; do sleep 60; done
echo "[$(date '+%F %T')] picture-model tuning"
venv/bin/python scripts/tune_picture.py 2>&1 | grep --line-buffered -v -i "warning\|W0000\|I0000" >> logs/tuning.log \
    || echo "[$(date '+%F %T')] FAILED"
echo "[$(date '+%F %T')] tuning done"
