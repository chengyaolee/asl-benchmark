#!/usr/bin/env bash
# Learning curves (skeletons + frozen pixel models) after the matched queue, so GPU timings stay comparable.
# Restart-safe (scripts/learning_curves.py skips finished triples).   -> results/learning_curves.json
cd "$(dirname "$0")/.."
while pgrep -f "scripts/run_matched_queue" > /dev/null; do sleep 60; done
echo "[$(date '+%F %T')] learning curves"
venv/bin/python scripts/learning_curves.py 2>&1 | grep --line-buffered -v -i "warning\|W0000\|I0000" >> logs/curves.log \
    || echo "[$(date '+%F %T')] FAILED"
echo "[$(date '+%F %T')] learning curves done"
