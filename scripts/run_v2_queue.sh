#!/usr/bin/env bash
# New runs for the short plan, after I3D has finished (the GPU runs one job at a time so timings stay comparable):
#   X1  hand-crop ResNet + wrist location (3 seeds, about 47 min, includes the shuffled-frame test)  -> results/v2_ladder.json
#   X5  end-to-end speed, machine idle                                                              -> results/end_to_end_latency.json
#   then the paired statistics, the figures and the summary table.
# Already run, on the CPU and a light GPU load: S1 rerun, X2, X3 (and X4 for them), in results/v2_ladder.json.
# Restart-safe: finished steps are skipped.
cd "$(dirname "$0")/.."
PY=venv/bin/python
stamp() { echo "[$(date '+%F %T')] $*"; }
while pgrep -f "scripts/run_baselines_queue|models\.i3d" > /dev/null; do sleep 60; done

has_key() { [ -f results/v2_ladder.json ] && $PY -c "import json,sys; sys.exit(0 if any(r.get('key')=='$1' for r in json.load(open('results/v2_ladder.json'))) else 1)"; }
if has_key X1; then stamp "X1 done, skipping"; else
    stamp "X1"
    $PY -m asl_benchmark.models.ladder --runs X1 2>&1 | grep --line-buffered -v -i "warning\|W0000\|I0000" >> logs/v2.log || stamp "X1 FAILED"
fi
if [ -f results/end_to_end_latency.json ]; then stamp "speed done, skipping"; else
    stamp "end-to-end speed"
    $PY scripts/end_to_end_latency.py 2>&1 | grep --line-buffered -v -i "warning\|W0000\|I0000\|INFO" >> logs/latency.log || stamp "speed FAILED"
fi
stamp "statistics and figures"
$PY scripts/paired_stats.py > /dev/null 2>&1 || stamp "stats FAILED"
$PY scripts/v2_figures.py >> logs/v2.log 2>&1 || stamp "figures FAILED"
$PY scripts/matched_summary.py > /dev/null 2>&1
stamp "v2 queue done"
