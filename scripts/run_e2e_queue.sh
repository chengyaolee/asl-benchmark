#!/usr/bin/env bash
# End-to-end timing, only after the P0 queue has finished (needs an idle machine).
cd "$(dirname "$0")/.."
while pgrep -f "scripts/run_p0_queue.sh|models.r2plus1d_32" > /dev/null; do sleep 20; done
venv/bin/python -u scripts/end_to_end_paper.py > logs/end_to_end_paper.log 2>&1 || echo "timing FAILED"
tail -8 logs/end_to_end_paper.log
