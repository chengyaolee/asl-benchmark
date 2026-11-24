#!/usr/bin/env bash
# GPU queue for the pixel-side experiments, one job at a time so training times stay comparable.
#   1. wait for the R(2+1)D learning-rate sweep (results/backbone_sweep.json)
#   2. ResNet-18 frame encoder + transformer head, all rungs x 3 seeds
#   3. R(2+1)D fine-tuning rungs 4-5 x 3 seeds, 60 epochs, cosine, best sweep LR (by val)
set -euo pipefail
cd "$(dirname "$0")/.."
PY=venv/bin/python

while pgrep -f "video_backbone.*backbone_sweep" > /dev/null; do sleep 60; done

$PY -m asl_benchmark.models.frame_encoder 2>&1 | grep --line-buffered -v Warning > logs/frame_encoder.log

LR=$($PY -c "
import json, re
rows = json.load(open('results/backbone_sweep.json'))
best = max(rows, key=lambda r: r['val_top1'])
print(re.search(r'lr=([0-9.e-]+)', best['rung']).group(1))")
echo "selected backbone lr=$LR" > logs/backbone_long.log

$PY -m asl_benchmark.models.video_backbone --rungs 4 5 --epochs 60 --patience 10 --cosine \
    --lr-backbone "$LR" --tag " [60ep cos lr=$LR]" --out results/backbone_long.json \
    2>&1 | grep --line-buffered -v Warning >> logs/backbone_long.log
echo "queue done"
