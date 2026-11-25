#!/usr/bin/env bash
# Transformer ladder first (GPU idle), then P0: R(2+1)D, letterbox, 32 frames, no augmentation, seed 0. Restart-safe.
cd "$(dirname "$0")/.."
venv/bin/python -u scripts/run_transformer_ladder.py > logs/transformer_ladder.log 2>&1 || echo "transformer ladder FAILED"
venv/bin/python -u -m asl_benchmark.models.r2plus1d_32 --no-augment --seeds 0 > logs/r2plus1d_32_noaug.log 2>&1 || echo "P0 FAILED"
echo "queue done $(date)"
