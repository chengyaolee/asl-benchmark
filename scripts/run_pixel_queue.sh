#!/usr/bin/env bash
# GPU queue for the pixel-side experiments, one job at a time so training times stay comparable.
# Safe to re-run: finished steps are skipped (checked against the results JSONs).
#
#   1. R(2+1)D learning-rate sweep: hand crop, seed 0, lr in {3e-5, 1e-4, 3e-4}  -> results/backbone_sweep.json
#   2. ResNet-18 frame encoder + transformer head, all rungs x 3 seeds          -> results/frame_encoder_ablations.json
#   3. R(2+1)D rungs 4-5 x 3 seeds, 60 epochs, cosine, best sweep LR by val      -> results/backbone_long.json
#   4. Summary of every new row                                                  -> logs/pixel_summary.txt
#
# Detached launch (survives closing the terminal / Claude session):
#   setsid nohup scripts/run_pixel_queue.sh > logs/queue.log 2>&1 < /dev/null &
cd "$(dirname "$0")/.."
PY=venv/bin/python
stamp() { echo "[$(date '+%F %T')] $*"; }

has_row() {  # has_row <json> <substring of rung name>
    $PY -c "import json,sys; rows=json.load(open('$1')); sys.exit(0 if any(sys.argv[1] in r['rung'] for r in rows) else 1)" "$2" 2>/dev/null
}

# Wait for any sweep already running (e.g. launched from an earlier session).
while pgrep -f "python -m asl_benchmark.models.video_backbone.*backbone_sweep" > /dev/null; do sleep 60; done

stamp "1. sweep"
for lr in 3e-5 1e-4 3e-4; do
    if has_row results/backbone_sweep.json "lr=$lr]"; then stamp "   lr=$lr done, skipping"; continue; fi
    stamp "   lr=$lr"
    $PY -m asl_benchmark.models.video_backbone --rungs 5 --seeds 0 --epochs 60 --patience 10 --cosine \
        --lr-backbone "$lr" --tag " [60ep cos lr=$lr]" --out results/backbone_sweep.json \
        2>&1 | grep --line-buffered -v Warning >> logs/backbone_sweep.log || stamp "   lr=$lr FAILED"
done

stamp "2. frame encoder"
for rung in 1 2 3 4 5; do
    if has_row results/frame_encoder_ablations.json "$rung. "; then stamp "   rung $rung done, skipping"; continue; fi
    stamp "   rung $rung"
    $PY -m asl_benchmark.models.frame_encoder --rungs "$rung" \
        2>&1 | grep --line-buffered -v Warning >> logs/frame_encoder.log || stamp "   rung $rung FAILED"
done

LR=$($PY -c "
import json, re
rows = json.load(open('results/backbone_sweep.json'))
best = max(rows, key=lambda r: r['val_top1'])
print(re.search(r'lr=([0-9.e-]+)', best['rung']).group(1))" 2>/dev/null || echo 1e-4)
stamp "3. R(2+1)D long fine-tuning, selected lr=$LR"
for rung in 4 5; do
    if has_row results/backbone_long.json "$rung. "; then stamp "   rung $rung done, skipping"; continue; fi
    stamp "   rung $rung"
    $PY -m asl_benchmark.models.video_backbone --rungs "$rung" --epochs 60 --patience 10 --cosine \
        --lr-backbone "$LR" --tag " [60ep cos lr=$LR]" --out results/backbone_long.json \
        2>&1 | grep --line-buffered -v Warning >> logs/backbone_long.log || stamp "   rung $rung FAILED"
done

stamp "4. summary"
$PY - > logs/pixel_summary.txt 2>&1 <<'EOF'
import json, os
for path in ("results/backbone_sweep.json", "results/frame_encoder_ablations.json", "results/backbone_long.json"):
    print(f"== {path}")
    if not os.path.exists(path):
        print("   (missing)"); continue
    for r in json.load(open(path)):
        print(f"   {r['rung']:<55} val {r['val_top1']:5.1f}  test {r['test_top1']:5.1f} ± {r['test_std']:3.1f}"
              f"  top5 {r['test_top5']:5.1f}  train {r['train_s']:7.1f}s  {r.get('latency_ms', float('nan')):.2f} ms")
EOF
stamp "queue done"
