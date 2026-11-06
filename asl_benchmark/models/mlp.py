"""MLP on hand keypoints for WLASL100.

Run from the repo root:
    python asl_benchmark/models/mlp.py                       # augmentation + location
    python asl_benchmark/models/mlp.py --no-augment --tag noaug
    python asl_benchmark/models/mlp.py --no-location --tag noloc
    python asl_benchmark/models/mlp.py --seed 1 --tag seed1

Results go to results/mlp_<tag>.json.
"""

import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from asl_benchmark.loader import load
from scripts.features import prep
from scripts.augment import augment

p = argparse.ArgumentParser()
p.add_argument("--tag", default="default")
p.add_argument("--seed", type=int, default=0)
p.add_argument("--T", type=int, default=32)
p.add_argument("--hidden", type=int, default=512)
p.add_argument("--dropout", type=float, default=0.5)
p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--wd", type=float, default=1e-2)
p.add_argument("--batch", type=int, default=64)
p.add_argument("--epochs", type=int, default=300)
p.add_argument("--patience", type=int, default=60)
p.add_argument("--layers", type=int, default=2)
p.add_argument("--in-dropout", type=float, default=0.0)
p.add_argument("--smoothing", type=float, default=0.1)
p.add_argument("--pool", default="none", choices=("none", "meanstd", "chunks4"))
p.add_argument("--velocity", action="store_true")
p.add_argument("--no-augment", dest="augment", action="store_false")
p.add_argument("--no-location", dest="location", action="store_false")
args = p.parse_args()

torch.manual_seed(args.seed)
rng = np.random.default_rng(args.seed)

# Raw keypoints per split.
kp_dir = ROOT / "data/wlasl100/keypoints"
raw = {"train": [], "val": [], "test": []}
labels = {"train": [], "val": [], "test": []}
for s in load("wlasl100", ROOT / "data", download=False).samples:
    f = kp_dir / f"{s.sample_id}.npy"
    if f.exists():
        a = np.load(f)
        if len(a) > 0:
            raw[s.split].append(a)
            labels[s.split].append(s.label)
y = {k: torch.tensor(v, dtype=torch.long) for k, v in labels.items()}

def features(clips, aug=False):
    return torch.tensor(np.stack([
        prep(augment(c, rng) if aug else c, T=args.T, location=args.location,
             pool=args.pool, velocity=args.velocity) for c in clips
    ]), dtype=torch.float32)

# Scaling uses clean train features only; val/test are prepared once and never augmented.
Xtr_clean = features(raw["train"])
mu, sd = Xtr_clean.mean(0), Xtr_clean.std(0) + 1e-6
Xva = (features(raw["val"]) - mu) / sd
Xte = (features(raw["test"]) - mu) / sd
Xtr_eval = (Xtr_clean - mu) / sd
print(f"train {len(Xtr_clean)}  val {len(Xva)}  test {len(Xte)}  features {Xtr_clean.shape[1]}"
      f"  augment={args.augment}  location={args.location}"
      f"  pool={args.pool}  velocity={args.velocity}")

# layers=0 is plain logistic regression; each extra layer halves the width.
blocks, d = [], Xtr_clean.shape[1]
if args.in_dropout > 0:
    blocks.append(nn.Dropout(args.in_dropout))
for i in range(args.layers):
    h = max(args.hidden // (2 ** i), 16)
    blocks += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(args.dropout)]
    d = h
blocks.append(nn.Linear(d, 100))
model = nn.Sequential(*blocks)
n_params = sum(p.numel() for p in model.parameters())
print(f"model: {args.layers} hidden layers, {n_params:,} parameters")
opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

def accuracy(X, yy, k=1):
    model.eval()
    with torch.no_grad():
        top = model(X).topk(k, 1).indices
    return (top == yy[:, None]).any(1).float().mean().item()

out_dir = ROOT / "results"; out_dir.mkdir(exist_ok=True)
ckpt = out_dir / f"mlp_{args.tag}.pt"
best, best_epoch, t0 = -1.0, 0, time.time()
for epoch in range(args.epochs):
    # A new random version of every training clip each epoch.
    Xtr = (features(raw["train"], aug=True) - mu) / sd if args.augment else Xtr_eval
    model.train()
    for i in torch.randperm(len(Xtr)).split(args.batch):
        loss = nn.functional.cross_entropy(model(Xtr[i]), y["train"][i], label_smoothing=args.smoothing)
        opt.zero_grad(); loss.backward(); opt.step()
    val = accuracy(Xva, y["val"])
    if val > best:
        best, best_epoch = val, epoch
        torch.save(model.state_dict(), ckpt)
    if epoch % 10 == 0:
        print(f"epoch {epoch:3d}  loss {loss.item():.3f}  "
              f"train {accuracy(Xtr_eval, y['train']):.3f}  val {val:.3f}")
    if epoch - best_epoch >= args.patience:
        print(f"stopping: no val improvement for {args.patience} epochs")
        break
train_time = time.time() - t0

model.load_state_dict(torch.load(ckpt)); model.eval()
with torch.no_grad():
    t1 = time.time()
    for _ in range(100):
        model(Xte[:1])
    latency_ms = (time.time() - t1) * 10

results = {
    "model": "mlp", "tag": args.tag, "params": n_params, "best_epoch": best_epoch,
    "val_top1": best, "test_top1": accuracy(Xte, y["test"]),
    "test_top5": accuracy(Xte, y["test"], k=5),
    "train_seconds": round(train_time, 1), "latency_ms_per_clip": round(latency_ms, 3),
    "n_train": len(Xtr_eval), "n_val": len(Xva), "n_test": len(Xte),
    "config": vars(args),
}
print(json.dumps(results, indent=2))
(out_dir / f"mlp_{args.tag}.json").write_text(json.dumps(results, indent=2))
