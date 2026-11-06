"""GRU / LSTM on per-frame hand keypoint features for WLASL100.

Same data, features, augmentation, early stopping and result format as mlp.py, but
the clip is kept as a sequence of frames (no temporal pooling): the recurrent
model reads it one frame at a time.

Run from the repo root:
    python asl_benchmark/models/rnn.py --cell gru --tag gru_base
    python asl_benchmark/models/rnn.py --cell lstm --bidirectional --readout mean --tag lstm_bi_mean

Results go to results/rnn_<tag>.json.
"""

import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from asl_benchmark.loader import load
from scripts.features import per_frame
from scripts.augment import augment

p = argparse.ArgumentParser()
p.add_argument("--tag", default="default")
p.add_argument("--cell", default="gru", choices=("gru", "lstm"))
p.add_argument("--hidden", type=int, default=128)
p.add_argument("--layers", type=int, default=1)
p.add_argument("--bidirectional", action="store_true")
p.add_argument("--readout", default="last", choices=("last", "mean"),
               help="last: final hidden state; mean: average of outputs over time")
p.add_argument("--T", type=int, default=32)
p.add_argument("--dropout", type=float, default=0.3)
p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--wd", type=float, default=1e-2)
p.add_argument("--batch", type=int, default=64)
p.add_argument("--epochs", type=int, default=200)
p.add_argument("--patience", type=int, default=40)
p.add_argument("--smoothing", type=float, default=0.1)
p.add_argument("--seed", type=int, default=0)
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


def sequence(clip):
    feat = per_frame(clip, location=args.location)                  # (frames, D)
    idx = np.linspace(0, len(feat) - 1, args.T).round().astype(int)
    return feat[idx]                                                 # (T, D)


def features(clips, aug=False):
    return torch.tensor(np.stack([sequence(augment(c, rng) if aug else c) for c in clips]),
                        dtype=torch.float32)


# Per-feature scaling from clean train frames; val/test prepared once, never augmented.
Xtr_clean = features(raw["train"])
mu = Xtr_clean.reshape(-1, Xtr_clean.shape[-1]).mean(0)
sd = Xtr_clean.reshape(-1, Xtr_clean.shape[-1]).std(0) + 1e-6
Xtr_eval = (Xtr_clean - mu) / sd
Xva = (features(raw["val"]) - mu) / sd
Xte = (features(raw["test"]) - mu) / sd
D = Xtr_clean.shape[-1]
print(f"train {len(Xtr_eval)}  val {len(Xva)}  test {len(Xte)}  sequence {args.T}x{D}  "
      f"cell={args.cell}  layers={args.layers}  hidden={args.hidden}  bi={args.bidirectional}  "
      f"readout={args.readout}  augment={args.augment}  location={args.location}")


class RNN(nn.Module):
    def __init__(self):
        super().__init__()
        rnn = nn.GRU if args.cell == "gru" else nn.LSTM
        self.drop_in = nn.Dropout(args.dropout)
        self.rnn = rnn(D, args.hidden, num_layers=args.layers, batch_first=True,
                       bidirectional=args.bidirectional,
                       dropout=args.dropout if args.layers > 1 else 0.0)
        width = args.hidden * (2 if args.bidirectional else 1)
        self.head = nn.Sequential(nn.Dropout(args.dropout), nn.Linear(width, 100))

    def forward(self, x):
        out, h = self.rnn(self.drop_in(x))
        if args.readout == "mean":
            z = out.mean(1)
        else:
            h = h[0] if isinstance(h, tuple) else h          # LSTM returns (h, c)
            dirs = 2 if args.bidirectional else 1
            z = torch.cat([h[-dirs + i] for i in range(dirs)], dim=1)  # last layer, all directions
        return self.head(z)


model = RNN()
n_params = sum(q.numel() for q in model.parameters())
print(f"model: {n_params:,} parameters")
opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)


def accuracy(X, yy, k=1):
    model.eval()
    with torch.no_grad():
        top = model(X).topk(k, 1).indices
    return (top == yy[:, None]).any(1).float().mean().item()


out_dir = ROOT / "results"; out_dir.mkdir(exist_ok=True)
ckpt = out_dir / f"rnn_{args.tag}.pt"
best, best_epoch, t0 = -1.0, 0, time.time()
history = []
for epoch in range(args.epochs):
    Xtr = (features(raw["train"], aug=True) - mu) / sd if args.augment else Xtr_eval
    model.train()
    batch_losses = []
    for i in torch.randperm(len(Xtr)).split(args.batch):
        loss = nn.functional.cross_entropy(model(Xtr[i]), y["train"][i], label_smoothing=args.smoothing)
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        batch_losses.append(loss.item())
    val = accuracy(Xva, y["val"])
    with torch.no_grad():
        val_loss = nn.functional.cross_entropy(model(Xva), y["val"]).item()
    train_acc = accuracy(Xtr_eval, y["train"])
    history.append({"epoch": epoch, "train_loss": float(np.mean(batch_losses)), "val_loss": val_loss,
                    "train_acc": train_acc, "val_acc": val})
    if val > best:
        best, best_epoch = val, epoch
        torch.save(model.state_dict(), ckpt)
    if epoch % 10 == 0:
        print(f"epoch {epoch:3d}  loss {np.mean(batch_losses):.3f}  train {train_acc:.3f}  val {val:.3f}")
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
    "model": args.cell, "tag": args.tag, "params": n_params, "best_epoch": best_epoch,
    "val_top1": best, "test_top1": accuracy(Xte, y["test"]), "test_top5": accuracy(Xte, y["test"], k=5),
    "train_seconds": round(train_time, 1), "latency_ms_per_clip": round(latency_ms, 3),
    "n_train": len(Xtr_eval), "n_val": len(Xva), "n_test": len(Xte),
    "config": vars(args), "history": history,
}
print(json.dumps({k: v for k, v in results.items() if k != "history"}, indent=2))
(out_dir / f"rnn_{args.tag}.json").write_text(json.dumps(results, indent=2))
