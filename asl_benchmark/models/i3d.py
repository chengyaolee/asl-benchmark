"""I3D baseline (Carreira & Zisserman; the WLASL authors' implementation and Kinetics weights), fine-tuned on the
matched 16 x 224 x 224 frames.

This is the image baseline published with WLASL (Li et al., WACV 2020). Same recipe as the other fine-tuned image models: backbone 1e-4, new head 1e-3,
AdamW (wd 1e-2), cosine, label smoothing 0.1, fp16, batch 8, up to 40 epochs, early stopping on validation accuracy, the
same clip augmentation. Per-frame class scores (T' = 2 for 16 frames) are averaged. No horizontal flip (it swaps the
signing hand).

    scripts/fetch_i3d.sh                       # once: 50 MB weights
    python -m asl_benchmark.models.i3d         # 3 seeds -> results/i3d.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np
import torch
from torch import nn

from asl_benchmark.models import frame_encoder as fe
from asl_benchmark.models import matched as m
from asl_benchmark.models import video_backbone as vb
from asl_benchmark.models.cnn import _topk_accuracy

ROOT = vb.ROOT
sys.path.insert(0, str(ROOT / "third_party/pytorch_i3d"))
from pytorch_i3d import InceptionI3d  # noqa: E402

WEIGHTS = ROOT / "third_party/pytorch_i3d/rgb_imagenet.pt"
EPOCHS, PATIENCE, BATCH = 40, 10, 8
LR_BACKBONE, LR_HEAD, WD, SMOOTHING = 1e-4, 1e-3, 1e-2, 0.1
T = fe.T


class I3D(nn.Module):
    def __init__(self, classes: int = 100, pretrained: bool = True):
        super().__init__()
        self.net = InceptionI3d(400, in_channels=3)
        if pretrained:
            self.net.load_state_dict(torch.load(WEIGHTS, map_location="cpu", weights_only=True))
        self.net.replace_logits(classes)

    def forward(self, x):                        # x: (B, 3, T, H, W) in [-1, 1]
        return self.net(x).mean(-1)              # (B, classes, T') -> average over time


def to_input(frames: np.ndarray) -> torch.Tensor:
    """(B, T, H, W, 3) uint8 or float in [0, 1] -> (B, 3, T, H, W) in [-1, 1]."""
    x = torch.from_numpy(np.ascontiguousarray(frames))
    x = x.float() / 255.0 if x.dtype == torch.uint8 else x
    return (x * 2 - 1).permute(0, 4, 1, 2, 3).contiguous()


@torch.inference_mode()
def predict(model, frames, dev, batch: int = 8) -> np.ndarray:
    model.eval()
    rows = []
    for i in range(0, len(frames), batch):
        with torch.autocast("cuda", dtype=torch.float16, enabled=dev.type == "cuda"):
            rows.append(model(to_input(frames[i : i + batch]).to(dev)).float().cpu().numpy())
    return np.concatenate(rows)


def fine_tune(train, val, seed, dev, epochs=EPOCHS):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    rng = np.random.default_rng(seed)
    (x, y), (vx, vy) = train, val
    model = I3D().to(dev)
    head = list(model.net.logits.parameters())
    ids = {id(p) for p in head}
    opt = torch.optim.AdamW([{"params": [p for p in model.parameters() if id(p) not in ids], "lr": LR_BACKBONE},
                             {"params": head, "lr": LR_HEAD}], weight_decay=WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    scaler = torch.amp.GradScaler(enabled=dev.type == "cuda")
    ty = torch.from_numpy(y)
    best, best_epoch, state = -1.0, 0, None
    for epoch in range(epochs):
        model.train()
        for idx in torch.randperm(len(ty)).split(BATCH):
            views = [fe.augment(x[i][:, None], np.zeros((T, 0), np.float32), rng)[0][:, 0] for i in idx.numpy()]
            inp = to_input(np.stack(views)).to(dev)
            opt.zero_grad()
            with torch.autocast("cuda", dtype=torch.float16, enabled=dev.type == "cuda"):
                loss = nn.functional.cross_entropy(model(inp), ty[idx].to(dev), label_smoothing=SMOOTHING)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
        sched.step()
        acc = float((predict(model, vx, dev).argmax(1) == vy).mean())
        print(f"    epoch {epoch:2d}  loss {loss.item():.3f}  val {acc * 100:5.1f}%", flush=True)
        if acc > best:
            best, best_epoch = acc, epoch
            state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        elif epoch - best_epoch >= PATIENCE:
            break
    model.load_state_dict(state)
    return model


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--out", default="results/i3d.json")
    args = p.parse_args(argv)
    torch.backends.cudnn.benchmark = True
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    samples, frames = m.load_frames()
    d = fe._by_split(samples, frames)
    (tr_x, tr_y), (va_x, va_y), (te_x, te_y) = d["train"], d["val"], d["test"]
    vals, tests, top5s, secs, lats, preds = [], [], [], [], [], []
    params = 0
    for seed in args.seeds:
        t0 = time.perf_counter()
        model = fine_tune((tr_x, tr_y), (va_x, va_y), seed, dev, args.epochs)
        secs.append(time.perf_counter() - t0)
        va_logits, te_logits = predict(model, va_x, dev), predict(model, te_x, dev)
        lats.append(fe._latency(lambda: (predict(model, te_x, dev, batch=1), len(te_x))[1], dev))
        params = sum(p.numel() for p in model.parameters())
        vals.append(float((va_logits.argmax(1) == va_y).mean()))
        tests.append(float((te_logits.argmax(1) == te_y).mean()))
        top5s.append(_topk_accuracy(te_logits, te_y))
        preds.append(te_logits.argmax(1).tolist())
        print(f"  seed {seed}: val {vals[-1] * 100:.1f}  test {tests[-1] * 100:.1f}  ({secs[-1]:.0f}s)", flush=True)
        del model
        torch.cuda.empty_cache()
    row = {
        "rung": "I3D (Kinetics), fine-tuned, 16 x 224 frames (+aug)",
        "val_top1": round(float(np.mean(vals)) * 100, 1), "test_top1": round(float(np.mean(tests)) * 100, 1),
        "test_std": round(float(np.std(tests)) * 100, 1), "test_top5": round(float(np.mean(top5s)) * 100, 1),
        "train_s": round(float(np.mean(secs)), 1), "latency_ms": round(float(np.mean(lats)), 3), "params": int(params),
        "seeds": list(args.seeds), "n_train": len(tr_y), "n_val": len(va_y), "n_test": len(te_y),
        "test_labels": te_y.tolist(), "test_preds": preds,
    }
    print(f"[I3D] {row['rung']} | Val: {row['val_top1']:.1f}% | Test: {row['test_top1']:.1f}% (std {row['test_std']:.1f}%) "
          f"| Top-5: {row['test_top5']:.1f}% | {row['train_s']:.0f}s | {row['latency_ms']:.2f} ms/clip", flush=True)
    out = ROOT / args.out
    out.write_text(json.dumps([row], indent=1))
    print(f"saved {out.relative_to(ROOT)}", flush=True)


if __name__ == "__main__":
    main()
