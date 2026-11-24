"""Pose-TGCN-style baseline on hand landmarks (Li et al., WLASL, WACV 2020), trained with our recipe.

Following the paper's description: the skeleton is a fully connected graph with K vertices; the node feature of a
keypoint is its whole trajectory (K x 2N for 2D coordinates over N frames); a graph layer is
H' = tanh(A H W) with a trainable adjacency A per layer; a residual block stacks two graph layers; blocks are stacked,
a final graph layer outputs class scores per vertex and these are average-pooled over vertices.

Differences from the original that a reader must know about:
  * K = 42 hand keypoints from MediaPipe Hands. The original used 55 OpenPose points (13 upper body + 2 x 21 hands).
  * N = 16 frames (the matched protocol), not 50 random consecutive frames.
  * Width, depth, BatchNorm and dropout are our choices (not given in the paper); depth is picked on validation.
  * Same training recipe as every other skeleton model here: AdamW 1e-3, wd 1e-2, label smoothing 0.1, batch 64,
    up to 200 epochs, early stopping on validation accuracy, the same clip augmentation.

    python -m asl_benchmark.models.tgcn            # all variants, 3 seeds each  -> results/tgcn.json
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
sys.path.insert(0, str(ROOT))
from scripts.augment import augment as augment_clip  # noqa: E402

K, T = 42, fe.T
HIDDEN, DROPOUT = 128, 0.3
BLOCKS = (2, 4, 8)                       # residual blocks tried; chosen by validation accuracy


class GraphLayer(nn.Module):
    """H' = tanh(BN(A H W)), A trainable and initialised near the identity."""

    def __init__(self, fin: int, fout: int):
        super().__init__()
        self.A = nn.Parameter(torch.eye(K) + 0.01 * torch.randn(K, K))
        self.W = nn.Linear(fin, fout)
        self.bn = nn.BatchNorm1d(K * fout)

    def forward(self, h):                                   # (B, K, F)
        h = self.W(torch.einsum("kj,bjf->bkf", self.A, h))
        return torch.tanh(self.bn(h.flatten(1)).view_as(h))


class ResBlock(nn.Module):
    def __init__(self, f: int, p: float):
        super().__init__()
        self.g1, self.g2, self.drop = GraphLayer(f, f), GraphLayer(f, f), nn.Dropout(p)

    def forward(self, h):
        return h + self.drop(self.g2(self.drop(self.g1(h))))


class PoseTGCN(nn.Module):
    def __init__(self, fin: int, classes: int = 100, blocks: int = 4, hidden: int = HIDDEN, p: float = DROPOUT):
        super().__init__()
        self.inp = GraphLayer(fin, hidden)
        self.blocks = nn.Sequential(*[ResBlock(hidden, p) for _ in range(blocks)])
        self.out = GraphLayer(hidden, classes)

    def forward(self, x):                                   # (B, K, fin) -> (B, classes)
        return self.out(self.blocks(self.inp(x))).mean(1)


# ---------------------------------------------------------------- node features
def node_features(clip: np.ndarray, variant: str) -> np.ndarray:
    """(frames, 42, 3) keypoints -> (42, F): each keypoint's trajectory over T frames, NaN -> 0.

    raw : 2D image coordinates, as the paper (F = 2T).
    norm: wrist-centred, palm-scaled xyz + wrist location + missing flag (our landmark features; F = 6T)."""
    clip = np.asarray(clip, np.float32)
    clip = clip[np.linspace(0, len(clip) - 1, T).round().astype(int)]
    if variant == "raw":
        return np.nan_to_num(clip[..., :2]).transpose(1, 0, 2).reshape(K, -1)
    x = clip.reshape(T, 2, 21, 3)
    size = np.linalg.norm(x[:, :, 9, :2] - x[:, :, 0, :2], axis=-1)[..., None, None]
    shape = (x - x[:, :, :1, :]) / (size + 1e-6)
    missing = np.isnan(shape).any(axis=(2, 3))                                   # (T, 2)
    loc = np.nan_to_num(x[:, :, 0, :2] - 0.5)                                    # (T, 2, 2) wrist location
    feats = np.concatenate([
        np.nan_to_num(shape).reshape(T, K, 3),
        np.repeat(loc[:, :, None, :], 21, axis=2).reshape(T, K, 2),
        np.repeat(missing[:, :, None], 21, axis=2).reshape(T, K, 1).astype(np.float32),
    ], axis=-1)                                                                  # (T, 42, 6)
    return feats.transpose(1, 0, 2).reshape(K, -1)


def keypoints(source: str):
    """(raw, labels): per split lists of (frames, 42, 3) arrays, in the shared sample order."""
    import scripts.run_ablations as ablations

    if source == "tracked":
        raw, labels, _ = ablations.load_wlasl()
        return raw, labels
    samples, frames = m.load_frames()
    kp = m.load_static_keypoints(samples, frames)
    raw = {k: [] for k in ("train", "val", "test")}
    labels = {k: [] for k in raw}
    for s, clip in zip(samples, kp):
        raw[s.split].append(clip)
        labels[s.split].append(s.label)
    return raw, labels


def run_graph_rung(name, raw, labels, *, variant: str, blocks: int, seeds, dev) -> dict:
    fn = lambda clip: node_features(clip, variant)
    clean = np.stack([fn(s) for s in raw["train"]])                              # (n, K, F)
    mu = clean.mean((0, 1))
    sd = clean.std((0, 1)) + 1e-6
    to_t = lambda a: torch.from_numpy(np.ascontiguousarray((a - mu) / sd, dtype=np.float32)).to(dev)
    va_x, te_x = to_t(np.stack([fn(s) for s in raw["val"]])), to_t(np.stack([fn(s) for s in raw["test"]]))
    tr_y = torch.from_numpy(np.asarray(labels["train"], np.int64)).to(dev)
    va_y, te_y = np.asarray(labels["val"], np.int64), np.asarray(labels["test"], np.int64)

    vals, tests, top5s, secs, preds = [], [], [], [], []
    for seed in seeds:
        t0 = time.perf_counter()
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        model = PoseTGCN(clean.shape[-1], 100, blocks).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
        best, best_epoch, state = -1.0, 0, None
        for epoch in range(200):
            views = [(fn(augment_clip(s, rng)) - mu) / sd for s in raw["train"]]
            x = torch.from_numpy(np.ascontiguousarray(np.stack(views), dtype=np.float32)).to(dev)
            model.train()
            for idx in torch.randperm(len(tr_y), device=dev).split(64):
                if len(idx) < 2:                                                 # BatchNorm needs more than one row
                    continue
                opt.zero_grad()
                nn.functional.cross_entropy(model(x[idx]), tr_y[idx], label_smoothing=0.1).backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            model.eval()
            with torch.inference_mode():
                acc = float((model(va_x).argmax(1).cpu().numpy() == va_y).mean())
            if acc > best:
                best, best_epoch = acc, epoch
                state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            elif epoch - best_epoch >= 40:
                break
        model.load_state_dict(state)
        model.eval()
        with torch.inference_mode():
            logits = model(te_x).cpu().numpy()
        vals.append(best)
        tests.append(float((logits.argmax(1) == te_y).mean()))
        top5s.append(_topk_accuracy(logits, te_y))
        secs.append(time.perf_counter() - t0)
        preds.append(logits.argmax(1).tolist())
    row = {
        "rung": name, "variant": variant, "blocks": blocks,
        "val_top1": round(float(np.mean(vals)) * 100, 1),
        "test_top1": round(float(np.mean(tests)) * 100, 1),
        "test_std": round(float(np.std(tests)) * 100, 1),
        "test_top5": round(float(np.mean(top5s)) * 100, 1),
        "train_s": round(float(np.mean(secs)), 1),
        "params": sum(p.numel() for p in model.parameters()),
        "seeds": list(seeds), "n_train": len(tr_y), "n_val": len(va_y), "n_test": len(te_y),
        "test_labels": te_y.tolist(), "test_preds": preds,
    }
    print(f"[TGCN] {name:<52} | Val: {row['val_top1']:5.1f}% | Test: {row['test_top1']:5.1f}% "
          f"(std {row['test_std']:4.1f}%) | Top-5: {row['test_top5']:5.1f}% | {row['train_s']:5.1f}s | "
          f"{row['params']:,} params", flush=True)
    return row


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--sources", nargs="+", default=["static", "tracked"], choices=["static", "tracked"])
    p.add_argument("--variants", nargs="+", default=["raw", "norm"], choices=["raw", "norm"])
    p.add_argument("--blocks", type=int, nargs="+", default=list(BLOCKS))
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--threads", type=int, default=3)
    p.add_argument("--out", default="results/tgcn.json")
    args = p.parse_args(argv)
    torch.set_num_threads(args.threads)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_path = ROOT / args.out
    rows = json.loads(out_path.read_text()) if out_path.exists() else []
    done = {r["rung"] for r in rows}
    for source in args.sources:
        raw, labels = keypoints(source)
        for variant in args.variants:
            for blocks in args.blocks:
                name = f"TGCN {variant} coords, {source} skeleton, {blocks} blocks"
                if name in done:
                    continue
                rows.append(run_graph_rung(name, raw, labels, variant=variant, blocks=blocks,
                                           seeds=tuple(args.seeds), dev=dev))
                out_path.write_text(json.dumps(rows, indent=1))
    print(f"saved {out_path.relative_to(ROOT)}", flush=True)


if __name__ == "__main__":
    main()
