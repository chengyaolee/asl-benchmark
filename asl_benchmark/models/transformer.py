"""Small transformer over per-frame hand keypoint features (scripts/features.py).

Uses the same per-frame feature representation (132 values: palm-size-normalized
wrist-relative hand shape, missing flags, and centered 2D wrist location),
32-frame resampling, per-feature standardization, augmentation, AdamW, label
smoothing, and early stopping as ``asl_benchmark/models/rnn.py``.

Run from the repo root:
    python asl_benchmark/models/transformer.py --tag base
    python asl_benchmark/models/transformer.py --seed 1 --tag seed1
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from asl_benchmark import compare as comparison  # noqa: E402
from asl_benchmark.loader import load  # noqa: E402
from asl_benchmark.models import hands as hand_landmarks  # noqa: E402
from scripts.augment import augment as augment_clip  # noqa: E402
from scripts.features import per_frame  # noqa: E402

SEEDS = (0, 1, 2)
MAX_FRAMES = 32
BATCH = 64
PATIENCE = 40
LR = 1e-3
WD = 1e-2
SMOOTHING = 0.1


def sequence_features(clip: np.ndarray, T: int = MAX_FRAMES, location: bool = True) -> np.ndarray:
    """Compute (T, 132) resampled per-frame features matching rnn.py."""
    pts = hand_landmarks.hands(clip)
    feat = per_frame(pts, location=location)
    idx = np.linspace(0, len(feat) - 1, T).round().astype(int)
    return feat[idx].astype(np.float32)


def pack_sequences(sequences, max_frames: int = MAX_FRAMES):
    """Pad or subsample ``(frames, features)`` rows. Mask is True on padding."""
    rows = [_fit_length(np.asarray(sequence, np.float32), max_frames) for sequence in sequences]
    width = rows[0].shape[1]
    packed = np.zeros((len(rows), max_frames, width), np.float32)
    pad = np.ones((len(rows), max_frames), dtype=bool)
    for index, row in enumerate(rows):
        if row.ndim != 2 or row.shape[1] != width:
            raise ValueError("feature width changed between samples")
        packed[index, : len(row)] = row
        pad[index, : len(row)] = False
    return packed, pad


class SmallTransformer(nn.Module):
    """Two encoder layers over the resampled hand-feature sequence."""

    def __init__(
        self,
        n_features: int,
        n_classes: int,
        max_frames: int = MAX_FRAMES,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.drop_in = nn.Dropout(dropout)
        self.proj = nn.Linear(n_features, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.pos = nn.Embedding(max_frames, d_model)
        nn.init.normal_(self.pos.weight, mean=0.0, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(d_model, n_classes))

    def forward(self, x, pad=None):
        steps = torch.arange(x.shape[1], device=x.device)
        h = self.norm(self.proj(self.drop_in(x))) + self.pos(steps)
        hidden = self.encoder(h, src_key_padding_mask=pad)
        if pad is None:
            pooled = hidden.mean(dim=1)
        else:
            keep = (~pad).unsqueeze(-1)
            pooled = (hidden * keep).sum(dim=1) / keep.sum(dim=1).clamp(min=1)
        return self.head(pooled)


def train(model, packed, pad, labels, epochs: int):
    """AdamW and cross-entropy for ``epochs``. Returns model, seconds, and device."""
    dev = _device()
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    model.to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    x = torch.from_numpy(np.ascontiguousarray(packed))
    mask = torch.from_numpy(np.ascontiguousarray(pad))
    labels = torch.from_numpy(np.ascontiguousarray(labels, dtype=np.int64))
    n = int(labels.shape[0])
    started = time.perf_counter()
    model.train()
    for _ in range(epochs):
        _steps(model, opt, x, mask, labels, n, dev, smoothing=0.0)
    return model, time.perf_counter() - started, dev


def infer(
    train_seq,
    train_y,
    test_seq,
    test_y,
    n_classes: int,
    epochs: int = 200,
    val_seq=None,
    val_y=None,
    seeds: tuple[int, ...] = SEEDS,
    augment: bool = True,
    T: int = MAX_FRAMES,
    location: bool = True,
) -> comparison.Result:
    """Train on the train sequences and score validation and test."""
    clean_train = np.stack([sequence_features(seq, T=T, location=location) for seq in train_seq])
    mu = clean_train.reshape(-1, clean_train.shape[-1]).mean(axis=0)
    sd = clean_train.reshape(-1, clean_train.shape[-1]).std(axis=0) + 1e-6

    clean_train_std = (clean_train - mu) / sd
    test_x = (np.stack([sequence_features(seq, T=T, location=location) for seq in test_seq]) - mu) / sd
    test_pad = np.zeros((len(test_x), T), dtype=bool)
    val_x = None
    val_pad = None
    if val_seq is not None and val_y is not None and len(val_y) > 0:
        val_x = (np.stack([sequence_features(seq, T=T, location=location) for seq in val_seq]) - mu) / sd
        val_pad = np.zeros((len(val_x), T), dtype=bool)

    top1, top5, val_top1, seconds, latency = [], [], [], [], []
    for seed in seeds:
        model, fit_seconds, dev = _fit(
            train_seq,
            train_y,
            clean_train_std,
            mu,
            sd,
            n_classes,
            epochs,
            seed,
            augment,
            val_x,
            val_pad,
            val_y,
            T=T,
            location=location,
        )
        logits = _logits(model, test_x, test_pad, dev)
        labels = np.asarray(test_y)
        top1.append(float((logits.argmax(axis=1) == labels).mean()))
        top5.append(_topk_accuracy(logits, labels))
        if val_x is not None and val_y is not None:
            val_logits = _logits(model, val_x, val_pad, dev)
            val_top1.append(float((val_logits.argmax(axis=1) == np.asarray(val_y)).mean()))
        seconds.append(fit_seconds)
        started = time.perf_counter()
        _logits(model, test_x, test_pad, dev)
        latency.append((time.perf_counter() - started) * 1000 / len(test_y))
    return comparison.Result(
        "transformer",
        float(np.mean(top1)),
        float(np.mean(seconds)),
        float(np.mean(latency)),
        len(train_y),
        len(test_y),
        None if not val_top1 else float(np.mean(val_top1)),
        float(np.mean(top5)),
        0 if val_y is None else len(val_y),
        float(np.std(top1)),
    )


def _fit(
    train_seq,
    train_y,
    clean_train_std: np.ndarray,
    mu: np.ndarray,
    sd: np.ndarray,
    n_classes: int,
    epochs: int,
    seed: int,
    augment: bool,
    val_x: np.ndarray | None,
    val_pad: np.ndarray | None,
    val_y,
    T: int = MAX_FRAMES,
    location: bool = True,
):
    dev = _device()
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    width = clean_train_std.shape[-1]
    dropout = 0.3 if augment else 0.0
    smoothing = SMOOTHING if augment else 0.0
    model = SmallTransformer(width, n_classes, max_frames=T, dropout=dropout).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD if augment else 0.0)
    labels = torch.from_numpy(np.ascontiguousarray(train_y, dtype=np.int64))
    rng = np.random.default_rng(seed)
    best_state = None
    best_score = -1.0
    best_epoch = 0
    pad_zeros = torch.zeros((len(train_y), T), dtype=torch.bool)
    started = time.perf_counter()
    for epoch in range(epochs):
        if augment:
            views = [
                (sequence_features(augment_clip(hand_landmarks.hands(seq), rng), T=T, location=location) - mu) / sd
                for seq in train_seq
            ]
            x = torch.from_numpy(np.ascontiguousarray(np.stack(views), dtype=np.float32))
        else:
            x = torch.from_numpy(np.ascontiguousarray(clean_train_std, dtype=np.float32))
        model.train()
        _steps(model, opt, x, pad_zeros, labels, len(train_y), dev, smoothing=smoothing)
        score = _val_score(model, val_x, val_pad, val_y, dev)
        if score is None or score > best_score:
            best_score = 0.0 if score is None else score
            best_epoch = epoch
            best_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
        elif epoch - best_epoch >= PATIENCE:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, time.perf_counter() - started, dev


def _val_score(model, val_x, val_pad, val_y, dev) -> float | None:
    if val_x is None or val_pad is None or val_y is None:
        return None
    logits = _logits(model, val_x, val_pad, dev)
    return float((logits.argmax(axis=1) == np.asarray(val_y)).mean())


def _steps(model, opt, x, mask, labels, n: int, dev, smoothing: float = SMOOTHING) -> None:
    for index in torch.randperm(n).split(BATCH):
        opt.zero_grad()
        out = model(x[index].to(dev), mask[index].to(dev))
        loss = nn.functional.cross_entropy(out, labels[index].to(dev), label_smoothing=smoothing)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _logits(model, packed, pad, dev) -> np.ndarray:
    x = torch.from_numpy(np.ascontiguousarray(packed))
    mask = torch.from_numpy(np.ascontiguousarray(pad))
    rows = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(packed), BATCH):
            stop = start + BATCH
            rows.append(model(x[start:stop].to(dev), mask[start:stop].to(dev)).cpu().numpy())
    return np.concatenate(rows)


def _topk_accuracy(scores: np.ndarray, labels: np.ndarray, k: int = 5) -> float:
    labels = np.asarray(labels)
    k = min(k, int(scores.shape[1]))
    chosen = np.argpartition(scores, -k, axis=1)[:, -k:]
    return float((chosen == labels.reshape(-1, 1)).any(axis=1).mean())


def _fit_length(row: np.ndarray, max_frames: int) -> np.ndarray:
    if len(row) <= max_frames:
        return row
    index = np.linspace(0, len(row) - 1, max_frames).round().astype(np.int64)
    return row[index]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tag", default="default")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--T", type=int, default=MAX_FRAMES)
    p.add_argument("--no-augment", dest="augment", action="store_false")
    p.add_argument("--no-location", dest="location", action="store_false")
    args = p.parse_args()

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

    res = infer(
        raw["train"],
        labels["train"],
        raw["test"],
        labels["test"],
        100,
        epochs=args.epochs,
        val_seq=raw["val"],
        val_y=labels["val"],
        seeds=(args.seed,),
        augment=args.augment,
        T=args.T,
        location=args.location,
    )
    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    payload = {
        "model": "transformer",
        "tag": args.tag,
        "val_top1": res.val_top1,
        "test_top1": res.top1,
        "test_top5": res.top5,
        "train_seconds": round(res.train_seconds, 2),
        "latency_ms_per_clip": round(res.latency_ms, 3),
        "n_train": res.n_train,
        "n_val": res.n_val,
        "n_test": res.n_test,
        "config": vars(args),
    }
    print(json.dumps(payload, indent=2))
    (out_dir / f"transformer_{args.tag}.json").write_text(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
