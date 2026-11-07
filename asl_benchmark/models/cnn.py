"""Appearance CNN on raw frames. It never sees the landmark values.

Training uses the same optimization and evaluation conventions as the pose
models: train-only augmentation, AdamW with weight decay, label smoothing,
early stopping on validation, and mean over three seeds.
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

SEEDS = (0, 1, 2)
N_CLIP_FRAMES = 8
FRAME_SIZE = 64
BATCH = 32
PATIENCE = 20
LR = 1e-3
WD = 1e-2
SMOOTHING = 0.1


def frame_span(n_frames: int, frame_start: int | None, frame_end: int | None) -> tuple[int, int]:
    """WLASL's 1-based inclusive window as a half-open frame index range.

    ``frame_end`` of -1, or a missing end, means the last frame.
    """
    start = 0 if frame_start is None else min(n_frames, max(0, int(frame_start) - 1))
    if frame_end is None or int(frame_end) < 0:
        end = n_frames
    else:
        end = min(n_frames, int(frame_end))
    if end <= start:
        return 0, n_frames
    return start, end


def read_clip(sample, n_frames: int = N_CLIP_FRAMES, size: int = FRAME_SIZE):
    """Sample ``n_frames`` RGB frames from the clip window. None if there is no file."""
    if sample.video_path is None or not sample.video_path.exists() or sample.video_path.stat().st_size == 0:
        return None
    import cv2

    cap = cv2.VideoCapture(str(sample.video_path))
    try:
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if count <= 0:
            return None
        start, end = frame_span(count, sample.frame_start, sample.frame_end)
        if end <= start:
            return None
        wanted = np.linspace(start, end - 1, n_frames).round().astype(np.int64)
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(start))
        pos = start
        last = None
        frames = []
        for index in wanted:
            while pos <= index:
                ok, frame = cap.read()
                pos += 1
                if not ok:
                    last = None
                    break
                last = frame
            if last is None:
                frames.append(np.zeros((size, size, 3), np.uint8))
                continue
            rgb = cv2.cvtColor(last, cv2.COLOR_BGR2RGB)
            frames.append(cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA))
        return np.stack(frames)
    finally:
        cap.release()


class CNN(nn.Module):
    """Three convolutions with BatchNorm, averaged over the sampled frames."""

    def __init__(self, n_classes: int, dropout: float = 0.3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(128, n_classes))

    def forward(self, x):
        batch, steps, channels, height, width = x.shape
        feat = self.conv(x.reshape(batch * steps, channels, height, width)).flatten(1)
        return self.head(feat.view(batch, steps, -1).mean(dim=1))


def train(model, batch, labels, epochs: int):
    """AdamW and cross-entropy for ``epochs``. Returns model, seconds, and device."""
    dev = _device()
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    model.to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    frames = torch.from_numpy(np.ascontiguousarray(batch))
    targets = torch.from_numpy(np.ascontiguousarray(labels, dtype=np.int64))
    n = int(targets.shape[0])
    started = time.perf_counter()
    model.train()
    for _ in range(epochs):
        _steps(model, opt, frames, targets, n, dev, smoothing=0.0)
    return model, time.perf_counter() - started, dev


def infer(
    train_clips,
    train_y,
    test_clips,
    test_y,
    n_classes: int,
    epochs: int = 50,
    val_clips=None,
    val_y=None,
    seeds: tuple[int, ...] = SEEDS,
    augment: bool = True,
) -> comparison.Result:
    """Train on the train clips and score validation and test."""
    test_x = _batch(test_clips)
    val_x = None if not val_clips else _batch(val_clips)
    top1, top5, val_top1, seconds, latency = [], [], [], [], []
    for seed in seeds:
        model, fit_seconds, dev = _fit(
            train_clips, train_y, val_x, val_y, n_classes, epochs, seed, augment
        )
        logits = _logits(model, test_x, dev)
        labels = np.asarray(test_y)
        top1.append(float((logits.argmax(axis=1) == labels).mean()))
        top5.append(_topk_accuracy(logits, labels))
        if val_x is not None and val_y is not None:
            val_logits = _logits(model, val_x, dev)
            val_top1.append(float((val_logits.argmax(axis=1) == np.asarray(val_y)).mean()))
        seconds.append(fit_seconds)
        started = time.perf_counter()
        _logits(model, test_x, dev)
        latency.append((time.perf_counter() - started) * 1000 / len(test_y))
    return comparison.Result(
        "cnn",
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


def _fit(train_clips, train_y, val_x, val_y, n_classes: int, epochs: int, seed: int, augment: bool):
    dev = _device()
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    dropout = 0.3 if augment else 0.0
    smoothing = SMOOTHING if augment else 0.0
    model = CNN(n_classes, dropout=dropout).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD if augment else 0.0)
    labels = torch.from_numpy(np.ascontiguousarray(train_y, dtype=np.int64))
    rng = np.random.default_rng(seed)
    best_state = None
    best_score = -1.0
    best_epoch = 0
    started = time.perf_counter()
    for epoch in range(epochs):
        views = [_view(clip, rng, augment) for clip in train_clips]
        frames = torch.from_numpy(np.ascontiguousarray(_batch(views)))
        model.train()
        _steps(model, opt, frames, labels, len(train_y), dev, smoothing=smoothing)
        score = None
        if val_x is not None and val_y is not None:
            guess = _logits(model, val_x, dev).argmax(axis=1)
            score = float((guess == np.asarray(val_y)).mean())
        if score is None or score > best_score:
            best_score = 0.0 if score is None else score
            best_epoch = epoch
            best_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
        elif epoch - best_epoch >= PATIENCE:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, time.perf_counter() - started, dev


def _view(clip: np.ndarray, rng: np.random.Generator, augment: bool) -> np.ndarray:
    if not augment:
        return clip
    return _augment_clip(np.asarray(clip), rng)


def _augment_clip(clip: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    import cv2

    frames = np.asarray(clip, dtype=np.float32)
    if float(frames.max()) > 1.5:
        frames = frames / 255.0
    count = len(frames)
    if count >= 4:
        start = int(rng.integers(0, int(count * 0.1) + 1))
        end = count - int(rng.integers(0, int(count * 0.1) + 1))
        if end - start >= max(2, count // 2):
            cropped = frames[start:end]
            index = np.linspace(0, len(cropped) - 1, count).round().astype(np.int64)
            frames = cropped[index]
    angle = float(rng.uniform(-15, 15))
    scale = float(rng.uniform(0.9, 1.1))
    shift = rng.uniform(-0.05, 0.05, size=2)
    height, width = frames.shape[1:3]
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, scale)
    matrix[0, 2] += float(shift[0]) * width
    matrix[1, 2] += float(shift[1]) * height
    out = np.empty_like(frames)
    for index, frame in enumerate(frames):
        warped = cv2.warpAffine(
            frame,
            matrix,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )
        noise = rng.normal(0.0, 0.003, size=warped.shape).astype(np.float32)
        out[index] = np.clip(warped + noise, 0.0, 1.0)
    return out


def _batch(clips) -> np.ndarray:
    """(n, frames, height, width, 3) uint8 or float, as (n, frames, 3, height, width)."""
    batch = np.stack([np.asarray(clip, dtype=np.float32) for clip in clips])
    if batch.ndim != 5 or batch.shape[-1] != 3:
        raise ValueError(f"clips must be (n, frames, height, width, 3), got {batch.shape}")
    if float(batch.max()) > 1.5:
        batch = batch / 255.0
    return np.transpose(batch, (0, 1, 4, 2, 3))


def _steps(model, opt, frames, labels, n: int, dev, smoothing: float = SMOOTHING) -> None:
    model.train()
    for index in torch.randperm(n).split(BATCH):
        opt.zero_grad()
        out = model(frames[index].to(dev))
        loss = nn.functional.cross_entropy(out, labels[index].to(dev), label_smoothing=smoothing)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _logits(model, batch, dev) -> np.ndarray:
    frames = torch.from_numpy(np.ascontiguousarray(batch))
    rows = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(batch), BATCH):
            logits = model(frames[start : start + BATCH].to(dev))
            rows.append(logits.cpu().numpy())
    return np.concatenate(rows)


def _topk_accuracy(scores: np.ndarray, labels: np.ndarray, k: int = 5) -> float:
    labels = np.asarray(labels)
    k = min(k, int(scores.shape[1]))
    chosen = np.argpartition(scores, -k, axis=1)[:, -k:]
    return float((chosen == labels.reshape(-1, 1)).any(axis=1).mean())


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tag", default="default")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--no-augment", dest="augment", action="store_false")
    args = p.parse_args()

    kp_dir = ROOT / "data/wlasl100/keypoints"
    if not kp_dir.exists():
        kp_dir = ROOT / "data/wlasl100/landmarks"
    clips = {"train": [], "val": [], "test": []}
    labels = {"train": [], "val": [], "test": []}
    for s in load("wlasl100", ROOT / "data", download=False).samples:
        if (kp_dir / f"{s.sample_id}.npy").exists():
            clip = read_clip(s)
            if clip is not None:
                clips[s.split].append(clip)
                labels[s.split].append(s.label)

    res = infer(
        clips["train"],
        labels["train"],
        clips["test"],
        labels["test"],
        100,
        epochs=args.epochs,
        val_clips=clips["val"],
        val_y=labels["val"],
        seeds=(args.seed,),
        augment=args.augment,
    )
    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    payload = {
        "model": "cnn",
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
    (out_dir / f"cnn_{args.tag}.json").write_text(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
