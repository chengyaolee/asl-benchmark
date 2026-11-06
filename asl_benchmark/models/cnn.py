"""Appearance CNN on raw frames. It never sees landmarks."""

from __future__ import annotations

import time

import numpy as np
import torch
from torch import nn

from asl_benchmark.compare import Result

N_CLIP_FRAMES = 8
FRAME_SIZE = 64
BATCH = 32


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
        # ponytail: one seek, then a forward read. Some codecs land off by a few frames.
        # Upgrade path: index the clip by presentation timestamp.
        wanted = np.linspace(start, end - 1, n_frames).astype(np.int64)
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
    def __init__(self, n_classes: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(64, n_classes)

    def forward(self, x):
        batch, steps, channels, height, width = x.shape
        feat = self.conv(x.reshape(batch * steps, channels, height, width)).flatten(1)
        return self.head(feat.view(batch, steps, -1).mean(dim=1))


def train(model, batch, labels, epochs: int):
    """Adam and cross-entropy on this architecture. Returns model, seconds, device."""
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    model.to(dev)
    opt = torch.optim.Adam(model.parameters())
    loss_fn = nn.CrossEntropyLoss()
    frames = torch.from_numpy(np.ascontiguousarray(batch))
    labels = torch.from_numpy(np.ascontiguousarray(labels, dtype=np.int64))
    n = int(labels.shape[0])
    started = time.perf_counter()
    model.train()
    for _ in range(epochs):
        for index in torch.randperm(n).split(BATCH):
            opt.zero_grad()
            loss_fn(model(frames[index].to(dev)), labels[index].to(dev)).backward()
            opt.step()
    return model, time.perf_counter() - started, dev


def infer(train_clips, train_y, test_clips, test_y, n_classes: int, epochs: int = 20) -> Result:
    """Train on the train clips and score the test clips."""
    model, train_seconds, dev = train(CNN(n_classes), _batch(train_clips), train_y, epochs)
    test_x = _batch(test_clips)
    _glosses(model, test_x[:1], dev)
    started = time.perf_counter()
    pred = _glosses(model, test_x, dev)
    latency_ms = (time.perf_counter() - started) * 1000 / len(test_y)
    return Result(
        "cnn",
        float((pred == np.asarray(test_y)).mean()),
        float(train_seconds),
        float(latency_ms),
        len(train_y),
        len(test_y),
    )


def _batch(clips) -> np.ndarray:
    """(n, frames, height, width, 3) uint8 or float, as (n, frames, 3, height, width)."""
    batch = np.stack([np.asarray(clip, dtype=np.float32) for clip in clips])
    if batch.ndim != 5 or batch.shape[-1] != 3:
        raise ValueError(f"clips must be (n, frames, height, width, 3), got {batch.shape}")
    if float(batch.max()) > 1.5:
        batch = batch / 255.0
    return np.transpose(batch, (0, 1, 4, 2, 3))


def _glosses(model, batch, dev) -> np.ndarray:
    frames = torch.from_numpy(np.ascontiguousarray(batch))
    preds = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(batch), BATCH):
            logits = model(frames[start : start + BATCH].to(dev))
            preds.append(logits.argmax(dim=1).cpu().numpy())
    return np.concatenate(preds)
