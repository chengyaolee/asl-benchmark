"""Small transformer over the hand-landmark sequence."""

from __future__ import annotations

import time

import numpy as np
import torch
from torch import nn

from asl_benchmark.compare import Result
from asl_benchmark.models.hands import flat

MAX_FRAMES = 48
BATCH = 32


def pack_sequences(sequences, max_frames: int = MAX_FRAMES):
    """Pad or subsample to (n, max_frames, features). Mask is True on padding."""
    # ponytail: missing points become 0, which is also a real coordinate.
    # Upgrade path: a mask token the attention can skip.
    flats = [_fit_length(np.nan_to_num(flat(seq)), max_frames) for seq in sequences]
    width = flats[0].shape[1]
    packed = np.zeros((len(flats), max_frames, width), np.float32)
    pad = np.ones((len(flats), max_frames), dtype=bool)
    for i, row in enumerate(flats):
        if row.shape[1] != width:
            raise ValueError("landmark width changed between samples")
        packed[i, : len(row)] = row
        pad[i, : len(row)] = False
    return packed, pad


class SmallTransformer(nn.Module):
    def __init__(self, n_features: int, n_classes: int, max_frames: int = MAX_FRAMES):
        super().__init__()
        d_model = 64
        self.proj = nn.Linear(n_features, d_model)
        self.pos = nn.Embedding(max_frames, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=8,
            dim_feedforward=128,
            batch_first=True,
        )
        # Padded clips stay on the dense mask path. Nested tensors are still a prototype.
        self.encoder = nn.TransformerEncoder(layer, num_layers=2, enable_nested_tensor=False)
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, x, pad):
        steps = torch.arange(x.shape[1], device=x.device)
        hidden = self.encoder(self.proj(x) + self.pos(steps), src_key_padding_mask=pad)
        keep = (~pad).unsqueeze(-1)
        pooled = (hidden * keep).sum(dim=1) / keep.sum(dim=1).clamp(min=1)
        return self.head(pooled)


def train(model, packed, pad, labels, epochs: int):
    """Adam and cross-entropy on this architecture. Returns model, seconds, device."""
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    model.to(dev)
    opt = torch.optim.Adam(model.parameters())
    loss_fn = nn.CrossEntropyLoss()
    x = torch.from_numpy(np.ascontiguousarray(packed))
    mask = torch.from_numpy(np.ascontiguousarray(pad))
    labels = torch.from_numpy(np.ascontiguousarray(labels, dtype=np.int64))
    n = int(labels.shape[0])
    started = time.perf_counter()
    model.train()
    for _ in range(epochs):
        for index in torch.randperm(n).split(BATCH):
            opt.zero_grad()
            loss_fn(model(x[index].to(dev), mask[index].to(dev)), labels[index].to(dev)).backward()
            opt.step()
    return model, time.perf_counter() - started, dev


def infer(train_seq, train_y, test_seq, test_y, n_classes: int, epochs: int = 20) -> Result:
    """Train on the train sequences and score the test sequences."""
    packed, pad = pack_sequences(train_seq)
    model, train_seconds, dev = train(
        SmallTransformer(packed.shape[-1], n_classes), packed, pad, train_y, epochs
    )
    test_x, test_pad = pack_sequences(test_seq)
    _glosses(model, test_x[:1], test_pad[:1], dev)
    started = time.perf_counter()
    pred = _glosses(model, test_x, test_pad, dev)
    latency_ms = (time.perf_counter() - started) * 1000 / len(test_y)
    return Result(
        "transformer",
        float((pred == np.asarray(test_y)).mean()),
        float(train_seconds),
        float(latency_ms),
        len(train_y),
        len(test_y),
    )


def _glosses(model, packed, pad, dev) -> np.ndarray:
    x = torch.from_numpy(np.ascontiguousarray(packed))
    mask = torch.from_numpy(np.ascontiguousarray(pad))
    preds = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(packed), BATCH):
            stop = start + BATCH
            logits = model(x[start:stop].to(dev), mask[start:stop].to(dev))
            preds.append(logits.argmax(dim=1).cpu().numpy())
    return np.concatenate(preds)


def _fit_length(row: np.ndarray, max_frames: int) -> np.ndarray:
    if len(row) <= max_frames:
        return row
    index = np.linspace(0, len(row) - 1, max_frames).astype(np.int64)
    return row[index]
