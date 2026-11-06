"""Turn one hand keypoint clip into a fixed-size vector. Same for every split.

Input: raw keypoints (frames, 42, 3) from scripts/extract.py, NaN = no hand.
Per frame and per hand:
  - shape: 21 points centered on the wrist and divided by hand size (63)
  - missing: 1 if the hand was not detected (1)
  - location: wrist x, y in the frame, centered on 0 (2), if location=True
The clip is resampled to T frames. Then:
  - velocity=True adds the frame-to-frame change of every per-frame feature
  - pool="none"    keeps all T frames side by side         (T x D)
  - pool="meanstd" keeps the mean and std over time         (2 x D)
  - pool="chunks4" keeps the mean of 4 consecutive chunks   (4 x D)
where D is the per-frame size (130 or 134, doubled with velocity).
"""

from __future__ import annotations

import numpy as np

POOLS = ("none", "meanstd", "chunks4")


def per_frame(x, location: bool = True) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if len(x) == 0:
        raise ValueError("empty clip")
    x = x.reshape(len(x), 2, 21, 3)                    # (frames, hand, point, xyz)

    wrist = x[:, :, :1, :]
    size = np.linalg.norm(x[:, :, 9, :2] - x[:, :, 0, :2], axis=-1)[..., None, None]
    shape = (x - wrist) / (size + 1e-6)
    missing = np.isnan(shape).any(axis=(2, 3)).astype(np.float32)   # (frames, 2)

    parts = [np.nan_to_num(shape).reshape(len(x), 126), missing]
    if location:
        parts.append(np.nan_to_num(x[:, :, 0, :2] - 0.5).reshape(len(x), 4))
    return np.concatenate(parts, axis=1)


def prep(x, T: int = 32, location: bool = True, pool: str = "none",
         velocity: bool = False) -> np.ndarray:
    if pool not in POOLS:
        raise ValueError(f"pool must be one of {POOLS}, got {pool!r}")
    feat = per_frame(x, location)
    idx = np.linspace(0, len(feat) - 1, T).round().astype(int)
    feat = feat[idx]                                   # (T, D)

    if velocity:
        vel = np.diff(feat, axis=0, prepend=feat[:1])  # first frame has zero change
        feat = np.concatenate([feat, vel], axis=1)     # (T, 2D)

    if pool == "none":
        return feat.reshape(-1)
    if pool == "meanstd":
        return np.concatenate([feat.mean(0), feat.std(0)])
    chunks = np.array_split(feat, 4, axis=0)
    return np.concatenate([c.mean(0) for c in chunks])


def feature_size(T: int = 32, location: bool = True, pool: str = "none",
                 velocity: bool = False) -> int:
    d = (126 + 2 + (4 if location else 0)) * (2 if velocity else 1)
    return {"none": T * d, "meanstd": 2 * d, "chunks4": 4 * d}[pool]
