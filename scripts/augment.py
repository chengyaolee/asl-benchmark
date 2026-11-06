"""Random augmentation for hand keypoint clips. Train split only.

Works on raw keypoints shaped (frames, 42, 3), before prep(). Missing hands
(NaN) stay NaN. Every call gives a different random version of the clip.
"""

from __future__ import annotations

import numpy as np


def augment(x, rng: np.random.Generator,
            crop=0.1, rotate_deg=15.0, scale=0.1, shift=0.05, noise=0.003):
    x = np.array(x, dtype=np.float32, copy=True)
    n = len(x)

    # Time: cut up to `crop` of the frames off each end (keep at least half).
    if n >= 4 and crop > 0:
        start = rng.integers(0, int(n * crop) + 1)
        end = n - rng.integers(0, int(n * crop) + 1)
        if end - start >= max(2, n // 2):
            x = x[start:end]

    # Space: one rotation / scale / shift for the whole clip, around the frame center.
    theta = np.deg2rad(rng.uniform(-rotate_deg, rotate_deg))
    s = rng.uniform(1 - scale, 1 + scale)
    c, si = np.cos(theta), np.sin(theta)
    xy = x[..., :2] - 0.5
    x[..., 0] = s * (c * xy[..., 0] - si * xy[..., 1]) + 0.5 + rng.uniform(-shift, shift)
    x[..., 1] = s * (si * xy[..., 0] + c * xy[..., 1]) + 0.5 + rng.uniform(-shift, shift)
    x[..., 2] *= s

    # Jitter: small independent noise on every point.
    x += rng.normal(0, noise, x.shape).astype(np.float32)
    return x
