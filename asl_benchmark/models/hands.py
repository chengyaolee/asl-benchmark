"""Both hands from a landmark frame. Pose models share this cut, not a trainer."""

from __future__ import annotations

import numpy as np

from asl_benchmark.loader import LANDMARK_LAYOUT, N_LANDMARKS


def hands(sequence):
    """(frames, 42, 3). A full 543-point frame is sliced to left hand, then right."""
    sequence = np.asarray(sequence, dtype=np.float32)
    if sequence.ndim != 3 or sequence.shape[-1] != 3:
        raise ValueError(f"landmarks must be (frames, points, 3), got {sequence.shape}")
    if sequence.shape[1] != N_LANDMARKS:
        return sequence
    parts = []
    base = 0
    for name, count in LANDMARK_LAYOUT:
        if name in ("left_hand", "right_hand"):
            parts.append(sequence[:, base : base + count])
        base += count
    return np.concatenate(parts, axis=1)


def flat(sequence) -> np.ndarray:
    sequence = hands(sequence)
    if sequence.shape[0] == 0 or sequence.shape[1] == 0:
        raise ValueError(f"empty landmark sequence {sequence.shape}")
    return sequence.reshape(sequence.shape[0], -1)
