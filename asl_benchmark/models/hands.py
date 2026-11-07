"""Both hands from a landmark frame. Pose models share this cut, not a trainer."""

from __future__ import annotations

import numpy as np

N_HAND_LANDMARKS = 42


def hands(sequence):
    """Validate and return (frames, 42, 3) float32 hand landmarks (left hand 0..20, right hand 21..41)."""
    sequence = np.asarray(sequence, dtype=np.float32)
    if sequence.ndim != 3 or sequence.shape[-1] != 3:
        raise ValueError(f"landmarks must be (frames, points, 3), got {sequence.shape}")
    return sequence


def flat(sequence) -> np.ndarray:
    sequence = hands(sequence)
    if sequence.shape[0] == 0 or sequence.shape[1] == 0:
        raise ValueError(f"empty landmark sequence {sequence.shape}")
    return sequence.reshape(sequence.shape[0], -1)
