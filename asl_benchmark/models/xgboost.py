"""XGBoost on pooled hand landmarks."""

from __future__ import annotations

import time

import numpy as np

from asl_benchmark.compare import Result
from asl_benchmark.models.hands import flat


def features(sequence) -> np.ndarray:
    """Mean, standard deviation, and mean velocity. Missing points are skipped."""
    values = flat(sequence)
    mask = np.isfinite(values)
    count = mask.sum(axis=0)
    filled = np.where(mask, values, 0.0)
    mean = np.zeros(values.shape[1], np.float64)
    std = np.zeros(values.shape[1], np.float64)
    vel = np.zeros(values.shape[1], np.float64)
    np.divide(filled.sum(axis=0), count, out=mean, where=count > 0)
    centered = np.where(mask, values - mean, 0.0)
    np.divide((centered * centered).sum(axis=0), count, out=std, where=count > 0)
    np.sqrt(std, out=std)
    if len(values) > 1:
        diff = values[1:] - values[:-1]
        dmask = np.isfinite(diff)
        dcount = dmask.sum(axis=0)
        dfilled = np.where(dmask, diff, 0.0)
        np.divide(dfilled.sum(axis=0), dcount, out=vel, where=dcount > 0)
    return np.concatenate([mean, std, vel]).astype(np.float32)


def train(matrix, labels):
    """Fit the default XGBClassifier. Returns the model, class ids, and fit seconds."""
    from xgboost import XGBClassifier

    labels = np.asarray(labels, np.int64)
    # XGBClassifier wants labels packed as 0..k-1. A split can omit glosses.
    classes, indexed = np.unique(labels, return_inverse=True)
    if len(classes) < 2:
        raise ValueError("xgboost needs at least two glosses in train")
    model = XGBClassifier(random_state=0)
    started = time.perf_counter()
    model.fit(matrix, indexed)
    return model, classes, time.perf_counter() - started


def infer(train_seq, train_y, test_seq, test_y) -> Result:
    """Train on the train sequences and score the test sequences."""
    train_x = np.stack([features(sequence) for sequence in train_seq])
    test_x = np.stack([features(sequence) for sequence in test_seq])
    model, classes, train_seconds = train(train_x, train_y)
    _glosses(model, classes, test_x[:1])
    started = time.perf_counter()
    pred = _glosses(model, classes, test_x)
    latency_ms = (time.perf_counter() - started) * 1000 / len(test_y)
    return Result(
        "xgboost",
        float((pred == np.asarray(test_y)).mean()),
        float(train_seconds),
        float(latency_ms),
        len(train_y),
        len(test_y),
    )


def _glosses(model, classes, matrix) -> np.ndarray:
    return classes[model.predict(matrix).astype(np.int64)]
