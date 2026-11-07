"""Call each architecture's infer and print the comparison.

    python -m asl_benchmark.compare wlasl100

Pose models read both hands: 21 points each, (x, y, z), 42 points per frame,
from ``data/wlasl100/keypoints/<id>.npy`` shaped ``(frames, 42, 3)``, left hand
then right, NaN where a hand is missing. The CNN reads the mp4 window on the
sample. All models are scored on the same clips.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np

from asl_benchmark.loader import SOURCES, load

EPOCHS = 200


@dataclass(frozen=True)
class Result:
    model: str
    top1: float
    train_seconds: float
    latency_ms: float
    n_train: int
    n_test: int
    val_top1: float | None = None
    top5: float | None = None
    n_val: int = 0
    top1_std: float = 0.0


def compare(source: str, root: str = "data", epochs: int = EPOCHS, limit: int | None = None):
    """Run every architecture this dataset can feed.

    Returns the score rows and the text of the chart plus the results table.
    """
    if epochs < 1:
        raise ValueError("epochs must be >= 1")
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    from asl_benchmark.models import cnn, transformer, xgboost

    dataset = load(source, root, download=False)
    train_samples = dataset.split("train")
    val_samples = dataset.split("val")
    test_samples = dataset.split("test")
    if limit is not None:
        train_samples = train_samples[:limit]
        val_samples = val_samples[:limit]
        test_samples = test_samples[:limit]
    n_train_all = len(train_samples)
    n_test_all = len(test_samples)
    train_rows = _load_split(dataset, train_samples, "train")
    val_rows = _load_split(dataset, val_samples, "val")
    test_rows = _load_split(dataset, test_samples, "test")
    use_pose = _has(train_rows, 1) and _has(test_rows, 1)
    use_frames = _has(train_rows, 2) and _has(test_rows, 2)
    if not use_pose and not use_frames:
        raise RuntimeError(
            f"{dataset.source}: {n_train_all} train and {n_test_all} test samples, "
            f"and none have the inputs these models read. {_hint(dataset)}"
        )
    if use_pose and use_frames:
        keep = lambda row: row[1] is not None and row[2] is not None
        mode = "xgboost, transformer, and cnn on the same clips"
    elif use_pose:
        keep = lambda row: row[1] is not None
        mode = "xgboost and transformer; the CNN needs the mp4 files"
    else:
        keep = lambda row: row[2] is not None
        mode = "cnn only; pose models need hand keypoints"
    train_rows = [row for row in train_rows if keep(row)]
    val_rows = [row for row in val_rows if keep(row)]
    test_rows = [row for row in test_rows if keep(row)]
    if not train_rows or not test_rows:
        raise RuntimeError(
            f"{dataset.source}: no {('train' if not train_rows else 'test')} clips with the inputs these models share. {_hint(dataset)}"
        )
    n_classes = len(dataset.glosses)
    train_y = [row[0] for row in train_rows]
    test_y = [row[0] for row in test_rows]
    val_y = [row[0] for row in val_rows]
    results = []
    if use_pose:
        train_seq = [row[1] for row in train_rows]
        test_seq = [row[1] for row in test_rows]
        val_seq = [row[1] for row in val_rows] or None
        held_y = val_y or None
        results.append(xgboost.infer(train_seq, train_y, test_seq, test_y, val_seq, held_y))
        results.append(
            transformer.infer(
                train_seq,
                train_y,
                test_seq,
                test_y,
                n_classes,
                epochs,
                val_seq,
                held_y,
            )
        )
    if use_frames:
        results.append(
            cnn.infer(
                [row[2] for row in train_rows],
                train_y,
                [row[2] for row in test_rows],
                test_y,
                n_classes,
                epochs,
                None if not val_rows else [row[2] for row in val_rows],
                val_y or None,
            )
        )
    note = (
        f"{dataset.source}: {len(train_rows)}/{n_train_all} train, "
        f"{len(val_rows)} val, and {len(test_rows)}/{n_test_all} test. {mode}. "
        "Mean of 3 seeds. Train clips are augmented; validation stops training early."
    )
    return results, report(results, note)


def report(results, note: str) -> str:
    """Top-1 chart, then the numeric results."""
    return "\n\n".join((note, _chart(results), _table(results)))


def _chart(results) -> str:
    width = 20
    lines = ["top-1"]
    for row in results:
        filled = min(width, max(0, int(round(row.top1 * width))))
        bar = "#" * filled + "." * (width - filled)
        lines.append(f"{row.model:<14}{bar}  {row.top1:.3f}")
    return "\n".join(lines)


def _table(results) -> str:
    lines = [
        f"{'model':<14}{'top1':>8}{'std':>8}{'top5':>8}{'val':>8}"
        f"{'train_s':>10}{'ms/clip':>10}{'n_train':>8}{'n_val':>8}{'n_test':>8}"
    ]
    for row in results:
        lines.append(
            f"{row.model:<14}{row.top1:8.3f}{row.top1_std:8.3f}{_cell(row.top5)}{_cell(row.val_top1)}"
            f"{row.train_seconds:10.2f}{row.latency_ms:10.2f}{row.n_train:8d}{row.n_val:8d}{row.n_test:8d}"
        )
    return "\n".join(lines)


def _cell(value: float | None) -> str:
    if value is None:
        return f"{'--':>8}"
    return f"{value:8.3f}"


def _load_split(dataset, samples, kind: str):
    from asl_benchmark.models.cnn import read_clip

    rows = []
    total = len(samples)
    for i, sample in enumerate(samples, start=1):
        rows.append((sample.label, _sequence(dataset, sample), read_clip(sample)))
        if total >= 2000 and i % 2000 == 0:
            print(f"{kind} {i}/{total}", flush=True)
    return rows


def _sequence(dataset, sample):
    from asl_benchmark.models import hands as hand_landmarks

    path = dataset.root / "keypoints" / f"{sample.sample_id}.npy"
    if not path.exists():
        return None
    seq = np.load(path)
    if seq.ndim != 3 or seq.shape[-1] != 3:
        raise ValueError(f"{path} must have shape (frames, points, 3), got {seq.shape}")
    if len(seq) == 0:
        return None
    return hand_landmarks.hands(seq)


def _has(rows, index: int) -> bool:
    return any(row[index] is not None for row in rows)


def _hint(dataset) -> str:
    return (
        "Run python scripts/extract.py first. Pose models read "
        "data/wlasl100/keypoints/<id>.npy shaped (frames, 42, 3). "
        "The CNN reads data/wlasl100/videos/<id>.mp4 on those same clips."
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Compare the architectures and print the chart.")
    parser.add_argument("source", nargs="?", default="wlasl100", choices=SOURCES)
    parser.add_argument("--root", default="data")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--limit", type=int, default=0, help="cap each split, for a first cheap run")
    args = parser.parse_args(argv)
    try:
        _results, text = compare(args.source, args.root, epochs=args.epochs, limit=args.limit or None)
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from None
    print(text)


if __name__ == "__main__":
    main()
