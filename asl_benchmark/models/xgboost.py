"""XGBoost on pooled hand keypoint features (scripts/features.py).

Each clip is transformed with ``scripts.features.prep(clip, T=32, location=True, pool="meanstd")``
(264 features: palm-size-normalized wrist-relative hand shape, missing flags,
and centered 2D wrist location, pooled by mean and std over 32 resampled frames)
and standardized using clean training statistics. Training clips are augmented
with ``scripts.augment.augment``.

Run from the repo root:
    python asl_benchmark/models/xgboost.py
    python asl_benchmark/models/xgboost.py --seed 1 --tag seed1
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = str(Path(__file__).resolve().parent)
if sys.path and sys.path[0] == SCRIPT_DIR:
    sys.path.pop(0)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from asl_benchmark import compare as comparison  # noqa: E402
from asl_benchmark.loader import load  # noqa: E402
from asl_benchmark.models import hands as hand_landmarks  # noqa: E402
from scripts.augment import augment as augment_clip  # noqa: E402
from scripts.features import prep  # noqa: E402

SEEDS = (0, 1, 2)
T_FRAMES = 32
N_AUGMENTS = 5
EARLY_STOP = 20


def features(sequence: np.ndarray, T: int = T_FRAMES, location: bool = True) -> np.ndarray:
    """Mean and standard deviation of the 32-frame hand features (264 values)."""
    pts = hand_landmarks.hands(sequence)
    return prep(pts, T=T, location=location, pool="meanstd").astype(np.float32)


def train(
    matrix: np.ndarray,
    labels: np.ndarray,
    seed: int = 0,
    val_x: np.ndarray | None = None,
    val_y: np.ndarray | None = None,
):
    """Fit one XGBClassifier. Returns the model, class ids, and fit seconds."""
    from xgboost import XGBClassifier

    labels = np.asarray(labels, np.int64)
    classes, indexed = np.unique(labels, return_inverse=True)
    if len(classes) < 2:
        raise ValueError("xgboost needs at least two glosses in train")
    eval_set = _eval_set(classes, val_x, val_y)
    # Regularized shallow trees when training on multi-class keypoints;
    # fall back to default splits on tiny synthetic toy sets in unit tests.
    if len(classes) > 10 and len(matrix) >= 100:
        model = XGBClassifier(
            n_estimators=300,
            max_depth=3,
            learning_rate=0.08,
            subsample=0.8,
            colsample_bytree=0.3,
            reg_lambda=2.0,
            random_state=seed,
            early_stopping_rounds=EARLY_STOP if eval_set else None,
            n_jobs=-1,
        )
    else:
        model = XGBClassifier(
            random_state=seed,
            early_stopping_rounds=EARLY_STOP if eval_set else None,
        )
    started = time.perf_counter()
    if eval_set:
        model.fit(matrix, indexed, eval_set=eval_set, verbose=False)
    else:
        model.fit(matrix, indexed, verbose=False)
    return model, classes, time.perf_counter() - started


def infer(
    train_seq,
    train_y,
    test_seq,
    test_y,
    val_seq=None,
    val_y=None,
    seeds: tuple[int, ...] = SEEDS,
    augment: bool = True,
) -> comparison.Result:
    """Train on the train sequences and score validation and test."""
    clean_train = np.stack([features(seq) for seq in train_seq])
    mu = clean_train.mean(axis=0)
    sd = clean_train.std(axis=0) + 1e-6
    clean_train_std = (clean_train - mu) / sd
    test_x = (np.stack([features(seq) for seq in test_seq]) - mu) / sd
    val_x = None if val_seq is None else (np.stack([features(seq) for seq in val_seq]) - mu) / sd

    top1, top5, val_top1, seconds, latency = [], [], [], [], []
    for seed in seeds:
        matrix, labels = _training_rows(train_seq, train_y, clean_train_std, mu, sd, seed, augment)
        model, classes, fit_seconds = train(
            matrix, labels, seed, val_x, None if val_y is None else np.asarray(val_y)
        )
        scores = _scores(model, classes, test_x)
        top1.append(_hits(scores, classes, test_y))
        top5.append(_topk_accuracy(scores, np.asarray(test_y, np.int64)))
        if val_x is not None and val_y is not None:
            val_scores = _scores(model, classes, val_x)
            val_top1.append(_hits(val_scores, classes, val_y))
        seconds.append(fit_seconds)
        started = time.perf_counter()
        _glosses(model, classes, test_x)
        latency.append((time.perf_counter() - started) * 1000 / len(test_y))
    return comparison.Result(
        "xgboost",
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


def _training_rows(train_seq, train_y, clean_std: np.ndarray, mu: np.ndarray, sd: np.ndarray, seed: int, augment: bool):
    rows = [clean_std]
    labels = [np.asarray(train_y, np.int64)]
    if augment:
        rng = np.random.default_rng(seed)
        for _ in range(N_AUGMENTS):
            viewed = [
                (features(augment_clip(hand_landmarks.hands(seq), rng)) - mu) / sd
                for seq in train_seq
            ]
            rows.append(np.stack(viewed))
            labels.append(np.asarray(train_y, np.int64))
    return np.concatenate(rows), np.concatenate(labels)


def _eval_set(classes: np.ndarray, val_x: np.ndarray | None, val_y: np.ndarray | None):
    if val_x is None or val_y is None or len(val_y) == 0:
        return None
    known = {int(gloss): index for index, gloss in enumerate(classes)}
    keep = [index for index, gloss in enumerate(val_y) if int(gloss) in known]
    if len(keep) < 2:
        return None
    indexed = np.array([known[int(val_y[index])] for index in keep], np.int64)
    return [(val_x[keep], indexed)]


def _scores(model, classes: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    proba = model.predict_proba(matrix)
    scores = np.full((len(matrix), int(classes.max()) + 1), -np.inf, np.float32)
    for column, gloss in enumerate(classes):
        scores[:, int(gloss)] = proba[:, column]
    return scores


def _hits(scores: np.ndarray, classes: np.ndarray, labels) -> float:
    pred = classes[np.argmax(scores[:, classes], axis=1)]
    return float((pred == np.asarray(labels)).mean())


def _topk_accuracy(scores: np.ndarray, labels: np.ndarray, k: int = 5) -> float:
    labels = np.asarray(labels)
    k = min(k, int(scores.shape[1]))
    chosen = np.argpartition(scores, -k, axis=1)[:, -k:]
    return float((chosen == labels.reshape(-1, 1)).any(axis=1).mean())


def _glosses(model, classes: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    return classes[model.predict(matrix).astype(np.int64)]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tag", default="default")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-augment", dest="augment", action="store_false")
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
        raw["val"],
        labels["val"],
        seeds=(args.seed,),
        augment=args.augment,
    )
    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    payload = {
        "model": "xgboost",
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
    (out_dir / f"xgboost_{args.tag}.json").write_text(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
