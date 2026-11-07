"""Each architecture infers on its own, and compare prints the chart."""

import json
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from asl_benchmark.compare import Result, compare, report  # noqa: E402
from asl_benchmark.models import cnn, transformer, xgboost  # noqa: E402
from asl_benchmark.models.cnn import frame_span  # noqa: E402
from asl_benchmark.models.hands import hands  # noqa: E402
from asl_benchmark.models.transformer import SmallTransformer, pack_sequences, train  # noqa: E402
from scripts.features import per_frame  # noqa: E402


def main() -> None:
    _check_chart()
    _check_hands_and_tree_features()
    _check_pack_and_window()
    _check_fit_updates_weights()
    _check_separable_pose_models()
    _check_appearance_runs()
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _check_wlasl_landmark_cache(root)
    print("ok")


def _check_chart() -> None:
    text = report(
        [
            Result("xgboost", 0.5, 1.0, 0.2, 10, 4),
            Result("transformer", 1.0, 2.0, 0.3, 10, 4),
        ],
        "note",
    )
    lines = text.splitlines()

    def bar(line):
        return line[14:34]

    trees = next(line for line in lines if line.startswith("xgboost"))
    net = next(line for line in lines if line.startswith("transformer"))
    assert bar(trees) == "#" * 10 + "." * 10
    assert bar(net) == "#" * 20
    assert "0.500" in text and "top1" in text


def _check_hands_and_tree_features() -> None:
    full = np.full((1, 42, 3), np.nan, np.float32)
    full[0, 0] = (0.25, 0.1, 0.0)
    full[0, 9] = (0.25, 0.2, 0.0)
    full[0, 21, 1] = 0.5
    both = hands(full)
    assert both.shape == (1, 42, 3)
    assert both[0, 0, 0] == np.float32(0.25)
    assert both[0, 21, 1] == np.float32(0.5)
    assert np.isnan(both[0, 2, 0])

    framed = per_frame(both, location=True)
    assert framed.shape == (1, 132)
    assert framed[0, 126] == 1.0
    assert framed[0, 127] == 1.0
    assert xgboost.features(full).shape == (264,)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        blank = per_frame(np.full((2, 42, 3), np.nan, np.float32), location=True)
    assert np.all(blank[:, 126] == 1.0)
    assert np.all(blank[:, 127] == 1.0)
    assert np.all(blank[:, 128:132] == 0.0)


def _check_pack_and_window() -> None:
    short = np.zeros((2, 3), np.float32)
    long = np.arange(18, dtype=np.float32).reshape(6, 3)
    packed, pad = pack_sequences([short, long], max_frames=4)
    assert packed.shape == (2, 4, 3)
    assert pad[0].tolist() == [False, False, True, True]
    assert not pad[1].any()
    assert frame_span(100, 1, 12) == (0, 12)
    assert frame_span(10, 1, -1) == (0, 10)
    assert frame_span(10, 5, -1) == (4, 10)
    assert frame_span(10, None, None) == (0, 10)
    assert frame_span(10, 8, 6) == (0, 10)


def _check_fit_updates_weights() -> None:
    import torch

    model = SmallTransformer(3, 2, max_frames=4)
    before = [param.detach().clone() for param in model.parameters()]
    x = np.zeros((8, 4, 3), np.float32)
    x[:4] = 1
    pad = np.zeros((8, 4), dtype=bool)
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    train(model, x, pad, y, epochs=3)
    changed = [
        not torch.equal(old, param.detach().cpu())
        for old, param in zip(before, model.parameters())
    ]
    assert any(changed)


def _check_separable_pose_models() -> None:
    def clip(value):
        seq = np.zeros((8, 42, 3), np.float32)
        seq[:, 0] = (0.2 + 0.2 * value, 0.3, 0.0)
        seq[:, 9] = (0.2 + 0.2 * value, 0.4, 0.0)
        seq[:, 5] = (0.2 + 0.2 * value + 0.05 * (value + 1), 0.5, 0.1)
        seq[:, 21] = (0.7, 0.2 + 0.1 * value, 0.0)
        seq[:, 30] = (0.7, 0.3 + 0.1 * value, 0.0)
        seq[:, 26] = (0.8, 0.2 + 0.1 * value, 0.2)
        return seq

    train_clips, labels = [], []
    for label in (0, 1, 2):
        train_clips.extend(clip(label) for _ in range(20))
        labels.extend([label] * 20)
    test = [clip(0), clip(1), clip(2)]
    boosted = xgboost.infer(train_clips, labels, test, [0, 1, 2], augment=False, seeds=(0,))
    assert boosted.model == "xgboost"
    assert boosted.top1 == 1.0
    assert boosted.top5 == 1.0
    assert boosted.n_train == 60 and boosted.n_test == 3
    assert boosted.train_seconds >= 0 and boosted.latency_ms >= 0
    shifted = xgboost.infer(
        train_clips, [label + 10 for label in labels], test, [10, 11, 12], augment=False, seeds=(0,)
    )
    assert shifted.top1 == 1.0
    net = transformer.infer(
        train_clips, labels, test, [0, 1, 2], n_classes=3, epochs=30, augment=False, seeds=(0,)
    )
    assert net.model == "transformer"
    assert net.top1 == 1.0
    assert net.n_test == boosted.n_test


def _check_appearance_runs() -> None:
    def block(value):
        return np.full((2, 8, 8, 3), value, np.uint8)

    train_clips = [block(0), block(0), block(255), block(255)]
    test = [block(0), block(255)]
    result = cnn.infer(
        train_clips, [0, 0, 1, 1], test, [0, 1], n_classes=2, epochs=15, augment=False, seeds=(0,)
    )
    assert result.model == "cnn"
    assert result.top1 == 1.0
    assert result.n_train == 4 and result.n_test == 2


def _check_wlasl_landmark_cache(root: Path) -> None:
    folder = root / "wlasl100"
    folder.mkdir()
    entries = []
    for label, gloss in enumerate(("book", "drink")):
        instances = []
        for split, sample_id in (("train", f"{label}0"), ("train", f"{label}1"), ("test", f"{label}2")):
            instances.append(
                {
                    "video_id": sample_id,
                    "url": "https://example.com/x.mp4",
                    "frame_start": 1,
                    "frame_end": 4,
                    "fps": 25,
                    "split": split,
                }
            )
            cache = folder / "keypoints" / f"{sample_id}.npy"
            cache.parent.mkdir(parents=True, exist_ok=True)
            landmarks = np.full((3, 42, 3), label, np.float32)
            np.save(cache, landmarks)
        entries.append({"gloss": gloss, "instances": instances})
    (folder / "WLASL_v0.3.json").write_text(json.dumps(entries))

    results, note = compare("wlasl100", root, epochs=1)
    assert [row.model for row in results] == ["xgboost", "transformer"]
    assert {row.n_train for row in results} == {4}
    assert {row.n_test for row in results} == {2}
    assert "mp4" in note


if __name__ == "__main__":
    main()
