"""Offline check that both sources yield the same sample record."""

import csv
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from asl_benchmark import load, load_landmarks  # noqa: E402
from asl_benchmark.loader import Sample, _apply_dotenv, ensure_video  # noqa: E402


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _check_wlasl(root)
        _check_google(root)
        _check_video_file_url(root)
        _check_refuses_unknown_and_missing(root)
        _check_dotenv(root)
    print("ok")


def _check_wlasl(root: Path) -> None:
    folder = root / "wlasl100"
    folder.mkdir()
    # 100 kept glosses, then one past the cutoff that must be dropped.
    entries = [
        {
            "gloss": "book",
            "instances": [
                {
                    "video_id": "10",
                    "url": "https://example.com/10.mp4",
                    "frame_start": 1,
                    "frame_end": 12,
                    "fps": 25,
                    "split": "train",
                }
            ],
        },
        {
            "gloss": "drink",
            "instances": [
                {
                    "video_id": "11",
                    "url": "https://example.com/11.mp4",
                    "frame_start": 3,
                    "frame_end": -1,
                    "fps": 25,
                    "split": "validation",
                }
            ],
        },
    ]
    for index in range(98):
        entries.append(
            {
                "gloss": f"filler{index}",
                "instances": [
                    {
                        "video_id": f"f{index}",
                        "url": "https://example.com/filler.mp4",
                        "frame_start": 1,
                        "frame_end": 2,
                        "fps": 25,
                        "split": "test",
                    }
                ],
            }
        )
    entries.append(
        {
            "gloss": "later",
            "instances": [
                {
                    "video_id": "99",
                    "url": "https://example.com/not-in-subset.mp4",
                    "frame_start": 1,
                    "frame_end": 2,
                    "fps": 25,
                    "split": "test",
                }
            ],
        }
    )
    (folder / "WLASL_v0.3.json").write_text(json.dumps(entries))

    data = load("wlasl100", root, download=False)
    assert data.glosses[0] == "book" and data.glosses[1] == "drink"
    assert len(data.glosses) == 100
    assert "later" not in data.glosses
    ids = {sample.sample_id for sample in data.samples}
    assert "10" in ids and "11" in ids and "99" not in ids
    train = data.split("train")
    assert len(train) == 1 and train[0].gloss == "book" and train[0].label == 0
    held = data.split("val")
    assert len(held) == 1
    assert held[0].sample_id == "11"
    assert held[0].label == 1
    assert held[0].frame_end == -1
    assert held[0].fps == 25
    assert held[0].video_path == folder / "videos" / "11.mp4"
    assert held[0].landmark_path is None


def _check_google(root: Path) -> None:
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq

    folder = root / "google_islr"
    files = folder / "train_landmark_files"
    (files / "1").mkdir(parents=True)
    (files / "0").mkdir()
    (files / "4").mkdir()
    rows = [
        ("path", "participant_id", "sequence_id", "sign"),
        ("train_landmark_files/1/20.parquet", "1", "20", "drink"),
        ("train_landmark_files/1/21.parquet", "1", "21", "book"),
        ("train_landmark_files/0/30.parquet", "0", "30", "book"),
        ("train_landmark_files/4/40.parquet", "4", "40", "drink"),
    ]
    with (folder / "train.csv").open("w", newline="") as handle:
        csv.writer(handle).writerows(rows)
    (folder / "sign_to_prediction_index_map.json").write_text(
        json.dumps({"book": 0, "drink": 1})
    )
    clip = files / "1" / "20.parquet"
    pq.write_table(
        pa.table(
            {
                "frame": [5, 5, 0],
                "row_id": ["a", "b", "c"],
                "type": ["pose", "nope", "face"],
                "landmark_index": [1, 0, 0],
                "x": [4.0, 9.0, 1.0],
                "y": [5.0, 9.0, 2.0],
                "z": [None, 9.0, 3.0],
            }
        ),
        clip,
    )
    for name in ("21.parquet", "30.parquet", "40.parquet"):
        target = files / ("1" if name.startswith("2") else "0" if name.startswith("3") else "4")
        pq.write_table(pq.read_table(clip), target / name)

    data = load("google_islr", root, download=False)
    assert data.glosses == ("book", "drink")
    by_id = {sample.sample_id: sample for sample in data.samples}
    assert by_id["20"].split == "train" and by_id["21"].split == "train"
    assert by_id["20"].label == 1 and by_id["21"].label == 0
    assert by_id["30"].split == "test"
    assert by_id["40"].split == "val"
    assert by_id["20"].video_path is None
    assert by_id["20"].landmark_path == clip

    landmarks = load_landmarks(by_id["20"])
    assert landmarks.shape == (2, 543, 3)
    assert landmarks.dtype == np.float32
    assert landmarks[0, 0].tolist() == [1.0, 2.0, 3.0]
    pose = 468 + 21 + 1
    assert landmarks[1, pose, 0] == 4.0
    assert np.isnan(landmarks[1, pose, 2])
    assert np.isnan(landmarks[0, 1, 0])


def _check_video_file_url(root: Path) -> None:
    source = root / "clip.bin"
    source.write_bytes(b"fake-video")
    dest = root / "videos" / "10.mp4"
    sample = Sample(
        source="wlasl100",
        sample_id="10",
        gloss="book",
        label=0,
        split="train",
        video_path=dest,
        url=source.as_uri(),
    )
    assert ensure_video(sample) == dest
    assert dest.read_bytes() == b"fake-video"
    assert ensure_video(sample) == dest


def _check_refuses_unknown_and_missing(root: Path) -> None:
    try:
        load("other", root, download=False)
    except ValueError as exc:
        assert "wlasl100" in str(exc)
    else:
        raise AssertionError("unknown source was accepted")

    empty = root / "empty"
    try:
        load("wlasl100", empty, download=False)
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("missing WLASL metadata was accepted")


def _check_dotenv(root: Path) -> None:
    path = root / "sample.env"
    path.write_text('export EXAMPLE_TOKEN="abc"\n# comment\n\n')
    os.environ.pop("EXAMPLE_TOKEN", None)
    _apply_dotenv(path)
    assert os.environ["EXAMPLE_TOKEN"] == "abc"
    os.environ["EXAMPLE_TOKEN"] = "kept"
    _apply_dotenv(path)
    assert os.environ["EXAMPLE_TOKEN"] == "kept"
    os.environ.pop("EXAMPLE_TOKEN", None)


if __name__ == "__main__":
    main()
