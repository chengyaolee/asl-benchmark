"""One loader for WLASL100 and Google Isolated Sign Language Recognition.

Both sources come back as the same Sample records, so later models can
iterate either dataset without knowing which files it was built from.

WLASL100 is raw RGB video plus the official 100-gloss split. The clip
window is the WLASL frame span (25 fps, frame_start is 1-based, frame_end
of -1 means the last frame). Google ISLR is precomputed MediaPipe Holistic
landmarks, 543 points by (x, y, z) per frame, and has no video.

Usage:
    from asl_benchmark import load, load_landmarks

    wlasl = load("wlasl100", "data")
    google = load("google_islr", "data")
    for sample in google.split("train"):
        landmarks = load_landmarks(sample)  # (T, 543, 3)
"""

from __future__ import annotations

import csv
import json
import shutil
import subprocess
import urllib.parse
import urllib.request
import zlib
from dataclasses import dataclass
from pathlib import Path

SOURCES = ("wlasl100", "google_islr")

# Official WLASL start kit. The 100-gloss subset is nslt_100.json.
WLASL_JSON_URL = (
    "https://raw.githubusercontent.com/dxli94/WLASL/master/start_kit/WLASL_v0.3.json"
)
WLASL_SPLIT_URL = (
    "https://raw.githubusercontent.com/dxli94/WLASL/master/start_kit/nslt_100.json"
)
WLASL_CLASSES_URL = (
    "https://raw.githubusercontent.com/dxli94/WLASL/master/start_kit/wlasl_class_list.txt"
)

# Kaggle competition "Google - Isolated Sign Language Recognition".
GOOGLE_COMPETITION = "asl-signs"

# Fixed column order so every frame lines up: 468 + 21 + 33 + 21 = 543.
LANDMARK_LAYOUT = (
    ("face", 468),
    ("left_hand", 21),
    ("pose", 33),
    ("right_hand", 21),
)
N_LANDMARKS = sum(count for _, count in LANDMARK_LAYOUT)
_LANDMARK_OFFSET = {}
_base = 0
for _name, _count in LANDMARK_LAYOUT:
    _LANDMARK_OFFSET[_name] = (_base, _count)
    _base += _count

_USER_AGENT = "asl-benchmark"


@dataclass(frozen=True)
class Sample:
    source: str
    sample_id: str
    gloss: str
    label: int
    split: str
    video_path: Path | None = None
    landmark_path: Path | None = None
    frame_start: int | None = None
    frame_end: int | None = None
    fps: int | None = None
    url: str | None = None


@dataclass(frozen=True)
class SignDataset:
    source: str
    root: Path
    glosses: tuple[str, ...]
    samples: tuple[Sample, ...]

    def split(self, name: str) -> tuple[Sample, ...]:
        if name not in ("train", "val", "test"):
            raise ValueError(f"split must be train, val, or test, got {name!r}")
        return tuple(sample for sample in self.samples if sample.split == name)


def load(source: str, root: str | Path = "data", *, download: bool = True) -> SignDataset:
    """Read one source from root, fetching published files when they are missing.

    root/wlasl100 holds the WLASL metadata and videos.
    root/google_islr holds the Kaggle asl-signs archive.
    Pass download=False to fail instead of touching the network.
    """
    source = _check_source(source)
    root = Path(root)
    if source == "wlasl100":
        return _load_wlasl(root / "wlasl100", download=download)
    return _load_google(root / "google_islr", download=download)


def load_landmarks(sample: Sample):
    """Return float32 landmarks shaped (frames, 543, 3).

    Rows follow LANDMARK_LAYOUT. Missing detections stay NaN. Frames are
    sorted by frame number; frames with no rows are left out.
    """
    if sample.landmark_path is None:
        raise ValueError(f"{sample.source} sample {sample.sample_id} has no landmarks")

    import numpy as np
    import pyarrow.parquet as pq

    table = pq.read_table(
        sample.landmark_path,
        columns=["frame", "type", "landmark_index", "x", "y", "z"],
    )
    frames = table.column("frame").to_pylist()
    types = table.column("type").to_pylist()
    indexes = table.column("landmark_index").to_pylist()
    xs = _float_list(table.column("x"))
    ys = _float_list(table.column("y"))
    zs = _float_list(table.column("z"))

    order = sorted({int(frame) for frame in frames if frame is not None})
    position = {frame: i for i, frame in enumerate(order)}
    out = np.full((len(order), N_LANDMARKS, 3), np.nan, dtype=np.float32)
    for frame, kind, index, x, y, z in zip(frames, types, indexes, xs, ys, zs):
        if frame is None or kind not in _LANDMARK_OFFSET or index is None:
            continue
        base, count = _LANDMARK_OFFSET[kind]
        index = int(index)
        if index < 0 or index >= count:
            continue
        row = position[int(frame)]
        out[row, base + index, 0] = x
        out[row, base + index, 1] = y
        out[row, base + index, 2] = z
    return out


def ensure_video(sample: Sample) -> Path:
    """Download one WLASL clip if videos/ does not already have it."""
    if sample.video_path is None:
        raise ValueError(f"{sample.source} sample {sample.sample_id} has no video")
    if sample.video_path.exists() and sample.video_path.stat().st_size > 0:
        return sample.video_path
    if not sample.url:
        raise ValueError(f"WLASL sample {sample.sample_id} has no source url")
    sample.video_path.parent.mkdir(parents=True, exist_ok=True)
    if _is_youtube(sample.url):
        _download_youtube(sample.url, sample.video_path)
        return sample.video_path
    partial = sample.video_path.with_suffix(".part")
    try:
        _download_url(sample.url, partial)
        if partial.stat().st_size == 0:
            raise ValueError(f"empty download for {sample.sample_id}")
        partial.replace(sample.video_path)
    except Exception:
        partial.unlink(missing_ok=True)
        raise
    return sample.video_path


def download_videos(dataset: SignDataset) -> tuple[list[Path], list[tuple[str, str]]]:
    """Fetch every WLASL video. Dead urls are returned, not raised.

    WLASL links rot. The first list is clips on disk; the second is
    (sample_id, error) for the ones that did not download.
    """
    if dataset.source != "wlasl100":
        raise ValueError("only WLASL100 publishes source videos")
    saved: list[Path] = []
    failed: list[tuple[str, str]] = []
    for sample in dataset.samples:
        try:
            saved.append(ensure_video(sample))
        except Exception as exc:
            failed.append((sample.sample_id, str(exc)))
    return saved, failed


def _load_wlasl(root: Path, *, download: bool) -> SignDataset:
    meta = root / "WLASL_v0.3.json"
    splits = root / "nslt_100.json"
    classes = root / "wlasl_class_list.txt"
    if download:
        _ensure_url(WLASL_JSON_URL, meta)
        _ensure_url(WLASL_SPLIT_URL, splits)
        _ensure_url(WLASL_CLASSES_URL, classes)
    missing = [path.name for path in (meta, splits) if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"WLASL100 metadata missing in {root}: {', '.join(missing)}. "
            "load('wlasl100', download=True) fetches it from the WLASL GitHub repo."
        )

    catalog = _wlasl_catalog(meta)
    split_rows = json.loads(splits.read_text())
    names = _wlasl_class_names(classes)

    samples = []
    for video_id, info in split_rows.items():
        video_id = _safe_id(str(video_id))
        action = info["action"]
        label = int(action[0] if isinstance(action, list) else action)
        subset = info["subset"]
        if subset == "validation":
            subset = "val"
        if subset not in ("train", "val", "test"):
            raise ValueError(f"WLASL sample {video_id} has subset {subset!r}")
        if label < 0 or label >= len(names) or not names[label]:
            raise ValueError(f"WLASL sample {video_id} has unknown class {label}")
        inst = catalog.get(video_id)
        samples.append(
            Sample(
                source="wlasl100",
                sample_id=video_id,
                gloss=names[label],
                label=label,
                split=subset,
                video_path=root / "videos" / f"{video_id}.mp4",
                frame_start=None if inst is None else _as_int(inst.get("frame_start")),
                frame_end=None if inst is None else _as_int(inst.get("frame_end")),
                fps=None if inst is None else _as_int(inst.get("fps")),
                url=None if inst is None else inst.get("url"),
            )
        )
    samples.sort(key=lambda sample: sample.sample_id)
    if not samples:
        raise ValueError(f"nslt_100.json in {root} has no samples")
    # The class list covers every WLASL gloss. Keep only the indexes this
    # subset uses, which are 0..99 for the official top-100 split.
    limit = max(sample.label for sample in samples) + 1
    missing = [index for index in range(limit) if index >= len(names) or not names[index]]
    if missing:
        raise ValueError(f"WLASL100 class list is missing indexes {missing[:8]}")
    return SignDataset("wlasl100", root, tuple(names[:limit]), tuple(samples))


def _load_google(root: Path, *, download: bool) -> SignDataset:
    found = _find_google_root(root)
    if found is None and download:
        _download_google(root)
        found = _find_google_root(root)
    if found is None:
        raise FileNotFoundError(
            f"Google ISLR files missing in {root}. "
            "load('google_islr', download=True) downloads the Kaggle "
            f"competition {GOOGLE_COMPETITION!r} (about 40 GB zipped)."
        )

    mapping = json.loads((found / "sign_to_prediction_index_map.json").read_text())
    glosses = _glosses_from_map(mapping)
    samples = []
    seen = set()
    with (found / "train.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            sample_id = _safe_id(str(row["sequence_id"]))
            if sample_id in seen:
                raise ValueError(f"duplicate Google sequence_id {sample_id}")
            seen.add(sample_id)
            gloss = row["sign"]
            if gloss not in mapping:
                raise ValueError(f"Google sequence {sample_id} has unknown sign {gloss!r}")
            rel = Path(row["path"])
            if rel.is_absolute() or ".." in rel.parts:
                raise ValueError(f"Google sequence {sample_id} has unsafe path {row['path']!r}")
            samples.append(
                Sample(
                    source="google_islr",
                    sample_id=sample_id,
                    gloss=gloss,
                    label=int(mapping[gloss]),
                    split=_participant_split(str(row["participant_id"])),
                    landmark_path=found / rel,
                )
            )
    samples.sort(key=lambda sample: sample.sample_id)
    if not samples:
        raise ValueError(f"train.csv in {found} has no samples")
    return SignDataset("google_islr", found, glosses, tuple(samples))


def _participant_split(participant_id: str) -> str:
    # ponytail: the public asl-signs release has no val/test split. This is a
    # signer-disjoint 80/10/10 cut so the same participant cannot leak across
    # splits. Upgrade path: replace the bucket with an official split file.
    bucket = zlib.crc32(participant_id.encode()) % 10
    if bucket < 8:
        return "train"
    return "val" if bucket == 8 else "test"


def _wlasl_catalog(path: Path) -> dict:
    catalog = {}
    for entry in json.loads(path.read_text()):
        for inst in entry["instances"]:
            catalog[str(inst["video_id"])] = inst
    return catalog


def _wlasl_class_names(path: Path) -> list[str]:
    names: dict[int, str] = {}
    if path.exists():
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            index, gloss = line.split(maxsplit=1)
            names[int(index)] = gloss
    if not names:
        raise FileNotFoundError(
            f"missing {path.name}. It maps WLASL class indexes to glosses."
        )
    size = max(names) + 1
    glosses = [""] * size
    for index, gloss in names.items():
        glosses[index] = gloss
    return glosses


def _glosses_from_map(mapping: dict) -> tuple[str, ...]:
    if not mapping:
        raise ValueError("sign_to_prediction_index_map.json is empty")
    glosses = [""] * (max(int(index) for index in mapping.values()) + 1)
    for gloss, index in mapping.items():
        index = int(index)
        if glosses[index]:
            raise ValueError(f"duplicate Google class index {index}")
        glosses[index] = gloss
    if any(not gloss for gloss in glosses):
        raise ValueError("sign_to_prediction_index_map.json has a gap")
    return tuple(glosses)


def _download_google(root: Path) -> None:
    kaggle = shutil.which("kaggle")
    if kaggle is None:
        raise RuntimeError(
            "Google ISLR is the Kaggle competition 'asl-signs'. Install the "
            "Kaggle CLI, accept the competition rules, and put an API token in "
            "~/.kaggle/kaggle.json. Or unpack the archive so train.csv and "
            f"train_landmark_files/ sit under {root}."
        )
    root.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [kaggle, "competitions", "download", "-c", GOOGLE_COMPETITION, "-p", str(root)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise RuntimeError(detail or "kaggle competitions download failed")
    archives = sorted(root.glob("*.zip"))
    if not archives:
        raise FileNotFoundError(f"kaggle wrote no zip into {root}")
    import zipfile

    with zipfile.ZipFile(archives[0]) as archive:
        archive.extractall(root)
    archives[0].unlink()


def _find_google_root(root: Path) -> Path | None:
    direct = root / "train.csv"
    if direct.exists():
        return root
    if not root.exists():
        return None
    found = sorted(root.rglob("train.csv"))
    return found[0].parent if found else None


def _download_youtube(url: str, dest: Path) -> None:
    yt_dlp = shutil.which("yt-dlp")
    if yt_dlp is None:
        raise RuntimeError(
            "yt-dlp is required for YouTube-hosted WLASL videos. "
            f"Could not fetch {url}"
        )
    result = subprocess.run(
        [yt_dlp, "-f", "mp4/best", "--no-playlist", "-o", str(dest), url],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0 or not dest.exists() or dest.stat().st_size == 0:
        dest.unlink(missing_ok=True)
        detail = (result.stderr or result.stdout).strip()
        raise RuntimeError(detail or f"yt-dlp failed for {url}")


def _download_url(url: str, dest: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(request, timeout=120) as response, dest.open("wb") as out:
        shutil.copyfileobj(response, out)


def _ensure_url(url: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_suffix(dest.suffix + ".part")
    try:
        _download_url(url, partial)
        partial.replace(dest)
    except Exception:
        partial.unlink(missing_ok=True)
        raise


def _is_youtube(url: str) -> bool:
    host = urllib.parse.urlparse(url).netloc.lower()
    return (
        host == "youtu.be"
        or host.endswith(".youtube.com")
        or host == "youtube.com"
        or host.endswith("youtube-nocookie.com")
    )


def _check_source(source: str) -> str:
    if source not in SOURCES:
        raise ValueError(f"source must be one of {SOURCES}, got {source!r}")
    return source


def _safe_id(value: str) -> str:
    allowed = set("0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_-")
    if not value or any(char not in allowed for char in value):
        raise ValueError(f"unsafe sample id {value!r}")
    return value


def _as_int(value):
    if value is None or value == "":
        return None
    return int(value)


def _float_list(column) -> list[float]:
    import numpy as np

    return [np.nan if value is None else float(value) for value in column.to_pylist()]
