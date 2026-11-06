"""One loader for WLASL100 and Google Isolated Sign Language Recognition.

Both sources come back as the same Sample records, so later models can
iterate either dataset without knowing which files it was built from.

WLASL100 is the first 100 glosses of WLASL_v0.3.json, each instance
keeping the split recorded in that file. The clip window is the WLASL
frame span (25 fps, frame_start is 1-based, frame_end of -1 means the last
frame). Google ISLR is precomputed MediaPipe Holistic
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
import os
import shutil
import subprocess
import urllib.error
import urllib.parse
import urllib.request
import zlib
from dataclasses import dataclass
from pathlib import Path

SOURCES = ("wlasl100", "google_islr")

# Official WLASL start kit. The file is ordered by gloss frequency.
# WLASL100 is the first 100 entries.
WLASL_JSON_URL = (
    "https://raw.githubusercontent.com/dxli94/WLASL/master/start_kit/WLASL_v0.3.json"
)
WLASL100_GLOSSES = 100

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
    n = table.num_rows
    if n == 0:
        return np.zeros((0, N_LANDMARKS, 3), np.float32)

    frame_missing = _null_mask(table.column("frame"))
    index_missing = _null_mask(table.column("landmark_index"))
    frames = table.column("frame").fill_null(0).to_numpy(zero_copy_only=False)
    indexes = table.column("landmark_index").fill_null(0).to_numpy(zero_copy_only=False)
    xs = _float_numpy(table.column("x"))
    ys = _float_numpy(table.column("y"))
    zs = _float_numpy(table.column("z"))
    base_of = np.full(n, -1, np.int32)
    count_of = np.zeros(n, np.int32)
    kinds = np.array(table.column("type").to_pylist())
    for name, (base, count) in _LANDMARK_OFFSET.items():
        hit = kinds == name
        base_of[hit] = base
        count_of[hit] = count

    present = ~frame_missing
    order = np.unique(frames[present])
    out = np.full((len(order), N_LANDMARKS, 3), np.nan, np.float32)
    ok = present & ~index_missing & (base_of >= 0) & (indexes >= 0) & (indexes < count_of)
    if len(order) == 0 or not ok.any():
        return out
    cols = base_of[ok] + indexes[ok]
    rows = np.searchsorted(order, frames[ok])
    # Duplicate cells keep the last row, matching a forward scan.
    out[rows, cols, 0] = xs[ok]
    out[rows, cols, 1] = ys[ok]
    out[rows, cols, 2] = zs[ok]
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
    if download:
        _ensure_url(WLASL_JSON_URL, meta)
    if not meta.exists():
        raise FileNotFoundError(
            f"WLASL100 metadata missing in {root}. "
            "load('wlasl100', download=True) fetches WLASL_v0.3.json."
        )

    entries = json.loads(meta.read_text())[:WLASL100_GLOSSES]
    if not entries:
        raise ValueError(f"WLASL_v0.3.json in {root} has no glosses")
    glosses = tuple(entry["gloss"] for entry in entries)
    samples = []
    seen = set()
    for label, entry in enumerate(entries):
        for inst in entry["instances"]:
            video_id = _safe_id(str(inst["video_id"]))
            if video_id in seen:
                raise ValueError(f"duplicate WLASL video_id {video_id}")
            seen.add(video_id)
            subset = inst["split"]
            if subset == "validation":
                subset = "val"
            if subset not in ("train", "val", "test"):
                raise ValueError(f"WLASL sample {video_id} has subset {subset!r}")
            samples.append(
                Sample(
                    source="wlasl100",
                    sample_id=video_id,
                    gloss=entry["gloss"],
                    label=label,
                    split=subset,
                    video_path=root / "videos" / f"{video_id}.mp4",
                    frame_start=_as_int(inst.get("frame_start")),
                    frame_end=_as_int(inst.get("frame_end")),
                    fps=_as_int(inst.get("fps")),
                    url=inst.get("url"),
                )
            )
    if not samples:
        raise ValueError(f"WLASL_v0.3.json in {root} has no instances")
    samples.sort(key=lambda sample: sample.sample_id)
    return SignDataset("wlasl100", root, glosses, tuple(samples))


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
    _apply_dotenv(Path(__file__).resolve().parents[1] / ".env")
    token = os.environ.get("KAGGLE_API_TOKEN", "").strip()
    root.mkdir(parents=True, exist_ok=True)
    if token:
        _download_google_token(root, token)
    else:
        _download_google_cli(root)
    archives = sorted(root.glob("*.zip"))
    if not archives:
        raise FileNotFoundError(f"Kaggle wrote no zip into {root}")
    import zipfile

    with zipfile.ZipFile(archives[0]) as archive:
        archive.extractall(root)
    archives[0].unlink()


def _download_google_token(root: Path, token: str) -> None:
    """Download with a KGAT token. The PyPI client on Python 3.9 cannot."""
    url = (
        "https://www.kaggle.com/api/v1/competitions/data/download-all/"
        + GOOGLE_COMPETITION
    )
    request = urllib.request.Request(
        url,
        headers={"Authorization": f"Bearer {token}", "User-Agent": _USER_AGENT},
    )
    dest = root / f"{GOOGLE_COMPETITION}.zip"
    partial = dest.with_suffix(".zip.part")
    try:
        with urllib.request.urlopen(request) as response, partial.open("wb") as out:
            shutil.copyfileobj(response, out, length=1024 * 1024)
        if partial.stat().st_size == 0:
            raise RuntimeError("Kaggle returned an empty archive")
        partial.replace(dest)
    except urllib.error.HTTPError as exc:
        partial.unlink(missing_ok=True)
        detail = exc.read().decode("utf-8", "replace").strip()
        raise RuntimeError(f"Kaggle download failed ({exc.code}): {detail}") from exc
    except Exception:
        partial.unlink(missing_ok=True)
        raise


def _download_google_cli(root: Path) -> None:
    if not os.environ.get("KAGGLE_USERNAME") or not os.environ.get("KAGGLE_KEY"):
        raise RuntimeError(
            "Google ISLR is the Kaggle competition 'asl-signs'. Accept the "
            "rules, then set KAGGLE_API_TOKEN in .env (Create New Token). "
            "Or unpack the archive so train.csv and "
            f"train_landmark_files/ sit under {root}."
        )
    kaggle = shutil.which("kaggle")
    if kaggle is None:
        raise RuntimeError("The kaggle command is not installed.")
    result = subprocess.run(
        [kaggle, "competitions", "download", "-c", GOOGLE_COMPETITION, "-p", str(root)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise RuntimeError(detail or "kaggle competitions download failed")


def _apply_dotenv(path: Path) -> None:
    """Fill missing environment variables from a KEY=value file."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export ") :]
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and value:
            os.environ.setdefault(key, value)


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


def _null_mask(column):
    import numpy as np

    return np.asarray(column.is_null().to_numpy(zero_copy_only=False), dtype=bool)


def _float_numpy(column):
    import numpy as np

    filled = column.fill_null(np.nan).to_numpy(zero_copy_only=False)
    return np.asarray(filled, dtype=np.float32)
