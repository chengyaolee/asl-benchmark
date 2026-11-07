"""Loader for WLASL100 isolated-sign benchmark.

WLASL100 is the first 100 glosses of WLASL_v0.3.json, each instance
keeping the split recorded in that file. The clip window is the WLASL
frame span (25 fps, frame_start is 1-based, frame_end of -1 means the last
frame).

Usage:
    from asl_benchmark import load

    wlasl = load("wlasl100", "data")
    for sample in wlasl.split("train"):
        print(sample.sample_id, sample.gloss, sample.video_path)
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

SOURCES = ("wlasl100",)

# Official WLASL start kit. The file is ordered by gloss frequency.
# WLASL100 is the first 100 entries.
WLASL_JSON_URL = (
    "https://raw.githubusercontent.com/dxli94/WLASL/master/start_kit/WLASL_v0.3.json"
)
WLASL100_GLOSSES = 100

_USER_AGENT = "asl-benchmark"


@dataclass(frozen=True)
class Sample:
    source: str
    sample_id: str
    gloss: str
    label: int
    split: str
    video_path: Path | None = None
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


def load(source: str = "wlasl100", root: str | Path = "data", *, download: bool = True) -> SignDataset:
    """Read WLASL100 from root, fetching published metadata when missing.

    root/wlasl100 holds the WLASL metadata and videos.
    Pass download=False to fail instead of touching the network.
    """
    source = _check_source(source)
    root = Path(root)
    return _load_wlasl(root / source, download=download)


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
