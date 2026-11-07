"""Fetch WLASL100 and Google ISLR into data/.

WLASL clips come from the Kaggle dataset risangbaskoro/wlasl-processed.
The zip is saved at data/raw/wlasl.zip, unpacked under data/raw, and the
mp4 files are copied to data/wlasl100/videos, where load() looks.
Credentials are ~/.kaggle/kaggle.json, or KAGGLE_API_TOKEN in .env.
Google ISLR is downloaded by load().

Run from the repository root:

    python3 load_data.py
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
import zipfile
from pathlib import Path

import asl_benchmark

WLASL_URL = (
    "https://www.kaggle.com/api/v1/datasets/download/"
    "risangbaskoro/wlasl-processed"
)


def main() -> None:
    """Download both datasets and print how many samples each one has."""
    root = Path("data")
    _download_wlasl_clips(root)
    for source in ("wlasl100", "google_islr"):
        dataset = asl_benchmark.load(source, root)
        print(
            f"{dataset.source}: {len(dataset.samples)} samples, "
            f"{len(dataset.glosses)} glosses -> {dataset.root}"
        )


def _download_wlasl_clips(root: Path) -> None:
    videos = root / "wlasl100" / "videos"
    raw = root / "raw"
    archive = raw / "wlasl.zip"
    if _mp4s(videos):
        print(f"wlasl100 videos already in {videos}", flush=True)
        return
    raw.mkdir(parents=True, exist_ok=True)
    if not _zip_ready(archive):
        _curl_resume(archive)
    if not _mp4s(raw / "videos"):
        print(f"unzipping {archive}", flush=True)
        with zipfile.ZipFile(archive) as zipped:
            zipped.extractall(raw)
    sources = _clip_files(raw)
    if not sources:
        raise FileNotFoundError(f"no mp4 files under {raw}")
    videos.mkdir(parents=True, exist_ok=True)
    for src in sources:
        dest = videos / src.name
        if dest.exists() and dest.stat().st_size > 0:
            continue
        shutil.copy2(src, dest)
    print(f"wlasl100: {len(_mp4s(videos))} videos -> {videos}", flush=True)


def _clip_files(raw: Path) -> list[Path]:
    direct = _mp4s(raw / "videos")
    if direct:
        return direct
    return sorted(raw.rglob("*.mp4"))


def _mp4s(folder: Path) -> list[Path]:
    if not folder.is_dir():
        return []
    return sorted(folder.glob("*.mp4"))


def _zip_ready(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        with zipfile.ZipFile(path) as archive:
            return bool(archive.namelist())
    except zipfile.BadZipFile:
        return False


def _curl_resume(dest: Path) -> None:
    auth = _curl_auth()
    while True:
        result = subprocess.run(
            [
                "curl",
                "-L",
                "-C",
                "-",
                "--fail",
                *auth,
                "-o",
                str(dest),
                WLASL_URL,
            ],
            check=False,
        )
        if result.returncode == 0 and _zip_ready(dest):
            return
        if result.returncode == 22:
            dest.unlink(missing_ok=True)
            raise RuntimeError(
                "Kaggle rejected the WLASL download. "
                "Check ~/.kaggle/kaggle.json or KAGGLE_API_TOKEN in .env."
            )
        if result.returncode == 0:
            dest.unlink(missing_ok=True)
            raise RuntimeError(f"Kaggle download finished but {dest.name} is not a zip.")
        print("dropped, resuming in 5s...", flush=True)
        time.sleep(5)


def _curl_auth() -> list[str]:
    legacy = Path.home() / ".kaggle" / "kaggle.json"
    if legacy.is_file():
        payload = json.loads(legacy.read_text())
        username = str(payload.get("username", "")).strip()
        key = str(payload.get("key", "")).strip()
        if not username or not key:
            raise ValueError(f"{legacy} needs username and key")
        return ["-u", f"{username}:{key}"]
    asl_benchmark.loader._apply_dotenv(Path(__file__).resolve().parent / ".env")
    token = os.environ.get("KAGGLE_API_TOKEN", "").strip()
    if not token:
        raise FileNotFoundError(
            "WLASL clips need ~/.kaggle/kaggle.json or KAGGLE_API_TOKEN in .env."
        )
    return ["-H", f"Authorization: Bearer {token}"]


if __name__ == "__main__":
    main()
