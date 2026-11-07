# WLASL100 setup: download + keypoint extraction

How to get the WLASL100 videos onto your machine and turn them into hand keypoints for the pose models (MLP, GRU, transformer).

## 0. Environment

From the repo root (`asl-benchmark/`):

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install "mediapipe==0.10.14" opencv-python kaggle
```

MediaPipe is pinned to 0.10.14 because it still has the `mp.solutions.hands` API that the extraction script uses.

You can ignore the `NotOpenSSLWarning` on macOS's built-in Python 3.9.

## 1. Kaggle credentials

1. On kaggle.com, go to your profile picture → **Settings** → **API** → **Create Legacy API Key**. This downloads `kaggle.json`.
   - Use the *legacy* key. The newer `KGAT_...` tokens don't work with the `kaggle` tool on Python 3.9, and you'll get `KeyError: 'username'`.
2. Put it in your **home folder**, not in the repo:
   ```bash
   mkdir -p ~/.kaggle
   mv ~/Downloads/kaggle.json ~/.kaggle/
   chmod 600 ~/.kaggle/kaggle.json
   ```
3. Never commit it. `.gitignore` should include:
   ```
   .kaggle/
   kaggle.json
   .env
   data/
   ```

## 2. Download the videos

We use the preprocessed copy on Kaggle: [`risangbaskoro/wlasl-processed`](https://www.kaggle.com/datasets/risangbaskoro/wlasl-processed). It's about 5.2 GB zipped, and you need about 11 GB of free disk (`df -h .` to check).

`kaggle datasets download ...` tends to drop the connection partway (around 3.1 GB) and can't resume. Use curl instead, which picks up where it stopped:

```bash
mkdir -p data/raw
USER=$(python3 -c "import json;print(json.load(open('$HOME/.kaggle/kaggle.json'))['username'])")
KEY=$(python3 -c "import json;print(json.load(open('$HOME/.kaggle/kaggle.json'))['key'])")

until curl -L -C - --fail -u "$USER:$KEY" \
  -o data/raw/wlasl.zip \
  https://www.kaggle.com/api/v1/datasets/download/risangbaskoro/wlasl-processed
do echo "dropped, resuming in 5s..."; sleep 5; done

unzip -q data/raw/wlasl.zip -d data/raw
```

If curl also keeps failing, download it in Chrome from the dataset page instead. Chrome can resume a failed download from its Downloads page.

Copy the videos to where the loader looks for them:

```bash
mkdir -p data/wlasl100/videos
cp data/raw/videos/*.mp4 data/wlasl100/videos/
```

The loader fetches `WLASL_v0.3.json` into `data/wlasl100/` by itself on first `load()`.

## 3. Check what you have

```bash
python - <<'EOF'
from pathlib import Path
from asl_benchmark.loader import load
print("videos in data/raw/videos:", len(list(Path("data/raw/videos").glob("*.mp4"))))
ds = load("wlasl100", "data")
have = [s for s in ds.samples if s.video_path.exists()]
print(len(have), "of", len(ds.samples), "WLASL100 videos present")
for split in ("train", "val", "test"):
    print(split, sum(s.split == split for s in have), "of", len(ds.split(split)))
EOF
```

**Expected output:**

```
videos in data/raw/videos: 11980
1013 of 2038 WLASL100 videos present
train 748 of 1442
val 165 of 338
test 100 of 258
```

Some things to know about these numbers:

- **The Kaggle copy only has about half of WLASL100.** The rest are dead links. Most of the missing ones are long YouTube videos that contain several signs.
- **No trimming is needed.** Every clip we have runs from frame 1 to the end of its file (none of them has an `frame_end` other than -1), so the extraction uses `TRIM = False`.
- **The official split counts are 1442 / 338 / 258.** We keep the official split assignment and only lose the videos we don't have. Report the counts we actually used.
- **Test accuracy will be noisy.** With 100 test videos, each one is 1% of the score.
- **The same signers are in train and test.** 52 of the 56 test signers also appear in training. That's how the official WLASL split works, but it's worth stating in the report.

## 4. Extract hand keypoints

```bash
python scripts/extract.py
```

- Expect roughly **1–3 hours on a laptop CPU** for about 1,000 videos.
- You can stop it with Ctrl+C at any point. It skips videos that are already done, so rerunning it continues from where it stopped and also picks up any new videos you add later.
- Output is one file per video: `data/wlasl100/keypoints/<video_id>.npy`, shaped **`(frames, 42, 3)`**.
  - Points 0–20 are the signer's **left** hand and points 21–41 are the **right** hand, each as (x, y, z).
  - x and y are normalized to the image size (0–1).
  - A hand that wasn't detected in a frame is **NaN**.
- Left and right are swapped relative to MediaPipe's own labels. MediaPipe Hands assumes a mirrored selfie image, so its "Left" is the signer's right hand on normal video. The script corrects for that so points 0–20 are the signer's left hand and points 21–41 are the right hand.

**Sanity checks while it runs:** each line prints `frames=` (usually 30–150) and `hand_seen=` (share of frames where a hand was detected, usually 60–100%). If `hand_seen` is 0% on most videos, something is wrong.