"""MediaPipe Holistic keypoints for WLASL100: face, hands and body in one pass.

Run from the repo root:
    python scripts/extract_holistic.py
Or split the work across terminals (each takes a different share of videos):
    python scripts/extract_holistic.py --shard 0 --num-shards 3
    python scripts/extract_holistic.py --shard 1 --num-shards 3
    python scripts/extract_holistic.py --shard 2 --num-shards 3

Output: data/wlasl100/holistic/<video_id>.npy, float32 (frames, 543, 3).
Rows follow LANDMARK_LAYOUT from the loader, the same layout as Google ISLR:
    face 0-467, left_hand 468-488, pose 489-521, right_hand 522-542.
Left/right are the signer's own left/right (Holistic's convention, no swap).
Anything not detected in a frame is NaN. Hands-only files in
data/wlasl100/keypoints/ are left untouched.
"""

import argparse, sys, time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from asl_benchmark.loader import LANDMARK_LAYOUT, N_LANDMARKS, load

p = argparse.ArgumentParser()
p.add_argument("--shard", type=int, default=0)
p.add_argument("--num-shards", type=int, default=1)
p.add_argument("--complexity", type=int, default=1, choices=(0, 1, 2),
               help="pose model size: 0 fast, 1 default, 2 most accurate")
args = p.parse_args()

assert [n for n, _ in LANDMARK_LAYOUT] == ["face", "left_hand", "pose", "right_hand"]
FIELD = {"face": "face_landmarks", "left_hand": "left_hand_landmarks",
         "pose": "pose_landmarks", "right_hand": "right_hand_landmarks"}

out_dir = ROOT / "data/wlasl100/holistic"
out_dir.mkdir(parents=True, exist_ok=True)
samples = load("wlasl100", ROOT / "data", download=False).samples
todo = [s for i, s in enumerate(samples)
        if i % args.num_shards == args.shard
        and s.video_path.exists() and not (out_dir / f"{s.sample_id}.npy").exists()]
print(f"shard {args.shard}/{args.num_shards}: {len(todo)} videos to process")


def frame_row(res) -> np.ndarray:
    row = np.full((N_LANDMARKS, 3), np.nan, np.float32)
    base = 0
    for name, count in LANDMARK_LAYOUT:
        lms = getattr(res, FIELD[name])
        if lms is not None:
            pts = [[q.x, q.y, q.z] for q in lms.landmark][:count]
            row[base:base + len(pts)] = pts
        base += count
    return row


t0 = time.time()
for k, s in enumerate(todo, 1):
    # A fresh tracker per video so tracking never carries over between clips.
    with mp.solutions.holistic.Holistic(static_image_mode=False,
                                        model_complexity=args.complexity,
                                        refine_face_landmarks=False) as holistic:
        cap, frames = cv2.VideoCapture(str(s.video_path)), []
        while True:
            ok, img = cap.read()
            if not ok:
                break
            frames.append(frame_row(holistic.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))))
        cap.release()

    arr = np.array(frames, np.float32).reshape(-1, N_LANDMARKS, 3)
    tmp = out_dir / f"{s.sample_id}.tmp.npy"
    np.save(tmp, arr)
    tmp.replace(out_dir / f"{s.sample_id}.npy")   # never leave a half-written file

    seen = lambda a, b: np.isfinite(arr[:, a:b, 0]).any(1).mean() if len(arr) else 0
    print(f"[{k}/{len(todo)}] {s.sample_id} frames={len(arr)} "
          f"pose={seen(489, 522):.0%} L={seen(468, 489):.0%} R={seen(522, 543):.0%} "
          f"face={seen(0, 468):.0%} elapsed={time.time() - t0:.0f}s")
