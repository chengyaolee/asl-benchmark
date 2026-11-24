"""Matched-input comparison: the image vs. the skeleton extracted from that same image (UPDATED PLAN.md).

Every clip is decoded once into 16 frames (rows chosen by fe.frame_indices) letterboxed to 224x224 RGB.
  * image models  : those frames (or crops of them) -> ResNet-18 -> transformer head
  * skeleton model: MediaPipe Hands, static_image_mode=True, run on exactly those frames -> the same
                    per-frame features and transformer head
No tracking, no neighbouring frames and no original resolution on the skeleton side.

    python -m asl_benchmark.models.matched --prepare            # build caches (CPU)
    python -m asl_benchmark.models.matched --rungs S1 S2 ...    # train

Writes results/matched_ablations.json.
"""

from __future__ import annotations

import argparse
import json
import time
from multiprocessing import Pool

import numpy as np
import torch

from asl_benchmark.models import frame_encoder as fe
from asl_benchmark.models import video_backbone as vb
from asl_benchmark.models.cnn import _topk_accuracy  # noqa: F401  (kept for parity with fe)

ROOT, CACHE = vb.ROOT, vb.CACHE
S, T, PATCH = 224, fe.T, vb.SIZE            # frame size, frames per clip, hand-patch size (112)

RUNGS = {
    "S1": dict(name="S1. Skeleton: static MediaPipe on the 16 frames", skeleton="static"),
    "S2": dict(name="S2. ResNet-18 frozen, full 224 frame", kind="full", finetune=False),
    "S3": dict(name="S3. ResNet-18 fine-tuned, full 224 frame (+aug)", kind="full", finetune=True),
    "S4": dict(name="S4. ResNet-18 frozen, hand patches at static boxes", kind="hands", finetune=False),
    "S5": dict(name="S5. ResNet-18 fine-tuned, hand patches at static boxes (+aug)", kind="hands", finetune=True),
    "C1": dict(name="C1. Skeleton: tracked full-resolution extraction, 16 frames", skeleton="tracked"),
    "C2a": dict(name="C2a. ResNet-18 frozen, skeleton drawn as an image", kind="skeleton_img", finetune=False),
    "C2b": dict(name="C2b. ResNet-18 fine-tuned, skeleton drawn as an image (+aug)", kind="skeleton_img", finetune=True),
    "C3": dict(name="C3. ResNet-18 frozen, hand patches blanked where the hand is missed", kind="hands_blank", finetune=False),
    "C4": dict(name="C4. Fusion: ResNet-18 frozen hand patches + skeleton features", kind="fusion", finetune=False),
    "C5": dict(name="C5. Skeleton: static MediaPipe, x/y only (no z)", skeleton="static_xy"),
}
ORDER = ["S1", "C1", "C5", "S2", "S4", "C3", "C4", "C2a", "S5", "C2b", "S3"]   # cheap runs first


# ------------------------------------------------------------------ frames (the shared input)
def letterbox(frame: np.ndarray, size: int = S) -> np.ndarray:
    """BGR frame -> square RGB `size` x `size`, aspect ratio kept, black bars."""
    import cv2

    h, w = frame.shape[:2]
    side = max(h, w)
    pad = np.zeros((side, side, 3), frame.dtype)
    y0, x0 = (side - h) // 2, (side - w) // 2
    pad[y0 : y0 + h, x0 : x0 + w] = frame
    return cv2.cvtColor(cv2.resize(pad, (size, size), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)


def read_frames(job) -> np.ndarray:
    video_path, kp_path = job
    wanted = fe.frame_indices(len(np.load(kp_path)))
    frames, _, _ = fe._decode(video_path, wanted)
    blank = np.zeros((S, S, 3), np.uint8)
    return np.stack([blank if f is None else letterbox(f) for f in frames])


def load_frames():
    """(samples, frames (n, T, S, S, 3) uint8), cached. Order = fe._samples()."""
    samples = fe._samples()
    path = CACHE / f"frames_kp{T}x{S}.npz"
    if not path.exists():
        t0 = time.perf_counter()
        jobs = [(s.video_path, vb.KP_DIR / f"{s.sample_id}.npy") for s in samples]
        with Pool(6) as pool:
            frames = np.stack(pool.map(read_frames, jobs, chunksize=8))
        np.savez(path, frames=frames)
        print(f"decoded {len(samples)} clips -> {path.name} in {time.perf_counter() - t0:.0f}s", flush=True)
    return samples, np.load(path)["frames"]


# --------------------------------------------------- static skeleton from exactly those frames
_HANDS = None


def _init_mediapipe():
    global _HANDS
    import mediapipe as mp

    _HANDS = mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5)


def extract_clip(frames: np.ndarray) -> np.ndarray:
    """(T, 42, 3) float32, NaN where a hand is missing. Same layout and handedness rule as scripts/extract.py."""
    out = np.full((len(frames), 42, 3), np.nan, np.float32)
    for t, img in enumerate(frames):
        res = _HANDS.process(img)                                  # one frame, no tracking
        if res.multi_hand_landmarks:
            for lm, hd in zip(res.multi_hand_landmarks, res.multi_handedness):
                start = 21 if hd.classification[0].label == "Left" else 0
                out[t, start : start + 21] = [[p.x, p.y, p.z] for p in lm.landmark]
    return out


def load_static_keypoints(samples, frames) -> np.ndarray:
    path = CACHE / f"kp_static{S}.npz"
    ids = np.array([s.sample_id for s in samples])
    if path.exists():
        z = np.load(path)
        assert np.array_equal(z["ids"], ids), "static keypoint cache does not match the sample order"
        return z["kp"]
    t0 = time.perf_counter()
    with Pool(5, initializer=_init_mediapipe) as pool:
        kp = np.stack(pool.map(extract_clip, list(frames), chunksize=4))
    np.savez(path, kp=kp, ids=ids)
    print(f"static MediaPipe on {len(frames) * T} frames -> {path.name} in {time.perf_counter() - t0:.0f}s", flush=True)
    return kp


# ---------------------------------------------------------------------- input variants
def make_patches(frames: np.ndarray, kp: np.ndarray, borrow: bool):
    """Hand patches cut from the shared 224 frames at the static skeleton's boxes.

    borrow=True : a missed hand takes the box of the nearest frame where it was found (flag 1).
    borrow=False: a missed hand gives a black patch (flag 1)."""
    n = len(frames)
    patches = np.zeros((n, T, 2, PATCH, PATCH, 3), np.uint8)
    missing = np.ones((n, T, 2), np.float32)
    for i in range(n):
        for hand in (0, 1):
            pts = kp[i, :, hand * 21 : (hand + 1) * 21, :2]
            found = ~np.isnan(pts).any((1, 2))
            rows = np.flatnonzero(found)
            if len(rows) == 0:
                continue
            for t in range(T):
                if found[t]:
                    src = t
                    missing[i, t, hand] = 0.0
                elif borrow:
                    src = rows[np.abs(rows - t).argmin()]
                else:
                    continue
                patches[i, t, hand] = vb._square(frames[i, t], fe._hand_box(pts[src], S, S))
    return patches, missing


def render_skeleton(kp: np.ndarray, size: int = PATCH) -> np.ndarray:
    """The static skeleton drawn on black, one colour per hand: (n, T, 1, size, size, 3) uint8."""
    import cv2
    import mediapipe as mp

    links = list(mp.solutions.hands.HAND_CONNECTIONS)
    colours = [(255, 120, 60), (60, 200, 120)]
    out = np.zeros((len(kp), T, 1, size, size, 3), np.uint8)
    for i in range(len(kp)):
        for t in range(T):
            img = out[i, t, 0]
            for hand in (0, 1):
                pts = kp[i, t, hand * 21 : (hand + 1) * 21, :2]
                if np.isnan(pts).any():
                    continue
                px = np.clip(pts * size, 0, size - 1).astype(int)
                for a, b in links:
                    cv2.line(img, tuple(px[a]), tuple(px[b]), colours[hand], 1)
                for p in px:
                    cv2.circle(img, tuple(p), 2, colours[hand], -1)
    return out


def skeleton_features(kp: np.ndarray, train_mask: np.ndarray) -> np.ndarray:
    """The landmark model's per-frame features (132-d), standardised with training statistics."""
    from asl_benchmark.models import hands as hand_landmarks
    from scripts.features import per_frame

    feat = np.stack([per_frame(hand_landmarks.hands(c), location=True) for c in kp]).astype(np.float32)
    flat = feat[train_mask].reshape(-1, feat.shape[-1])
    return (feat - flat.mean(0)) / (flat.std(0) + 1e-6)


def inputs(kind: str):
    """Per split: (images (n, T, streams, H, W, 3), flags (n, T, F), labels)."""
    samples, frames = load_frames()
    kp = load_static_keypoints(samples, frames)
    n = len(samples)
    if kind == "full":
        return fe._by_split(samples, frames[:, :, None], np.zeros((n, T, 0), np.float32))
    if kind == "skeleton_img":
        return fe._by_split(samples, render_skeleton(kp), np.zeros((n, T, 0), np.float32))
    patches, missing = make_patches(frames, kp, borrow=kind != "hands_blank")
    if kind == "fusion":
        train = np.array([s.split == "train" for s in samples])
        missing = np.concatenate([missing, skeleton_features(kp, train)], -1)
    return fe._by_split(samples, patches, missing)


# --------------------------------------------------------------------- skeleton rungs
def run_skeleton(cfg, seeds) -> dict:
    """The landmark transformer (as fe rung 1) on the chosen keypoints, 16 frames."""
    import scripts.run_ablations as ablations

    if cfg["skeleton"] == "tracked":
        raw, labels, _ = ablations.load_wlasl()
    else:
        samples, frames = load_frames()
        kp = load_static_keypoints(samples, frames)
        raw = {k: [] for k in ("train", "val", "test")}
        labels = {k: [] for k in raw}
        for s, clip in zip(samples, kp):
            raw[s.split].append(clip)
            labels[s.split].append(s.label)
    ablations.SEEDS = tuple(seeds)
    original = ablations.per_frame
    if cfg["skeleton"] == "static_xy":
        def per_frame_xy(x, location=True):
            f = original(x, location)
            f[:, 2:126:3] = 0.0           # the z of each of the 42 points; zeroed after augmentation so it stays constant
            return f
        ablations.per_frame = per_frame_xy
    try:
        row = ablations.run_tf_rung(cfg["name"], raw, labels, location=True, augment=True, num_layers=1,
                                    readout="mean", T=T)
    finally:
        ablations.per_frame = original
    row["params"] = sum(p.numel() for p in fe.head(132).parameters())
    return row


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--prepare", action="store_true", help="build the frame and keypoint caches, then exit")
    p.add_argument("--rungs", nargs="+", default=ORDER, choices=sorted(RUNGS))
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--out", default="results/matched_ablations.json")
    args = p.parse_args(argv)
    if args.prepare:
        samples, frames = load_frames()
        kp = load_static_keypoints(samples, frames)
        seen = (~np.isnan(kp[..., 0])).any(-1).mean()
        print(f"frames {frames.shape}, static keypoints {kp.shape}, hand seen in {seen:.1%} of (clip, frame, hand-slot) rows")
        return
    torch.backends.cudnn.benchmark = True
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_path = ROOT / args.out
    rows = json.loads(out_path.read_text()) if out_path.exists() else []
    for key in args.rungs:
        cfg = RUNGS[key]
        row = run_skeleton(cfg, tuple(args.seeds)) if "skeleton" in cfg else \
            fe.run_rung(None, tuple(args.seeds), dev, cfg=cfg, data=inputs(cfg["kind"]))
        row["key"] = key
        rows = [r for r in rows if r.get("key") != key] + [row]
        rows.sort(key=lambda r: ORDER.index(r["key"]))
        out_path.write_text(json.dumps(rows, indent=1))
    print(f"saved {out_path.relative_to(ROOT)}", flush=True)


if __name__ == "__main__":
    main()
