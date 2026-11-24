"""The information ladder, new runs X1-X4 (UPDATED PLAN.md, section 3b).

    X1  ResNet-18 fine-tuned on hand crops + the same 4 wrist-location numbers the keypoint model has
    X2  keypoint transformer + upper-body points (MediaPipe Pose: nose, shoulders, elbows, wrists, hips)
    X3  keypoint transformer without wrist location
    S1x the matched keypoint transformer again, so that it also gets the shuffled-frame test
    X4  every run above is also tested on test clips with the 16 frames shuffled (3 fixed shuffles, the same for all
        models): same trained model, only the test order changes

Same 16 shared 224x224 frames, same split and same final transformer as the earlier matched runs.

    python -m asl_benchmark.models.ladder --prepare               # body keypoints (CPU, a few minutes)
    python -m asl_benchmark.models.ladder --runs X3 X2 S1x X1     # 3 seeds each -> results/v2_ladder.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from multiprocessing import Pool

import numpy as np
import torch

from asl_benchmark.models import frame_encoder as fe
from asl_benchmark.models import matched as m
from asl_benchmark.models import video_backbone as vb
from asl_benchmark.models.cnn import _topk_accuracy

ROOT, CACHE, T = vb.ROOT, vb.CACHE, fe.T
sys.path.insert(0, str(ROOT))
import scripts.run_ablations as ablations  # noqa: E402

POSE_POINTS = [0, 11, 12, 13, 14, 15, 16, 23, 24]    # nose, shoulders, elbows, wrists, hips (MediaPipe Pose indices)
SHUFFLES = 3
NAMES = {
    "S1x": "S1. Keypoint transformer, matched (rerun with the shuffle test)",
    "X1": "X1. ResNet-18 fine-tuned, hand crops + wrist location (+aug)",
    "X2": "X2. Keypoint transformer + upper-body points",
    "X3": "X3. Keypoint transformer without wrist location",
}
ORDER = ["X3", "X2", "S1x", "X1"]


# ------------------------------------------------------------ body keypoints from the shared frames
_POSE = None


def _init_pose():
    global _POSE
    import mediapipe as mp

    _POSE = mp.solutions.pose.Pose(static_image_mode=True, model_complexity=1, min_detection_confidence=0.5)


def extract_pose_clip(frames: np.ndarray) -> np.ndarray:
    """(T, 9, 3) float32: x, y (fraction of the 224 frame) and relative depth of the 9 upper-body points; NaN if no person."""
    out = np.full((len(frames), len(POSE_POINTS), 3), np.nan, np.float32)
    for t, img in enumerate(frames):
        res = _POSE.process(img)                                   # one frame, no tracking
        if res.pose_landmarks:
            lm = res.pose_landmarks.landmark
            out[t] = [[lm[i].x, lm[i].y, lm[i].z] for i in POSE_POINTS]
    return out


def load_pose(samples, frames) -> np.ndarray:
    path = CACHE / f"kp_pose{m.S}.npz"
    ids = np.array([s.sample_id for s in samples])
    if path.exists():
        z = np.load(path)
        assert np.array_equal(z["ids"], ids), "pose cache does not match the sample order"
        return z["kp"]
    t0 = time.perf_counter()
    with Pool(5, initializer=_init_pose) as pool:
        kp = np.stack(pool.map(extract_pose_clip, list(frames), chunksize=4))
    np.savez(path, kp=kp, ids=ids)
    print(f"MediaPipe Pose on {len(frames) * T} frames -> {path.name} in {time.perf_counter() - t0:.0f}s", flush=True)
    return kp


# ------------------------------------------------------------------------- keypoint variants
def body_per_frame(original):
    """Per-frame features = the usual 132 hand numbers + 9 upper-body points (x, y) relative to the shoulder midpoint,
    divided by shoulder width (18 numbers) + a 'body found' flag. The clip array is (frames, 42 + 9, 3)."""

    def fn(x, location=True):
        x = np.asarray(x, np.float32)
        base = original(x[:, :42], location)
        body = x[:, 42:]
        mid = (body[:, 1, :2] + body[:, 2, :2]) / 2
        width = np.linalg.norm(body[:, 1, :2] - body[:, 2, :2], axis=-1)[:, None, None]
        rel = (body[:, :, :2] - mid[:, None, :]) / (width + 1e-6)
        missing = np.isnan(rel).any(axis=(1, 2)).astype(np.float32)[:, None]
        return np.concatenate([base, np.nan_to_num(rel).reshape(len(x), -1), missing], axis=1)

    return fn


def run_skeleton_variant(key: str, seeds) -> dict:
    samples = fe._samples()
    clips = m.load_static_keypoints(samples, None)                 # (n, 16, 42, 3); the cache exists after matched --prepare
    if key == "X2":
        clips = np.concatenate([clips, load_pose(samples, None)], axis=2)
    raw = {k: [] for k in ("train", "val", "test")}
    labels = {k: [] for k in raw}
    for s, clip in zip(samples, clips):
        raw[s.split].append(clip)
        labels[s.split].append(s.label)
    ablations.SEEDS = tuple(seeds)
    original = ablations.per_frame
    if key == "X2":
        ablations.per_frame = body_per_frame(original)
    try:
        row = ablations.run_tf_rung(NAMES[key], raw, labels, location=key != "X3", augment=True, num_layers=1,
                                    readout="mean", T=T, shuffle_test=SHUFFLES)
    finally:
        ablations.per_frame = original
    dim = {"S1x": 132, "X3": 128, "X2": 132 + 18 + 1}[key]
    row["params"] = sum(p.numel() for p in fe.head(dim).parameters())
    return row


# ------------------------------------------------------------------------------ X1: crops + location
def wrist_location(kp: np.ndarray) -> np.ndarray:
    """The 4 numbers the keypoint model uses: each wrist's (x, y) minus the frame centre; 0 where the hand is missing."""
    return np.nan_to_num(kp[:, :, [0, 21], :2] - 0.5).reshape(len(kp), kp.shape[1], 4).astype(np.float32)


def make_augment(mu: np.ndarray, sd: np.ndarray):
    """fe.augment plus, on the location numbers, the same random rotation / zoom / shift / noise the keypoints get
    (scripts/augment.py), zeroed again where the hand is missing, then standardised with the training statistics."""
    original = fe.augment

    def augment(images, flags, rng):
        out, fl = original(images, flags, rng)                     # temporal trim + per-crop affine; flags follow the trim
        fl = fl.copy()
        loc = fl[:, 2:6].reshape(len(fl), 2, 2)
        theta, s = np.deg2rad(rng.uniform(-15, 15)), rng.uniform(0.9, 1.1)
        c, si = np.cos(theta), np.sin(theta)
        x, y = loc[..., 0], loc[..., 1]
        new = np.stack([s * (c * x - si * y) + rng.uniform(-0.05, 0.05),
                        s * (si * x + c * y) + rng.uniform(-0.05, 0.05)], -1)
        new = new + rng.normal(0, 0.003, new.shape)
        new[fl[:, 0:2] > 0.5] = 0.0                                # a missing hand has no location
        fl[:, 2:6] = (new.reshape(len(fl), 4) - mu) / sd
        return out, fl.astype(np.float32)

    return augment


def run_x1(seeds, dev) -> dict:
    samples = fe._samples()
    kp = m.load_static_keypoints(samples, None)
    d = m.inputs("hands")                                          # (crops, missing flags, labels) per split
    loc = wrist_location(kp)
    idx = {k: [i for i, s in enumerate(samples) if s.split == k] for k in ("train", "val", "test")}
    clean = loc[idx["train"]].reshape(-1, 4)
    mu, sd = clean.mean(0), clean.std(0) + 1e-6
    std = lambda a: ((a - mu) / sd).astype(np.float32)
    (tr_x, tr_m, tr_y), (va_x, va_m, va_y), (te_x, te_m, te_y) = d["train"], d["val"], d["test"]
    train = (tr_x, np.concatenate([tr_m, loc[idx["train"]]], -1).astype(np.float32), tr_y)   # raw: augmented, then standardised
    va_f = np.concatenate([va_m, std(loc[idx["val"]])], -1).astype(np.float32)
    te_f = np.concatenate([te_m, std(loc[idx["test"]])], -1).astype(np.float32)

    vals, tests, top5s, secs, preds, shuffled = [], [], [], [], [], []
    original = fe.augment
    fe.augment = make_augment(mu, sd)
    try:
        for seed in seeds:
            t0 = time.perf_counter()
            model = fe.fine_tune(train, (va_x, va_f, va_y), seed, dev)
            secs.append(time.perf_counter() - t0)
            va_logits, te_logits = fe.predict(model, va_x, va_f, dev), fe.predict(model, te_x, te_f, dev)
            accs = []
            for k in range(SHUFFLES):
                perm = np.random.default_rng(1234 + k).permutation(T)
                accs.append(float((fe.predict(model, te_x[:, perm], te_f[:, perm], dev).argmax(1) == te_y).mean()))
            shuffled.append(float(np.mean(accs)))
            vals.append(float((va_logits.argmax(1) == va_y).mean()))
            tests.append(float((te_logits.argmax(1) == te_y).mean()))
            top5s.append(_topk_accuracy(te_logits, te_y))
            preds.append(te_logits.argmax(1).tolist())
            params = sum(p.numel() for p in model.parameters())
            print(f"  seed {seed}: val {vals[-1] * 100:.1f}  test {tests[-1] * 100:.1f}  shuffled {shuffled[-1] * 100:.1f}"
                  f"  ({secs[-1]:.0f}s)", flush=True)
            del model
            torch.cuda.empty_cache()
    finally:
        fe.augment = original
    row = {
        "rung": NAMES["X1"], "val_top1": round(float(np.mean(vals)) * 100, 1),
        "test_top1": round(float(np.mean(tests)) * 100, 1), "test_std": round(float(np.std(tests)) * 100, 1),
        "test_top5": round(float(np.mean(top5s)) * 100, 1), "train_s": round(float(np.mean(secs)), 1),
        "params": int(params), "seeds": list(seeds), "n_train": len(tr_y), "n_val": len(va_y), "n_test": len(te_y),
        "test_labels": te_y.tolist(), "test_preds": preds,
        "shuffled_test_top1": round(float(np.mean(shuffled)) * 100, 1),
        "shuffled_test_by_seed": [round(x * 100, 1) for x in shuffled],
    }
    print(f"[X1] {row['rung']} | Val: {row['val_top1']}% | Test: {row['test_top1']}% (std {row['test_std']}%) | "
          f"Top-5: {row['test_top5']}% | shuffled {row['shuffled_test_top1']}% | {row['train_s']:.0f}s", flush=True)
    return row


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--prepare", action="store_true", help="extract the upper-body keypoints, then exit")
    p.add_argument("--runs", nargs="+", default=ORDER, choices=ORDER)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--out", default="results/v2_ladder.json")
    args = p.parse_args(argv)
    if args.prepare:
        samples, frames = m.load_frames()
        kp = load_pose(samples, frames)
        found = (~np.isnan(kp[..., 0])).all(-1).mean(-1)               # per clip: share of frames with a person
        for split in ("train", "val", "test"):
            sel = np.array([s.split == split for s in samples])
            print(f"{split:5s}: person found in {found[sel].mean():.1%} of frames; clips with none: "
                  f"{int((found[sel] == 0).sum())} of {int(sel.sum())}")
        return
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = ROOT / args.out
    rows = json.loads(out.read_text()) if out.exists() else []
    for key in args.runs:
        row = run_x1(tuple(args.seeds), dev) if key == "X1" else run_skeleton_variant(key, args.seeds)
        row["key"] = key
        rows = [r for r in rows if r.get("key") != key] + [row]
        rows.sort(key=lambda r: ORDER.index(r["key"]))
        out.write_text(json.dumps(rows, indent=1))
    print(f"saved {out.relative_to(ROOT)}", flush=True)


if __name__ == "__main__":
    main()
