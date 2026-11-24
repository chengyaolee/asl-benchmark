"""End-to-end inference latency: raw video file -> prediction, one clip at a time, nothing cached.

Every pipeline pays for everything it needs at deployment: decoding and resizing the frames, MediaPipe where the
pipeline uses it (skeletons, and hand patches because their crops come from the skeleton), cropping or drawing,
and the network. Latency does not depend on trained weights, so each model is a freshly initialised network of the
trained architecture. Stages that several pipelines share (decode, MediaPipe) are measured once per clip and charged to
each pipeline that needs them. Hand-patch pipelines are charged one static MediaPipe pass, as trained.

    python scripts/end_to_end_latency.py [--clips 100]      # run with the GPU and CPU otherwise idle

Writes results/end_to_end_latency.json. Neural parts run on the GPU (fp16, batch 1), MediaPipe on the CPU.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from asl_benchmark.models import frame_encoder as fe  # noqa: E402
from asl_benchmark.models import hands as hand_landmarks  # noqa: E402
from asl_benchmark.models import i3d as i3d_mod  # noqa: E402
from asl_benchmark.models import tgcn as tgcn_mod  # noqa: E402
from asl_benchmark.models import matched as m  # noqa: E402
from asl_benchmark.models import ladder  # noqa: E402
from scripts.features import per_frame  # noqa: E402

T = fe.T
# pipeline -> (rung keys it applies to, stages it pays for)
PIPELINES = {
    "skeleton_static": (["S1", "S1x", "C5", "X3"], ["decode", "mp_static", "skeleton_head"]),
    "skeleton_body": (["X2"], ["decode", "mp_static", "mp_pose", "skeleton_head"]),
    "skeleton_tracked": (["C1"], ["mp_tracked", "skeleton_head"]),
    "image_full": (["S2", "S3"], ["decode", "net_full"]),
    "image_patches": (["S4", "S5", "C3", "X1"], ["decode", "mp_static", "crop", "net_patches"]),
    "skeleton_image": (["C2a", "C2b"], ["decode", "mp_static", "render", "net_skeleton_image"]),
    "fusion": (["C4"], ["decode", "mp_static", "crop", "features", "net_fusion"]),
    "i3d_full": (["I3D"], ["decode", "net_i3d"]),
    "tgcn_static": (["TGCN"], ["decode", "mp_static", "tgcn_static"]),
    "tgcn_tracked": ([], ["mp_tracked", "tgcn_tracked"]),
}


def sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize()


def clock(dev, fn):
    """(result, milliseconds) with the GPU drained before and after."""
    sync(dev)
    t0 = time.perf_counter()
    out = fn()
    sync(dev)
    return out, (time.perf_counter() - t0) * 1000


def decode_letterbox(path):
    """16 evenly spaced frames, letterboxed to 224x224 RGB. Frame count from the header (no keypoint file exists)."""
    import cv2

    cap = cv2.VideoCapture(str(path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    frames, _, _ = fe._decode(path, np.linspace(0, n - 1, T).round().astype(np.int64))
    blank = np.zeros((m.S, m.S, 3), np.uint8)
    return np.stack([blank if f is None else m.letterbox(f) for f in frames])


def tracked_keypoints(hands, path):
    """scripts/extract.py on one video: MediaPipe in tracking mode over every frame at the original resolution."""
    import cv2

    cap, rows = cv2.VideoCapture(str(path)), []
    while True:
        ok, img = cap.read()
        if not ok:
            break
        res = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        row = np.full((42, 3), np.nan, np.float32)
        if res.multi_hand_landmarks:
            for lm, hd in zip(res.multi_hand_landmarks, res.multi_handedness):
                start = 21 if hd.classification[0].label == "Left" else 0
                row[start : start + 21] = [[p.x, p.y, p.z] for p in lm.landmark]
        rows.append(row)
    cap.release()
    return np.stack(rows)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--clips", type=int, default=100, help="test clips to time (the whole test set by default)")
    ap.add_argument("--out", default="results/end_to_end_latency.json")
    args = ap.parse_args(argv)
    import mediapipe as mp

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m._init_mediapipe()                                         # static_image_mode=True, as trained
    ladder._init_pose()
    tracked = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

    skeleton_head = fe.head(132).to(dev).eval()
    nets = {  # name -> (model, streams)
        "net_full": fe.FrameModel(1, 0), "net_patches": fe.FrameModel(2, 2),
        "net_skeleton_image": fe.FrameModel(1, 0), "net_fusion": fe.FrameModel(2, 2 + 132),
    }
    nets = {k: v.to(dev).eval() for k, v in nets.items()}
    i3d_model = i3d_mod.I3D(pretrained=False).to(dev).eval()
    tgcn_model = tgcn_mod.PoseTGCN(tgcn_mod.T * 6, 100, blocks=4).to(dev).eval()   # latency is depth-insensitive here

    def run_i3d(frames):
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16, enabled=dev.type == "cuda"):
            return i3d_model(i3d_mod.to_input(frames[None]).to(dev)).float().cpu()

    def run_tgcn(kp):
        with torch.inference_mode():
            x = torch.from_numpy(tgcn_mod.node_features(kp, "norm")[None]).to(dev)
            return tgcn_model(x).cpu()


    def run_net(name, images, flags):
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16, enabled=dev.type == "cuda"):
            return nets[name](fe.to_tensor(images).to(dev), torch.from_numpy(flags).to(dev)).float().cpu()

    def run_skeleton_head(kp):
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16, enabled=dev.type == "cuda"):
            feat = per_frame(hand_landmarks.hands(kp), location=True)
            feat = feat[np.linspace(0, len(feat) - 1, T).round().astype(int)].astype(np.float32)
            return skeleton_head(torch.from_numpy(feat)[None].to(dev)).float().cpu()

    clips = [s for s in fe._samples() if s.split == "test"][: args.clips]
    stages: list[dict] = []
    for i, s in enumerate(clips):
        path = s.video_path
        st = {}
        frames, st["decode"] = clock(dev, lambda: decode_letterbox(path))
        kp, st["mp_static"] = clock(dev, lambda: m.extract_clip(frames))
        kp_tracked, st["mp_tracked"] = clock(dev, lambda: tracked_keypoints(tracked, path))
        (patches, missing), st["crop"] = clock(dev, lambda: m.make_patches(frames[None], kp[None], borrow=True))
        skel_img, st["render"] = clock(dev, lambda: m.render_skeleton(kp[None]))
        _, st["skeleton_head"] = clock(dev, lambda: run_skeleton_head(kp))
        _, st["features_head"] = clock(dev, lambda: run_skeleton_head(kp_tracked))
        feats, st["features"] = clock(dev, lambda: per_frame(hand_landmarks.hands(kp), True).astype(np.float32)[None])
        _, st["net_full"] = clock(dev, lambda: run_net("net_full", frames[None, :, None], np.zeros((1, T, 0), np.float32)))
        _, st["net_patches"] = clock(dev, lambda: run_net("net_patches", patches, missing))
        _, st["net_skeleton_image"] = clock(dev, lambda: run_net("net_skeleton_image", skel_img, np.zeros((1, T, 0), np.float32)))
        fusion_flags = np.concatenate([missing, feats], -1).astype(np.float32)
        _, st["net_i3d"] = clock(dev, lambda: run_i3d(frames))
        _, st["tgcn_static"] = clock(dev, lambda: run_tgcn(kp))
        _, st["tgcn_tracked"] = clock(dev, lambda: run_tgcn(kp_tracked))
        _, st["net_fusion"] = clock(dev, lambda: run_net("net_fusion", patches, fusion_flags))
        _, st["mp_pose"] = clock(dev, lambda: ladder.extract_pose_clip(frames))
        if i > 0:                                                   # clip 0 is warm-up (cuDNN, allocator)
            stages.append(st)
        if i % 10 == 0:
            print(f"  clip {i + 1}/{len(clips)}", flush=True)

    # run_skeleton_head on tracked keypoints is the tracked pipeline's head; report it under that name
    for st in stages:
        st["skeleton_head_tracked"] = st["features_head"]
    result = {"n_clips": len(stages), "hardware": {
        "cpu": platform.processor() or "unknown", "gpu": torch.cuda.get_device_name(0) if dev.type == "cuda" else "none",
        "torch_threads": torch.get_num_threads()}, "pipelines": {}}
    for name, (keys, parts) in PIPELINES.items():
        parts = ["skeleton_head_tracked" if (name == "skeleton_tracked" and p == "skeleton_head") else p for p in parts]
        totals = np.array([sum(st[p] for p in parts) for st in stages])
        result["pipelines"][name] = {
            "rungs": keys, "ms_mean": round(float(totals.mean()), 1), "ms_median": round(float(np.median(totals)), 1),
            "ms_p95": round(float(np.percentile(totals, 95)), 1),
            "stages_mean": {p: round(float(np.mean([st[p] for st in stages])), 2) for p in parts},
        }
        r = result["pipelines"][name]
        print(f"{name:<18} {r['ms_mean']:>8.1f} ms (median {r['ms_median']:.1f}, p95 {r['ms_p95']:.1f})  "
              + "  ".join(f"{p} {v:.1f}" for p, v in r["stages_mean"].items()), flush=True)
    (ROOT / args.out).write_text(json.dumps(result, indent=1))
    print(f"saved {args.out}", flush=True)


if __name__ == "__main__":
    main()
