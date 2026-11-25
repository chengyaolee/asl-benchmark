"""R(2+1)D-18 (Kinetics), fine-tuned on whole frames: 32 frames, 112x112, no MediaPipe.

Pure image model. The 32 frames are the ones the pose models use (evenly spaced over all frames of the full-resolution
video, frame_encoder.frame_indices); each whole frame is letterboxed straight to 112x112. The keypoint files supply only
the clip's frame count (and the same 1,013-clip subset); no keypoint value reaches the crop, the pixels or the network.

--view letterbox (default): black bars to a square. --view center: the central square of the frame (side = the shorter
image side), no bars, no detector; whatever lies outside it is lost.
--no-augment: the base rung P0 (letterbox) of the pixel ladder; everything else is unchanged.

Training is video_backbone.fine_tune, unchanged: batch 8, AdamW, up to 60 epochs, patience 10, cosine, lr 1e-4 (backbone)
and 1e-3 (head), best epoch on validation accuracy.

    python -m asl_benchmark.models.r2plus1d_32 --seeds 0     # -> results/r2plus1d_32x112.json
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
from asl_benchmark.models.cnn import _topk_accuracy

ROOT, CACHE = vb.ROOT, vb.CACHE
T, S = 32, vb.SIZE
EPOCHS, PATIENCE = 60, 10


def read_clip(job) -> np.ndarray:
    """(T, S, S, 3) uint8 RGB: whole frames with black bars (letterbox) or the central square (center)."""
    import cv2

    video_path, kp_path, view = job
    frames, w, h = fe._decode(video_path, fe.frame_indices(len(np.load(kp_path)), T))   # frame count only
    side = max(w, h) if view == "letterbox" else min(w, h)
    box = ((w - side) // 2, (h - side) // 2, side)
    blank = np.zeros((S, S, 3), np.uint8)
    return np.stack([blank if f is None else cv2.cvtColor(vb._square(f, box), cv2.COLOR_BGR2RGB) for f in frames])


def load(view: str = "letterbox"):
    samples = fe._samples()
    path = CACHE / f"clips_{'full' if view == 'letterbox' else view}_posefr_{T}x{S}.npy"
    if not path.exists():
        t0 = time.perf_counter()
        with Pool(6) as pool:
            clips = np.stack(pool.map(read_clip, [(s.video_path, vb.KP_DIR / f"{s.sample_id}.npy", view) for s in samples], chunksize=8))
        np.save(path, clips)
        print(f"decoded {len(samples)} clips -> {path.name} in {time.perf_counter() - t0:.0f}s", flush=True)
    return fe._by_split(samples, np.load(path))


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--view", choices=("letterbox", "center"), default="letterbox")
    p.add_argument("--no-augment", dest="augment", action="store_false")
    p.add_argument("--out", default=None, help="default: results/r2plus1d_32x112[_center][_noaug].json")
    args = p.parse_args(argv)
    tag = ("" if args.view == "letterbox" else f"_{args.view}") + ("" if args.augment else "_noaug")
    vb.FT_EPOCHS, vb.FT_PATIENCE, vb.COSINE = EPOCHS, PATIENCE, True
    torch.backends.cudnn.benchmark = True
    dev = torch.device("cuda")
    d = load(args.view)
    train, val, (te_x, te_y) = d["train"], d["val"], d["test"]
    print(f"train {train[0].shape} val {val[0].shape} test {te_x.shape} on {torch.cuda.get_device_name()}", flush=True)
    out = ROOT / (args.out or f"results/r2plus1d_32x112{tag}.json")
    rows = json.loads(out.read_text()) if out.exists() else []
    for seed in args.seeds:
        if any(r["seed"] == seed for r in rows):
            print(f"seed {seed} done, skipping", flush=True)
            continue
        t0 = time.perf_counter()
        model = vb.fine_tune(train, val, seed, dev, augment=args.augment)
        secs = time.perf_counter() - t0
        va, te = vb.predict(model, val[0], dev), vb.predict(model, te_x, dev)
        row = {"name": f"R(2+1)D fine-tuned, {'whole frame' if args.view == 'letterbox' else 'centre square crop'}, 32 frames, 112, no MediaPipe" + ("" if args.augment else ", no augmentation"),
               "view": args.view, "augment": args.augment, "seed": seed,
               "val_top1": round(float((va.argmax(1) == val[1]).mean()) * 100, 1),
               "test_top1": round(float((te.argmax(1) == te_y).mean()) * 100, 1),
               "test_top5": round(_topk_accuracy(te, te_y) * 100, 1),
               "train_s": round(secs, 1), "params": vb.n_params(model), "epochs_cap": EPOCHS, "patience": PATIENCE,
               "latency_ms": round(vb.latency_ms(model, te_x, dev), 2), "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 2**30, 1),
               "val_labels": val[1].tolist(), "val_preds": va.argmax(1).tolist(),
               "test_labels": te_y.tolist(), "test_preds": te.argmax(1).tolist()}
        rows = [r for r in rows if r["seed"] != seed] + [row]
        out.write_text(json.dumps(rows, indent=1))
        ckpt = ROOT / f"data/wlasl100/checkpoints/r2plus1d_32x112{tag}_seed{seed}.pt"      # gitignored, for inference
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), ckpt)
        print(f"seed {seed}: val {row['val_top1']}  test {row['test_top1']}  top5 {row['test_top5']}  ({secs:.0f}s)", flush=True)
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
