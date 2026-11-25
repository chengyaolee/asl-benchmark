"""The best image model (X1: fine-tuned ResNet-18 + small transformer) on the *pose models' own input*.

The pose models read 32 frames per clip, chosen from every frame of the full-resolution video, with tracked MediaPipe
Hands keypoints (scripts/extract.py). Here the image model gets those same 32 frames from the same full-resolution video:

    M   with MediaPipe   : two 112x112 hand crops per frame, cut from the full-resolution frame at the tracked boxes,
                           + missing-hand flags + the 4 wrist-location numbers (exactly X1's inputs, other source)
    W   without MediaPipe: the whole frame, letterboxed from full resolution to 224x224; no crops, flags or location

Recipe = X1 / S3 (frame_encoder.fine_tune): same optimiser, schedule, augmentation, early stopping on validation. The
only code difference is that augmentation runs in threads with one random stream per clip and normalisation is done on
the GPU, so training times are not comparable with the earlier runs.

    python -m asl_benchmark.models.pose_input --prepare
    python -m asl_benchmark.models.pose_input --arms M W          # 3 seeds -> results/pose_input_image.json
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool

import numpy as np
import torch
from torch import nn

from asl_benchmark.models import frame_encoder as fe
from asl_benchmark.models import ladder
from asl_benchmark.models import matched as m
from asl_benchmark.models import video_backbone as vb
from asl_benchmark.models.cnn import _topk_accuracy

ROOT, CACHE = vb.ROOT, vb.CACHE
T, PATCH, S = 32, vb.SIZE, 224
fe.T = T                                   # the transformer head reads this many frames
ARMS = {
    "M": "ResNet-18 fine-tuned, hand crops from full-resolution tracked MediaPipe + wrist location, 32 frames (+aug)",
    "W": "ResNet-18 fine-tuned, whole 224 frame, no MediaPipe, same 32 frames (+aug)",
}


# ------------------------------------------------------------------------------ inputs
def read_m(job):
    """(T, 2, P, P, 3) crops, (T, 2) missing flags, (T, 4) wrist location. As fe.read_hands, 32 frames."""
    import cv2

    video_path, kp_path = job
    kp = np.load(kp_path)
    seen = ~np.isnan(kp[..., :2]).any(-1)
    found = [np.flatnonzero(seen[:, h * 21 : (h + 1) * 21].all(1)) for h in (0, 1)]
    wanted = fe.frame_indices(len(kp), T)
    frames, w, h = fe._decode(video_path, wanted)
    crops = np.zeros((T, 2, PATCH, PATCH, 3), np.uint8)
    missing = np.ones((T, 2), np.float32)
    for t, (row, frame) in enumerate(zip(wanted, frames)):
        if frame is None:
            continue
        for hand in (0, 1):
            if len(found[hand]) == 0:
                continue
            near = found[hand][np.abs(found[hand] - row).argmin()]
            missing[t, hand] = float(near != row)
            box = fe._hand_box(kp[near, hand * 21 : (hand + 1) * 21, :2], w, h)
            crops[t, hand] = cv2.cvtColor(vb._square(frame, box), cv2.COLOR_BGR2RGB)
    return crops, missing, ladder.wrist_location(kp[wanted][None])[0]


def read_w(job):
    video_path, kp_path = job
    frames, _, _ = fe._decode(video_path, fe.frame_indices(len(np.load(kp_path)), T))
    blank = np.zeros((S, S, 3), np.uint8)
    return np.stack([blank if f is None else m.letterbox(f) for f in frames])


def build(arm: str):
    """Decode every clip once into .npy caches (memory-mapped on reading)."""
    img_path, flag_path = CACHE / f"poseinput_{arm}_{T}.npy", CACHE / f"poseinput_{arm}_{T}_flags.npy"
    if img_path.exists() and flag_path.exists():
        return np.load(img_path, mmap_mode="r"), np.load(flag_path)
    samples = fe._samples()
    jobs = [(s.video_path, vb.KP_DIR / f"{s.sample_id}.npy") for s in samples]
    shape = (len(jobs), T, 2, PATCH, PATCH, 3) if arm == "M" else (len(jobs), T, 1, S, S, 3)
    img = np.lib.format.open_memmap(img_path, "w+", np.uint8, shape)
    flags = np.zeros((len(jobs), T, 6 if arm == "M" else 0), np.float32)
    t0 = time.perf_counter()
    with Pool(6) as pool:
        for i, out in enumerate(pool.imap(read_m if arm == "M" else read_w, jobs, chunksize=4)):
            if arm == "M":
                img[i] = out[0]
                flags[i] = np.concatenate([out[1], out[2]], -1)
            else:
                img[i, :, 0] = out
    img.flush()
    np.save(flag_path, flags)
    print(f"[{arm}] decoded {len(jobs)} clips in {time.perf_counter() - t0:.0f}s -> {img_path.name}", flush=True)
    return np.load(img_path, mmap_mode="r"), flags


def load_arm(arm: str):
    """Split indices, images (memmap), flags with location standardised on train, and the augmentation function."""
    img, flags = build(arm)
    samples = fe._samples()
    idx = {k: np.array([i for i, s in enumerate(samples) if s.split == k]) for k in ("train", "val", "test")}
    y = {k: np.array([samples[i].label for i in v], np.int64) for k, v in idx.items()}
    if arm == "W":
        return img, flags, idx, y, fe.augment
    mu = flags[idx["train"], :, 2:].reshape(-1, 4).mean(0)
    sd = flags[idx["train"], :, 2:].reshape(-1, 4).std(0) + 1e-6
    aug = ladder.make_augment(mu, sd)                 # augments and standardises the location itself
    std = flags.copy()
    std[:, :, 2:] = (flags[:, :, 2:] - mu) / sd      # evaluation inputs
    return img, (flags, std), idx, y, aug


# ---------------------------------------------------------------------------- training
def fine_tune(img, flags, idx, y, aug, seed, dev, epochs, workers=8):
    """frame_encoder.fine_tune with threaded augmentation and GPU normalisation."""
    import cv2

    cv2.setNumThreads(1)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    rng = np.random.default_rng(seed)
    raw_flags, eval_flags = flags if isinstance(flags, tuple) else (flags, flags)
    model = fe.FrameModel(img.shape[2], raw_flags.shape[-1]).to(dev)
    opt = torch.optim.AdamW([{"params": model.encoder.parameters(), "lr": fe.LR_ENC},
                             {"params": model.head.parameters(), "lr": fe.LR_HEAD}], weight_decay=fe.WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=fe.FT_EPOCHS)
    scaler = torch.amp.GradScaler(enabled=dev.type == "cuda")
    ty = torch.from_numpy(y["train"])
    mean = torch.tensor(fe.MEAN, device=dev).view(1, 1, 1, 3, 1, 1)
    std = torch.tensor(fe.STD, device=dev).view(1, 1, 1, 3, 1, 1)

    def view(args):
        pos, s = args
        i = idx["train"][pos]
        return aug(np.asarray(img[i]), raw_flags[i], np.random.default_rng(s))

    best, best_epoch, state = -1.0, 0, None
    with ThreadPoolExecutor(workers) as pool:
        for epoch in range(epochs):
            model.train()
            t0 = time.perf_counter()
            batches = list(torch.randperm(len(ty)).split(fe.FT_BATCH))
            queue = []
            for bi, b in enumerate(batches):
                seeds = rng.integers(2**31, size=len(b))
                queue.append((b, [pool.submit(view, a) for a in zip(b.numpy(), seeds)]))
                if len(queue) < 3 and bi + 1 < len(batches):     # keep two batches in flight
                    continue
                while queue and (len(queue) >= 3 or bi + 1 == len(batches)):
                    bb, futs = queue.pop(0)
                    out = [f.result() for f in futs]
                    x = torch.from_numpy(np.stack([o[0] for o in out])).to(dev, non_blocking=True)
                    x = (x.permute(0, 1, 2, 5, 3, 4) - mean) / std
                    f = torch.from_numpy(np.stack([o[1] for o in out])).to(dev)
                    opt.zero_grad()
                    with torch.autocast("cuda", dtype=torch.float16, enabled=dev.type == "cuda"):
                        loss = nn.functional.cross_entropy(model(x, f), ty[bb].to(dev), label_smoothing=fe.SMOOTHING)
                    scaler.scale(loss).backward()
                    scaler.unscale_(opt)
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(opt)
                    scaler.update()
            sched.step()
            acc = float((fe.predict(model, img[idx["val"]], eval_flags[idx["val"]], dev, batch=8).argmax(1) == y["val"]).mean())
            print(f"    epoch {epoch:2d}  loss {loss.item():.3f}  val {acc * 100:5.1f}%  ({time.perf_counter() - t0:.0f}s)", flush=True)
            if acc > best:
                best, best_epoch = acc, epoch
                state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            elif epoch - best_epoch >= fe.FT_PATIENCE:
                break
    model.load_state_dict(state)
    return model, best, best_epoch


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--prepare", action="store_true")
    p.add_argument("--arms", nargs="+", default=["M", "W"], choices=list(ARMS))
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--epochs", type=int, default=fe.FT_EPOCHS)
    p.add_argument("--out", default="results/pose_input_image.json")
    args = p.parse_args(argv)
    if args.prepare:
        for arm in args.arms:
            img, fl = build(arm)
            print(arm, img.shape, fl.shape, f"missing-hand rate {fl[..., :2].mean():.3f}" if arm == "M" else "", flush=True)
        return
    torch.backends.cudnn.benchmark = True
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = ROOT / args.out
    rows = json.loads(out.read_text()) if out.exists() else []
    for arm in args.arms:
        img, flags, idx, y, aug = load_arm(arm)
        eval_flags = flags[1] if isinstance(flags, tuple) else flags
        for seed in args.seeds:
            if any(r["arm"] == arm and r["seed"] == seed for r in rows):
                print(f"[{arm}] seed {seed} done, skipping", flush=True)
                continue
            t0 = time.perf_counter()
            model, val, ep = fine_tune(img, flags, idx, y, aug, seed, dev, args.epochs)
            secs = time.perf_counter() - t0
            te = fe.predict(model, img[idx["test"]], eval_flags[idx["test"]], dev, batch=8)
            rows.append({"arm": arm, "name": ARMS[arm], "seed": seed, "val_top1": round(val * 100, 1),
                         "test_top1": round(float((te.argmax(1) == y["test"]).mean()) * 100, 1),
                         "test_top5": round(_topk_accuracy(te, y["test"]) * 100, 1), "best_epoch": ep,
                         "train_s": round(secs, 1), "epochs_cap": args.epochs,
                         "params": sum(q.numel() for q in model.parameters()),
                         "test_labels": y["test"].tolist(), "test_preds": te.argmax(1).tolist()})
            out.write_text(json.dumps(rows, indent=1))
            r = rows[-1]
            print(f"[{arm}] seed {seed}: val {r['val_top1']} test {r['test_top1']} top5 {r['test_top5']} ({secs:.0f}s)", flush=True)
            del model
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
