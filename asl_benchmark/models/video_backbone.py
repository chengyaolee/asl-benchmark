"""Pretrained video backbone on raw frames: R(2+1)D-18 with Kinetics-400 weights.

The appearance-side counterpart to the landmark models, on the same 1,013-clip
WLASL100 subset (748 / 165 / 100). It never reads landmark *values*; the
hand-crop variant only uses the keypoints to decide *where* to crop, which
separates "knowing where the hands are" from "the skeletal abstraction".

Rungs (3 seeds each, selected on validation, early stopping as in cnn.py):
    1. frozen backbone + linear head, full frame, 8 frames
    2. frozen backbone + linear head, full frame, 16 frames
    3. frozen backbone + linear head, hand crop, 16 frames
    4. fine-tuned end to end, full frame, 16 frames, augmentation
    5. fine-tuned end to end, hand crop, 16 frames, augmentation

Run from the repo root:
    python -m asl_benchmark.models.video_backbone            # all rungs
    python -m asl_benchmark.models.video_backbone --rungs 1 2 3 --seeds 0

Writes results/backbone_ablations.json (same row format as
results/ablations_xgb_tf_cnn.json, plus params and latency) and caches the
decoded clips in data/wlasl100/cache/.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from asl_benchmark.loader import load  # noqa: E402
from asl_benchmark.models import cnn as cnn_mod  # noqa: E402

SEEDS = (0, 1, 2)
SIZE = 112
MEAN = np.array([0.43216, 0.394666, 0.37645], np.float32)
STD = np.array([0.22803, 0.22145, 0.216989], np.float32)
CROP_PAD = 1.25          # hand box side = 1.25 x the larger side of the hand-union box
BATCH = 8                # fine-tuning batch (12 GB GPU, fp16)
FEAT_BATCH = 32
FT_EPOCHS, FT_PATIENCE = 30, 6
HEAD_EPOCHS, HEAD_PATIENCE = 300, 30
LR_BACKBONE, LR_HEAD, WD = 1e-4, 1e-3, 1e-2
SMOOTHING = 0.1
KP_DIR = ROOT / "data/wlasl100/keypoints"
CACHE = ROOT / "data/wlasl100/cache"

RUNGS = {
    1: dict(name="1. Frozen, full frame, 8 frames", frames=8, crop=False, finetune=False),
    2: dict(name="2. Frozen, full frame, 16 frames", frames=16, crop=False, finetune=False),
    3: dict(name="3. Frozen, hand crop, 16 frames", frames=16, crop=True, finetune=False),
    4: dict(name="4. Fine-tuned, full frame, 16 frames (+aug)", frames=16, crop=False, finetune=True),
    5: dict(name="5. Fine-tuned, hand crop, 16 frames (+aug)", frames=16, crop=True, finetune=True),
}


# --------------------------------------------------------------------------- data
def hand_box(keypoints: np.ndarray, width: int, height: int) -> tuple[int, int, int]:
    """Square crop (x0, y0, side) in pixels around both hands over the whole clip.

    Uses the 2nd-98th percentile of all detected keypoints so one stray
    detection cannot blow the box up. Falls back to the full frame.
    """
    xy = keypoints[..., :2].reshape(-1, 2)
    xy = xy[~np.isnan(xy).any(axis=1)]
    if len(xy) == 0:
        side = max(width, height)
        return (width - side) // 2, (height - side) // 2, side
    lo = np.percentile(xy, 2, axis=0) * (width, height)
    hi = np.percentile(xy, 98, axis=0) * (width, height)
    cx, cy = (lo + hi) / 2
    side = int(max(hi - lo) * CROP_PAD)
    side = max(side, int(0.25 * min(width, height)))   # never zoom in past a quarter frame
    side = min(side, max(width, height))
    return int(cx - side / 2), int(cy - side / 2), side


def _square(frame: np.ndarray, box: tuple[int, int, int]) -> np.ndarray:
    """Crop the square box (zero-padded where it leaves the frame) and resize to SIZE."""
    import cv2

    x0, y0, side = box
    h, w = frame.shape[:2]
    out = np.zeros((side, side, 3), frame.dtype)
    sx0, sy0 = max(0, x0), max(0, y0)
    sx1, sy1 = min(w, x0 + side), min(h, y0 + side)
    if sx1 > sx0 and sy1 > sy0:
        out[sy0 - y0 : sy1 - y0, sx0 - x0 : sx1 - x0] = frame[sy0:sy1, sx0:sx1]
    return cv2.resize(out, (SIZE, SIZE), interpolation=cv2.INTER_AREA)


def read_clip(job) -> np.ndarray:
    """(frames, SIZE, SIZE, 3) uint8 RGB: letterboxed full frame or hand crop."""
    import cv2

    video_path, frame_start, frame_end, kp_path, n_frames, crop = job
    cap = cv2.VideoCapture(str(video_path))
    try:
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if crop:
            box = hand_box(np.load(kp_path), w, h)
        else:
            side = max(w, h)
            box = ((w - side) // 2, (h - side) // 2, side)
        start, end = cnn_mod.frame_span(count, frame_start, frame_end)
        wanted = np.linspace(start, end - 1, n_frames).round().astype(np.int64)
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(start))
        pos, last, frames = start, None, []
        for index in wanted:
            while pos <= index:
                ok, frame = cap.read()
                pos += 1
                if ok:
                    last = frame
                else:
                    break
            if last is None:
                frames.append(np.zeros((SIZE, SIZE, 3), np.uint8))
            else:
                frames.append(cv2.cvtColor(_square(last, box), cv2.COLOR_BGR2RGB))
        return np.stack(frames)
    finally:
        cap.release()


def load_split_clips(n_frames: int, crop: bool):
    """Clips, labels and sample ids per split for every sample with keypoints. Cached."""
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"clips_{'crop' if crop else 'full'}_{n_frames}x{SIZE}.npz"
    samples = [s for s in load("wlasl100", ROOT / "data", download=False).samples
               if (KP_DIR / f"{s.sample_id}.npy").exists()]
    if path.exists():
        z = np.load(path)
        clips = z["clips"]
    else:
        jobs = [(s.video_path, s.frame_start, s.frame_end, KP_DIR / f"{s.sample_id}.npy", n_frames, crop)
                for s in samples]
        t0 = time.perf_counter()
        with Pool(6) as pool:
            clips = np.stack(pool.map(read_clip, jobs, chunksize=8))
        np.savez(path, clips=clips)
        print(f"decoded {len(clips)} clips -> {path.name} in {time.perf_counter() - t0:.0f}s", flush=True)
    out = {}
    for split in ("train", "val", "test"):
        idx = [i for i, s in enumerate(samples) if s.split == split]
        out[split] = (clips[idx], np.array([samples[i].label for i in idx], np.int64),
                      [samples[i].sample_id for i in idx])
    return out


def to_tensor(clips: np.ndarray) -> torch.Tensor:
    """(n, T, H, W, 3) uint8 or [0, 1] float -> normalized (n, 3, T, H, W) float32."""
    x = np.asarray(clips, np.float32)
    if x.max() > 1.5:
        x = x / 255.0
    x = (x - MEAN) / STD
    return torch.from_numpy(np.ascontiguousarray(x.transpose(0, 4, 1, 2, 3)))


# -------------------------------------------------------------------------- model
def backbone() -> nn.Module:
    from torchvision.models.video import R2Plus1D_18_Weights, r2plus1d_18

    model = r2plus1d_18(weights=R2Plus1D_18_Weights.KINETICS400_V1)
    model.fc = nn.Identity()          # 512-d pooled feature
    return model


class Classifier(nn.Module):
    def __init__(self, n_classes: int = 100, dropout: float = 0.5):
        super().__init__()
        self.backbone = backbone()
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(512, n_classes))

    def forward(self, x):
        return self.head(self.backbone(x))


def n_params(module: nn.Module, trainable_only: bool = False) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad or not trainable_only)


@torch.inference_mode()
def predict(model: nn.Module, clips: np.ndarray, dev, batch: int = FEAT_BATCH) -> np.ndarray:
    model.eval()
    rows = []
    for i in range(0, len(clips), batch):
        with torch.autocast("cuda", dtype=torch.float16, enabled=dev.type == "cuda"):
            rows.append(model(to_tensor(clips[i : i + batch]).to(dev)).float().cpu().numpy())
    return np.concatenate(rows)


def latency_ms(model: nn.Module, clips: np.ndarray, dev) -> float:
    """Batched inference time per clip on the test set (same convention as cnn.py)."""
    predict(model, clips[:FEAT_BATCH], dev)                 # warm-up
    if dev.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    predict(model, clips, dev)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000 / len(clips)


# ----------------------------------------------------------------------- training
def _seed(seed: int) -> np.random.Generator:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return np.random.default_rng(seed)


def fit_linear_head(feats, y, val_feats, val_y, seed: int, dev) -> nn.Module:
    """Linear head on frozen features, AdamW, early stopping on validation."""
    _seed(seed)
    mu, sd = feats.mean(0), feats.std(0) + 1e-6
    head = nn.Sequential(nn.Dropout(0.3), nn.Linear(feats.shape[1], 100)).to(dev)
    opt = torch.optim.AdamW(head.parameters(), lr=LR_HEAD, weight_decay=WD)
    x = torch.from_numpy((feats - mu) / sd).float().to(dev)
    vx = torch.from_numpy((val_feats - mu) / sd).float().to(dev)
    t, vy = torch.from_numpy(y).to(dev), val_y
    best, best_epoch, state = -1.0, 0, None
    for epoch in range(HEAD_EPOCHS):
        head.train()
        for idx in torch.randperm(len(t), device=dev).split(64):
            opt.zero_grad()
            nn.functional.cross_entropy(head(x[idx]), t[idx], label_smoothing=SMOOTHING).backward()
            opt.step()
        head.eval()
        with torch.no_grad():
            acc = float((head(vx).argmax(1).cpu().numpy() == vy).mean())
        if acc > best:
            best, best_epoch = acc, epoch
            state = {k: v.detach().clone() for k, v in head.state_dict().items()}
        elif epoch - best_epoch >= HEAD_PATIENCE:
            break
    head.load_state_dict(state)
    head.eval()
    norm = nn.Module()
    norm.mu, norm.sd = torch.from_numpy(mu).float().to(dev), torch.from_numpy(sd).float().to(dev)
    return head, norm


def fine_tune(train, val, seed: int, dev, augment: bool = True) -> nn.Module:
    rng = _seed(seed)
    model = Classifier().to(dev)
    opt = torch.optim.AdamW([
        {"params": model.backbone.parameters(), "lr": LR_BACKBONE},
        {"params": model.head.parameters(), "lr": LR_HEAD},
    ], weight_decay=WD)
    scaler = torch.amp.GradScaler(enabled=dev.type == "cuda")
    clips, y = train[0], torch.from_numpy(train[1])
    best, best_epoch, state = -1.0, 0, None
    for epoch in range(FT_EPOCHS):
        model.train()
        for idx in torch.randperm(len(y)).split(BATCH):
            batch = clips[idx.numpy()]
            if augment:
                batch = np.stack([cnn_mod._augment_clip(c, rng) for c in batch])
            opt.zero_grad()
            with torch.autocast("cuda", dtype=torch.float16, enabled=dev.type == "cuda"):
                loss = nn.functional.cross_entropy(model(to_tensor(batch).to(dev)), y[idx].to(dev),
                                                   label_smoothing=SMOOTHING)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
        acc = float((predict(model, val[0], dev).argmax(1) == val[1]).mean())
        print(f"    epoch {epoch:2d}  loss {loss.item():.3f}  val {acc * 100:5.1f}%", flush=True)
        if acc > best:
            best, best_epoch = acc, epoch
            state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        elif epoch - best_epoch >= FT_PATIENCE:
            break
    model.load_state_dict(state)
    return model


def run_rung(number: int, seeds, dev) -> dict:
    cfg = RUNGS[number]
    data = load_split_clips(cfg["frames"], cfg["crop"])
    (tr_x, tr_y, _), (va_x, va_y, _), (te_x, te_y, te_ids) = data["train"], data["val"], data["test"]
    vals, tests, top5s, secs, lats, preds = [], [], [], [], [], []
    params = trainable = 0
    for seed in seeds:
        t0 = time.perf_counter()
        if cfg["finetune"]:
            model = fine_tune((tr_x, tr_y), (va_x, va_y), seed, dev)
            fit_s = time.perf_counter() - t0
            params = trainable = n_params(model)
            va_logits, te_logits = predict(model, va_x, dev), predict(model, te_x, dev)
            lats.append(latency_ms(model, te_x, dev))
        else:
            net = backbone().to(dev)
            feats = {k: predict(net, x, dev) for k, x in (("tr", tr_x), ("va", va_x), ("te", te_x))}
            head, norm = fit_linear_head(feats["tr"], tr_y, feats["va"], va_y, seed, dev)
            fit_s = time.perf_counter() - t0
            params, trainable = n_params(net) + n_params(head), n_params(head)

            def logits(f):
                with torch.no_grad():
                    x = (torch.from_numpy(f).float().to(dev) - norm.mu) / norm.sd
                    return head(x).cpu().numpy()

            va_logits, te_logits = logits(feats["va"]), logits(feats["te"])
            lats.append(latency_ms(net, te_x, dev))     # head cost is negligible
        vals.append(float((va_logits.argmax(1) == va_y).mean()))
        tests.append(float((te_logits.argmax(1) == te_y).mean()))
        top5s.append(cnn_mod._topk_accuracy(te_logits, te_y))
        secs.append(fit_s)
        preds.append(te_logits.argmax(1).tolist())
        print(f"  seed {seed}: val {vals[-1] * 100:.1f}  test {tests[-1] * 100:.1f}  ({fit_s:.0f}s)", flush=True)
        if dev.type == "cuda":
            torch.cuda.empty_cache()
    row = {
        "rung": cfg["name"],
        "val_top1": round(float(np.mean(vals)) * 100, 1),
        "test_top1": round(float(np.mean(tests)) * 100, 1),
        "test_std": round(float(np.std(tests)) * 100, 1),
        "test_top5": round(float(np.mean(top5s)) * 100, 1),
        "train_s": round(float(np.mean(secs)), 1),
        "latency_ms": round(float(np.mean(lats)), 3),
        "params": int(params),
        "trainable_params": int(trainable),
        "seeds": list(seeds),
        "n_train": len(tr_y), "n_val": len(va_y), "n_test": len(te_y),
        "test_ids": te_ids,
        "test_labels": te_y.tolist(),
        "test_preds": preds,
    }
    print(f"[R(2+1)D] {cfg['name']:<46} | Val: {row['val_top1']:5.1f}% | Test: {row['test_top1']:5.1f}% "
          f"(std {row['test_std']:4.1f}%) | Top-5: {row['test_top5']:5.1f}% | {row['train_s']:6.1f}s "
          f"| {row['latency_ms']:.2f} ms/clip", flush=True)
    return row


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--rungs", type=int, nargs="+", default=sorted(RUNGS))
    p.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    p.add_argument("--out", default="results/backbone_ablations.json")
    args = p.parse_args(argv)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {torch.cuda.get_device_name(0) if dev.type == 'cuda' else 'cpu'}", flush=True)
    out_path = ROOT / args.out
    rows = json.loads(out_path.read_text()) if out_path.exists() else []
    for number in args.rungs:
        row = run_rung(number, tuple(args.seeds), dev)
        rows = [r for r in rows if r["rung"] != row["rung"]] + [row]
        rows.sort(key=lambda r: r["rung"])
        out_path.write_text(json.dumps(rows, indent=1))   # save after every rung
    print(f"saved {out_path.relative_to(ROOT)}", flush=True)


if __name__ == "__main__":
    main()
