"""Per-frame image encoder + the landmark transformer head: a like-for-like pixel model.

The landmark transformer reads one 132-d landmark vector per frame. Here the
same transformer head (scripts/run_ablations.AblationTransformer: 1 layer,
d=128, 4 heads, mean readout) reads one ImageNet ResNet-18 feature per frame
instead, so the per-frame representation is the only thing that changes.

Per-frame inputs (16 frames, 112 x 112, the same 1,013-clip split):
    full   one letterboxed full frame                       -> 512-d
    hands  two patches tracked around each hand every frame  -> 2 x 512-d + 2 missing flags
The hand patches use the landmarks only to place the crop (like the R(2+1)D
hand-crop rung), but per frame and per hand rather than one box per clip.

Rungs (3 seeds, early stopping on validation):
    1. landmarks, 16 frames (control: same head and frame count, + augmentation)
    2. ResNet-18 frozen, full frame
    3. ResNet-18 frozen, hand patches
    4. ResNet-18 fine-tuned, full frame (+ augmentation)
    5. ResNet-18 fine-tuned, hand patches (+ augmentation)

    python -m asl_benchmark.models.frame_encoder               # all rungs
    python -m asl_benchmark.models.frame_encoder --rungs 3 --seeds 0

Writes results/frame_encoder_ablations.json.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from multiprocessing import Pool

import numpy as np
import torch
from torch import nn

from asl_benchmark.models import video_backbone as vb
from asl_benchmark.models.cnn import _topk_accuracy, frame_span

ROOT = vb.ROOT
sys.path.insert(0, str(ROOT))
from scripts.run_ablations import AblationTransformer, load_wlasl, run_tf_rung  # noqa: E402

T, SIZE = 16, vb.SIZE
MEAN = np.array([0.485, 0.456, 0.406], np.float32)   # ImageNet
STD = np.array([0.229, 0.224, 0.225], np.float32)
HAND_PAD, MIN_SIDE = 1.6, 32       # patch side = 1.6 x hand bbox side, at least 32 px
HEAD_EPOCHS, HEAD_PATIENCE = 200, 40                  # as the landmark transformer
FT_EPOCHS, FT_PATIENCE, FT_BATCH = 40, 10, 8
LR_ENC, LR_HEAD, WD, SMOOTHING = 1e-4, 1e-3, 1e-2, 0.1

RUNGS = {
    1: dict(name="1. Landmarks, 16 frames (control)", kind="landmarks"),
    2: dict(name="2. ResNet-18 frozen, full frame", kind="full", finetune=False),
    3: dict(name="3. ResNet-18 frozen, hand patches", kind="hands", finetune=False),
    4: dict(name="4. ResNet-18 fine-tuned, full frame (+aug)", kind="full", finetune=True),
    5: dict(name="5. ResNet-18 fine-tuned, hand patches (+aug)", kind="hands", finetune=True),
}


# --------------------------------------------------------------------- hand patches
def _hand_box(points: np.ndarray, width: int, height: int):
    """Square (x0, y0, side) in pixels around one hand's 21 normalized points."""
    lo = points.min(0) * (width, height)
    hi = points.max(0) * (width, height)
    cx, cy = (lo + hi) / 2
    side = max(int(max(hi - lo) * HAND_PAD), MIN_SIDE)
    return int(cx - side / 2), int(cy - side / 2), side


def frame_indices(n_keypoint_rows: int, t: int = T) -> np.ndarray:
    """The t frames every model sees for a clip, as rows of its MediaPipe keypoint file.

    Keypoints were extracted from every frame of the video (scripts/extract.py, TRIM = False), so
    row i is video frame i. The landmark loaders use the same rule on the same array, which makes
    "the skeleton of frame i" and "the pixels of frame i" the same sample. The header frame count
    over-reports the decodable frames by 1-3 on some clips, so it is never used here.
    """
    return np.linspace(0, n_keypoint_rows - 1, t).round().astype(np.int64)


def _decode(video_path, wanted: np.ndarray):
    """BGR frames at the sorted indices `wanted` (the last readable frame stands in past the end),
    plus the frame width and height."""
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    try:
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames, pos, last = [], 0, None
        for index in wanted:
            while pos <= index:
                ok, frame = cap.read()
                pos += 1
                if not ok:
                    break
                last = frame
            frames.append(last)
        return frames, w, h
    finally:
        cap.release()


def read_hands(job):
    """(T, 2, SIZE, SIZE, 3) uint8 patches (signer's left, right) and (T, 2) missing flags.

    A hand not detected in a sampled frame takes its box from the nearest frame
    where it was detected (flag 1); a hand never detected gives a black patch.
    """
    import cv2

    video_path, kp_path = job
    kp = np.load(kp_path)                                   # (frames, 42, 3)
    seen = ~np.isnan(kp[..., :2]).any(-1)                   # (frames, 42)
    detected = [np.flatnonzero(seen[:, h * 21 : (h + 1) * 21].all(1)) for h in (0, 1)]
    wanted = frame_indices(len(kp))
    frames, w, h = _decode(video_path, wanted)
    patches = np.zeros((T, 2, SIZE, SIZE, 3), np.uint8)
    missing = np.ones((T, 2), np.float32)
    for t, (row, frame) in enumerate(zip(wanted, frames)):
        if frame is None:
            continue
        for hand in (0, 1):
            rows = detected[hand]
            if len(rows) == 0:
                continue
            nearest = rows[np.abs(rows - row).argmin()]
            missing[t, hand] = float(nearest != row)
            box = _hand_box(kp[nearest, hand * 21 : (hand + 1) * 21, :2], w, h)
            patches[t, hand] = cv2.cvtColor(vb._square(frame, box), cv2.COLOR_BGR2RGB)
    return patches, missing


def read_full(job):
    """(T, SIZE, SIZE, 3) uint8 letterboxed full frames at the same indices as read_hands."""
    import cv2

    video_path, kp_path = job
    wanted = frame_indices(len(np.load(kp_path)))
    frames, w, h = _decode(video_path, wanted)
    side = max(w, h)
    box = ((w - side) // 2, (h - side) // 2, side)
    blank = np.zeros((SIZE, SIZE, 3), np.uint8)
    return np.stack([blank if f is None else cv2.cvtColor(vb._square(f, box), cv2.COLOR_BGR2RGB) for f in frames])


def _samples():
    """Clips that have a keypoint file, in one fixed order shared by every stream and split."""
    return [s for s in vb.load("wlasl100", ROOT / "data", download=False).samples
            if (vb.KP_DIR / f"{s.sample_id}.npy").exists()]


def _by_split(samples, *arrays):
    out = {}
    for name in ("train", "val", "test"):
        idx = [i for i, s in enumerate(samples) if s.split == name]
        out[name] = tuple(a[idx] for a in arrays) + (np.array([samples[i].label for i in idx], np.int64),)
    return out


def _cached(name: str, reader, samples, keys: tuple[str, ...]):
    """Decode every clip with `reader` (frames chosen by frame_indices) once, then reuse the npz."""
    path = vb.CACHE / f"{name}_kp{T}x{SIZE}.npz"
    if path.exists():
        z = np.load(path)
        return [z[k] for k in keys]
    jobs = [(s.video_path, vb.KP_DIR / f"{s.sample_id}.npy") for s in samples]
    t0 = time.perf_counter()
    with Pool(6) as pool:
        out = pool.map(reader, jobs, chunksize=8)
    arrays = [np.stack([o[i] for o in out]) if len(keys) > 1 else np.stack(out) for i in range(len(keys))]
    np.savez(path, **dict(zip(keys, arrays)))
    print(f"decoded {name} -> {path.name} in {time.perf_counter() - t0:.0f}s", flush=True)
    return arrays


def load_hands():
    """Per split: (patches (n, T, 2, S, S, 3), missing flags (n, T, 2), labels)."""
    samples = _samples()
    patches, missing = _cached("hands", read_hands, samples, ("patches", "missing"))
    return _by_split(samples, patches, missing)


def load_inputs(kind: str):
    """Per split: (images (n, T, S, H, W, 3) uint8, flags (n, T, F), labels). S = 1 or 2 streams."""
    if kind == "hands":
        return load_hands()
    samples = _samples()
    (frames,) = _cached("full", read_full, samples, ("frames",))
    data = _by_split(samples, frames)
    return {k: (x[:, :, None], np.zeros((len(x), T, 0), np.float32), y) for k, (x, y) in data.items()}


# ------------------------------------------------------------------------ model
def resnet() -> nn.Module:
    from torchvision.models import ResNet18_Weights, resnet18

    net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    net.fc = nn.Identity()
    return net


def to_tensor(images: np.ndarray) -> torch.Tensor:
    """(..., H, W, 3) uint8 or [0, 1] float -> ImageNet-normalized (..., 3, H, W)."""
    x = np.asarray(images, np.float32)
    if x.max() > 1.5:
        x = x / 255.0
    x = (x - MEAN) / STD
    return torch.from_numpy(np.ascontiguousarray(np.moveaxis(x, -1, -3)))


def head(n_features: int) -> nn.Module:
    """Exactly the selected landmark transformer: 1 layer, d=128, 4 heads, mean readout."""
    return AblationTransformer(n_features, 100, max_frames=T, d_model=128, nhead=4,
                               num_layers=1, dropout=0.3, readout="mean")


class FrameModel(nn.Module):
    def __init__(self, streams: int, n_flags: int):
        super().__init__()
        self.encoder = resnet()
        self.head = head(streams * 512 + n_flags)

    def forward(self, images, flags):
        b, t, s = images.shape[:3]
        feats = self.encoder(images.flatten(0, 2)).view(b, t, s * 512)
        return self.head(torch.cat([feats, flags], dim=-1))


@torch.inference_mode()
def encode(net, images: np.ndarray, dev, batch: int = 16) -> np.ndarray:
    """Frozen ResNet features, (n, T, S*512)."""
    net.eval()
    rows = []
    for i in range(0, len(images), batch):
        x = to_tensor(images[i : i + batch]).to(dev)
        b, t, s = x.shape[:3]
        with torch.autocast("cuda", dtype=torch.float16, enabled=dev.type == "cuda"):
            rows.append(net(x.flatten(0, 2)).float().view(b, t, s * 512).cpu().numpy())
    return np.concatenate(rows)


@torch.inference_mode()
def predict(model, images, flags, dev, batch: int = 16) -> np.ndarray:
    model.eval()
    rows = []
    for i in range(0, len(images), batch):
        with torch.autocast("cuda", dtype=torch.float16, enabled=dev.type == "cuda"):
            out = model(to_tensor(images[i : i + batch]).to(dev), torch.from_numpy(flags[i : i + batch]).to(dev))
        rows.append(out.float().cpu().numpy())
    return np.concatenate(rows)


# --------------------------------------------------------------------- training
def augment(images: np.ndarray, flags: np.ndarray, rng: np.random.Generator, *,
            crop: float = 0.1, rotate: float = 15.0, scale: float = 0.1, shift: float = 0.05, jitter: float = 0.0):
    """One shared temporal crop, then one random affine per stream (as the landmark augmentation).

    The defaults are the augmentation every earlier run used. `jitter` > 0 adds one brightness and contrast change per
    clip (used only by scripts/tune_picture.py)."""
    import cv2

    t = len(images)
    start = int(rng.integers(0, int(t * crop) + 1))
    end = t - int(rng.integers(0, int(t * crop) + 1))
    idx = np.linspace(start, end - 1, t).round().astype(np.int64)
    images, flags = images[idx].astype(np.float32) / 255.0, flags[idx]
    out = np.empty_like(images)
    size = images.shape[-2]                   # 112 for patches, 224 for the matched full frames
    for s in range(images.shape[1]):
        m = cv2.getRotationMatrix2D((size / 2, size / 2), float(rng.uniform(-rotate, rotate)),
                                    float(rng.uniform(1 - scale, 1 + scale)))
        m[:, 2] += rng.uniform(-shift, shift, size=2) * size
        for f in range(t):
            out[f, s] = cv2.warpAffine(images[f, s], m, (size, size), borderMode=cv2.BORDER_REFLECT_101)
    if jitter > 0:
        out = (out - 0.5) * float(rng.uniform(1 - jitter, 1 + jitter)) + 0.5 + float(rng.uniform(-jitter / 2, jitter / 2))
    out += rng.normal(0, 0.003, out.shape).astype(np.float32)
    return np.clip(out, 0, 1), flags


def _seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return np.random.default_rng(seed)


def train_head(x, y, vx, vy, seed, dev):
    """Transformer head on frozen features, same recipe as run_ablations.run_tf_rung."""
    _seed(seed)
    mu = x.reshape(-1, x.shape[-1]).mean(0)
    sd = x.reshape(-1, x.shape[-1]).std(0) + 1e-6
    tx = torch.from_numpy((x - mu) / sd).float().to(dev)
    vxt = torch.from_numpy((vx - mu) / sd).float().to(dev)
    ty = torch.from_numpy(y).to(dev)
    model = head(x.shape[-1]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=WD)
    best, best_epoch, state = -1.0, 0, None
    for epoch in range(HEAD_EPOCHS):
        model.train()
        for idx in torch.randperm(len(ty), device=dev).split(64):
            opt.zero_grad()
            nn.functional.cross_entropy(model(tx[idx]), ty[idx], label_smoothing=SMOOTHING).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        with torch.inference_mode():
            acc = float((model(vxt).argmax(1).cpu().numpy() == vy).mean())
        if acc > best:
            best, best_epoch = acc, epoch
            state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        elif epoch - best_epoch >= HEAD_PATIENCE:
            break
    model.load_state_dict(state)
    model.eval()

    def logits(f):
        with torch.inference_mode():
            return model(torch.from_numpy((f - mu) / sd).float().to(dev)).cpu().numpy()

    return model, logits


def fine_tune(train, val, seed, dev):
    rng = _seed(seed)
    (images, flags, y), (vimg, vflags, vy) = train, val
    model = FrameModel(images.shape[2], flags.shape[-1]).to(dev)
    opt = torch.optim.AdamW([{"params": model.encoder.parameters(), "lr": LR_ENC},
                             {"params": model.head.parameters(), "lr": LR_HEAD}], weight_decay=WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=FT_EPOCHS)
    scaler = torch.amp.GradScaler(enabled=dev.type == "cuda")
    ty = torch.from_numpy(y)
    best, best_epoch, state = -1.0, 0, None
    for epoch in range(FT_EPOCHS):
        model.train()
        for idx in torch.randperm(len(ty)).split(FT_BATCH):
            views = [augment(images[i], flags[i], rng) for i in idx.numpy()]
            x = to_tensor(np.stack([v[0] for v in views])).to(dev)
            f = torch.from_numpy(np.stack([v[1] for v in views])).to(dev)
            opt.zero_grad()
            with torch.autocast("cuda", dtype=torch.float16, enabled=dev.type == "cuda"):
                loss = nn.functional.cross_entropy(model(x, f), ty[idx].to(dev), label_smoothing=SMOOTHING)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
        sched.step()
        acc = float((predict(model, vimg, vflags, dev).argmax(1) == vy).mean())
        print(f"    epoch {epoch:2d}  loss {loss.item():.3f}  val {acc * 100:5.1f}%", flush=True)
        if acc > best:
            best, best_epoch = acc, epoch
            state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        elif epoch - best_epoch >= FT_PATIENCE:
            break
    model.load_state_dict(state)
    return model


def _latency(fn, dev) -> float:
    fn()                                       # warm-up
    if dev.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    n = fn()
    if dev.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000 / n


def run_rung(number, seeds, dev, cfg=None, data=None) -> dict:
    """`cfg` and `data` let scripts/matched.py reuse this loop with its own inputs."""
    cfg = cfg or RUNGS[number]
    if cfg["kind"] == "landmarks":
        import scripts.run_ablations as ablations

        raw, labels, _ = load_wlasl()
        ablations.SEEDS = tuple(seeds)           # run_tf_rung loops over the module's SEEDS
        row = run_tf_rung(cfg["name"], raw, labels, location=True, augment=True, num_layers=1,
                          readout="mean", T=T)
        row["params"] = sum(p.numel() for p in head(132).parameters())
        return row
    data = data or load_inputs(cfg["kind"])
    (tr_x, tr_f, tr_y), (va_x, va_f, va_y), (te_x, te_f, te_y) = data["train"], data["val"], data["test"]
    vals, tests, top5s, secs, lats, preds = [], [], [], [], [], []
    params = 0
    for seed in seeds:
        t0 = time.perf_counter()
        if cfg["finetune"]:
            model = fine_tune((tr_x, tr_f, tr_y), (va_x, va_f, va_y), seed, dev)
            fit_s = time.perf_counter() - t0
            va_logits, te_logits = predict(model, va_x, va_f, dev), predict(model, te_x, te_f, dev)
            lats.append(_latency(lambda: (predict(model, te_x, te_f, dev), len(te_x))[1], dev))
            params = sum(p.numel() for p in model.parameters())
        else:
            net = resnet().to(dev)
            feats = {k: np.concatenate([encode(net, x, dev), f], -1)
                     for k, x, f in (("tr", tr_x, tr_f), ("va", va_x, va_f), ("te", te_x, te_f))}
            hd, logits = train_head(feats["tr"], tr_y, feats["va"], va_y, seed, dev)
            fit_s = time.perf_counter() - t0
            va_logits, te_logits = logits(feats["va"]), logits(feats["te"])
            lats.append(_latency(lambda: (logits(np.concatenate([encode(net, te_x, dev), te_f], -1)), len(te_x))[1], dev))
            params = sum(p.numel() for p in net.parameters()) + sum(p.numel() for p in hd.parameters())
        vals.append(float((va_logits.argmax(1) == va_y).mean()))
        tests.append(float((te_logits.argmax(1) == te_y).mean()))
        top5s.append(_topk_accuracy(te_logits, te_y))
        secs.append(fit_s)
        preds.append(te_logits.argmax(1).tolist())
        print(f"  seed {seed}: val {vals[-1] * 100:.1f}  test {tests[-1] * 100:.1f}  ({fit_s:.0f}s)", flush=True)
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
        "seeds": list(seeds),
        "n_train": len(tr_y), "n_val": len(va_y), "n_test": len(te_y),
        "test_labels": te_y.tolist(),
        "test_preds": preds,
    }
    print(f"[ResNet] {cfg['name']:<46} | Val: {row['val_top1']:5.1f}% | Test: {row['test_top1']:5.1f}% "
          f"(std {row['test_std']:4.1f}%) | Top-5: {row['test_top5']:5.1f}% | {row['train_s']:6.1f}s "
          f"| {row['latency_ms']:.2f} ms/clip", flush=True)
    return row


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--rungs", type=int, nargs="+", default=sorted(RUNGS))
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--out", default="results/frame_encoder_ablations.json")
    args = p.parse_args(argv)
    torch.backends.cudnn.benchmark = True
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_path = ROOT / args.out
    rows = json.loads(out_path.read_text()) if out_path.exists() else []
    for number in args.rungs:
        row = run_rung(number, tuple(args.seeds), dev)
        rows = [r for r in rows if r["rung"] != row["rung"]] + [row]
        rows.sort(key=lambda r: r["rung"])
        out_path.write_text(json.dumps(rows, indent=1))
    print(f"saved {out_path.relative_to(ROOT)}", flush=True)


if __name__ == "__main__":
    main()
