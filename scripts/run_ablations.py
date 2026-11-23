"""Run the full ablation ladders for XGBoost, Transformer, and Appearance CNN.

Evaluated over seeds 0, 1, and 2 on the WLASL100 split (748 / 165 / 100 clips).
Writes results/ablations_xgb_tf_cnn.json and logs/ablations.log.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from asl_benchmark.loader import load  # noqa: E402
from asl_benchmark.models import cnn as cnn_mod  # noqa: E402
from asl_benchmark.models import hands as hand_landmarks  # noqa: E402
from asl_benchmark.models import xgboost as xgb_mod  # noqa: E402
from scripts.augment import augment as augment_clip  # noqa: E402
from scripts.features import per_frame, prep  # noqa: E402

SEEDS = (0, 1, 2)


def load_wlasl():
    ds = load("wlasl100", ROOT / "data", download=False)
    kp_dir = ROOT / "data/wlasl100/keypoints"
    raw = {"train": [], "val": [], "test": []}
    labels = {"train": [], "val": [], "test": []}
    samples = {"train": [], "val": [], "test": []}
    for s in ds.samples:
        f = kp_dir / f"{s.sample_id}.npy"
        if f.exists():
            a = np.load(f)
            if len(a) > 0:
                raw[s.split].append(a)
                labels[s.split].append(s.label)
                samples[s.split].append(s)
    return raw, labels, samples


def run_xgb_rung(name, raw, labels, *, location: bool, augment: bool, pool: str, velocity: bool):
    t0 = time.perf_counter()
    fn = lambda seq: prep(hand_landmarks.hands(seq), T=32, location=location, pool=pool, velocity=velocity).astype(np.float32)
    clean_tr = np.stack([fn(s) for s in raw["train"]])
    mu = clean_tr.mean(0)
    sd = clean_tr.std(0) + 1e-6
    tr_std = (clean_tr - mu) / sd
    va_x = (np.stack([fn(s) for s in raw["val"]]) - mu) / sd
    te_x = (np.stack([fn(s) for s in raw["test"]]) - mu) / sd
    tr_y = np.asarray(labels["train"], np.int64)
    va_y = np.asarray(labels["val"], np.int64)
    te_y = np.asarray(labels["test"], np.int64)

    val_accs, test_accs, top5_accs = [], [], []
    for seed in SEEDS:
        rows = [tr_std]
        ys = [tr_y]
        if augment:
            rng = np.random.default_rng(seed)
            for _ in range(xgb_mod.N_AUGMENTS):
                viewed = [(fn(augment_clip(hand_landmarks.hands(s), rng)) - mu) / sd for s in raw["train"]]
                rows.append(np.stack(viewed))
                ys.append(tr_y)
        mat = np.concatenate(rows)
        lab = np.concatenate(ys)
        model, classes, _ = xgb_mod.train(mat, lab, seed, va_x, va_y)
        va_scores = xgb_mod._scores(model, classes, va_x)
        te_scores = xgb_mod._scores(model, classes, te_x)
        val_accs.append(xgb_mod._hits(va_scores, classes, va_y))
        test_accs.append(xgb_mod._hits(te_scores, classes, te_y))
        top5_accs.append(xgb_mod._topk_accuracy(te_scores, te_y))

    elapsed = (time.perf_counter() - t0) / len(SEEDS)
    res = {
        "rung": name,
        "val_top1": round(float(np.mean(val_accs)) * 100, 1),
        "test_top1": round(float(np.mean(test_accs)) * 100, 1),
        "test_std": round(float(np.std(test_accs)) * 100, 1),
        "test_top5": round(float(np.mean(top5_accs)) * 100, 1),
        "train_s": round(elapsed, 1),
    }
    print(
        f"[XGB] {name:<42} | Val: {res['val_top1']:5.1f}% | Test: {res['test_top1']:5.1f}% "
        f"(std {res['test_std']:4.1f}%) | Top-5: {res['test_top5']:5.1f}% | {res['train_s']:5.1f}s",
        flush=True,
    )
    return res


class AblationTransformer(nn.Module):
    def __init__(self, n_features: int, n_classes: int = 100, max_frames: int = 32, d_model: int = 128, nhead: int = 4, num_layers: int = 2, dropout: float = 0.3, readout: str = "mean"):
        super().__init__()
        self.readout = readout
        self.drop_in = nn.Dropout(dropout)
        self.proj = nn.Linear(n_features, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.pos = nn.Embedding(max_frames, d_model)
        nn.init.normal_(self.pos.weight, mean=0.0, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(d_model, n_classes))

    def forward(self, x):
        steps = torch.arange(x.shape[1], device=x.device)
        h = self.norm(self.proj(self.drop_in(x))) + self.pos(steps)
        hidden = self.encoder(h)
        if self.readout == "last":
            pooled = hidden[:, -1, :]
        else:
            pooled = hidden.mean(dim=1)
        return self.head(pooled)


def run_tf_rung(name, raw, labels, *, location: bool, augment: bool, num_layers: int, readout: str):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    T = 32

    def seq_fn(clip):
        feat = per_frame(hand_landmarks.hands(clip), location=location)
        idx = np.linspace(0, len(feat) - 1, T).round().astype(int)
        return feat[idx].astype(np.float32)

    clean_tr = np.stack([seq_fn(s) for s in raw["train"]])
    mu = clean_tr.reshape(-1, clean_tr.shape[-1]).mean(0)
    sd = clean_tr.reshape(-1, clean_tr.shape[-1]).std(0) + 1e-6
    tr_std = (clean_tr - mu) / sd
    va_x = torch.from_numpy(np.ascontiguousarray((np.stack([seq_fn(s) for s in raw["val"]]) - mu) / sd, dtype=np.float32)).to(dev)
    te_x = torch.from_numpy(np.ascontiguousarray((np.stack([seq_fn(s) for s in raw["test"]]) - mu) / sd, dtype=np.float32)).to(dev)
    tr_y = torch.from_numpy(np.asarray(labels["train"], np.int64)).to(dev)
    va_y = np.asarray(labels["val"], np.int64)
    te_y = np.asarray(labels["test"], np.int64)

    val_accs, test_accs, top5_accs, times, preds = [], [], [], [], []
    for seed in SEEDS:
        t0 = time.perf_counter()
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        rng = np.random.default_rng(seed)
        model = AblationTransformer(
            clean_tr.shape[-1],
            100,
            max_frames=T,
            d_model=128,
            nhead=4,
            num_layers=num_layers,
            dropout=0.3,
            readout=readout,
        ).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
        best_val = -1.0
        best_epoch = 0
        best_state = None
        for epoch in range(200):
            if augment:
                views = [(seq_fn(augment_clip(hand_landmarks.hands(s), rng)) - mu) / sd for s in raw["train"]]
                x = torch.from_numpy(np.ascontiguousarray(np.stack(views), dtype=np.float32)).to(dev)
            else:
                x = torch.from_numpy(np.ascontiguousarray(tr_std, dtype=np.float32)).to(dev)
            model.train()
            for idx in torch.randperm(len(tr_y), device=dev).split(64):
                opt.zero_grad()
                loss = nn.functional.cross_entropy(model(x[idx]), tr_y[idx], label_smoothing=0.1)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            model.eval()
            with torch.inference_mode():
                va_logits = model(va_x).cpu().numpy()
            val_acc = float((va_logits.argmax(1) == va_y).mean())
            if val_acc > best_val:
                best_val = val_acc
                best_epoch = epoch
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            elif epoch - best_epoch >= 40:
                break
        model.load_state_dict(best_state)
        model.eval()
        with torch.inference_mode():
            te_logits = model(te_x).cpu().numpy()
        val_accs.append(best_val)
        test_accs.append(float((te_logits.argmax(1) == te_y).mean()))
        top5_accs.append(cnn_mod._topk_accuracy(te_logits, te_y))
        times.append(time.perf_counter() - t0)
        preds.append(te_logits.argmax(1).tolist())

    res = {
        "rung": name,
        "val_top1": round(float(np.mean(val_accs)) * 100, 1),
        "test_top1": round(float(np.mean(test_accs)) * 100, 1),
        "test_std": round(float(np.std(test_accs)) * 100, 1),
        "test_top5": round(float(np.mean(top5_accs)) * 100, 1),
        "train_s": round(float(np.mean(times)), 1),
        "test_labels": te_y.tolist(),
        "test_preds": preds,          # one list per seed, for error analysis
    }
    print(
        f"[TF ] {name:<42} | Val: {res['val_top1']:5.1f}% | Test: {res['test_top1']:5.1f}% "
        f"(std {res['test_std']:4.1f}%) | Top-5: {res['test_top5']:5.1f}% | {res['train_s']:5.1f}s",
        flush=True,
    )
    return res


class AblationCNN(nn.Module):
    def __init__(self, n_classes: int = 100, use_bn: bool = True, pool: str = "mean", dropout: float = 0.3):
        super().__init__()
        self.pool = pool
        layers = []
        in_c = 3
        for out_c, do_pool in ((32, True), (64, True), (128, False)):
            layers.append(nn.Conv2d(in_c, out_c, 3, padding=1))
            if use_bn:
                layers.append(nn.BatchNorm2d(out_c))
            layers.append(nn.ReLU())
            if do_pool:
                layers.append(nn.MaxPool2d(2))
            in_c = out_c
        layers.append(nn.AdaptiveAvgPool2d(1))
        self.conv = nn.Sequential(*layers)
        head_in = 128 * 2 if pool == "meanstd" else 128
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(head_in, n_classes))

    def forward(self, x):
        b, t, c, h, w = x.shape
        feat = self.conv(x.reshape(b * t, c, h, w)).flatten(1).view(b, t, -1)
        if self.pool == "meanstd":
            z = torch.cat([feat.mean(1), feat.std(1)], dim=1)
        else:
            z = feat.mean(1)
        return self.head(z)


def run_cnn_rung(name, clips_dict, labels, *, use_bn: bool, augment: bool, pool: str):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    va_x = cnn_mod._batch(clips_dict["val"])
    te_x = cnn_mod._batch(clips_dict["test"])
    tr_y = torch.from_numpy(np.asarray(labels["train"], np.int64))
    va_y = np.asarray(labels["val"], np.int64)
    te_y = np.asarray(labels["test"], np.int64)

    val_accs, test_accs, top5_accs, times = [], [], [], []
    for seed in SEEDS:
        t0 = time.perf_counter()
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        rng = np.random.default_rng(seed)
        model = AblationCNN(100, use_bn=use_bn, pool=pool, dropout=0.3 if augment else 0.0).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2 if augment else 0.0)
        best_val = -1.0
        best_epoch = 0
        best_state = None
        for epoch in range(50):
            views = [cnn_mod._view(c, rng, augment) for c in clips_dict["train"]]
            frames = torch.from_numpy(np.ascontiguousarray(cnn_mod._batch(views)))
            cnn_mod._steps(model, opt, frames, tr_y, len(tr_y), dev, smoothing=0.1 if augment else 0.0)
            va_logits = cnn_mod._logits(model, va_x, dev)
            val_acc = float((va_logits.argmax(1) == va_y).mean())
            if val_acc > best_val:
                best_val = val_acc
                best_epoch = epoch
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            elif epoch - best_epoch >= 20:
                break
        model.load_state_dict(best_state)
        te_logits = cnn_mod._logits(model, te_x, dev)
        val_accs.append(best_val)
        test_accs.append(float((te_logits.argmax(1) == te_y).mean()))
        top5_accs.append(cnn_mod._topk_accuracy(te_logits, te_y))
        times.append(time.perf_counter() - t0)

    res = {
        "rung": name,
        "val_top1": round(float(np.mean(val_accs)) * 100, 1),
        "test_top1": round(float(np.mean(test_accs)) * 100, 1),
        "test_std": round(float(np.std(test_accs)) * 100, 1),
        "test_top5": round(float(np.mean(top5_accs)) * 100, 1),
        "train_s": round(float(np.mean(times)), 1),
    }
    print(
        f"[CNN] {name:<42} | Val: {res['val_top1']:5.1f}% | Test: {res['test_top1']:5.1f}% "
        f"(std {res['test_std']:4.1f}%) | Top-5: {res['test_top5']:5.1f}% | {res['train_s']:5.1f}s",
        flush=True,
    )
    return res


def main():
    raw, labels, samples = load_wlasl()
    out = {"xgboost": [], "transformer": [], "cnn": []}

    print("=== 1. XGBoost Ablation Ladder (matching MLP Table 4) ===", flush=True)
    xgb_rungs = [
        ("1. Hand shape, 32 frames flattened", dict(location=False, augment=False, pool="none", velocity=False)),
        ("2. + wrist location", dict(location=True, augment=False, pool="none", velocity=False)),
        ("3. + augmentation", dict(location=True, augment=True, pool="none", velocity=False)),
        ("4a. + temporal pooling (mean, std)", dict(location=True, augment=True, pool="meanstd", velocity=False)),
        ("4b. + temporal pooling (4 chunks)", dict(location=True, augment=True, pool="chunks4", velocity=False)),
        ("5. 4a + velocity", dict(location=True, augment=True, pool="meanstd", velocity=True)),
    ]
    for name, kw in xgb_rungs:
        out["xgboost"].append(run_xgb_rung(name, raw, labels, **kw))

    print("\n=== 2. Transformer Ablation Ladder (matching GRU/LSTM Table 5) ===", flush=True)
    tf_rungs = [
        ("0. Base: 1 layer, 128d, last token, no aug", dict(location=True, augment=False, num_layers=1, readout="last")),
        ("1. Base: 1 layer, 128d, last token (+aug)", dict(location=True, augment=True, num_layers=1, readout="last")),
        ("2. + mean over time instead of last token", dict(location=True, augment=True, num_layers=1, readout="mean")),
        ("3. + second layer (2 layers, mean over time)", dict(location=True, augment=True, num_layers=2, readout="mean")),
        ("4. Rung 3 without wrist location", dict(location=False, augment=True, num_layers=2, readout="mean")),
    ]
    for name, kw in tf_rungs:
        out["transformer"].append(run_tf_rung(name, raw, labels, **kw))

    print("\n=== 3. Appearance CNN Ablation Ladder ===", flush=True)
    print("Loading 8x64x64 video clips...", flush=True)
    clips8 = {sp: [cnn_mod.read_clip(s, n_frames=8, size=64) for s in samples[sp]] for sp in ("train", "val", "test")}
    cnn_rungs = [
        ("0. Base: 3-layer ConvNet (8x64x64), no BN, no aug", dict(use_bn=False, augment=False, pool="mean")),
        ("1. + BatchNorm, no augmentation", dict(use_bn=True, augment=False, pool="mean")),
        ("2. + augmentation (mean over 8 frames)", dict(use_bn=True, augment=True, pool="mean")),
        ("3. + temporal pooling (mean, std)", dict(use_bn=True, augment=True, pool="meanstd")),
    ]
    for name, kw in cnn_rungs:
        out["cnn"].append(run_cnn_rung(name, clips8, labels, **kw))

    print("Loading 16x64x64 video clips for Rung 4...", flush=True)
    clips16 = {sp: [cnn_mod.read_clip(s, n_frames=16, size=64) for s in samples[sp]] for sp in ("train", "val", "test")}
    out["cnn"].append(
        run_cnn_rung("4. + 16 frames (16x64x64, mean pool)", clips16, labels, use_bn=True, augment=True, pool="mean")
    )

    res_dir = ROOT / "results"
    res_dir.mkdir(exist_ok=True)
    (res_dir / "ablations_xgb_tf_cnn.json").write_text(json.dumps(out, indent=2))
    print("\nSaved results/ablations_xgb_tf_cnn.json", flush=True)


if __name__ == "__main__":
    main()
