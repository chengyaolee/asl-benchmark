"""Transformer ladder in the order of the MLP and GRU/LSTM ladders, on the paper's protocol.

Tracked full-resolution MediaPipe Hands, T = 32 frames, 132 features, WLASL100 748 / 165 / 100, 3 seeds. Training is the
recipe of scripts/run_ablations.run_tf_rung (same seeding and random streams, so rungs T1, T2, T3a reproduce its rows).
The six rungs are those of the MLP ladder: T0 hand shape, T1 + location, T2 + augmentation, T3a mean readout (4a),
T3b 4-chunk readout (4b), T4 velocity (5). Rows T5 to T8 in the results file are earlier extra checks, not run here. Rung T4 (velocity) changes one thing in the rung selected on validation among T2, T3a, T3b (ties:
the earlier, smaller rung). Every row stores val and test predictions per seed.

    python scripts/run_transformer_ladder.py          # -> results/transformer_ladder.json (restart-safe)
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
sys.path.insert(0, str(ROOT))
from asl_benchmark.models import cnn as cnn_mod  # noqa: E402
from scripts.augment import augment as augment_clip  # noqa: E402
from scripts.features import per_frame  # noqa: E402
from scripts.run_ablations import AblationTransformer, load_wlasl  # noqa: E402

SEEDS = (0, 1, 2)
OUT = ROOT / "results/transformer_ladder.json"
BASE = dict(location=True, augment=True, layers=1, readout="last", velocity=False, pos=True, T=32)


class Model(AblationTransformer):
    """AblationTransformer plus a 4-chunk readout and an optional missing positional embedding."""

    def __init__(self, n, T, layers, readout, pos):
        super().__init__(n, 100, max_frames=T, d_model=128, nhead=4, num_layers=layers, dropout=0.3,
                         readout="last" if readout == "last" else "mean")
        self.chunks, self.use_pos = readout == "chunks4", pos
        if self.chunks:
            self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(4 * 128, 100))

    def forward(self, x):
        h = self.norm(self.proj(self.drop_in(x)))
        if self.use_pos:
            h = h + self.pos(torch.arange(x.shape[1], device=x.device))
        h = self.encoder(h)
        if self.chunks:
            return self.head(torch.cat([c.mean(1) for c in torch.tensor_split(h, 4, dim=1)], 1))
        return self.head(h[:, -1] if self.readout == "last" else h.mean(1))


def run(key, name, cfg, raw, labels, dev):
    T = cfg["T"]

    def feats(clip):
        f = per_frame(clip, location=cfg["location"])
        f = f[np.linspace(0, len(f) - 1, T).round().astype(int)].astype(np.float32)
        return np.concatenate([f, np.diff(f, axis=0, prepend=f[:1])], 1) if cfg["velocity"] else f

    tr = np.stack([feats(s) for s in raw["train"]])
    mu, sd = tr.reshape(-1, tr.shape[-1]).mean(0), tr.reshape(-1, tr.shape[-1]).std(0) + 1e-6
    ten = lambda a: torch.from_numpy(np.ascontiguousarray((a - mu) / sd, dtype=np.float32)).to(dev)
    va_x, te_x = (ten(np.stack([feats(s) for s in raw[k]])) for k in ("val", "test"))
    y = torch.from_numpy(np.asarray(labels["train"], np.int64)).to(dev)
    va_y, te_y = (np.asarray(labels[k], np.int64) for k in ("val", "test"))
    r = dict(val=[], test=[], top5=[], secs=[], shuffled=[], val_preds=[], test_preds=[])
    for seed in SEEDS:
        t0 = time.perf_counter()
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        rng = np.random.default_rng(seed)
        model = Model(tr.shape[-1], T, cfg["layers"], cfg["readout"], cfg["pos"]).to(dev)
        n_params = sum(p.numel() for p in model.parameters()) - (0 if cfg["pos"] else model.pos.weight.numel())
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
        best, best_epoch, state = -1.0, 0, None
        for epoch in range(200):
            x = ten(np.stack([feats(augment_clip(s, rng)) for s in raw["train"]])) if cfg["augment"] else ten(tr)
            model.train()
            for idx in torch.randperm(len(y), device=dev).split(64):
                opt.zero_grad()
                nn.functional.cross_entropy(model(x[idx]), y[idx], label_smoothing=0.1).backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            model.eval()
            with torch.inference_mode():
                acc = float((model(va_x).argmax(1).cpu().numpy() == va_y).mean())
            if acc > best:
                best, best_epoch, state = acc, epoch, {k: v.detach().clone() for k, v in model.state_dict().items()}
            elif epoch - best_epoch >= 40:
                break
        model.load_state_dict(state)
        model.eval()
        with torch.inference_mode():
            lv, lt = model(va_x).cpu().numpy(), model(te_x).cpu().numpy()
            shuf = [float((model(te_x[:, np.random.default_rng(1234 + k).permutation(T)]).argmax(1).cpu().numpy() == te_y).mean())
                    for k in range(3)]
        r["val"].append(best); r["test"].append(float((lt.argmax(1) == te_y).mean()))
        r["top5"].append(cnn_mod._topk_accuracy(lt, te_y)); r["secs"].append(time.perf_counter() - t0)
        r["shuffled"].append(float(np.mean(shuf)))
        r["val_preds"].append(lv.argmax(1).tolist()); r["test_preds"].append(lt.argmax(1).tolist())
    pct = lambda v: round(float(np.mean(v)) * 100, 1)
    row = dict(key=key, rung=name, config=cfg, params=n_params, val_top1=pct(r["val"]), val_std=round(float(np.std(r["val"])) * 100, 1),
               test_top1=pct(r["test"]), test_std=round(float(np.std(r["test"])) * 100, 1), test_top5=pct(r["top5"]),
               shuffled_test_top1=pct(r["shuffled"]), train_s=round(float(np.mean(r["secs"])), 1), seeds=list(SEEDS),
               val_by_seed=[round(v * 100, 1) for v in r["val"]], test_by_seed=[round(v * 100, 1) for v in r["test"]],
               val_correct=int(round(sum(r["val"]) * len(va_y))), val_labels=va_y.tolist(), test_labels=te_y.tolist(),
               val_preds=r["val_preds"], test_preds=r["test_preds"])
    print(f"[TF] {key:4s} {name:46s} val {row['val_top1']:5.1f}  test {row['test_top1']:5.1f} +- {row['test_std']:3.1f}  top5 {row['test_top5']:5.1f}"
          f"  shuffled {row['shuffled_test_top1']:5.1f}  {row['train_s']:5.1f}s  {n_params:,}", flush=True)
    return row


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raw, labels, _ = load_wlasl()
    rows = json.loads(OUT.read_text()) if OUT.exists() else []
    done = {r["key"]: r for r in rows}

    def go(key, name, **change):
        cfg = {**BASE, **change}
        if key in done:
            assert done[key]["config"] == cfg, f"{key} was run with another config"
        else:
            done[key] = run(key, name, cfg, raw, labels, dev)
            rows.append(done[key])
            OUT.write_text(json.dumps(rows, indent=1))
        return cfg

    go("T0", "Hand shape only, last token, no augmentation", location=False, augment=False)
    go("T1", "+ wrist location", augment=False)
    cands = {"T2": go("T2", "+ augmentation"),
             "T3a": go("T3a", "+ mean readout", readout="mean"),
             "T3b": go("T3b", "T2 + 4-chunk readout", readout="chunks4")}
    sel = max(cands, key=lambda k: (done[k]["val_correct"], -list(cands).index(k)))
    base = {k: v for k, v in cands[sel].items()}
    print(f"selected on validation: {sel}", flush=True)
    go("T4", f"{sel} + velocity", **{**base, "velocity": True})


if __name__ == "__main__":
    main()
