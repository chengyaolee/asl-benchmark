"""Learning curves for the matched comparison (UPDATED PLAN.md, extensions).

Train on a class-stratified fraction of the training clips (25 / 50 / 75 / 100%), evaluate on the same
val and test sets. The seed controls both the subset and the training run. One row per (model, fraction, seed)
-> results/learning_curves.json. Restart-safe: finished (model, fraction, seed) triples are skipped.

    python scripts/learning_curves.py                    # skeletons + frozen pixel models (minutes)
    python scripts/learning_curves.py --models S5        # a fine-tuned model (slow: ~17 min/seed at 100%)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from asl_benchmark.models import frame_encoder as fe  # noqa: E402
from asl_benchmark.models import matched as m  # noqa: E402

FRACTIONS = (0.25, 0.5, 0.75, 1.0)
DEFAULT_MODELS = ("S1", "C1", "S2", "C3")      # static skeleton, tracked skeleton, frozen full frame, frozen blanked patches


def stratified(labels: np.ndarray, frac: float, seed: int) -> np.ndarray:
    """Indices of a class-stratified subset: ceil(frac * n_c) clips per class, at least one."""
    rng = np.random.default_rng(1000 + seed)
    keep = []
    for c in np.unique(labels):
        idx = np.flatnonzero(labels == c)
        keep.extend(rng.permutation(idx)[: max(1, int(np.ceil(frac * len(idx))))])
    return np.sort(np.array(keep))


def run_skeleton_subset(cfg, frac, seed):
    import scripts.run_ablations as ablations

    if cfg["skeleton"] == "tracked":
        raw, labels, _ = ablations.load_wlasl()
    else:
        samples, frames = m.load_frames()
        kp = m.load_static_keypoints(samples, frames)
        raw = {k: [] for k in ("train", "val", "test")}
        labels = {k: [] for k in raw}
        for s, clip in zip(samples, kp):
            raw[s.split].append(clip)
            labels[s.split].append(s.label)
    keep = stratified(np.asarray(labels["train"]), frac, seed)
    raw["train"] = [raw["train"][i] for i in keep]
    labels["train"] = [labels["train"][i] for i in keep]
    ablations.SEEDS = (seed,)
    row = ablations.run_tf_rung(cfg["name"], raw, labels, location=True, augment=True, num_layers=1,
                                readout="mean", T=m.T)
    return row, len(keep)


def run_pixel_subset(cfg, frac, seed, dev, data):
    x, f, y = data["train"]
    keep = stratified(y, frac, seed)
    sub = dict(data)
    sub["train"] = (x[keep], f[keep], y[keep])
    return fe.run_rung(None, (seed,), dev, cfg=cfg, data=sub), len(keep)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS), choices=sorted(m.RUNGS))
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--fractions", type=float, nargs="+", default=list(FRACTIONS))
    p.add_argument("--out", default="results/learning_curves.json")
    args = p.parse_args(argv)
    torch.backends.cudnn.benchmark = True
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_path = ROOT / args.out
    rows = json.loads(out_path.read_text()) if out_path.exists() else []
    done = {(r["key"], r["fraction"], r["seed"]) for r in rows}
    for key in args.models:
        cfg = m.RUNGS[key]
        data = None if "skeleton" in cfg else m.inputs(cfg["kind"])
        for frac in args.fractions:
            for seed in args.seeds:
                if (key, frac, seed) in done:
                    continue
                if "skeleton" in cfg:
                    row, n = run_skeleton_subset(cfg, frac, seed)
                else:
                    row, n = run_pixel_subset(cfg, frac, seed, dev, data)
                rows.append({"key": key, "fraction": frac, "seed": seed, "n_train": n,
                             "val_top1": row["val_top1"], "test_top1": row["test_top1"],
                             "test_top5": row["test_top5"]})
                out_path.write_text(json.dumps(rows, indent=1))
                print(f"[curve] {key} frac {frac:.2f} seed {seed}  n_train {n}  val {row['val_top1']:.1f}  "
                      f"test {row['test_top1']:.1f}", flush=True)
    print(f"saved {out_path.relative_to(ROOT)}", flush=True)


if __name__ == "__main__":
    main()
