"""Dataset statistics and report figures for WLASL100.

Run from the repo root (after scripts/extract.py and the curve runs):
    python scripts/report_figures.py

Prints dataset statistics (full WLASL100 vs. the subset we have) and writes:
    results/dataset_stats.json
    results/fig_dataset.png   clips per sign (train) and clip lengths
    results/fig_curves.png    train/val accuracy per epoch (needs results/mlp_curve_*.json)
"""

import json, sys
from collections import Counter
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from asl_benchmark.loader import load

RES = ROOT / "results"; RES.mkdir(exist_ok=True)
ds = load("wlasl100", ROOT / "data", download=False)
meta = json.loads((ROOT / "data/wlasl100/WLASL_v0.3.json").read_text())
signer_of = {str(i["video_id"]): i["signer_id"] for e in meta for i in e["instances"]}
kp = ROOT / "data/wlasl100/keypoints"

full = list(ds.samples)
have = [s for s in full if (kp / f"{s.sample_id}.npy").exists()]
SPLITS = ("train", "val", "test")


def describe(samples):
    out = {"videos": len(samples)}
    for sp in SPLITS:
        out[f"{sp}_videos"] = sum(s.split == sp for s in samples)
    per = Counter(s.label for s in samples if s.split == "train")
    counts = np.array([per.get(c, 0) for c in range(100)])
    out["train_per_sign_min"] = int(counts.min())
    out["train_per_sign_mean"] = round(float(counts.mean()), 1)
    out["train_per_sign_max"] = int(counts.max())
    out["signs_with_no_train"] = int((counts == 0).sum())
    out["signs_in_test"] = len({s.label for s in samples if s.split == "test"})
    sig = {sp: {signer_of[s.sample_id] for s in samples if s.split == sp} for sp in SPLITS}
    out["signers"] = len(set().union(*sig.values()))
    for sp in SPLITS:
        out[f"{sp}_signers"] = len(sig[sp])
    out["test_signers_seen_in_train"] = len(sig["test"] & sig["train"])
    out["test_clips_with_seen_signer"] = sum(
        s.split == "test" and signer_of[s.sample_id] in sig["train"] for s in samples)
    return out, counts


stats_full, counts_full = describe(full)
stats_have, counts_have = describe(have)

# Keypoint statistics (only exist for the videos we have).
lengths, left, right, anyh, both = [], [], [], [], []
for s in have:
    a = np.load(kp / f"{s.sample_id}.npy").reshape(-1, 42, 3)
    if len(a) == 0:
        continue
    lengths.append(len(a))
    l = np.isfinite(a[:, :21, 0]).any(1); r = np.isfinite(a[:, 21:, 0]).any(1)
    left.append(l.mean()); right.append(r.mean()); anyh.append((l | r).mean()); both.append((l & r).mean())
lengths = np.array(lengths)
stats_have.update(
    frames_median=int(np.median(lengths)), frames_p10=int(np.percentile(lengths, 10)),
    frames_p90=int(np.percentile(lengths, 90)), frames_min=int(lengths.min()), frames_max=int(lengths.max()),
    seconds_median=round(float(np.median(lengths)) / 25, 2),
    left_hand_detected=round(float(np.mean(left)), 3), right_hand_detected=round(float(np.mean(right)), 3),
    any_hand_detected=round(float(np.mean(anyh)), 3), both_hands_detected=round(float(np.mean(both)), 3),
)

print(f"{'statistic':32s} {'full WLASL100':>14s} {'available':>10s}")
for k in stats_have:
    print(f"{k:32s} {str(stats_full.get(k, '')):>14s} {str(stats_have[k]):>10s}")
(RES / "dataset_stats.json").write_text(json.dumps({"full": stats_full, "available": stats_have}, indent=2))

# Figure: dataset.
fig, ax = plt.subplots(1, 2, figsize=(8, 2.8))
bins = np.arange(0, max(counts_full.max(), 1) + 2) - 0.5
ax[0].hist(counts_full, bins=bins, alpha=0.5, label="full WLASL100")
ax[0].hist(counts_have, bins=bins, alpha=0.8, label="available")
ax[0].set_xlabel("training clips per sign"); ax[0].set_ylabel("signs"); ax[0].legend(fontsize=8)
ax[1].hist(lengths, bins=30, color="C1")
ax[1].set_xlabel("clip length (frames, 25 fps)"); ax[1].set_ylabel("clips")
fig.tight_layout(); fig.savefig(RES / "fig_dataset.png", dpi=200)
print("saved results/fig_dataset.png")

# Figure: training curves.
runs = {"curve_raw_mlp": "MLP, rung 1 (raw)", "curve_aug_mlp": "MLP, rung 3 (+aug)",
        "curve_pool_mlp": "MLP, rung 4 (+pool)", "curve_pool_linear": "Linear, rung 4 (+pool)"}
found = {k: json.loads((RES / f"mlp_{k}.json").read_text()) for k in runs if (RES / f"mlp_{k}.json").exists()}
if found:
    fig, ax = plt.subplots(1, 2, figsize=(8, 2.8))
    for i, (k, r) in enumerate(found.items()):
        h = r["history"]
        e = [x["epoch"] for x in h]
        ax[0].plot(e, [x["train_acc"] for x in h], color=f"C{i}", lw=1, ls="--")
        ax[0].plot(e, [x["val_acc"] for x in h], color=f"C{i}", lw=1.2, label=runs[k])
        ax[1].plot(e, [x["val_loss"] for x in h], color=f"C{i}", lw=1.2, label=runs[k])
    ax[0].set_xlabel("epoch"); ax[0].set_ylabel("accuracy"); ax[0].set_ylim(0, 1)
    ax[0].set_title("train (dashed) vs. val (solid)", fontsize=9); ax[0].legend(fontsize=7)
    ax[1].set_xlabel("epoch"); ax[1].set_ylabel("val loss"); ax[1].set_title("validation loss", fontsize=9)
    fig.tight_layout(); fig.savefig(RES / "fig_curves.png", dpi=200)
    print("saved results/fig_curves.png")
else:
    print("no curve runs found; run the three --tag curve_* commands first")