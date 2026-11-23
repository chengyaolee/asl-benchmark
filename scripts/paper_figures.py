"""Figures for paper/paper.tex, written to paper/figs/.

Run from the repo root after scripts/run_ablations.py and
asl_benchmark.models.video_backbone:
    python scripts/paper_figures.py

    crops.png           full frame vs. hand crop inputs to the video backbone
    acc_vs_latency.png  test top-1 vs. inference latency, selected config per family
    confusions.png      most-confused sign pairs, best landmark vs. best pixel model

The transformer's test predictions are not in the ablation JSON from earlier
runs; the selected configuration is re-run once (3 seeds, ~2 min) and cached in
results/transformer_selected_preds.json.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from asl_benchmark.loader import load  # noqa: E402

RES, FIGS = ROOT / "results", ROOT / "paper/figs"
CACHE = ROOT / "data/wlasl100/cache"
FIGS.mkdir(parents=True, exist_ok=True)

# Reference categorical palette, slots 1-2 (validated pair): landmarks vs. pixels.
LANDMARK, PIXEL = "#2a78d6", "#eb6834"
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e4e3dd"
plt.rcParams.update({
    "font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

ds = load("wlasl100", ROOT / "data", download=False)
GLOSS = ds.glosses


def best_by_val(rows):
    return max(rows, key=lambda r: (r["val_top1"], -r.get("params", 0)))


# ---------------------------------------------------------------- crops figure
def crops(n_clips: int = 4, frames=(2, 6, 10, 14)):
    full = np.load(CACHE / "clips_full_16x112.npz")["clips"]
    crop = np.load(CACHE / "clips_crop_16x112.npz")["clips"]
    pick = np.linspace(0, len(full) - 1, n_clips + 2).astype(int)[1:-1]
    gap = np.full((112, 6, 3), 255, np.uint8)
    rows = []
    for i in pick:
        left = np.concatenate([full[i, f] for f in frames], axis=1)
        right = np.concatenate([crop[i, f] for f in frames], axis=1)
        rows.append(np.concatenate([left, gap, right], axis=1))
        rows.append(np.full((4, rows[-1].shape[1], 3), 255, np.uint8))
    img = np.concatenate(rows[:-1])
    fig, ax = plt.subplots(figsize=(3.3, 3.3 * img.shape[0] / img.shape[1] + 0.15))
    ax.imshow(img)
    ax.axis("off")
    w = img.shape[1]
    ax.text(w * 0.25, -6, "full frame (letterboxed)", ha="center", va="bottom", color=INK, fontsize=6)
    ax.text(w * 0.75, -6, "landmark-guided hand crop", ha="center", va="bottom", color=INK, fontsize=6)
    fig.savefig(FIGS / "crops.png")
    plt.close(fig)


# ------------------------------------------------------------ frontier figure
def _seed_latency(model: str) -> float:
    vals = [json.loads(p.read_text())["latency_ms_per_clip"] for p in RES.glob(f"{model}_seed*.json")]
    return float(np.mean(vals))


def frontier():
    abl = json.loads((RES / "ablations_xgb_tf_cnn.json").read_text())
    bb = json.loads((RES / "backbone_ablations.json").read_text())
    points = []  # (label, latency, top1, std, colour)
    for key, label in (("xgboost", "XGBoost"), ("transformer", "Transformer")):
        r = best_by_val(abl[key])
        points.append((label, _seed_latency(key), r["test_top1"], r["test_std"], LANDMARK))
    r = best_by_val(abl["cnn"])
    points.append(("CNN (scratch)", _seed_latency("cnn"), r["test_top1"], r["test_std"], PIXEL))
    frozen = best_by_val([r for r in bb if "Frozen" in r["rung"]])
    for r in [frozen] + [r for r in bb if "Fine" in r["rung"]]:
        short = ("R(2+1)D " + ("FT" if "Fine" in r["rung"] else "frozen")
                 + (", crop" if "crop" in r["rung"] else ", full") + f", {16 if '16' in r['rung'] else 8}f")
        points.append((short, r["latency_ms"], r["test_top1"], r["test_std"], PIXEL))
    fig, ax = plt.subplots(figsize=(3.3, 2.4))
    for label, lat, acc, sd, colour in points:
        ax.errorbar(lat, acc, yerr=sd, fmt="o", ms=5, color=colour, mec="white", mew=1,
                    ecolor=colour, elinewidth=1, capsize=0, zorder=3)
        ax.annotate(label, (lat, acc), xytext=(5, 2), textcoords="offset points", fontsize=6, color=INK)
    ax.set_xscale("log")
    ax.set_xlabel("inference latency (ms / clip, batched, log scale)")
    ax.set_ylabel("test top-1 (%)")
    ax.set_ylim(0, 80)
    ax.axhline(1, color=MUTED, lw=0.8, ls=":")
    ax.text(ax.get_xlim()[0] * 1.1, 2.5, "chance", fontsize=6, color=MUTED)
    handles = [plt.Line2D([], [], marker="o", ls="", color=c, label=l)
               for c, l in ((LANDMARK, "landmarks"), (PIXEL, "pixels"))]
    ax.legend(handles=handles, frameon=False, loc="upper right", fontsize=7)
    fig.savefig(FIGS / "acc_vs_latency.png")
    plt.close(fig)
    return points


# ---------------------------------------------------------- confusion figure
def transformer_preds():
    path = RES / "transformer_selected_preds.json"
    if path.exists():
        return json.loads(path.read_text())
    from scripts.run_ablations import load_wlasl, run_tf_rung

    abl = json.loads((RES / "ablations_xgb_tf_cnn.json").read_text())
    name = best_by_val(abl["transformer"])["rung"]
    config = {  # must match the rung definitions in scripts/run_ablations.py
        "0.": dict(location=True, augment=False, num_layers=1, readout="last"),
        "1.": dict(location=True, augment=True, num_layers=1, readout="last"),
        "2.": dict(location=True, augment=True, num_layers=1, readout="mean"),
        "3.": dict(location=True, augment=True, num_layers=2, readout="mean"),
        "4.": dict(location=False, augment=True, num_layers=2, readout="mean"),
    }[name[:2]]
    raw, labels, _ = load_wlasl()
    res = run_tf_rung(name, raw, labels, **config)
    path.write_text(json.dumps(res))
    return res


def confused_pairs(labels, preds_per_seed, k=8):
    c = Counter()
    for preds in preds_per_seed:
        c.update((t, p) for t, p in zip(labels, preds) if t != p)
    return c.most_common(k)


def confusions():
    tf = transformer_preds()
    bb = best_by_val(json.loads((RES / "backbone_ablations.json").read_text()))
    models = (("Transformer (landmarks)", tf, LANDMARK), (f"R(2+1)D (pixels)", bb, PIXEL))
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.2), sharex=True)
    for ax, (title, r, colour) in zip(axes, models):
        pairs = confused_pairs(r["test_labels"], r["test_preds"])
        names = [f"{GLOSS[t]} → {GLOSS[p]}" for (t, p), _ in pairs][::-1]
        counts = [n for _, n in pairs][::-1]
        ax.barh(names, counts, color=colour, height=0.6)
        for y, n in enumerate(counts):
            ax.text(n + 0.05, y, str(n), va="center", fontsize=6, color=INK)
        ax.set_title(title, fontsize=8, color=INK, loc="left")
        ax.set_xlabel("errors over 3 seeds")
        ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(FIGS / "confusions.png")
    plt.close(fig)

    # Do the two paradigms fail on the same clips? (majority vote over seeds)
    y = np.asarray(tf["test_labels"])
    assert y.tolist() == bb["test_labels"], "test order differs between runs"
    wrong = lambda r: (np.asarray(r["test_preds"]) != y).mean(0) > 0.5
    a, b = wrong(tf), wrong(bb)
    stats = {"transformer_wrong": int(a.sum()), "backbone_wrong": int(b.sum()),
             "both_wrong": int((a & b).sum()), "n_test": len(y),
             "transformer_pairs": [[GLOSS[t], GLOSS[p], n] for (t, p), n in confused_pairs(y, tf["test_preds"])],
             "backbone_pairs": [[GLOSS[t], GLOSS[p], n] for (t, p), n in confused_pairs(y, bb["test_preds"])],
             "backbone_rung": bb["rung"]}
    (RES / "error_overlap.json").write_text(json.dumps(stats, indent=1))
    return stats


if __name__ == "__main__":
    crops()
    print("crops.png")
    for p in frontier():
        print(f"  {p[0]:<28} {p[1]:8.3f} ms  {p[2]:5.1f}%")
    print("acc_vs_latency.png")
    print(json.dumps(confusions(), indent=1))
    print("confusions.png")
