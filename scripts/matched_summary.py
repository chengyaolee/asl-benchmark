"""Table and figures for the matched skeleton-vs-image comparison (UPDATED PLAN.md).

    python scripts/matched_summary.py

Reads results/matched_ablations.json and, if present, results/learning_curves.json. Rungs that have not
finished are skipped, so it can be run on partial results.

    logs/matched_summary.txt        every finished row
    paper/figs/matched_ladder.png   test top-1 +- seed std, one bar per rung
    paper/figs/learning_curves.png  test top-1 vs. training clips, skeletons vs. pixels
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES, FIGS = ROOT / "results", ROOT / "paper/figs"
FIGS.mkdir(parents=True, exist_ok=True)

LANDMARK, PIXEL, FUSION = "#2a78d6", "#eb6834", "#8a8f98"      # skeleton / image / both
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e4e3dd"
plt.rcParams.update({
    "font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

# key -> (short label, family, hatch). Order = top to bottom in the ladder figure.
LADDER = [
    ("S2", "Full 224 frame, ResNet frozen", "pixel", ""),
    ("S3", "Full 224 frame, ResNet fine-tuned", "pixel", ""),
    ("S4", "Hand patches (borrowed boxes), frozen", "pixel", ""),
    ("C3", "Hand patches (blanked if missed), frozen", "pixel", ""),
    ("S5", "Hand patches, fine-tuned", "pixel", ""),
    ("C2a", "Skeleton drawn as image, frozen", "pixel", "//"),
    ("C2b", "Skeleton drawn as image, fine-tuned", "pixel", "//"),
    ("C4", "Fusion: patches + skeleton, frozen", "fusion", ""),
    ("S1", "Skeleton, static MediaPipe (matched)", "skeleton", ""),
    ("C5", "Skeleton, x/y only", "skeleton", ""),
    ("C1", "Skeleton, tracked full-res", "skeleton", ""),
]
COLOUR = {"pixel": PIXEL, "skeleton": LANDMARK, "fusion": FUSION}


def load_rows():
    path = RES / "matched_ablations.json"
    return {r["key"]: r for r in json.loads(path.read_text())} if path.exists() else {}


def end_to_end() -> dict:
    """rung key -> end-to-end ms per clip (video file to prediction, nothing cached), if measured."""
    path = RES / "end_to_end_latency.json"
    if not path.exists():
        return {}
    return {k: p["ms_mean"] for p in json.loads(path.read_text())["pipelines"].values() for k in p["rungs"]}


def table(rows) -> str:
    e2e = end_to_end()
    out = [f"{'rung':<72}{'val':>6}{'test':>8}{'+-':>6}{'top5':>7}{'train s':>9}{'net ms':>8}{'e2e ms':>9}{'params':>12}"]
    for key, label, *_ in LADDER:
        r = rows.get(key)
        if r:
            out.append(f"{r['rung']:<72}{r['val_top1']:>6.1f}{r['test_top1']:>8.1f}{r['test_std']:>6.1f}"
                       f"{r['test_top5']:>7.1f}{r['train_s']:>9.0f}{r.get('latency_ms', float('nan')):>8.2f}"
                       f"{e2e.get(key, float('nan')):>9.0f}{r.get('params', 0):>12,}")
        else:
            out.append(f"{key:<72}  (not finished)")
    return "\n".join(out)


def ladder(rows) -> None:
    items = [(k, lab, fam, h) for k, lab, fam, h in LADDER if k in rows]
    fig, ax = plt.subplots(figsize=(5.4, 0.3 * len(items) + 0.9))
    y = np.arange(len(items))[::-1]
    for yi, (k, lab, fam, hatch) in zip(y, items):
        r = rows[k]
        ax.barh(yi, r["test_top1"], height=0.62, color=COLOUR[fam], hatch=hatch, edgecolor="white", linewidth=0.0)
        ax.errorbar(r["test_top1"], yi, xerr=r["test_std"], color=INK, capsize=2, linewidth=0.8)
        ax.text(r["test_top1"] + r["test_std"] + 1.2, yi, f"{r['test_top1']:.1f}", va="center", fontsize=7, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels([lab for _, lab, _, _ in items])
    ax.set_xlim(0, 85)
    ax.set_xlabel("Test top-1 (%), mean $\\pm$ std over 3 seeds (100 test clips)")
    ax.grid(axis="y", visible=False)
    ax.set_axisbelow(True)
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLOUR[f]) for f in ("skeleton", "pixel", "fusion")]
    ax.legend(handles, ["Skeleton (vector input)", "Image input", "Both"], loc="upper right", frameon=False, fontsize=7)
    fig.savefig(FIGS / "matched_ladder.png")
    plt.close(fig)


def curves() -> None:
    path = RES / "learning_curves.json"
    if not path.exists():
        return
    rows = json.loads(path.read_text())
    series = [  # key, label, colour, linestyle
        ("C1", "Skeleton, tracked", LANDMARK, "--"),
        ("S1", "Skeleton, static (matched)", LANDMARK, "-"),
        ("C3", "Image: hand patches, frozen", PIXEL, "--"),
        ("S2", "Image: full frame, frozen", PIXEL, "-"),
    ]
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    for key, label, colour, ls in series:
        sub = [r for r in rows if r["key"] == key]
        if not sub:
            continue
        xs = sorted({r["n_train"] for r in sub})
        by = {x: [r["test_top1"] for r in sub if r["n_train"] == x] for x in xs}
        mean = np.array([np.mean(by[x]) for x in xs])
        std = np.array([np.std(by[x]) for x in xs])
        ax.plot(xs, mean, ls, color=colour, marker="o", markersize=3.5, linewidth=1.4, label=label)
        ax.fill_between(xs, mean - std, mean + std, color=colour, alpha=0.12, linewidth=0)
    ax.set_xlabel("Training clips (class-stratified subset)")
    ax.set_ylabel("Test top-1 (%)")
    ax.set_ylim(0, 80)
    ax.legend(frameon=False, fontsize=6.5, loc="upper left")
    fig.savefig(FIGS / "learning_curves.png")
    plt.close(fig)


# Reported on the FULL WLASL100 (2,038 clips, official split); our subset has 1,013 clips, so these are context only.
REPORTED = [
    ("Pose-TGCN (Li et al. 2020, OpenPose body + hands)", 55.43, 78.68),
    ("I3D (Li et al. 2020, Kinetics)", 65.89, 84.11),
]


def baselines_table() -> str:
    """TGCN (depth picked on validation, per source and input variant) and I3D, plus the published numbers."""
    out = ["", "Baselines on the 1,013-clip subset (3 seeds); published numbers are on the full 2,038-clip WLASL100",
           f"{'model':<72}{'val':>6}{'test':>8}{'+-':>6}{'top5':>7}{'params':>12}"]
    path = RES / "tgcn.json"
    if path.exists():
        groups = {}
        for r in json.loads(path.read_text()):
            groups.setdefault((r["variant"], r["rung"].split(",")[1].strip()), []).append(r)
        for (variant, source), rs in sorted(groups.items()):
            r = max(rs, key=lambda r: r["val_top1"])                           # depth chosen by validation accuracy
            out.append(f"{'TGCN ' + variant + ' coords, ' + source + ', ' + str(r['blocks']) + ' blocks (val-selected of ' + str(len(rs)) + ')':<72}"
                       f"{r['val_top1']:>6.1f}{r['test_top1']:>8.1f}{r['test_std']:>6.1f}{r['test_top5']:>7.1f}{r['params']:>12,}")
    path = RES / "i3d.json"
    if path.exists():
        r = json.loads(path.read_text())[0]
        out.append(f"{r['rung']:<72}{r['val_top1']:>6.1f}{r['test_top1']:>8.1f}{r['test_std']:>6.1f}{r['test_top5']:>7.1f}{r['params']:>12,}")
    for name, top1, top5 in REPORTED:
        out.append(f"{name + '  [published, full WLASL100]':<72}{'':>6}{top1:>8.1f}{'':>6}{top5:>7.1f}")
    return "\n".join(out)


def main() -> None:
    rows = load_rows()
    text = table(rows) + "\n" + baselines_table()
    (ROOT / "logs").mkdir(exist_ok=True)
    (ROOT / "logs/matched_summary.txt").write_text(text + "\n")
    print(text)
    if rows:
        ladder(rows)
    curves()


if __name__ == "__main__":
    main()
