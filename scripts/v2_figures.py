"""Figures for paper/paper_v2.tex, written to paper/figs/. Runs on whatever results exist; rerun as results arrive.

    python scripts/v2_figures.py

    ladder_diagram.png   the information ladder (what each step changes)
    ladder_bars.png      test accuracy of each step with a 95% interval over the 100 test videos (needs results/v2_stats.json)
    examples.png         one test video: the shared frames, the hand crops, the keypoints (needs the caches)
    acc_vs_time.png      accuracy against end-to-end time per video (needs results/end_to_end_latency.json)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RES, FIGS = ROOT / "results", ROOT / "paper/figs"
FIGS.mkdir(parents=True, exist_ok=True)

SKELETON, PICTURE = "#2a78d6", "#eb6834"
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e4e3dd"
plt.rcParams.update({
    "font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "savefig.dpi": 300, "savefig.bbox": "tight",
})

STEPS = [  # key, label, group
    ("S3", "Whole frame", "picture"), ("S5", "Hand crops", "picture"), ("X1", "Hand crops\n+ location", "picture"),
    ("S1", "Hand keypoints\n+ location", "skeleton"), ("X2", "Hand + body\nkeypoints", "skeleton"),
]
COLOUR = {"picture": PICTURE, "skeleton": SKELETON}


def ladder_diagram() -> None:
    fig, ax = plt.subplots(figsize=(7.0, 1.9))
    ax.axis("off")
    ax.set_xlim(0, 10.2)
    ax.set_ylim(0, 2.4)
    notes = ["where to look", "where the hand is", "what is represented:\npixels or skeleton", "body context"]
    for i, (key, label, group) in enumerate(STEPS):
        x = 0.1 + i * 2.05
        ax.add_patch(FancyBboxPatch((x, 0.85), 1.75, 0.95, boxstyle="round,pad=0.02,rounding_size=0.08",
                                    facecolor=COLOUR[group], edgecolor="none", alpha=0.92))
        ax.text(x + 0.875, 1.32, label, ha="center", va="center", color="white", fontsize=7.5, fontweight="bold")
        ax.text(x + 0.875, 0.55, key, ha="center", va="center", color=MUTED, fontsize=7)
        if i < len(notes):
            ax.add_patch(FancyArrowPatch((x + 1.78, 1.32), (x + 2.07, 1.32), arrowstyle="-|>", mutation_scale=8, color=INK, lw=0.9))
            ax.text(x + 1.925, 2.05, notes[i], ha="center", va="center", fontsize=6.3, color=INK)
    ax.text(3.2, 0.12, "pixels  (ResNet-18 + the same small transformer)", ha="center", fontsize=6.5, color=PICTURE)
    ax.text(8.0, 0.12, "keypoints  (the same small transformer)", ha="center", fontsize=6.5, color=SKELETON)
    fig.savefig(FIGS / "ladder_diagram.png")
    plt.close(fig)


def ladder_bars() -> None:
    path = RES / "v2_stats.json"
    if not path.exists():
        return
    models = json.loads(path.read_text())["models"]
    steps = [s for s in STEPS if s[0] in models]
    if "I3D" in models:
        steps.append(("I3D", "I3D (video network,\nwhole frame)", "picture"))
    fig, ax = plt.subplots(figsize=(3.4, 0.42 * len(steps) + 0.7))
    ys = np.arange(len(steps))[::-1]
    for y, (key, label, group) in zip(ys, steps):
        m = models[key]
        ax.barh(y, m["accuracy"], height=0.62, color=COLOUR[group], zorder=2)
        ax.errorbar(m["accuracy"], y, xerr=[[m["accuracy"] - m["ci95"][0]], [m["ci95"][1] - m["accuracy"]]],
                    color=INK, capsize=2, lw=0.8, zorder=3)
        ax.text(m["ci95"][1] + 1.5, y, f"{m['accuracy']:.0f}", va="center", fontsize=7, color=INK)
    ax.set_yticks(ys)
    ax.set_yticklabels([lab.replace("\n", " ") if "I3D" not in lab else lab for _, lab, _ in steps], fontsize=6.5)
    ax.set_xlabel("Correct out of 100 test videos")
    ax.set_xlim(0, 100)
    ax.grid(axis="y", visible=False)
    ax.set_axisbelow(True)
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLOUR[g]) for g in ("picture", "skeleton")]
    ax.legend(handles, ["Pixels", "Keypoints"], frameon=False, fontsize=6.5, loc="lower right")
    fig.savefig(FIGS / "ladder_bars.png")
    plt.close(fig)


def examples() -> None:
    import cv2
    import mediapipe as mp

    from asl_benchmark.models import frame_encoder as fe
    from asl_benchmark.models import matched as m
    from asl_benchmark.models import video_backbone as vb

    cache = vb.CACHE
    if not (cache / "kp_static224.npz").exists() or not (cache / "kp_pose224.npz").exists():
        return
    samples = fe._samples()
    hands = np.load(cache / "kp_static224.npz")["kp"]
    pose = np.load(cache / "kp_pose224.npz")["kp"]
    test = [i for i, s in enumerate(samples) if s.split == "test"]
    both = [(~np.isnan(hands[i, :, [0, 21], 0])).all(0).sum() for i in test]          # frames with both hands found
    i = test[int(np.argmax(both))]
    s = samples[i]
    frames = m.read_frames((s.video_path, vb.KP_DIR / f"{s.sample_id}.npy"))
    patches, _ = m.make_patches(frames[None], hands[i][None], borrow=True)
    cols = [1, 5, 9, 13]
    links = list(mp.solutions.hands.HAND_CONNECTIONS)
    body_links = [(1, 2), (1, 3), (3, 5), (2, 4), (4, 6), (1, 7), (2, 8), (7, 8)]       # indices into the 9 body points
    rows = [[], [], []]
    for t in cols:
        rows[0].append(frames[t])
        crop = np.concatenate([patches[0, t, 0], patches[0, t, 1]], 0)
        rows[1].append(cv2.resize(crop, (224, 448)))
        img = np.zeros_like(frames[t])
        for a, b in body_links:
            if not np.isnan(pose[i, t, [a, b], 0]).any():
                cv2.line(img, tuple((pose[i, t, a, :2] * 223).astype(int)), tuple((pose[i, t, b, :2] * 223).astype(int)), (120, 120, 120), 2)
        for hand, colour in ((0, (255, 140, 70)), (1, (70, 200, 130))):
            pts = hands[i, t, hand * 21 : (hand + 1) * 21, :2]
            if np.isnan(pts).any():
                continue
            px = (pts * 223).astype(int)
            for a, b in links:
                cv2.line(img, tuple(px[a]), tuple(px[b]), colour, 1)
            for p in px:
                cv2.circle(img, tuple(p), 2, colour, -1)
        rows[2].append(img)
    fig, axes = plt.subplots(3, len(cols), figsize=(6.4, 7.0), gridspec_kw={"height_ratios": [1, 2, 1]})
    titles = ["Shared frame (224x224)", "Hand crops (112x112 each, left above right)", "Keypoints found in that frame"]
    for r in range(3):
        for c in range(len(cols)):
            axes[r, c].imshow(rows[r][c])
            axes[r, c].axis("off")
        axes[r, 0].set_title(titles[r], loc="left", fontsize=7, color=INK)
    fig.savefig(FIGS / "examples.png")
    plt.close(fig)


OFFSETS = {"Hands": (-5, 3, "right"), "Whole frame": (5, -9, "left"), "I3D": (5, 4, "left"), "Hands + body": (0, -11, "center"),
           "Hands, tracked": (2, 5, "right")}


def acc_vs_time() -> None:
    path = RES / "end_to_end_latency.json"
    stats = RES / "v2_stats.json"
    if not path.exists() or not stats.exists():
        return
    pipes = json.loads(path.read_text())["pipelines"]
    acc = json.loads(stats.read_text())["models"]
    points = [  # label, pipeline, accuracy key, group
        ("Hands", "skeleton_static", "S1", "skeleton"), ("Hands + body", "skeleton_body", "X2", "skeleton"),
        ("Hands, tracked", "skeleton_tracked", "C1", "skeleton"), ("Whole frame", "image_full", "S3", "picture"),
        ("Hand crops", "image_patches", "X1", "picture"), ("I3D", "i3d_full", "I3D", "picture"),
    ]
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    for label, pipe, key, group in points:
        if pipe in pipes and key in acc:
            x, y = pipes[pipe]["ms_mean"], acc[key]["accuracy"]
            ax.scatter(x, y, color=COLOUR[group], s=22, zorder=3)
            dx, dy, ha = OFFSETS.get(label, (4, 3, "left"))
            ax.annotate(label, (x, y), textcoords="offset points", xytext=(dx, dy), fontsize=6, color=INK, ha=ha)
    handles = [plt.Line2D([], [], marker="o", ls="", color=COLOUR[g], markersize=4) for g in ("skeleton", "picture")]
    ax.legend(handles, ["Keypoints (transformer)", "Pixels (ResNet, I3D)"], frameon=False, fontsize=6, loc="lower right")
    ax.set_xscale("log")
    ax.set_xlabel("Time from video file to answer (ms, log scale)")
    ax.set_ylabel("Correct out of 100")
    ax.set_ylim(0, 100)
    fig.savefig(FIGS / "acc_vs_time.png")
    plt.close(fig)


if __name__ == "__main__":
    for fn in (ladder_diagram, ladder_bars, examples, acc_vs_time):
        try:
            fn()
            print("ok", fn.__name__)
        except Exception as e:  # keep going: figures that need unfinished results are skipped
            print("skipped", fn.__name__, "-", type(e).__name__, e)
