"""Paired comparisons with 95% intervals (UPDATED PLAN.md, section 4). No training.

For each model, the score of a test video is the share of the model's seeds that got it right. A comparison is the mean
difference of those scores over the 100 test videos, with a 95% interval from a paired bootstrap over videos
(10,000 draws). The same 100 videos are used for every model, so the pairing removes video difficulty from the difference.

Three PRIMARY comparisons are fixed here, before the new runs finish; everything else is exploratory.

    python scripts/paired_stats.py        # -> results/v2_stats.json, logs/v2_stats.txt, paper/tables/paired.tex
"""

from __future__ import annotations

import json
import zlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
DRAWS, SEED = 10_000, 0

# plain name, file, how to pick the row
MODELS = {
    "S1": ("Keypoint transformer, hands (matched)", "matched_ablations.json", "S1"),
    "C1": ("Keypoint transformer, tracked full-resolution", "matched_ablations.json", "C1"),
    "C5": ("Keypoint transformer, no depth", "matched_ablations.json", "C5"),
    "S3": ("ResNet, whole frame, fine-tuned", "matched_ablations.json", "S3"),
    "S5": ("ResNet, hand crops, fine-tuned", "matched_ablations.json", "S5"),
    "S2": ("ResNet, whole frame, frozen", "matched_ablations.json", "S2"),
    "S4": ("ResNet, hand crops (borrowed box), frozen", "matched_ablations.json", "S4"),
    "C3": ("ResNet, hand crops (black if missed), frozen", "matched_ablations.json", "C3"),
    "X1": ("ResNet, hand crops + wrist location, fine-tuned", "v2_ladder.json", "X1"),
    "X2": ("Keypoint transformer, hands + upper body", "v2_ladder.json", "X2"),
    "X3": ("Keypoint transformer, no wrist location", "v2_ladder.json", "X3"),
    "I3D": ("I3D, whole frame, fine-tuned", "i3d.json", None),
}
PRIMARY = [("S1", "X1", "pose vs. image, same location"), ("X1", "S5", "value of location for the image model"),
           ("X2", "S1", "value of the upper body for the keypoint model")]
EXPLORATORY = [("S1", "S5", "keypoints vs. hand crops without location"), ("S1", "S3", "keypoints vs. whole frame"),
               ("S5", "S3", "hand crops vs. whole frame"), ("S1", "X3", "value of location for the keypoint model"),
               ("C1", "S1", "tracking and full resolution"), ("C5", "S1", "removing depth"),
               ("S1", "I3D", "keypoints vs. a video network"), ("X2", "X1", "keypoints + body vs. crops + location"),
               ("C3", "S2", "hand crops (black if missed) vs. whole frame, frozen"),
               ("C3", "S4", "black vs. borrowed box when a hand is missed, frozen"),
               ("S3", "S2", "fine-tuning the whole-frame ResNet"), ("S5", "S4", "fine-tuning the hand-crop ResNet")]


def load(key: str):
    """(per-video score vector, labels) or None if the run is not finished."""
    _, fname, rung = MODELS[key]
    path = RES / fname
    if not path.exists():
        return None
    rows = json.loads(path.read_text())
    row = rows[0] if rung is None else next((r for r in rows if r.get("key") == rung), None)
    if row is None:
        return None
    labels = np.asarray(row["test_labels"])
    correct = np.mean([np.asarray(p) == labels for p in row["test_preds"]], axis=0)
    return correct, row


def interval(d: np.ndarray, name: str) -> tuple[float, float]:
    """Bootstrap interval; the random stream depends only on `name`, so adding rows never changes the others."""
    rng = np.random.default_rng(SEED + zlib.crc32(name.encode()))
    idx = rng.integers(0, len(d), size=(DRAWS, len(d)))
    means = d[idx].mean(1) * 100
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main() -> None:
    scores = {k: load(k) for k in MODELS}
    out = {"models": {}, "comparisons": []}
    for k, v in scores.items():
        if v is None:
            continue
        lo, hi = interval(v[0], f"model {k}")
        out["models"][k] = {"name": MODELS[k][0], "accuracy": round(float(v[0].mean()) * 100, 1), "ci95": [round(lo, 1), round(hi, 1)],
                            "seeds": len(v[1]["test_preds"])}
    lines = [f"{'comparison':<62}{'difference':>11}{'95% interval':>20}"]
    tex = []
    for kind, pairs in (("primary", PRIMARY), ("exploratory", EXPLORATORY)):
        lines.append(f"-- {kind} --")
        for a, b, why in pairs:
            va, vb = scores[a], scores[b]
            label = f"{a} - {b}  ({why})"
            if va is None or vb is None:
                lines.append(f"{label:<62}{'pending':>11}")
                tex.append((kind, a, b, why, None))
                continue
            d = va[0] - vb[0]
            lo, hi = interval(d, f"{a} - {b}")
            diff = float(d.mean()) * 100
            out["comparisons"].append({"kind": kind, "a": a, "b": b, "why": why, "difference": round(diff, 1),
                                       "ci95": [round(lo, 1), round(hi, 1)]})
            lines.append(f"{label:<62}{diff:>+11.1f}{f'{lo:+.1f} to {hi:+.1f}':>20}")
            tex.append((kind, a, b, why, (diff, lo, hi)))
    (RES / "v2_stats.json").write_text(json.dumps(out, indent=1))
    text = "\n".join(lines)
    (ROOT / "logs").mkdir(exist_ok=True)
    (ROOT / "logs/v2_stats.txt").write_text(text + "\n")
    print(text)

    # LaTeX rows for the paper (\input{tables/paired}); unfinished comparisons are marked.
    names = {k: MODELS[k][0] for k in MODELS}
    rows = []
    for kind, a, b, why, res in tex:
        mark = r"\textbf{" if kind == "primary" else ""
        end = "}" if kind == "primary" else ""
        cell = "pending" if res is None else f"${res[0]:+.1f}$ & ${res[1]:+.1f}$ to ${res[2]:+.1f}$"
        if res is None:
            cell = r"\multicolumn{2}{c}{pending}"
        rows.append(f"{mark}{names[a]}{end} $-$ {names[b]} & {cell} \\\\")
        if kind == "primary" and (a, b, why) == PRIMARY[-1]:
            rows.append(r"\midrule")
    (ROOT / "paper/tables").mkdir(parents=True, exist_ok=True)
    (ROOT / "paper/tables/paired.tex").write_text("\n".join(rows) + "\n")


if __name__ == "__main__":
    main()
