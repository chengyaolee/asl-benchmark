"""Error analysis from saved predictions (no training): overlap, confused pairs, and accuracy against input quality.

Transformer T3a (3 seeds), R(2+1)D P0 / P1 / C0 / P2 (seed 0), ResNet-18 rungs (3 seeds). Test split, 100 clips.
Keys here (they seed the bootstrap, so do not rename): P0 letterbox, P1 letterbox + augmentation, C0 centre crop,
P2 centre crop + augmentation. In the paper's table: P0, P0 + augmentation, P1, P2.
    python scripts/error_analysis.py      # -> results/error_analysis.json
"""
import json, sys, zlib
from collections import Counter
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cv2
from asl_benchmark.models import frame_encoder as fe
from asl_benchmark.models import video_backbone as vb

R = ROOT / "results"
def rows(f): return json.loads((R / f).read_text())
def pick(rs, key=None):
    r = rs[0] if key is None else next(x for x in rs if x.get("key") == key or str(x.get("rung", "")).startswith(key))
    return np.array(r["test_labels"]), np.atleast_2d(r["test_preds"])
M = {"T3a": pick(rows("transformer_ladder.json"), "T3a"), "T0": pick(rows("transformer_ladder.json"), "T0"),
     "T2": pick(rows("transformer_ladder.json"), "T2"),
     "P0": pick(rows("r2plus1d_32x112_noaug.json")), "P1": pick(rows("r2plus1d_32x112.json")),
     "P2": pick(rows("r2plus1d_32x112_center.json")), "C0": pick(rows("r2plus1d_32x112_center_noaug.json")),
     "Rfull": pick(rows("frame_encoder_ablations.json"), "4."), "Rhands": pick(rows("frame_encoder_ablations.json"), "5."),
     "Rfull_frozen": pick(rows("frame_encoder_ablations.json"), "2."), "Rhands_frozen": pick(rows("frame_encoder_ablations.json"), "3.")}
y = M["T3a"][0]
assert all((v[0] == y).all() for v in M.values())
score = {k: (v[1] == y).mean(0) for k, v in M.items()}
def boot(d, name):
    rng = np.random.default_rng(zlib.crc32(name.encode())); s = d[rng.integers(0, len(d), (10000, len(d)))].mean(1)
    return [round(float(x) * 100, 1) for x in (d.mean(), np.percentile(s, 2.5), np.percentile(s, 97.5))]
out = {"paired": {f"{a}-{b}": boot(score[a] - score[b], f"{a}-{b}") for a, b in
       [("Rhands", "Rfull"), ("Rhands_frozen", "Rfull_frozen"), ("Rhands", "Rhands_frozen"), ("P2", "P1"), ("P2", "P0"), ("P1", "P0"),
        ("T3a", "P2"), ("T3a", "P0"), ("T3a", "Rhands"), ("T3a", "Rfull"), ("P2", "Rfull"), ("P2", "Rhands"), ("T0", "P2"), ("T2", "T0"), ("T3a", "T0"),
        ("C0", "P0"), ("P2", "C0"), ("C0", "P1"), ("T3a", "C0"), ("T3a", "P1"), ("C0", "Rhands"), ("C0", "Rfull")]}}
maj = {k: (v > 0.5) for k, v in score.items()}
out["overlap"] = {}
for a, b in [("T3a", "P2"), ("T3a", "P0"), ("T3a", "Rhands"), ("P2", "Rhands"), ("P1", "P2"), ("P0", "P2"), ("T3a", "C0"), ("C0", "P2"), ("P0", "C0")]:
    A, B = maj[a], maj[b]
    out["overlap"][f"{a}|{b}"] = dict(both=int((A & B).sum()), only_a=int((A & ~B).sum()), only_b=int((~A & B).sum()), neither=int((~A & ~B).sum()))
A = [maj["T3a"], maj["P2"], maj["Rhands"]]
out["union_T3a_P2"] = int((A[0] | A[1]).sum())
out["union_T3a_C0"] = int((maj["T3a"] | maj["C0"]).sum())
# confusions (pooled over seeds); clips the model gets wrong
samples = [s for s in fe._samples() if s.split == "test"]
gl = {s.label: s.gloss if hasattr(s, "gloss") else str(s.label) for s in fe._samples()}
out["confused"] = {}
for k in ("T3a", "P2", "C0"):
    c = Counter((gl[t], gl[p]) for run in M[k][1] for t, p in zip(y, run) if t != p)
    out["confused"][k] = [[*a, n] for a, n in c.most_common(8)]
# covariates of each test clip
cov = {k: [] for k in ("any_hand", "both_hands", "hand_px_letter112", "hand_px_center112", "outside_center", "n_frames")}
for s in samples:
    k = np.load(vb.KP_DIR / f"{s.sample_id}.npy"); cap = cv2.VideoCapture(str(s.video_path)); w, h = int(cap.get(3)), int(cap.get(4)); cap.release()
    seen = ~np.isnan(k[..., 0]).reshape(len(k), 2, 21).any(2)
    xy = k[..., :2].reshape(-1, 2, 21, 2)[..., :] * np.array([w, h]); b = xy[seen]
    cov["any_hand"].append(seen.any(1).mean()); cov["both_hands"].append(seen.all(1).mean()); cov["n_frames"].append(len(k))
    if len(b):
        sz = np.median((b.max(1) - b.min(1)).max(1)); m = min(w, h)
        out_ = 1 - ((abs(b[..., 0] - w / 2) <= m / 2) & (abs(b[..., 1] - h / 2) <= m / 2)).mean()
    else:
        sz, out_, m = np.nan, np.nan, min(w, h)
    cov["hand_px_letter112"].append(sz / max(w, h) * 112); cov["hand_px_center112"].append(sz / m * 112); cov["outside_center"].append(out_)
out["by_covariate"] = {}
for name, v in cov.items():
    v = np.array(v); ok = ~np.isnan(v); med = np.median(v[ok]); lo, hi = ok & (v <= med), ok & (v > med)
    out["by_covariate"][name] = {"median": round(float(med), 3), "n_lo": int(lo.sum()), "n_hi": int(hi.sum()),
        **{m_: [round(float(score[m_][lo].mean()) * 100, 1), round(float(score[m_][hi].mean()) * 100, 1)] for m_ in ("T3a", "P0", "P1", "C0", "P2", "Rhands", "Rfull")}}
(R / "error_analysis.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=0).replace("\n", " ").replace('"paired"', '\n"paired"').replace('"overlap"', '\n"overlap"').replace('"confused"', '\n"confused"').replace('"by_cov', '\n"by_cov'))
