"""How separable are the WLASL100 hand keypoint features?

Run from the repo root:
    python scripts/analyze_features.py

Uses clean (not augmented) features from scripts/features.py and answers:
  1. Are clips of the same sign close together?          (nearest-neighbour accuracy)
  2. Can a straight line split the classes?             (linear probe = logistic regression)
  3. Which feature groups carry the signal?             (shape / missing / location, alone and together)
  4. Does the order of frames matter?                   (time-averaged vs full sequence)
  5. Do neighbours share the sign or just the signer?   (1-NN same label vs same signer)
  6. Which signs get confused with each other?          (top confused pairs)
  7. A 2-D picture of 10 signs                           (t-SNE plot)

Writes results/feature_analysis.json and results/tsne.png.
"""

import json, sys, warnings
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from asl_benchmark.loader import load
from scripts.features import prep

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

warnings.filterwarnings("ignore")
T, PER_FRAME = 32, 132          # per frame: 126 shape + 2 missing + 4 location

ds = load("wlasl100", ROOT / "data", download=False)
meta = json.loads((ROOT / "data/wlasl100/WLASL_v0.3.json").read_text())
signer_of = {str(i["video_id"]): i["signer_id"] for e in meta for i in e["instances"]}

X, y, split, signer = [], [], [], []
for s in ds.samples:
    f = ROOT / "data/wlasl100/keypoints" / f"{s.sample_id}.npy"
    if f.exists():
        a = np.load(f)
        if len(a):
            X.append(prep(a, T=T, location=True)); y.append(s.label)
            split.append(s.split); signer.append(signer_of[s.sample_id])
X, y, split, signer = np.array(X), np.array(y), np.array(split), np.array(signer)
tr, ev = split == "train", np.isin(split, ["val", "test"])   # evaluate on val+test together (265 clips)
mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
Z = (X - mu) / sd
print(f"{tr.sum()} train clips, {ev.sum()} val+test clips, {X.shape[1]} features\n")
out = {}

def cols(groups, time_avg=False):
    per = {"shape": range(0, 126), "missing": range(126, 128), "location": range(128, 132)}
    keep = np.zeros(PER_FRAME, bool)
    for g in groups:
        keep[list(per[g])] = True
    M = Z.reshape(len(Z), T, PER_FRAME)[:, :, keep]
    return M.mean(1) if time_avg else M.reshape(len(Z), -1)

def probe(F):
    # pick regularisation on val only, report on val+test
    va = split == "val"
    best = max((LogisticRegression(C=C, max_iter=2000).fit(F[tr], y[tr]).score(F[va], y[va]), C)
               for C in (0.001, 0.01, 0.1, 1.0))
    clf = LogisticRegression(C=best[1], max_iter=2000).fit(F[tr], y[tr])
    return clf, clf.score(F[ev], y[ev])

# 1. nearest neighbours
print("1. Nearest neighbours (cosine), val+test accuracy")
for k in (1, 5):
    knn = KNeighborsClassifier(n_neighbors=k, metric="cosine").fit(Z[tr], y[tr])
    acc = knn.score(Z[ev], y[ev]); out[f"knn{k}"] = acc
    print(f"   {k}-NN: {acc:.3f}")
print("   (chance = 0.010)\n")

# 2-4. linear probes
print("2-4. Linear probe (logistic regression), val+test accuracy")
rows = [("all features", ["shape", "missing", "location"], False),
        ("shape only", ["shape"], False),
        ("location only", ["location"], False),
        ("missing-hand flags only", ["missing"], False),
        ("shape + missing (no location)", ["shape", "missing"], False),
        ("all, averaged over time", ["shape", "missing", "location"], True)]
clf_all = None
for name, groups, avg in rows:
    clf, acc = probe(cols(groups, avg))
    out[f"probe: {name}"] = acc
    if clf_all is None:
        clf_all = clf
    print(f"   {name:32s} {acc:.3f}")
print()

# 5. same sign or same signer?
nn = KNeighborsClassifier(n_neighbors=1, metric="cosine").fit(Z[tr], np.arange(tr.sum()))
idx = nn.predict(Z[ev])
same_label = (y[tr][idx] == y[ev]).mean()
same_signer = (signer[tr][idx] == signer[ev]).mean()
base_signer = np.mean([np.mean(signer[tr] == s) for s in signer[ev]])
out.update(nn_same_label=same_label, nn_same_signer=same_signer, signer_chance=base_signer)
print("5. Nearest training clip of each val+test clip")
print(f"   same sign:   {same_label:.3f}")
print(f"   same signer: {same_signer:.3f}   (by chance: {base_signer:.3f})")
print("   If 'same signer' is far above chance, the features also encode who is signing.\n")

# 6. confused pairs
pred = clf_all.predict(cols(["shape", "missing", "location"])[ev])
cm = confusion_matrix(y[ev], pred, labels=range(100))
np.fill_diagonal(cm, 0)
pairs = Counter()
for a in range(100):
    for b in range(100):
        if cm[a, b]:
            pairs[tuple(sorted((a, b)))] += cm[a, b]
print("6. Most confused sign pairs (linear probe, val+test)")
out["confused_pairs"] = []
for (a, b), n in pairs.most_common(10):
    print(f"   {ds.glosses[a]:>12s} <-> {ds.glosses[b]:<12s} {n}")
    out["confused_pairs"].append([ds.glosses[a], ds.glosses[b], int(n)])
print()

# 7. t-SNE of the 10 signs with the most clips
try:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    top = [c for c, _ in Counter(y).most_common(10)]
    m = np.isin(y, top)
    E = TSNE(n_components=2, perplexity=min(30, m.sum() // 4), init="pca",
             random_state=0).fit_transform(Z[m])
    plt.figure(figsize=(7, 6))
    for c in top:
        k = y[m] == c
        plt.scatter(E[k, 0], E[k, 1], s=18, label=ds.glosses[c])
    plt.legend(fontsize=8, markerscale=1.5)
    plt.title("t-SNE of hand keypoint features, 10 most common signs")
    plt.xticks([]); plt.yticks([])
    (ROOT / "results").mkdir(exist_ok=True)
    plt.savefig(ROOT / "results/tsne.png", dpi=150, bbox_inches="tight")
    print("7. Saved results/tsne.png")
except ImportError:
    print("7. Skipped t-SNE (pip install matplotlib)")

(ROOT / "results").mkdir(exist_ok=True)
(ROOT / "results/feature_analysis.json").write_text(json.dumps(out, indent=2, default=float))
print("Saved results/feature_analysis.json")
