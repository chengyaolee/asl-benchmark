"""End-to-end inference time of the paper's models: video file -> predicted sign, one test clip at a time, nothing cached.

Same conventions as scripts/end_to_end_latency.py (neural nets on the GPU, fp16, batch 1; MediaPipe and trees on the CPU;
the first clip is a warm-up), but for the paper's setup: MediaPipe Hands tracked over every frame of the full-resolution
video for the keypoint models, and 32 (R(2+1)D) or 16 (ResNet, ConvNet) frames for the pixel models. Network latency does
not depend on trained weights, so networks are untrained copies. Trees are the exception (their cost depends on how many
trees early stopping kept), so configuration 4b is fitted once (seed 0) before timing. Keypoints are charged once per clip.
MLP, GRU and LSTM are untrained copies of the configurations in mlp.py and rnn.py (those scripts train when imported).

    python scripts/end_to_end_paper.py [out.json]   # run with the machine idle -> results/end_to_end_paper.json
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from asl_benchmark.models import frame_encoder as fe  # noqa: E402
from asl_benchmark.models import video_backbone as vb  # noqa: E402
from asl_benchmark.models import xgboost as xgb_mod  # noqa: E402
from scripts.augment import augment as augment_clip  # noqa: E402
from scripts.end_to_end_latency import clock, tracked_keypoints  # noqa: E402
from scripts.features import per_frame, prep  # noqa: E402
from scripts.run_ablations import AblationCNN, AblationTransformer, load_wlasl  # noqa: E402

PIPELINES = {  # name -> stages charged
    "trees": ["mp_tracked", "features_trees", "predict_trees"],
    "mlp": ["mp_tracked", "features_mlp", "net_mlp"],
    "gru": ["mp_tracked", "features_tf", "net_gru"],
    "lstm": ["mp_tracked", "features_tf", "net_lstm"],
    "transformer": ["mp_tracked", "features_tf", "net_tf"],
    "r2plus1d_center": ["decode_center32", "net_r2plus1d"],
    "resnet_full": ["decode_full16", "net_resnet_full"],
    "resnet_hands": ["mp_tracked", "decode_hands16", "net_resnet_hands"],
    "convnet": ["decode_conv16", "net_convnet"],
}


class RNN(torch.nn.Module):
    """rnn.py: bidirectional, hidden 128, one layer, mean readout."""

    def __init__(self, cell):
        super().__init__()
        self.rnn = cell(132, 128, batch_first=True, bidirectional=True)
        self.head = torch.nn.Sequential(torch.nn.Dropout(0.3), torch.nn.Linear(256, 100))
        self.drop_in = torch.nn.Dropout(0.3)

    def forward(self, x):
        return self.head(self.rnn(self.drop_in(x))[0].mean(1))


def mlp_net():
    """mlp.py: two hidden layers (512, 256) on mean and std over time."""
    L, R, D = torch.nn.Linear, torch.nn.ReLU, torch.nn.Dropout
    return torch.nn.Sequential(L(264, 512), R(), D(0.5), L(512, 256), R(), D(0.5), L(256, 100))


def n_frames(path):
    import cv2

    cap = cv2.VideoCapture(str(path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n


def decode(path, t):
    return fe._decode(path, np.linspace(0, n_frames(path) - 1, t).round().astype(np.int64))


def rgb(frame, box):
    import cv2

    return cv2.cvtColor(vb._square(frame, box), cv2.COLOR_BGR2RGB)


def letterbox_box(w, h):
    s = max(w, h)
    return (w - s) // 2, (h - s) // 2, s


def center_box(w, h):
    s = min(w, h)
    return (w - s) // 2, (h - s) // 2, s


def main():
    import cv2
    import mediapipe as mp

    dev = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    raw, labels, samples = load_wlasl()

    # trees: configuration 4b, seed 0 (needed only because latency depends on the fitted trees)
    fn = lambda s: prep(s, T=32, location=True, pool="chunks4").astype(np.float32)
    clean = np.stack([fn(s) for s in raw["train"]])
    mu, sd = clean.mean(0), clean.std(0) + 1e-6
    rng, rows, ys = np.random.default_rng(0), [(clean - mu) / sd], [np.asarray(labels["train"], np.int64)]
    for _ in range(xgb_mod.N_AUGMENTS):
        rows.append(np.stack([(fn(augment_clip(s, rng)) - mu) / sd for s in raw["train"]])); ys.append(ys[0])
    va = (np.stack([fn(s) for s in raw["val"]]) - mu) / sd
    tree, classes, _ = xgb_mod.train(np.concatenate(rows), np.concatenate(ys), 0, va, np.asarray(labels["val"], np.int64))
    trees_kept = int(tree.best_iteration) + 1
    print(f"trees kept: {trees_kept}", flush=True)

    seq = np.stack([per_frame(s, True)[np.linspace(0, len(s) - 1, 32).round().astype(int)] for s in raw["train"]])
    tf_mu, tf_sd = seq.reshape(-1, 132).mean(0), seq.reshape(-1, 132).std(0) + 1e-6
    tf = AblationTransformer(132, 100, max_frames=32, num_layers=1, readout="mean").to(dev).eval()
    r2p = vb.Classifier().to(dev).eval()
    res_full, res_hands = fe.FrameModel(1, 0).to(dev).eval(), fe.FrameModel(2, 2).to(dev).eval()
    conv = AblationCNN(100, use_bn=True, pool="mean").to(dev).eval()
    mlp = mlp_net().to(dev).eval()
    gru, lstm = RNN(torch.nn.GRU).to(dev).eval(), RNN(torch.nn.LSTM).to(dev).eval()
    mfn = lambda s: prep(s, T=32, location=True, pool="meanstd").astype(np.float32)
    mclean = np.stack([mfn(s) for s in raw["train"]])
    m_mu, m_sd = mclean.mean(0), mclean.std(0) + 1e-6
    n_params = {"mlp": mlp, "gru": gru, "lstm": lstm, "transformer": tf, "r2plus1d_center": r2p, "resnet_full": res_full, "resnet_hands": res_hands, "convnet": conv}
    n_params = {k: sum(p.numel() for p in v.parameters()) for k, v in n_params.items()}
    hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
    amp = lambda: torch.autocast("cuda", dtype=torch.float16)

    def sq16(kp):                                       # ResNet hand crops, as frame_encoder.read_hands on fresh keypoints
        seen = ~np.isnan(kp[..., :2]).any(-1)
        det = [np.flatnonzero(seen[:, h * 21 : (h + 1) * 21].all(1)) for h in (0, 1)]
        rows_ = fe.frame_indices(len(kp))
        frames, w, h = fe._decode(path, rows_)
        out, flag = np.zeros((16, 2, 112, 112, 3), np.uint8), np.ones((16, 2), np.float32)
        for i, (row, fr) in enumerate(zip(rows_, frames)):
            for hand in (0, 1):
                if fr is None or len(det[hand]) == 0:
                    continue
                near = det[hand][np.abs(det[hand] - row).argmin()]
                flag[i, hand] = float(near != row)
                out[i, hand] = rgb(fr, fe._hand_box(kp[near, hand * 21 : (hand + 1) * 21, :2], w, h))
        return out, flag

    acc = {k: [] for p in PIPELINES.values() for k in p}
    for j, s in enumerate(samples["test"]):
        path, ms = s.video_path, {}
        kp, ms["mp_tracked"] = clock(dev, lambda: tracked_keypoints(hands, path))
        f, ms["features_trees"] = clock(dev, lambda: ((fn(kp) - mu) / sd)[None])
        _, ms["predict_trees"] = clock(dev, lambda: xgb_mod._scores(tree, classes, f))

        def tf_feats():
            z = per_frame(kp, True)[np.linspace(0, len(kp) - 1, 32).round().astype(int)]
            return torch.from_numpy(((z - tf_mu) / tf_sd).astype(np.float32))[None].to(dev)
        x, ms["features_tf"] = clock(dev, tf_feats)
        _, ms["net_tf"] = clock(dev, lambda: run(lambda: tf(x), amp))
        _, ms["net_gru"] = clock(dev, lambda: run(lambda: gru(x), amp))
        _, ms["net_lstm"] = clock(dev, lambda: run(lambda: lstm(x), amp))
        xm, ms["features_mlp"] = clock(dev, lambda: torch.from_numpy(((mfn(kp) - m_mu) / m_sd).astype(np.float32))[None].to(dev))
        _, ms["net_mlp"] = clock(dev, lambda: run(lambda: mlp(xm), amp))

        def center32():
            frames, w, h = decode(path, 32)
            return np.stack([rgb(fr, center_box(w, h)) for fr in frames])[None]
        clip, ms["decode_center32"] = clock(dev, center32)
        _, ms["net_r2plus1d"] = clock(dev, lambda: run(lambda: r2p(vb.to_tensor(clip).to(dev)), amp))

        def full16():
            frames, w, h = decode(path, 16)
            return np.stack([rgb(fr, letterbox_box(w, h)) for fr in frames])[None, :, None]
        im, ms["decode_full16"] = clock(dev, full16)
        _, ms["net_resnet_full"] = clock(dev, lambda: run(lambda: res_full(fe.to_tensor(im).to(dev), torch.zeros(1, 16, 0, device=dev)), amp))
        (cr, fl), ms["decode_hands16"] = clock(dev, lambda: sq16(kp))
        _, ms["net_resnet_hands"] = clock(dev, lambda: run(lambda: res_hands(fe.to_tensor(cr[None]).to(dev), torch.from_numpy(fl[None]).to(dev)), amp))

        def conv16():
            frames, w, h = decode(path, 16)
            z = np.stack([cv2.resize(fr, (64, 64)) for fr in frames])[..., ::-1].astype(np.float32) / 255
            return torch.from_numpy(np.ascontiguousarray(z.transpose(0, 3, 1, 2)))[None].to(dev)
        xc, ms["decode_conv16"] = clock(dev, conv16)
        _, ms["net_convnet"] = clock(dev, lambda: run(lambda: conv(xc), amp))
        if j > 0:                                               # clip 0 is a warm-up
            for k, v in ms.items():
                acc[k].append(v)
        if j % 10 == 0:
            print(f"  clip {j}/{len(samples['test'])}", flush=True)

    out = {"n_clips": len(next(iter(acc.values()))), "gpu": torch.cuda.get_device_name(), "trees_kept": trees_kept,
           "params": n_params, "pipelines": {}}
    for name, stages in PIPELINES.items():
        tot = np.sum([acc[s] for s in stages], axis=0)
        out["pipelines"][name] = dict(ms_mean=round(float(tot.mean()), 1), ms_median=round(float(np.median(tot)), 1),
                                      ms_p95=round(float(np.percentile(tot, 95)), 1),
                                      stages_mean={s: round(float(np.mean(acc[s])), 1) for s in stages})
        print(name, out["pipelines"][name], flush=True)
    (ROOT / (sys.argv[1] if len(sys.argv) > 1 else "results/end_to_end_paper.json")).write_text(json.dumps(out, indent=1))


@torch.inference_mode()
def run(fn, amp):
    with amp():
        return fn().float().cpu()


if __name__ == "__main__":
    main()
