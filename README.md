# ASL Benchmark

A reproducible benchmark for isolated-word American Sign Language recognition on WLASL100. A short video of one sign goes in; a gloss from a fixed 100-sign vocabulary comes out. The point of the project is the comparison around that prediction: how much accuracy a compact MediaPipe hand-landmark representation gives up, and how much compute it saves, against models that learn directly from pixels.

The same clip is scored two ways. Landmarks feed four pose models; raw frames feed the appearance models. Both paths are judged on accuracy, training time, and inference latency.

![Workflow: one RGB clip enters WLASL100 and splits into MediaPipe hand landmarks and raw frames. Landmarks feed gradient-boosted trees, an MLP, a GRU/LSTM, and a small transformer. Raw frames feed appearance-based models. All models meet at a shared evaluation of top-1 accuracy, training time, and latency, then ablations that choose the final model.](assets/workflow.jpg)

**CS289A** · Narelle Ho, Cheng Yao Lee

## Setup

The loader reads WLASL100, cached under `data/wlasl100/` (gitignored):

| Source | What `load` fetches | What each sample carries |
| --- | --- | --- |
| `wlasl100` | WLASL100 metadata from the [WLASL](https://github.com/dxli94/WLASL) repo: 2,038 clips, 100 signs, official train/val/test split | A video path plus the frame span. Video files are downloaded separately. |

```bash
pip install -r requirements.txt
```

```python
from asl_benchmark import load, download_videos

wlasl = load("wlasl100")    # metadata into data/wlasl100

train = wlasl.split("train")  # also "val" and "test"
saved, failed = download_videos(wlasl)  # raw clips; failed is dead urls
```

Many original WLASL source links have rotted, so `load_data.py` fetches the preprocessed video archive (`risangbaskoro/wlasl-processed`) from Kaggle (1,013 of the 2,038 WLASL100 clips; 748 train / 165 val / 100 test):

1. On [kaggle.com/settings](https://www.kaggle.com/settings), create an API token (`~/.kaggle/kaggle.json` or `KAGGLE_API_TOKEN` in `.env`).
2. Run the download and keypoint extraction scripts:

```bash
cp .env.example .env
python3 load_data.py
python3 scripts/extract.py
```

## Overview

More than 70 million deaf people use a sign language, and most speech and captioning tools still assume a spoken utterance. Isolated-word recognition is a narrow, measurable version of that gap. The signer performs one sign, and the system names it.

The domain is computer vision for human motion. We compare two paradigms on the exact same video clips:
1. **Pose-based (learning from MediaPipe landmarks):** Each frame is reduced to 3D hand keypoints, and a classifier reads that trajectory. Four model families (gradient-boosted trees, MLP, GRU/LSTM, and Transformer) share that input so differences in accuracy come from the model architecture and temporal aggregation, not from the data.
2. **Appearance-based (learning from pixels):** Convolutional and video backbones trained directly on raw RGB frames, evaluating what the keypoint bottleneck gains or loses in accuracy, sample efficiency, signer invariance, and compute cost.

The deliverable is the comparison itself. Every family is trained and scored on the same splits, with training time and inference latency recorded beside accuracy, and with ablations that justify the final models.

## How a clip is scored

1. **Input.** One RGB video of a single sign.
2. **Representation.** Either cached hand landmarks (pose models) or the RGB frames themselves (appearance models).
3. **Classifier.** One of the model families below.
4. **Output.** Top-1 and top-5 gloss predictions, plus training time and per-clip inference latency.

Landmark extraction runs once and is reused. Pose models then train in under a minute on CPU. Appearance models learn directly from pixels and trade higher compute cost for end-to-end visual representation learning.

## Components

### WLASL100

2,038 raw RGB videos across 100 signs (1,013 available clips in the preprocessed archive: 748 train, 165 val, 100 test). Having raw video for every sample makes the landmark-vs-pixel comparison exact: pose models and appearance models see the exact same clips.

### Landmark extraction

MediaPipe Hands, applied to WLASL100 (`scripts/extract.py`). 21 keypoints per hand, three coordinates each (`(frames, 42, 3)`), normalized per hand by wrist position and palm size with missing-hand indicators and centered 2D wrist coordinates (`scripts/features.py`, 132 values per frame). This is a one-time CPU pass; every pose model reads the cached `.npy` files.

### Gradient-boosted trees

Pose model (`asl_benchmark/models/xgboost.py`). Mean and standard deviation summarize the 32-frame keypoint sequence into a 264-dimensional vector, and a regularized XGBoost ensemble classifies that vector. This is the non-neural tabular baseline.

### MLP

Pose model (`asl_benchmark/models/mlp.py`). A feed-forward network on the keypoint representation across an ablation ladder from flattened 32-frame sequences to temporally pooled (`mean, std`) features. It tests how much of neural network performance comes from non-linear feature combinations versus explicit temporal sequence modeling.

### GRU / LSTM

Pose model (`asl_benchmark/models/rnn.py`). Recurrent networks over the 32-frame per-frame keypoint sequence, ablated across unidirectionality vs. bidirectionality, final hidden state vs. temporal mean readout, and depth.

### Small transformer

Pose model (`asl_benchmark/models/transformer.py`). Multi-head self-attention over the 32-frame keypoint sequence with LayerNorm and learned positional embeddings, testing whether self-attention captures temporal alignment better than recurrent or pooled baselines.

### Appearance models (learning from pixels)

Pixel-based models (`asl_benchmark/models/cnn.py`) trained directly on WLASL100 video frames without access to landmarks:
- **From-scratch 2D ConvNet baseline:** 3-layer ConvNet on $8 \times 64 \times 64$ and $16 \times 64 \times 64$ clips with temporal pooling.
- **Pretrained 2D & 3D video backbones:** Fine-tuning pretrained spatial/video backbones (frozen vs. end-to-end fine-tuned, frame sampling, and hand-cropped ROI vs. full-frame pixels) to measure the true ceiling and compute cost of learning directly from pixels in a low-data regime (7.5 training clips per sign).

### Evaluation

Shared across all families on the same WLASL100 splits (`asl_benchmark/compare.py`, `scripts/run_ablations.py`):

- **Accuracy.** Validation top-1, test top-1, and test top-5 gloss accuracy (mean and std over 3 seeds).
- **Cost.** Training time and per-clip inference latency for every model.
- **Ablations & feature analysis.** Feature separability probes, signer-leakage analysis (`scripts/analyze_features.py`), and cross-model ablation ladders over feature groups, augmentation, and temporal pooling.

## What the project accomplishes

A single, rerunnable benchmark of isolated-word ASL recognition comparing learning from MediaPipe hand landmarks against learning directly from video pixels. Pose models answer which classifier and temporal aggregation work best once hands are extracted as skeletal keypoints; pixel models answer how end-to-end visual learning compares in accuracy, overfitting under small sample sizes, and compute cost.
