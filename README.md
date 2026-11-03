# ASL Benchmark

A reproducible benchmark for isolated-word American Sign Language recognition. A short video of one sign goes in; a gloss from a fixed vocabulary comes out. The point of the project is the comparison around that prediction: how much accuracy a compact hand-landmark representation gives up, and how much compute it saves, against models that learn from pixels.

The same clip is scored two ways. Landmarks feed four pose models; raw frames feed the appearance model. Both paths are judged on accuracy and on cost.

![Workflow: one RGB clip enters WLASL100 and splits into MediaPipe hand landmarks and raw frames. Google ISLR joins the landmark path only. Landmarks feed gradient-boosted trees, an MLP, a GRU/LSTM, and a small transformer. Raw frames feed an appearance CNN. All five models meet at a shared evaluation of top-1 accuracy, training time, and latency, then ablations that choose the final model.](assets/workflow.jpg)

**CS298A** · Narelle Ho, Cheng Yao Lee

## Setup

The loader reads two datasets, both cached under `data/` (gitignored):

| Source | What `load` fetches | What each sample carries |
| --- | --- | --- |
| `wlasl100` | WLASL100 metadata from the [WLASL](https://github.com/dxli94/WLASL) repo: 2,038 clips, 100 signs, official train/val/test split | A video path plus the frame span. The file itself is downloaded separately. |
| `google_islr` | [Google Isolated Sign Language Recognition](https://www.kaggle.com/competitions/asl-signs) (`asl-signs` on Kaggle): 94,481 sequences, 250 signs, MediaPipe Holistic landmarks | A parquet of 543 landmarks `(x, y, z)` per frame. No video. |

```bash
pip install -r requirements.txt
```

```python
from asl_benchmark import load, load_landmarks, download_videos

wlasl = load("wlasl100")    # metadata only, into data/wlasl100
google = load("google_islr")  # full archive, into data/google_islr

train = wlasl.split("train")  # also "val" and "test"
saved, failed = download_videos(wlasl)  # raw clips; failed is dead urls

sequence = load_landmarks(google.split("train")[0])  # (frames, 543, 3)
```

WLASL video links are a mix of direct files and YouTube. YouTube clips need `yt-dlp` on `PATH`. `download_videos` keeps going when a url is dead and returns those ids in `failed`.

Google ISLR is only published on Kaggle, about 40 GB zipped. Download it with the API. `kaggle` is installed with the requirements above. A Kaggle account is required.

1. Accept the rules at [kaggle.com/competitions/asl-signs/rules](https://www.kaggle.com/competitions/asl-signs/rules) while logged in as the account that will create the token. Until that account has accepted, the download returns 403: "You must accept this competition's rules before you'll be able to download files."
2. On [kaggle.com/settings](https://www.kaggle.com/settings), choose **Create New Token**. Copy the token value. It starts with `KGAT`.
3. Copy the example and put that token in `KAGGLE_API_TOKEN`. `.env` is gitignored. `python3 load_data.py` reads it, so the variables do not need to be exported first.

```bash
cp .env.example .env
python3 load_data.py
```

`load("google_islr")` downloads the archive into `data/google_islr` and unpacks it.

If the zip is already on disk, unzip it so `train.csv`, `sign_to_prediction_index_map.json`, and `train_landmark_files/` sit under `data/google_islr`. The loader then skips Kaggle. The public release has no val/test split. The loader holds out signers with a fixed 80/10/10 cut, so the same participant never appears in two splits. Landmark columns are face, left hand, pose, then right hand. Missing points are NaN.

## Overview

More than 70 million deaf people use a sign language, and most speech and captioning tools still assume a spoken utterance. Isolated-word recognition is a narrow, measurable version of that gap. The signer performs one sign, and the system names it.

The domain is computer vision for human motion. The main line of work is pose-based: each frame is reduced to 3D hand keypoints, and a sequence classifier reads that trajectory. Four classifiers share that input so differences in accuracy come from the model, not from the data. An appearance-based model, trained on the same raw video, is the reference for what the keypoint bottleneck costs.

The deliverable is the comparison itself. Every family is trained and scored on the same splits, with training time and inference latency recorded beside accuracy, and with ablations that justify the final model.

## How a clip is scored

1. **Input.** One RGB video of a single sign.
2. **Representation.** Either cached hand landmarks (pose models) or the frames themselves (appearance model).
3. **Classifier.** One of the five families below.
4. **Output.** One gloss, plus the time that prediction took.

Landmark extraction runs once and is reused. Pose models then train in minutes on CPU. The appearance model is the expensive run and needs a GPU.

## Components

### WLASL100

Primary dataset. 2,038 raw RGB videos, 100 signs. Raw video is what makes the comparison fair: pose models and the appearance model see the same clips. Publicly available.

### Google Isolated Sign Language Recognition

Secondary dataset for pose models only. 94,481 sequences, 250 signs, released as precomputed MediaPipe Holistic landmarks (543 points per frame, each an `x, y, z`). There are no pixels, so it cannot host the appearance comparison. Useful as a larger check that a pose model still holds on a bigger vocabulary. Publicly available.

### Landmark extraction

MediaPipe Hands, applied to WLASL100. 21 keypoints per hand, three coordinates each, so 126 values per frame for two hands. This is a one-time CPU pass (on the order of five hours for the full subset). Every pose model reads the cache, so extraction cost is not paid again at training time.

### Gradient-boosted trees

Pose model. Engineered features summarize the keypoint sequence (the motion collapsed into a fixed vector), and a boosted-tree ensemble classifies that vector. This is the non-sequential baseline. If it matches the sequence models, temporal structure is not earning its complexity.

### MLP

Pose model. A small feed-forward network on the keypoint representation. Same information budget as the trees, with a learned non-linear boundary. It shows whether the gain of a neural net depends on modeling time, or just on a more flexible classifier.

### GRU / LSTM

Pose model. A recurrent network over the per-frame keypoint vector. It keeps a running state across a variable-length clip instead of requiring the sequence to be pooled first. This is the default architecture for short motion trajectories, and the one the other pose models are measured against.

### Small transformer

Pose model. Self-attention over the same frame sequence. Included to test whether attention helps on clips this short, and what that costs in training time and latency relative to the recurrent net.

### Appearance model

A compact convolutional network trained on raw WLASL frames. It never sees the landmarks. A few GPU-hours per run, against minutes for each pose model. Its accuracy is the ceiling the keypoint representation is being judged against: the gap is what the compact input gives up, and the logged timings are what it saves.

### Evaluation

Shared across all five families, on the same splits.

- **Accuracy.** Top-1 gloss accuracy on held-out videos.
- **Cost.** Training time and inference latency for every model, so compute is a result alongside accuracy.
- **Ablations.** On the leading family, variations in input and model capacity that show which choices actually move accuracy. The final model is the one those ablations support.

## What the project accomplishes

A single, rerunnable account of isolated-word ASL recognition under two representations. Pose models answer which sequence classifier is worth using once the hands are already landmarks. The appearance model answers whether that shortcut was expensive in accuracy. Together they pick a model for a reason: measured recognition quality, measured cost, and a short set of ablations that explain the choice.
