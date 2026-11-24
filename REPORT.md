# Report: image or skeleton? What was done and what it shows

Branch `cy-first-draft`. Work by Cheng Yao Lee; the MLP, GRU and LSTM results cited as context are Narelle Ho's.
Written 2026-11-24 after all planned runs had finished. The paper text for the methods, evaluation and ablation
sections is `paper/paper_v2.tex`.

**How to read the scores.** A score is how many of the 100 official test videos the model got right (guessing gives about 1).
"± y" is the spread over 3 random starts (seeds). It does **not** include the uncertainty from having only 100 test videos
(about ±9 points for one score), so models are compared with a paired 95% interval over the same 100 videos (section 3.5).
"Top-5" is how many videos had the right sign among the model's five best guesses.

## 0. Summary

With only about 7 training videos per sign, we compared learning from **pixels** with learning from a **skeleton** (hand
keypoints found by MediaPipe), after making the comparison as fair as the data allowed: the same videos, the same 16 frames,
the same final transformer, and the keypoints found in the very frames the picture models see.

| | Result |
|---|---|
| Main comparison | Keypoint transformer **59.0 ± 2.9** against the best picture model with the same location information **41.7 ± 1.2**: a gap of **+17.3 points (95% interval +9.0 to +25.3)** |
| Is the gap just "knowing where the hand is"? | No, not detectably: giving the crop model the wrist location changed it by +1.3 (−3.7 to +6.3) |
| Upper body | Adds +6.0 (+1.0 to +11.3) to the keypoint model: 65.0 ± 1.6 |
| A video network | I3D (Kinetics) scores 26.3 ± 0.5, about the same as a whole-frame ResNet (24.3 ± 1.9) |
| Movement | Shuffling the 16 frames does not hurt the keypoint models (59.0 to 62.1): they use which hand poses occur, not their order |
| Cost | Keypoints are cheap to **train** (20 s vs. 958 s per seed) but not to **run** from a video file: 624 ms vs. 102 ms for a whole-frame ResNet, because MediaPipe takes 541 ms. The hand-crop ResNet pays the same MediaPipe cost (635 ms). |

Two qualifications run through everything below. First, MediaPipe is itself a model trained on hands and people, so this
compares **pose pipelines with pixel pipelines**, not representations in general. Second, the picture models received no
tuning search (one recipe per group), so "the picture models lose" means "under this recipe at this data size".

## 1. Question and rationale

> With only about 7 training videos per sign, should a sign recognizer learn from pixels or from a skeleton? And where does the difference come from?

**Why this is hard to answer from the literature.** Published comparisons change the data, the preprocessing and the model at
the same time, so a score difference does not say where it comes from. Our first attempt had the same flaw (section 6).

**The information ladder.** Each step changes one thing, on the same videos and the same 16 frames, and the size of each
step says what matters:

```
 whole frame   ->   hand crops   ->   hand crops        ->   hand keypoints   ->   hands + body
 (pixels)           (pixels)          + location (pixels)      + location             keypoints
   S3                 S5                X1                       S1                     X2
      WHERE to look      WHERE the hand is     WHAT: pixels vs. skeleton    BODY context
```

All five end in **the same small transformer**, so only what is fed in changes. S1 against X1 is the cleanest pose-versus-image
step because both receive the hand position. Two more checks: I3D as the strongest picture baseline, and shuffled frames to
see whether movement is used.

## 2. Data

- The 1,013 of the 2,038 WLASL100 videos that could be downloaded, with the official split restricted to them: **748 train / 165 validation / 100 test** (about 7.5 training videos per sign).
- Median video: 656×400 pixels, 65 frames; 92% have at least 32 frames; 24% end on a black frame, which every model sees.
- The 100 test videos cover **72 of the 100 signs**, with at most 3 videos per sign. Of 34 test signers, 31 also appear in training (97 of the 100 videos).
- **One duplicate:** test video 24960 ("go") is an exact copy of training video 24961 (identical frames). Every test video was compared with all training and validation videos by frame content; no other same-sign duplicate exists. The official test set was kept, so this can raise a score by at most one point.
- MediaPipe Hands finds two hands in 23% of the 16 sampled frames, one in 41%, none in 36% (many frames are rests between signs). MediaPipe Pose finds a person in 98.5% of frames and in every video.
- The median hand is only **23.6 pixels** on its larger side in the 224×224 frame (10th to 90th percentile: 16 to 34).

## 3. Methods

### 3.1 Shared input
Each video is reduced once to **16 evenly spaced frames**, resized to 224×224 with black bars to keep the shape. Both groups use exactly these frames (`frames_kp16x224.npz`, 1013×16×224×224×3).

- **Pixels:** the 224×224 frame, or two 112×112 crops per frame (one per hand): a square of 1.6 times the hand's bounding box (at least 32 pixels), cut from the 224 frame. The crops are raw pixels; nothing is drawn on them.
- **Keypoints:** MediaPipe Hands is run on **each of the 16 frames separately, with no tracking** and no neighbouring frames, so the keypoints cannot contain anything the picture model cannot see.

*Why 224 pixels.* On a 50-video test, MediaPipe on a 224 frame found 0.92 hands per frame, the same as the old full-resolution tracked extraction (0.92), while at 112 pixels it dropped to 0.78. Over all videos, the per-hand detection rates of the matched keypoints (0.30 and 0.57) are within 0.02 of the tracked ones (0.31 and 0.58). So nothing is lost on the keypoint side.

*Why one rule for choosing frames.* The first version chose frames by the video header's frame count for pixels and by keypoint rows for skeletons; the header over-counts by 1 to 3 frames on 178 videos. Both now use the keypoint array's length.

### 3.2 Keypoint models
Each frame becomes **132 numbers**: for each hand the 21 points relative to the wrist and divided by palm length (126), a "hand found" flag per hand (2), and the position of each wrist in the frame (4). A missing hand is zero with its flag set. In the upper-body model (X2) we append 9 points from MediaPipe Pose (nose, shoulders, elbows, wrists, hips) relative to the shoulder midpoint and divided by shoulder width (18 numbers) and a "body found" flag. Features are standardised with training statistics.

The **small transformer**: one layer, width 128, 4 heads, feed-forward width 256, dropout 0.3, learned position embeddings over the 16 frames, mean over frames, linear layer to 100 classes. **165 thousand parameters**, trained from scratch.

### 3.3 Picture models
- **ResNet-18** pretrained on ImageNet turns each frame (or each hand crop, same weights for both hands) into 512 numbers; the 16 vectors (plus hand-found flags) go through the **same** transformer as above. Fine-tuned end to end (11.4 to 11.5 million parameters).
- **X1** gives the crop model the same four wrist-position numbers the keypoint model has: standardised, zero when the hand is missing, and given the same random rotation, zoom and shift as the keypoints, so the only remaining difference is pixels versus keypoints.
- **I3D** (the WLASL authors' implementation and Kinetics weights, vendored in `third_party/pytorch_i3d/`, Apache 2.0) fine-tuned on the 16 whole frames (12.4M parameters). Its code is from https://github.com/piergiaj/pytorch-i3d; the 50 MB weights are fetched with `scripts/fetch_i3d.sh` and are not in the repository.

### 3.4 Training recipe
All models: label smoothing 0.1, AdamW with weight decay 0.01, gradient clipping 1.0, the best epoch chosen on **validation** accuracy, seeds 0, 1, 2.
Keypoint models: learning rate 1e-3, batch 64, up to 200 epochs, patience 40. Fine-tuned picture models: 1e-4 for the pretrained network and 1e-3 for the new layers, cosine schedule, mixed precision, batch 8, up to 40 epochs, patience 10. Augmentation is one random transform per video (trim up to 10% of frames at each end, rotation ±15°, zoom ±10%, shift ±5%, small noise); keypoints get the same transform in coordinates. **No horizontal flip** (it would swap the signing hand). Hyper-parameters were not searched.

*Rationale for no search.* The aim is a comparison, not the best score, and the GPU time was limited (one RTX 3060). A tuning experiment was designed and written (`scripts/tune_picture.py`) but cancelled before it ran; the consequence is stated in the limits.

### 3.5 Statistics
For each model, the score of a test video is the share of its seeds that got it right. A comparison is the mean difference over the same 100 videos, with a **paired bootstrap** 95% interval over videos (10,000 draws; the random stream depends only on the comparison's name, so adding rows never changes the others). **Three primary comparisons were fixed before the new runs finished**: S1 − X1, X1 − S5, X2 − S1. Everything else is exploratory. An interval containing 0 means "not detectable at this sample size", not "equal".

### 3.6 End-to-end speed
For each of the 100 test videos, everything from the video file to the answer is timed, one video at a time with nothing cached: reading and resizing the frames, MediaPipe where the pipeline needs it, cropping, and the network. MediaPipe runs on the CPU (AMD Ryzen 5 5500), the networks on the GPU (RTX 3060). Latency does not depend on trained weights, so untrained copies of the same networks were timed; the first video is a warm-up (99 timed). The machine was idle.

### 3.7 Frame-order test
The trained model is evaluated on test videos whose 16 frames are shuffled (3 fixed permutations, the same for all models, averaged); it is not retrained. This is one-sided: shuffled videos differ from the training data, so a drop would be hard to interpret, and no drop does not show that order carries no information for a model trained to use it.

## 4. Results

All numbers: matched protocol (16 frames of 224×224), 3 seeds, test videos, unless a section says otherwise.

### 4.1 Main ladder

| Model | What it reads | Test | Top-5 | Parameters | Training per seed | Video to answer |
|---|---|---|---|---|---|---|
| ResNet, whole frame (S3) | the whole frame | 24.3 ± 1.9 | 54.0 | 11.4M | 2058 s | 102 ms |
| I3D | 16 whole frames | 26.3 ± 0.5 | 58.7 | 12.4M | 1371 s | 101 ms |
| ResNet, hand crops (S5) | two hand crops | 40.3 ± 2.1 | 68.7 | 11.5M | 942 s | 635 ms |
| ResNet, hand crops + location (X1) | two crops + wrist position | 41.7 ± 1.2 | 71.7 | 11.5M | 958 s | 635 ms |
| **Keypoint transformer (S1)** | hand keypoints + wrist position | **59.0 ± 2.9** | 83.7 | 0.16M | 20 s | 624 ms |
| Keypoint transformer + upper body (X2) | hands + 9 body points | 65.0 ± 1.6 | 85.0 | 0.17M | 23 s | 1356 ms |
| *(supplement)* tracked full-resolution keypoints | hands, tracked through the full-resolution video | 67.0 ± 2.4 | 88.3 | 0.16M | 37 s | 2123 ms |

### 4.2 Paired comparisons (difference in points, 95% interval)

| Comparison | Difference | Interval |
|---|---|---|
| **Keypoints vs. crops with the same location** (S1 − X1), primary | **+17.3** | +9.0 to +25.3 |
| **Value of location for the crop model** (X1 − S5), primary | +1.3 | −3.7 to +6.3 |
| **Value of the upper body for keypoints** (X2 − S1), primary | +6.0 | +1.0 to +11.3 |
| Keypoints vs. crops without location (S1 − S5) | +18.7 | +9.7 to +27.3 |
| Keypoints vs. whole frame (S1 − S3) | +34.7 | +25.0 to +44.0 |
| Crops vs. whole frame (S5 − S3) | +16.0 | +9.0 to +23.3 |
| Keypoints vs. I3D (S1 − I3D) | +32.7 | +22.3 to +43.0 |
| Keypoints + body vs. crops + location (X2 − X1) | +23.3 | +15.3 to +31.3 |
| Value of location for keypoints (S1 − no-location model) | +2.0 | −3.0 to +7.0 |
| Tracking and full resolution (tracked − S1) | +8.0 | +2.0 to +14.3 |
| Removing depth (no-depth − S1) | +3.3 | −1.0 to +7.7 |
| Black crop vs. borrowed box when a hand is missed, frozen | +8.3 | +2.0 to +15.0 |
| Fine-tuning, whole frame | +8.7 | +2.0 to +15.3 |
| Fine-tuning, hand crops | +24.7 | +17.0 to +32.3 |

### 4.3 Cost: training and end to end

Time per video in milliseconds (mean over 99 test videos):

| Pipeline | Test | Decode | MediaPipe | Network | Total |
|---|---|---|---|---|---|
| Whole frame, ResNet | 24.3 | 82 | – | 21 | 102 |
| I3D | 26.3 | 82 | – | 19 | 101 |
| Hand crops + location, ResNet | 41.7 | 82 | 541 | 12 | 635 |
| Keypoints, hands | 59.0 | 82 | 541 | 1 | 624 |
| Keypoints, hands + body | 65.0 | 82 | 1272 (Hands 541 + Pose 731) | 1 | 1356 |
| Keypoints, tracked full resolution | 67.0 | – | 2122 (one pass over every frame) | 1 | 2123 |

The network is never the cost: 1.3 ms for the keypoint transformer, 10.5 ms for the crop ResNet, 20.5 ms for the whole-frame ResNet. MediaPipe is (a Python CPU implementation at batch size one; an optimised or GPU deployment would change the absolute numbers).

### 4.4 Ablations

**What the keypoint model is given** (matched, 3 seeds; difference from the base 59.0):

| Variant | Test | Difference | 95% interval |
|---|---|---|---|
| Base: hands + wrist location | 59.0 ± 2.9 | – | – |
| No wrist location | 57.0 ± 2.9 | −2.0 | −7.0 to +3.0 |
| No depth coordinate | 62.3 ± 0.9 | +3.3 | −1.0 to +7.7 |
| + upper-body points | 65.0 ± 1.6 | +6.0 | +1.0 to +11.3 |
| Tracked, full-resolution keypoints | 67.0 ± 2.4 | +8.0 | +2.0 to +14.3 |
| Graph network instead of the transformer | 26.7 ± 1.2 | – | – |

Only the upper body gives a detectable gain among the inputs. Location and depth are within the interval. Tracking and full resolution are worth about 8 points to the keypoint side, so a comparison that gives the skeleton full-resolution tracked video and the images a shrunken frame overstates the skeleton's lead by about that much.

**Frame count:** 32 frames instead of 16 does not change the tracked-keypoint transformer (66.7 ± 1.9 vs. 67.0 ± 2.4; same configuration, earlier run).

**Graph network** in the style of Pose-TGCN (Li et al., 2020), twelve settings (matched or tracked keypoints × raw coordinates or our normalised features × 2, 4 or 8 blocks). Our normalised features beat the raw coordinates the original used at every depth. Tracked keypoints beat matched ones by 11 to 17 points (the transformer gained 8). More blocks do not help consistently, and some settings are unstable across seeds (standard deviations up to 11 points). The settings chosen on validation: matched, normalised, 2 blocks (26.7 ± 1.2); tracked, normalised, 8 blocks (39.3 ± 4.0). Full grid: `results/tgcn.json` and the appendix of `paper/paper_v2.tex`.

**Choices in the picture models:**

| Variant (ResNet-18) | Test | Top-5 |
|---|---|---|
| Whole frame, frozen | 15.7 ± 1.7 | 30.7 |
| Whole frame, fine-tuned | 24.3 ± 1.9 | 54.0 |
| Hand crops, frozen, box borrowed from a nearby frame when the hand is missed | 15.7 ± 0.5 | 39.0 |
| Hand crops, frozen, black when missed | 24.0 ± 5.1 | 48.0 |
| Hand crops, fine-tuned | 40.3 ± 2.1 | 68.7 |
| Hand crops + wrist location, fine-tuned | 41.7 ± 1.2 | 71.7 |

Fine-tuning matters most (+8.7 for the whole frame, +24.7 for crops). A borrowed box shows whatever lies where the hand used to be, so blanking it helps a frozen network (+8.3). Wrist location does not detectably help the crop model.

**Is movement used?** Test accuracy with the 16 frames in order, then shuffled:

| Model | In order | Shuffled | Change |
|---|---|---|---|
| Keypoints, hands + location | 59.0 | 62.1 | +3.1 |
| Keypoints, no wrist location | 57.0 | 56.8 | −0.2 |
| Keypoints, hands + upper body | 65.0 | 65.1 | +0.1 |
| ResNet crops + location | 41.7 | 39.9 | −1.8 |

None of the models loses accuracy that exceeds the noise (see 3.7 for what this does and does not show).

**Earlier protocol and other model types** (tracked full-resolution keypoints, 32 frames; or 112-pixel frames; kept apart and never mixed with the matched numbers). Keypoints: XGBoost 39.3 ± 1.2; MLP 60.0, GRU 60.7, LSTM 63.3 (Narelle's, each the mean of three seeds, per-video predictions not in this repository); transformer 66.7 ± 1.9 (selected on validation), 68.0 ± 2.4 base without augmentation, 70.0 ± 2.2 base with augmentation. Pixels: ResNet hand crops cut from the full-resolution video 47.0 ± 3.7; ResNet whole frame 15.7 ± 1.2; R(2+1)D whole frame 38.7 ± 3.1; R(2+1)D one crop per video 34.3 ± 3.4; frozen R(2+1)D 10.7 to 12.3 whatever the frames (8 or 16) or input; a small ConvNet from scratch 1.0 to 1.7 (chance is 1). R(2+1)D crop with a longer schedule and three learning rates (3e-5, 1e-4, 3e-4, one seed each): 27, 46 and 36.

*Design choices of the earlier keypoint models* (test, before → after adding the choice): augmentation helps flat-vector models most (linear 18.7 → 36.0, MLP 30.3 → 44.0, XGBoost 37.3 → 45.7) and the transformer least (68.0 → 70.0); temporal aggregation is the largest gain for the MLP (44.0 → 60.0), linear model (36.0 → 52.7), LSTM (51.0 → 63.3) and GRU (53.3 → 60.7), but costs XGBoost (45.7 → 41.3) and the transformer (70.0 → 66.7, where validation rises while test falls); wrist location adds 2 to 6 points for every model tested both ways.

### 4.5 Where the models differ
Counting a video as correct if a model gets it right in at least two of its three seeds: the keypoint model is right on 60 of the 100 videos, the crop ResNet with location on 41. They share 36. The keypoint model gets 24 videos the crop model misses, the crop model gets 5 the keypoint model misses, and 35 videos neither gets. So the crop model's correct videos are nearly a subset of the keypoint model's. The same pattern holds against the whole-frame ResNet (44 only the keypoint model, 4 only the ResNet), I3D (43 and 5), and, for the upper-body model against the crop model, 29 and 3. With at most three videos per sign, per-sign results would not be reliable and are not reported.

## 5. Analysis

**What the results support**
1. **At matched input, keypoints beat pixels at this data size.** The gap to the best picture model is +17.3 (9.0 to 25.3), and 35 points to the whole-frame model and to I3D. The intervals exclude zero by a wide margin.
2. **Knowing where the hand is does not explain the gap.** Cutting out the hands helps a lot (+16.0 over the whole frame), but giving the crop model the same wrist position as the keypoint model adds only 1.3 (−3.7 to +6.3). This is the main change from our first draft, which said localization explained little: it did, but through cropping, not through the location numbers.
3. **The upper body carries information that hands alone leave unused** (+6.0, 1.0 to 11.3). Many signs depend on where on the body the hand is.
4. **The keypoint models use hand poses, not their order.** Shuffling frames does not hurt them. This is consistent with the aggregation results of the earlier protocol (mean pooling mattered most).
5. **The skeleton's advantage in cost is in training, not in running.** 20 s against 958 s per seed, but 624 ms against 102 ms from a video file, because MediaPipe takes 541 ms. A picture pipeline that needs no detector is six times faster end to end and 35 points less accurate. A picture pipeline that crops with the detector (the hand-crop ResNet) pays the same MediaPipe cost and is 17 points less accurate.
6. **Pixel models stay far below; a video network does not rescue them** (I3D 26.3). Pretraining is essential (a small ConvNet from scratch stays at chance), and fine-tuning adds the most (+8.7 for the whole frame, +24.7 for crops).

**What the results do not show**
- *Why* picture models lose. Candidates: too little data, too little tuning, too little resolution (the hand is about 24 pixels wide), only 16 frames, the pretraining of the detector. We did not isolate these. The evidence on tuning is mixed: an R(2+1)D crop model reached 27, 46 or 36 depending only on the learning rate (one seed each), so tuning can move a video network by more than most gaps between rows here; but the same recipe was applied to every picture model, none got a search, and none came close to the keypoint model. Learning curves, which would separate "more data" from "more tuning", were cut.
- Anything about **representations in general**. MediaPipe was trained for hands and people; the picture networks come from everyday photos (ImageNet) or human actions (Kinetics). The comparison is between pipelines.
- Whether location helps. Neither the crop model (+1.3) nor the keypoint model (+2.0) shows a detectable effect; with 100 videos we cannot rule out a small one. (In the earlier protocol, removing location cost the transformer 4.3 points.)
- Anything about **new signers**: 97 of the 100 test videos are by signers seen in training.

**Alternative explanations to keep in mind.** The keypoint and crop models share detector failures (a missed hand gives both nothing); the whole-frame models do not depend on the Hands detector. The keypoint features were designed (wrist-centred, palm-scaled) by us, with some effort; the pixel side used a standard recipe. The duplicate test video can account for at most one point.

## 6. How the design changed (process, corrections)

**First design** (the progress report): hand landmarks from full-resolution tracked video against a frozen or fine-tuned R(2+1)D on 112×112 frames. It said landmarks beat pixels by about 30 points. Problems found in review:
- The skeleton saw **more** than the picture models: full resolution, and tracking across frames. We measured this advantage at about 8 points (tracked 67.0 vs. matched 59.0).
- Frames for pixels and keypoints were chosen by different rules (178 videos differed by 1 to 3 frames).
- Hand crops were placed by the tracked keypoints, so the pixel side borrowed skeleton output; and crops got no location, while the skeleton had it.
- The pixel models had a different recipe (frozen networks without augmentation, different epochs and patience).
→ We built the **matched protocol** (section 3.1): both groups start from the same 16 shared 224-pixel frames, with static detection.

**Preliminary matched results** (hands only, no location for pixels) gave a keypoint lead of about 19 points. In review, several claims from that stage were **withdrawn**:
- "A keypoint drawing fed to a ResNet shows the network is the problem" (the drawing was unreadable: hands about 12 pixels wide, joints drawn as 5-pixel dots).
- "Joining pixel and keypoint features makes things worse" (only one poor way of joining, 1,024 frozen picture numbers next to 132 keypoint numbers, was tested).
- "Cropping adds 8 (frozen) and 30 (fine-tuned) points": the first is a different protocol; in the matched protocol borrowed-box crops add nothing when frozen.
- "Differences under 5 points are noise": the paired intervals are wider (about ±4 to ±9).
- "0.92 hands per frame in both" (measured on a 50-video sample; per-hand rates over all videos agree within 0.02).
- "If tuned accuracy stays at 40 to 45, it is not a tuning problem": the search was too narrow to say so.

**Plan cuts for the first draft.** A larger design (upper-body crop from original resolution, 32 frames, equal tuning search, signer-grouped cross-validation, sign-type labels, two-stream and stronger pretrained picture models, learning curves, late fusion, face keypoints) was reduced to the ladder, because the GPU budget allowed about one hour of new training. The cut items are listed in section 8.

**Corrections made while writing up**
- A statement that Narelle's MLP, GRU and LSTM were "one training each" was wrong; the progress report says mean over three seeds. Fixed in the draft; her per-video predictions are not in this repository, so her models are not in the paired comparisons.
- Bootstrap intervals shifted by a few tenths when a comparison was added (shared random stream); each interval is now seeded by its own name.
- A bug in the x/y-only keypoint run (zeroed depth perturbed by augmentation, then divided by a near-zero standard deviation) collapsed it to 3%; fixed by zeroing the depth features after augmentation.
- The test-set duplicate (section 2) was found by checking, not assumed.
- Two figures had overlapping labels and were redrawn.
- A machine reboot during the work lost about 3.5 hours of GPU time; the queue scripts are restart-safe and none of the reported numbers were affected.

## 7. Limits

- One recipe per group and **no tuning search**; the picture side received less tuning than the keypoint side over the course of the work (the skeleton design went through about a dozen validated variants in the earlier protocol).
- 16 frames of 224×224 (the WLASL authors used 64); hands about 24 pixels wide; I3D at 16 frames.
- Hands, and in one model the upper body, but **no face**; mouth movements can matter in ASL.
- Half of WLASL100; a 100-video test set covering 72 signs with 97 of its videos by seen signers; one duplicate test video.
- MediaPipe is trained on hands and people; the picture networks are not.
- Epoch selection used the official validation videos; no signer-grouped cross-validation was run.
- The graph network is "in the style of" Pose-TGCN: 42 hand points (the original used 13 body points plus both hands), 16 frames (original: 50), our own width and depth.
- Speed numbers are for a Python CPU implementation of MediaPipe at batch size one, with untrained copies of the networks.

## 8. What was not done

Cut for the GPU budget (full design kept in the plan files, which are not committed): equal tuning search for both groups; upper-body crop from the original resolution with 32 frames; R(2+1)D and two-stream picture models retrained; a stronger or hand-pretrained image encoder; learning curves; late fusion of probabilities; signer-grouped cross-validation and per-sign-type results; face keypoints (the Holistic extraction script does not run as it stands); retraining the frame-order test with shuffled training data.
Not finished: the abstract, introduction, related work and conclusion of the paper; the author email; verification of six references added from memory; compiling the PDF and checking the 8-page limit (LaTeX is not installed on this machine); the peer-review write-up.

## 9. Reproducing and files

**Hardware:** one RTX 3060 (12 GB), AMD Ryzen 5 5500, 14 GB RAM. All training runs are restart-safe queue scripts.

| What | Where |
|---|---|
| Shared frames, static keypoints, matched pixel and keypoint runs | `asl_benchmark/models/matched.py` |
| New runs: upper body, no location, hand crops + location, frame-order test | `asl_benchmark/models/ladder.py` |
| Graph network | `asl_benchmark/models/tgcn.py` |
| I3D | `asl_benchmark/models/i3d.py`, `third_party/pytorch_i3d/`, `scripts/fetch_i3d.sh` |
| Paired statistics, figures, speed test, summary table | `scripts/paired_stats.py`, `scripts/v2_figures.py`, `scripts/end_to_end_latency.py`, `scripts/matched_summary.py` |
| Queue scripts | `scripts/run_matched_queue.sh`, `run_baselines_queue.sh`, `run_v2_queue.sh` (and the cancelled `run_curves_queue.sh`, `run_tuning_queue.sh`, `run_latency_queue.sh`, `run_shared_frames_queue.sh`) |
| Results | `results/*.json` (every run stores its per-video test predictions); `results/v2_stats.json` has the paired intervals |
| Paper text and figures | `paper/paper_v2.tex`, `paper/figs/`, `paper/tables/paired.tex`, `paper/refs.bib` |

Typical order: `python -m asl_benchmark.models.matched --prepare`, `python -m asl_benchmark.models.ladder --prepare`, the queue scripts, then `python scripts/paired_stats.py` and `python scripts/v2_figures.py`. Data, caches and logs are not in the repository.

### Key to the labels used in the result files

| Plain name | Label |
|---|---|
| ResNet whole frame, frozen / fine-tuned | S2 / S3 |
| ResNet hand crops (borrowed box), frozen / fine-tuned | S4 / S5 |
| ResNet hand crops left black, frozen | C3 |
| Keypoint transformer, hands + location | S1 (rerun with the shuffle test: S1x) |
| Keypoint transformer, tracked full resolution / no depth | C1 / C5 |
| Keypoint transformer, no wrist location | X3 |
| Keypoint transformer + upper body | X2 |
| ResNet hand crops + wrist location | X1 |
| Keypoint drawing, frozen / fine-tuned; fusion (withdrawn controls) | C2a / C2b; C4 |
