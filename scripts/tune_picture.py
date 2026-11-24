"""Tuning budget for the picture models (UPDATED PLAN.md, section 5).

The picture models got less tuning than the skeleton models. This gives the better of the two picture models
(ResNet-18 on hand crops, or I3D on the whole frame, chosen by validation accuracy) a comparable search, then reports the
best setting over 3 seeds. Everything is chosen by VALIDATION accuracy; test accuracy is only reported.

Stage 1  learning rate of the pretrained network in {3e-5, 1e-4, 3e-4}   (60 epochs, no early stopping)
Stage 2  at the best rate: (a) stronger augmentation, (b) crops left black where no hand is found  [ResNet only]
Stage 3  if both (a) and (b) helped: the two combined;  then the best setting trained for 100 epochs
Final    the best setting by validation, seeds 0, 1, 2 (seed 0 is reused from the search)

Each search run uses seed 0, so single-run validation noise (about 4 points on 165 clips) applies to the choices.

    python scripts/tune_picture.py                 # picks the model automatically
    python scripts/tune_picture.py --model resnet  # or i3d
    python scripts/tune_picture.py --dry-run       # show the choice and the first stage, train nothing

Writes results/tuning.json. Restart-safe: finished runs are skipped.
"""

from __future__ import annotations

import argparse
import functools
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from asl_benchmark.models import frame_encoder as fe  # noqa: E402
from asl_benchmark.models import i3d as i3d_mod  # noqa: E402
from asl_benchmark.models import matched as m  # noqa: E402
from asl_benchmark.models.cnn import _topk_accuracy  # noqa: E402

STRONG = dict(crop=0.2, rotate=25.0, scale=0.2, shift=0.1, jitter=0.3)   # vs. the default 0.1 / 15 / 0.1 / 0.05 / none
BASE = dict(lr=1e-4, epochs=60, strong_aug=False, crops="hands")
ORIGINAL_AUGMENT = fe.augment
OUT = ROOT / "results/tuning.json"


def choose_model() -> tuple[str, dict]:
    """The better picture model by validation accuracy: ResNet on hand crops (fine-tuned) or I3D (fine-tuned)."""
    seen = {}
    p = ROOT / "results/matched_ablations.json"
    if p.exists():
        seen["resnet"] = next((r["val_top1"] for r in json.loads(p.read_text()) if r.get("key") == "S5"), None)
    p = ROOT / "results/i3d.json"
    if p.exists():
        seen["i3d"] = json.loads(p.read_text())[0]["val_top1"]
    valid = {k: v for k, v in seen.items() if v is not None}
    if not valid:
        return "resnet", seen
    return max(valid, key=valid.get), seen


def label(cfg: dict) -> str:
    return (f"lr={cfg['lr']:g}, {cfg['epochs']} epochs, {'strong' if cfg['strong_aug'] else 'default'} augmentation"
            + (", crops blanked when missed" if cfg["crops"] == "hands_blank" else ""))


_CACHE: dict = {}


def run_trial(model: str, cfg: dict, seed: int, dev) -> dict:
    """Train once with this setting; returns validation / test accuracy."""
    fe.augment = functools.partial(ORIGINAL_AUGMENT, **STRONG) if cfg["strong_aug"] else ORIGINAL_AUGMENT
    t0 = time.perf_counter()
    if model == "resnet":
        fe.LR_ENC, fe.FT_EPOCHS, fe.FT_PATIENCE = cfg["lr"], cfg["epochs"], cfg["epochs"]   # patience = epochs: no early stop
        if cfg["crops"] not in _CACHE:
            _CACHE[cfg["crops"]] = m.inputs(cfg["crops"])
        row = fe.run_rung(None, (seed,), dev, cfg=dict(name=label(cfg), kind=cfg["crops"], finetune=True),
                          data=_CACHE[cfg["crops"]])
        return dict(val=row["val_top1"], test=row["test_top1"], top5=row["test_top5"], train_s=row["train_s"])
    if "i3d" not in _CACHE:
        samples, frames = m.load_frames()
        _CACHE["i3d"] = fe._by_split(samples, frames)
    d = _CACHE["i3d"]
    (tr_x, tr_y), (va_x, va_y), (te_x, te_y) = d["train"], d["val"], d["test"]
    i3d_mod.LR_BACKBONE, i3d_mod.PATIENCE = cfg["lr"], cfg["epochs"]
    net = i3d_mod.fine_tune((tr_x, tr_y), (va_x, va_y), seed, dev, cfg["epochs"])
    va, te = i3d_mod.predict(net, va_x, dev), i3d_mod.predict(net, te_x, dev)
    out = dict(val=round(float((va.argmax(1) == va_y).mean()) * 100, 1),
               test=round(float((te.argmax(1) == te_y).mean()) * 100, 1),
               top5=round(_topk_accuracy(te, te_y) * 100, 1), train_s=round(time.perf_counter() - t0, 1))
    del net
    torch.cuda.empty_cache()
    return out


def plan_stages(model: str, trials: list[dict]):
    """Generator of the search runs, decided from the validation results so far (`trials` grows while we iterate)."""
    search = lambda: [t for t in trials if t["stage"] != "final"]
    best = lambda ts: max(ts, key=lambda t: t["val"])
    for lr in (3e-5, 1e-4, 3e-4):
        yield {**BASE, "lr": lr}
    s1 = best([t for t in search() if t["stage"] == 1])
    yield {**s1["cfg"], "strong_aug": True}
    if model == "resnet":
        yield {**s1["cfg"], "crops": "hands_blank"}
    cfg, helped = dict(s1["cfg"]), 0
    for t in [t for t in search() if t["stage"] == 2]:
        if t["val"] > s1["val"]:
            changed = {k: v for k, v in t["cfg"].items() if v != s1["cfg"][k]}   # only the field this run changed
            cfg.update(changed)
            helped += 1
    if helped == 2:
        yield cfg                                          # both helped: try them together
    yield {**best(search())["cfg"], "epochs": 100}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", choices=["auto", "resnet", "i3d"], default="auto")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    model, seen = choose_model() if args.model == "auto" else (args.model, {})
    print(f"tuning model: {model}  (validation accuracy of the untuned models: {seen})", flush=True)
    if args.dry_run:
        print("stage 1 runs:", [label({**BASE, "lr": lr}) for lr in (3e-5, 1e-4, 3e-4)])
        return
    torch.backends.cudnn.benchmark = True
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = json.loads(OUT.read_text()) if OUT.exists() else {"model": model, "selection": seen, "trials": [], "final": None}
    if state["model"] != model:
        raise SystemExit(f"{OUT.name} holds a search for {state['model']}, not {model}; move it aside first")
    trials = state["trials"]

    def key_of(cfg):
        return json.dumps(cfg, sort_keys=True)

    gen = plan_stages(model, trials)
    n = 0
    for cfg in gen:                                        # the generator re-reads `trials`, which we extend
        stage = 1 if n < 3 else (2 if n < (4 if model == "i3d" else 5) else 3)
        n += 1
        k = key_of(cfg)
        done = next((t for t in trials if t["cfg_key"] == k and t["seed"] == 0), None)
        if done is None:
            print(f"[search {n}] {label(cfg)}", flush=True)
            res = run_trial(model, cfg, 0, dev)
            done = dict(stage=stage, cfg=cfg, cfg_key=k, seed=0, **res)
            trials.append(done)
            OUT.write_text(json.dumps(state, indent=1))
        print(f"   -> val {done['val']:.1f}  test {done['test']:.1f}  ({done['train_s']:.0f}s)", flush=True)

    top = max([t for t in trials if t["stage"] != "final"], key=lambda t: t["val"])
    print(f"best by validation: {label(top['cfg'])}  (val {top['val']:.1f})", flush=True)
    runs = [top] + [t for t in trials if t.get("final_of") == top["cfg_key"]]
    have = {t["seed"] for t in runs}
    for seed in (1, 2):
        if seed in have:
            continue
        res = run_trial(model, top["cfg"], seed, dev)
        rec = dict(stage="final", cfg=top["cfg"], cfg_key=top["cfg_key"], final_of=top["cfg_key"], seed=seed, **res)
        trials.append(rec)
        runs.append(rec)
        OUT.write_text(json.dumps(state, indent=1))
        print(f"[final] seed {seed}: val {res['val']:.1f}  test {res['test']:.1f}", flush=True)
    tests = np.array([t["test"] for t in runs])
    state["final"] = dict(
        setting=label(top["cfg"]), cfg=top["cfg"], seeds=[t["seed"] for t in runs],
        val_top1=round(float(np.mean([t["val"] for t in runs])), 1), test_top1=round(float(tests.mean()), 1),
        test_std=round(float(tests.std()), 1), test_top5=round(float(np.mean([t["top5"] for t in runs])), 1))
    OUT.write_text(json.dumps(state, indent=1))
    f = state["final"]
    print(f"[TUNED {model}] {f['setting']} | Val: {f['val_top1']}% | Test: {f['test_top1']}% (std {f['test_std']}%) "
          f"| Top-5: {f['test_top5']}%", flush=True)


if __name__ == "__main__":
    main()
