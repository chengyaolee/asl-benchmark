#!/usr/bin/env bash
# Download the Kinetics-pretrained I3D RGB weights used by the WLASL baseline (50 MB, not committed to git).
cd "$(dirname "$0")/.."
curl -fL -m 600 -o third_party/pytorch_i3d/rgb_imagenet.pt \
    https://raw.githubusercontent.com/piergiaj/pytorch-i3d/master/models/rgb_imagenet.pt
