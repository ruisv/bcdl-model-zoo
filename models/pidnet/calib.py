#!/usr/bin/env python3
"""[2/3] Calibration set for PIDNet-S — and the reason the `_v3` build exists.

THIS IS THE TRAP THIS MODEL IS NAMED AFTER. The calibration arrays MUST be
pre-normalised, and the only way to guarantee it is to route them through
common/calib_pack.pack(), which applies the exact normalisation config.yaml
declares (data_mean_and_scale, ImageNet mean/std).

Why it matters, restated so this file stands alone: when `cal_data_type` is
float32, the compiler's `norm_type` / `mean_value` / `scale_value` are applied to
the RUNTIME input path only — they are NOT applied to the calibration data. Feed
raw 0-255 pixels to a config that declares a mean and scale and the activation
statistics are gathered on a distribution the deployed model never sees. The
input thresholds come out wrong, and the model compiles WITHOUT A SINGLE WARNING,
loads, runs at full speed, and decodes to garbage. Earlier PIDNet builds did
exactly this; `_v3` is the one calibrated on properly pre-normalised data.

pack() takes raw 0-255 RGB CHW and normalises it here, so this script must NOT
normalise a second time — it hands pack() raw pixels and lets the one entry point
do the maths. That is the whole defence: there is no second way to write
calibration data.

Cityscapes val frames are the natural calibration domain (urban street scenes at
2048x1024). Any road-scene set works; domain match matters more than count.

Usage:
    python calib.py --config config.yaml --images /path/to/cityscapes_frames --limit 64
"""

import argparse
import glob
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "common"))
from calib_pack import load_config, pack  # noqa: E402

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--images", required=True, help="directory or glob of street frames")
    ap.add_argument("--limit", type=int, default=64)
    args = ap.parse_args()

    cfg = load_config(args.config)
    if len(cfg.inputs) != 1:
        sys.exit(f"{args.config}: expected a single input, got {len(cfg.inputs)}")
    spec = cfg.inputs[0]
    H, W = spec.h or 1024, spec.w or 2048

    pat = args.images if glob.has_magic(args.images) else os.path.join(args.images, "*")
    files = [f for f in sorted(glob.glob(pat))
             if os.path.splitext(f)[1].lower() in IMAGE_EXTS][: args.limit]
    if not files:
        sys.exit(f"no images under {args.images!r}")

    arrays, sources = [], []
    for i, f in enumerate(files):
        img = cv2.imread(f, cv2.IMREAD_COLOR)
        if img is None:
            continue
        # Plain resize to the model resolution: Cityscapes is already 2:1, so no
        # letterbox is needed; the runtime feeds a full-frame 2048x1024.
        img = cv2.resize(img, (W, H), interpolation=cv2.INTER_LINEAR)
        rgb = img[:, :, ::-1]  # BGR -> RGB; config declares rgb, hands raw 0-255
        arrays.append(np.ascontiguousarray(rgb.transpose(2, 0, 1)).astype(np.float32))
        sources.append({"index": i, "source": os.path.basename(f)})

    print(f"[calib] {len(arrays)} street frames, {H}x{W}, pre-normalised via calib_pack")
    pack(spec, arrays, sources=sources)


if __name__ == "__main__":
    main()
