#!/usr/bin/env python3
"""[2/3] PTQ calibration data for PP-OCRv6 detection and recognition.

A note on where normalisation lives, because this model is the exception.

These models are compiled with `input_type_rt: featuremap` and
`norm_type: no_preprocess`, matching PP-OCRv5 so the runtime's existing OCR
path works unchanged: the BPU takes an already-normalised float tensor and the
CPU side does the scaling. That means `common/calib_pack.py` cannot enforce the
normalisation from config.yaml the way it does elsewhere — with
`no_preprocess`, its normalize() is a pass-through by definition.

So for these two models the contract is between THIS FILE and the runtime's
preprocessing, not between the yaml and the compiler. The constants below must
match what the runtime feeds at inference time. They are PaddleOCR's own
preprocessing, read off the `inference.yml` shipped with the weights:

    det : (x/255 - [0.485,0.456,0.406]) / [0.229,0.224,0.225]   (ImageNet)
    rec : (x/255 - 0.5) / 0.5

Writing still goes through calib_pack.pack() so the file naming, dtype and
manifest stay identical to every other model here.

Usage:
    python calib.py --config config_det.yaml --task det --images <page images>
    python calib.py --config config_rec.yaml --task rec --images <line crops>
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "common"))
from calib_pack import load_config, pack  # noqa: E402

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# Must match the runtime's CPU preprocessing. See the module docstring.
NORM = {
    "det": (np.array([0.485, 0.456, 0.406], np.float32),
            np.array([0.229, 0.224, 0.225], np.float32)),
    "rec": (np.array([0.5, 0.5, 0.5], np.float32),
            np.array([0.5, 0.5, 0.5], np.float32)),
}


def preprocess(path: str, task: str, h: int, w: int) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"could not read {path}")
    img = img[:, :, ::-1]                                   # BGR -> RGB
    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
    x = img.astype(np.float32) / 255.0
    mean, std = NORM[task]
    x = (x - mean) / std
    return np.ascontiguousarray(x.transpose(2, 0, 1))        # CHW


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--task", required=True, choices=["det", "rec"])
    ap.add_argument("--images", required=True,
                    help="directory or glob. det: page images. rec: line crops.")
    ap.add_argument("--limit", type=int, default=64)
    a = ap.parse_args()

    cfg = load_config(a.config)
    if len(cfg.inputs) != 1:
        sys.exit(f"{a.config}: expected 1 input, got {len(cfg.inputs)}")
    spec = cfg.inputs[0]
    if not (spec.h and spec.w):
        sys.exit(f"{a.config}: input_shape must be set so calibration matches "
                 f"the compiled shape exactly")

    if os.path.isfile(a.images):
        files = [a.images]
    else:
        pat = a.images if glob.has_magic(a.images) else os.path.join(a.images, "*")
        files = [f for f in sorted(glob.glob(pat))
                 if os.path.splitext(f)[1].lower() in IMAGE_EXTS][: a.limit]
    if not files:
        sys.exit(f"no images found under {a.images!r}")

    # Domain match matters more than count: calibrate detection on pages that
    # look like the pages you will run on, and recognition on REAL line crops
    # (ideally cut by the detector), not on whole pages scaled down to 48 high.
    print(f"[calib] {a.task}: {len(files)} images -> {spec.h}x{spec.w}")
    arrays = (preprocess(f, a.task, spec.h, spec.w) for f in files)
    srcs = [{"index": i, "source": os.path.basename(f), "task": a.task,
             "sha256": hashlib.sha256(open(f, "rb").read()).hexdigest()[:16]}
            for i, f in enumerate(files)]
    pack(spec, arrays, sources=srcs)


if __name__ == "__main__":
    main()
