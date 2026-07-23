#!/usr/bin/env python3
"""[2/3] PTQ calibration data for the two face models.

Two sub-models, two different normalisation stories, both written through the
one entry point common/calib_pack.pack():

  det (SCRFD)   config = rgb / data_mean_and_scale (mean 127.5, scale 1/128).
                So calib.py hands pack() RAW 0-255 RGB CHW and the CONFIG applies
                the (x-127.5)/128 normalisation -- exactly like the calib_pack
                CLI image path. Calibrate on face-containing scenes at 640x640.

  rec (ArcFace) config = featuremap / no_preprocess. The compiler applies
                NOTHING, so the normalisation is intrinsic to the array and this
                file does it: (x-127.5)/128 -> [-1,1]. calib.py hands pack() the
                already-normalised arrays; pack()'s normalize() is a pass-through
                by definition. Same pattern as osnet and ppocr's featuremap
                inputs.

THE `_aligned` STORY (read this before regenerating the rec set). ArcFace does
not see a face box -- it sees a 112x112 chip produced by a 5-point SIMILARITY
TRANSFORM that warps the eyes/nose/mouth onto ArcFace's canonical template
(insightface's `norm_crop`, driven by the landmarks SCRFD emits). The shipped
recognition build is calibrated on those ALIGNED chips, which is why the file is
named `..._aligned_...`. A plain centre-crop of a detector box is a DIFFERENT,
WORSE distribution: it compiles and runs identically, and quietly degrades the
embedding. So this script takes a directory of already-aligned 112x112 chips and
refuses to invent them from boxes -- produce them upstream with insightface
(`FaceAnalysis` -> `face.kps` -> `norm_crop`) so calibration matches deployment.

A note on the normalisation constant: insightface's own recognition wrapper uses
(x-127.5)/127.5; the shipped build used (x-127.5)/128 to match SCRFD's
scale_value (0.0078125 == 1/128). The runtime's face preprocessing is the source
of truth -- keep this in step with it. --rec-std overrides the divisor.

Usage:
    python calib.py --config config_det.yaml --task det --images <face scenes>
    python calib.py --config config_rec.yaml --task rec --aligned-dir <112 chips>
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

# ArcFace intrinsic normalisation. mean/std match SCRFD's mean_value/scale_value
# (127.5, 1/128); see the module docstring on the 127.5-vs-128 divisor.
REC_MEAN = 127.5
REC_STD = 128.0


def list_images(spec_path: str, limit: int) -> list:
    if os.path.isfile(spec_path):
        return [spec_path]
    pat = spec_path if glob.has_magic(spec_path) else os.path.join(spec_path, "*")
    files = [f for f in sorted(glob.glob(pat))
             if os.path.splitext(f)[1].lower() in IMAGE_EXTS]
    return files[:limit] if limit else files


def det_array(path: str, h: int, w: int) -> np.ndarray:
    """SCRFD: raw 0-255 RGB CHW. The config's data_mean_and_scale does the norm."""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"could not read {path}")
    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
    img = img[:, :, ::-1]                                    # BGR -> RGB
    return np.ascontiguousarray(img.transpose(2, 0, 1)).astype(np.float32)


def rec_array(path: str, h: int, w: int, std: float) -> np.ndarray:
    """ArcFace: aligned chip -> (x-127.5)/std RGB CHW, fully normalised here.

    featuremap/no_preprocess means the compiler applies nothing, so the array
    handed to pack() must already be what the board feeds the BPU.
    """
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"could not read {path}")
    if img.shape[:2] != (h, w):
        # Aligned chips should already be 112x112; resize only guards odd inputs.
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
    img = img[:, :, ::-1].astype(np.float32)                # BGR -> RGB
    x = (img - REC_MEAN) / std                              # -> ~[-1, 1]
    return np.ascontiguousarray(x.transpose(2, 0, 1))


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--task", required=True, choices=["det", "rec"])
    ap.add_argument("--images", help="det: directory or glob of face-scene images")
    ap.add_argument("--aligned-dir",
                    help="rec: directory of ALIGNED 112x112 face chips "
                         "(insightface norm_crop output), NOT raw detector boxes")
    ap.add_argument("--rec-std", type=float, default=REC_STD,
                    help="ArcFace normalisation divisor (default 128; "
                         "insightface's own wrapper uses 127.5)")
    ap.add_argument("--limit", type=int, default=64)
    a = ap.parse_args()

    cfg = load_config(a.config)
    if len(cfg.inputs) != 1:
        sys.exit(f"{a.config}: expected 1 input, got {len(cfg.inputs)}")
    spec = cfg.inputs[0]
    if not (spec.h and spec.w):
        sys.exit(f"{a.config}: input_shape must be set so calibration matches "
                 f"the compiled shape exactly")

    if a.task == "det":
        if not a.images:
            sys.exit("det needs --images (face-containing scenes)")
        files = list_images(a.images, a.limit)
        if not files:
            sys.exit(f"no images found under {a.images!r}")
        print(f"[calib] det: {len(files)} scenes -> {spec.h}x{spec.w} "
              f"(config applies (x-127.5)/128)")
        arrays = (det_array(f, spec.h, spec.w) for f in files)
    else:
        if not a.aligned_dir:
            sys.exit("rec needs --aligned-dir (aligned 112x112 face chips)")
        files = list_images(a.aligned_dir, a.limit)
        if not files:
            sys.exit(f"no aligned chips found under {a.aligned_dir!r}. Produce "
                     f"them with insightface norm_crop, not a centre-crop.")
        print(f"[calib] rec: {len(files)} ALIGNED chips -> {spec.h}x{spec.w} "
              f"(this file applies (x-127.5)/{a.rec_std:g})")
        arrays = (rec_array(f, spec.h, spec.w, a.rec_std) for f in files)

    srcs = [{"index": i, "source": os.path.basename(f), "task": a.task,
             "sha256": hashlib.sha256(open(f, "rb").read()).hexdigest()[:16]}
            for i, f in enumerate(files)]
    pack(spec, arrays, sources=srcs)


if __name__ == "__main__":
    main()
