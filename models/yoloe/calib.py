#!/usr/bin/env python3
"""[2/3] PTQ calibration data for YOLOE (detection and segmentation).

Both builds take the identical input -- one 640x640 RGB image -- so one calib
set serves either config. Pick the config to write the manifest next to the
right cal_data_dir.

Preprocessing MUST match deployment. bcdl feeds the BPU an nv12 frame that the
compiler converts to RGB and scales by 1/255 (config: input_type_rt=nv12,
input_type_train=rgb, norm_type=data_scale). So the calibration path here is:

    letterbox to 640x640 (aspect-preserving resize + pad 114) -> RGB, 0..255 CHW

and calib_pack.pack() applies the /255 from config.yaml. We hand pack() raw
0..255 pixels, exactly as calib_pack expects for an image input, so the
normalisation stays defined in one place (the yaml) and cannot drift from what
the compiler bakes. Do NOT pre-divide by 255 here -- that would double-scale.

Pad value 114 is YOLO's standard letterbox grey; matching it matters because the
padded border is part of the activation distribution the calibrator sees.

Calibrate on real COCO images (the vocabulary is COCO-80). 20-100 is plenty;
domain match beats count.

Usage:
    python calib.py --config config_det.yaml --images <coco val images> --limit 64
    python calib.py --config config_seg.yaml --images <coco val images> --limit 64
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
PAD = 114  # YOLO letterbox grey


def letterbox_chw(path: str, h: int, w: int) -> np.ndarray:
    """Aspect-preserving resize + pad to HxW, returned as RGB 0..255 CHW.

    Mirrors bcdl's YOLO preprocessing: scale by the smaller ratio, centre the
    image, pad the remainder with 114. Returns 0..255 -- pack() does the /255.
    """
    img = cv2.imread(path, cv2.IMREAD_COLOR)  # BGR HWC uint8
    if img is None:
        raise SystemExit(f"could not read {path}")
    ih, iw = img.shape[:2]
    r = min(h / ih, w / iw)
    nh, nw = int(round(ih * r)), int(round(iw * r))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((h, w, 3), PAD, dtype=np.uint8)
    top, left = (h - nh) // 2, (w - nw) // 2
    canvas[top:top + nh, left:left + nw] = resized
    canvas = canvas[:, :, ::-1]  # BGR -> RGB
    return np.ascontiguousarray(canvas.transpose(2, 0, 1)).astype(np.float32)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True,
                    help="config_det.yaml or config_seg.yaml")
    ap.add_argument("--images", required=True,
                    help="directory or glob of COCO images")
    ap.add_argument("--limit", type=int, default=64)
    a = ap.parse_args()

    cfg = load_config(a.config)
    if len(cfg.inputs) != 1:
        sys.exit(f"{a.config}: expected 1 input, got {len(cfg.inputs)}")
    spec = cfg.inputs[0]
    if not (spec.h and spec.w):
        sys.exit(f"{a.config}: input_shape must be set so calibration matches "
                 f"the compiled shape exactly")

    pat = a.images if glob.has_magic(a.images) else os.path.join(a.images, "*")
    files = [f for f in sorted(glob.glob(pat))
             if os.path.splitext(f)[1].lower() in IMAGE_EXTS][: a.limit]
    if not files:
        sys.exit(f"no images found under {a.images!r}")

    print(f"[calib] {len(files)} images -> {spec.h}x{spec.w} (letterbox, pad {PAD})")
    arrays = (letterbox_chw(f, spec.h, spec.w) for f in files)
    srcs = [{"index": i, "source": os.path.basename(f),
             "sha256": hashlib.sha256(open(f, "rb").read()).hexdigest()[:16]}
            for i, f in enumerate(files)]
    pack(spec, arrays, sources=srcs)


if __name__ == "__main__":
    main()
