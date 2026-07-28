#!/usr/bin/env python3
"""[2/3] Calibration set for YOLO26n-sem: letterboxed 640x640 street frames.

YOLO26n-sem is pretrained on Cityscapes (19-class urban street scenes). PTQ
calibration domain must match that closely -- general detection imagery (the
COCO set ../yolo26/calib.py uses) is obviously the wrong domain, but even a
DIFFERENT road-scene domain measurably underperforms the real thing: this
recipe first calibrated on nuScenes CAM_FRONT frames (a different camera
geometry/FOV and city/continent) and measured 85% pixel-level argmax
agreement with the float model on a held-out image; recalibrating with real
Cityscapes validation frames -- same int16 config, same 24-image count --
raised that to 96% (see README.md). "Any road scene" is not close enough here;
use the model's actual training domain when you can get it. If you don't have
an official Cityscapes account (cityscapes-dataset.com requires one), an
ungated Parquet mirror works for calibration purposes:
huggingface.co/datasets/Chris1/cityscapes (columns `image` /
`semantic_segmentation`; decode the `image` column's embedded bytes to files
and point --images at that directory). See ../pidnet/calib.py, which hits the
same domain-match requirement for a different Cityscapes model.

Letterbox, not a plain resize: BCDL's runtime preprocessing for every yolo26
head (det/seg/pose/obb) letterboxes to a square input, and the calibration
distribution has to match what the deployed model actually sees, padding
included. Cityscapes/road frames are wide (16:9 or 2:1); squashing them instead of
letterboxing would calibrate on a geometry the runtime never produces.

The write goes through common/calib_pack.pack(), which applies exactly the
normalisation config.yaml declares -- here data_scale 1/255, the same
convention as ../yolo26/ (see export.py: the graph does not bake /255). This
file hands pack() raw 0-255 RGB CHW; it must NOT scale a second time.

Usage:
    python calib.py --config config.yaml --images /path/to/street_frames --limit 64
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


def letterbox_640(img_bgr):
    """Aspect-preserving resize to 640 + pad 114, matching BCDL's runtime path."""
    h, w = img_bgr.shape[:2]
    s = 640 / max(h, w)
    nh, nw = int(round(h * s)), int(round(w * s))
    r = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((640, 640, 3), 114, np.uint8)
    top, left = (640 - nh) // 2, (640 - nw) // 2
    canvas[top:top + nh, left:left + nw] = r
    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--images", required=True,
                    help="directory or glob of street-scene frames (Cityscapes, "
                         "nuScenes CAM_FRONT, BDD100K, or similar)")
    ap.add_argument("--limit", type=int, default=64)
    args = ap.parse_args()

    cfg = load_config(args.config)
    if len(cfg.inputs) != 1:
        sys.exit(f"{args.config}: expected a single input, got {len(cfg.inputs)}")
    spec = cfg.inputs[0]

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
        rgb = letterbox_640(img)[:, :, ::-1]  # BGR -> RGB; config declares rgb
        arrays.append(np.ascontiguousarray(rgb.transpose(2, 0, 1)).astype(np.float32))
        sources.append({"index": i, "source": os.path.basename(f)})

    print(f"[calib] {len(arrays)} street frames, 640x640 letterboxed")
    pack(spec, arrays, sources=sources)


if __name__ == "__main__":
    main()
