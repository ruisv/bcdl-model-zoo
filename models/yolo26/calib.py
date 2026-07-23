#!/usr/bin/env python3
"""[2/3] Calibration set for YOLO26s: letterboxed 640x640 frames.

COCO val (or any general detection imagery) is the natural domain. The write goes
through common/calib_pack.pack(), which applies exactly the normalisation
config.yaml declares -- here data_scale 1/255. So this file hands pack() raw
0-255 RGB CHW and pack() scales it; it must NOT scale a second time. When
cal_data_type is float32 the compiler's norm does not touch calibration data,
which is the trap the single packer entry point closes.

Usage:
    python calib.py --config config.yaml --images /path/to/coco_val --limit 64
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
    """Aspect-preserving resize to 640 + pad 114, as YOLO preprocessing does."""
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
    ap.add_argument("--images", required=True, help="directory or glob of frames")
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

    print(f"[calib] {len(arrays)} frames, 640x640 letterboxed")
    pack(spec, arrays, sources=sources)


if __name__ == "__main__":
    main()
