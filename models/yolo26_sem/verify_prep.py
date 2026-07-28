#!/usr/bin/env python3
"""Accuracy verification step 1a (run OUTSIDE the OE docker -- it carries no
OpenCV; run this in any env with cv2, e.g. the model's own export env).

Letterboxes one real street-scene image to the model's input size and writes
the two feeds the next step needs: NV12 y/uv planes (uint8, what .bc/.hbm
consume) and a normalised RGB float32 NCHW array (what the float ONNX
consumes). Keeping this split out of verify_gen_ref.py is deliberate --
common/calib_pack.py notes the OE image carries no OpenCV, and this is the
same constraint.

Usage:
    python verify_prep.py <image.jpg> --out ./out_verify --size 640
"""
import argparse
import os
import sys

import cv2
import numpy as np


def letterbox(img_bgr, size):
    h, w = img_bgr.shape[:2]
    s = size / max(h, w)
    nh, nw = int(round(h * s)), int(round(w * s))
    r = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), 114, np.uint8)
    top, left = (size - nh) // 2, (size - nw) // 2
    canvas[top:top + nh, left:left + nw] = r
    return canvas


def bgr_to_nv12(img_bgr):
    """HxWx3 BGR uint8 -> (y [H,W,1], uv [H/2,W/2,2] U/V-interleaved)."""
    yuv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YUV_I420)
    h, w = img_bgr.shape[:2]
    y = yuv[:h].reshape(h, w, 1)
    u = yuv[h:h + h // 4].reshape(h // 2, w // 2)
    v = yuv[h + h // 4:].reshape(h // 2, w // 2)
    uv = np.stack([u, v], axis=-1)
    return y, uv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--out", default="./out_verify")
    ap.add_argument("--size", type=int, default=640)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    img = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if img is None:
        sys.exit(f"could not read {args.image}")
    canvas = letterbox(img, args.size)

    y, uv = bgr_to_nv12(canvas)
    np.save(os.path.join(args.out, "in_y.npy"), y[None].astype(np.uint8))
    np.save(os.path.join(args.out, "in_uv.npy"), uv[None].astype(np.uint8))

    rgb01 = canvas[:, :, ::-1].astype(np.float32) / 255.0
    chw = np.ascontiguousarray(rgb01.transpose(2, 0, 1))[None]
    np.save(os.path.join(args.out, "in_rgb01_chw.npy"), chw)

    print(f"wrote {args.out}/in_y.npy {y.shape}, in_uv.npy {uv.shape}, "
          f"in_rgb01_chw.npy {chw.shape}")


if __name__ == "__main__":
    main()
