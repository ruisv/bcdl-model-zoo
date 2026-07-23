#!/usr/bin/env python3
"""[2/3] PTQ calibration data for XFeat: 640x480, grayscale, instance-normalised.

The descriptor map is the part of this model PTQ can quietly degrade, so the set
is deliberately mixed in TEXTURE rather than just count — street scenes (KITTI),
facades and terrain (ETH3D — the repetitive structure that stresses a
descriptor) and general objects (COCO). Domain match beats sample count; ~100
frames is plenty.

The input the board sees is NOT pixels. XFeat's InstanceNorm was lifted out of
the graph into CPU preprocessing (see export.py), so calibration must reproduce
that preprocessing exactly:

    grayscale by CHANNEL MEAN — x.mean(dim=1), a plain average, NOT a luma
    weighting — then InstanceNorm over the whole image, which for one channel is
    a plain standardisation (subtract mean, divide by sqrt(var + 1e-5), biased
    variance, InstanceNorm2d's default eps).

Get the grayscale weighting wrong (luma vs mean) and the calibration domain
drifts off what the runtime feeds, and — as everywhere in this repo — the model
still compiles clean and runs at full speed.

Writing goes through common/calib_pack.py so the one calibration-writing path is
used even though this input is `no_preprocess`: the arrays handed to pack() are
already standardised here (a featuremap input, so pack applies no normalisation),
and pack() still writes the manifest that lets a rebuild prove its inputs.

Usage:
    python calib.py --config config.yaml \\
        --kitti /path/to/kitti/image_2 --eth3d /path/to/eth3d \\
        --coco /path/to/coco/images --limit 100
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

W, H = 640, 480


def preprocess(bgr):
    """What the CPU does at runtime, reproduced for calibration.

    grayscale by channel mean (XFeat uses x.mean(dim=1), not a luma weighting),
    resize to WxH, then InstanceNorm over the whole single-channel image =
    standardisation with biased variance and eps 1e-5.
    """
    g = bgr.astype(np.float32).mean(axis=2)
    g = cv2.resize(g, (W, H), interpolation=cv2.INTER_LINEAR)
    return ((g - g.mean()) / np.sqrt(g.var() + 1e-5)).astype(np.float32)[None]  # 1xHxW


def gather(pat, want):
    """Take up to `want` readable images from a directory or glob."""
    files = (sorted(glob.glob(os.path.join(pat, "**", "*"), recursive=True))
             if os.path.isdir(pat) else sorted(glob.glob(pat)))
    files = [f for f in files
             if os.path.splitext(f)[1].lower() in (".png", ".jpg", ".jpeg", ".bmp")]
    out = []
    for f in files:
        if len(out) >= want:
            break
        img = cv2.imread(f)
        if img is not None:
            out.append((os.path.basename(f), img))
    return out


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--kitti", help="KITTI image dir/glob (street scenes)")
    ap.add_argument("--eth3d", help="ETH3D image dir/glob (facades/terrain)")
    ap.add_argument("--coco", help="COCO image dir/glob (general objects)")
    ap.add_argument("--images", help="fallback: any image dir/glob")
    ap.add_argument("--kitti-n", type=int, default=40)
    ap.add_argument("--eth3d-n", type=int, default=30)
    ap.add_argument("--coco-n", type=int, default=30)
    ap.add_argument("--limit", type=int, default=0, help="hard cap over all sources")
    a = ap.parse_args()

    cfg = load_config(a.config)
    if len(cfg.inputs) != 1:
        sys.exit(f"{a.config}: expected 1 input (image), got {len(cfg.inputs)}")
    spec = cfg.inputs[0]

    plan = [(a.kitti, a.kitti_n, "kitti"),
            (a.eth3d, a.eth3d_n, "eth3d"),
            (a.coco, a.coco_n, "coco")]
    if a.images:
        plan.append((a.images, a.limit or 100, "images"))
    plan = [(p, n, tag) for p, n, tag in plan if p]
    if not plan:
        sys.exit("need at least one of --kitti/--eth3d/--coco/--images")

    arrays, srcs = [], []
    for pat, want, tag in plan:
        got = gather(pat, want)
        for name, img in got:
            arrays.append(preprocess(img))
            srcs.append({"index": len(arrays) - 1, "source": name, "set": tag})
        print("  %-8s %d" % (tag, len(got)))
        if a.limit and len(arrays) >= a.limit:
            arrays, srcs = arrays[:a.limit], srcs[:a.limit]
            break

    if not arrays:
        sys.exit("no calibration images found")
    print(f"[calib] {len(arrays)} frames, {H}x{W}, grayscale + instance-norm")
    pack(spec, arrays, fmt="npy", sources=srcs)


if __name__ == "__main__":
    main()
