#!/usr/bin/env python3
"""[2/3] PTQ calibration data for GLASS: NORMAL images of the one product.

    python calib.py --config config.yaml --good /path/to/mvtec/bottle/train/good --limit 48

**The calibration domain is "normal", and that is not a compromise.** A one-class
detector is trained on defect-free examples only — it has never seen a defect and
is not supposed to. Calibrating on normal images is therefore the honest
distribution, not a convenient approximation of one, which is unusual enough to
say out loud.

It also has to be the SAME product the heads were trained on. GLASS's backbone is
frozen and only two small per-product heads are trained, so a checkpoint is a
bottle detector or a screw detector, never both.

The tensors are ImageNet-normalised here because the graph expects them that way
(`no_preprocess` in the config): BGR is converted to RGB, divided by 255, then
mean/std applied. That is exactly what BCDL's `anomalyPreprocess` does at run
time, and a mismatch is invisible — it compiles, runs, and quietly scores
against a distribution the runtime never produces.
"""
import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "common"))
from calib_pack import load_config, pack  # noqa: E402

SIZE = 288
MEAN = np.array([0.485, 0.456, 0.406], np.float32).reshape(3, 1, 1)
STD = np.array([0.229, 0.224, 0.225], np.float32).reshape(3, 1, 1)


def preprocess(bgr):
    """BGR uint8 -> ImageNet-normalised RGB CHW, the tensor the graph expects."""
    import cv2

    img = cv2.resize(bgr, (SIZE, SIZE), interpolation=cv2.INTER_LINEAR)
    rgb = img[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
    return (rgb - MEAN) / STD


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--good", required=True,
                    help="a directory of DEFECT-FREE images of the product this "
                         "checkpoint was trained on")
    ap.add_argument("--limit", type=int, default=48)
    a = ap.parse_args()

    import cv2

    cfg = load_config(a.config)
    files = sorted(glob.glob(os.path.join(a.good, "*.png")) +
                   glob.glob(os.path.join(a.good, "*.jpg")))[:a.limit]
    if not files:
        sys.exit("no images under %s" % a.good)

    arrays, srcs = [], []
    for f in files:
        img = cv2.imread(f)
        if img is None:
            continue
        arrays.append(preprocess(img))
        srcs.append({"index": len(arrays) - 1, "source": os.path.basename(f)})
    print("[calib] %d normal images, %dx%d, ImageNet-normalised" % (len(arrays), SIZE, SIZE))
    pack(cfg.inputs[0], arrays, fmt="npy", sources=srcs)


if __name__ == "__main__":
    main()
