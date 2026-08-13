#!/usr/bin/env python3
"""[2/3] PTQ calibration data for NeuFlow v2: consecutive frame PAIRS.

    python calib.py --config config.yaml --sweeps /path/to/nuscenes/sweeps --limit 60

The model takes two inputs, so this writes two aligned directories — `img0` and
`img1`, sample *i* in each being the two halves of one pair. hb_compile matches
them positionally through the `;`-separated `cal_data_dir`, so a mismatch in
count or order silently calibrates the model on pairs that are not pairs.

**Use high-rate frames, not keyframes.** This is the choice that matters here.
nuScenes sweeps run at 12 Hz and its keyframes at 2 Hz — half a second apart, by
which time the car has moved several metres. Calibrating on keyframe pairs
describes a displacement range the model is never asked about at run time, and
the activation thresholds come out sized for it. 30 logs x 2 pairs = 60 pairs is
plenty; domain match beats sample count.

**The tensors are BGR in 0-255, not normalised.** The `/255` lives inside the
graph, which is why the compile config says `no_preprocess` — hand it normalised
data and you have calibrated a different function than the one that runs.
"""
import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "common"))
from calib_pack import load_config, pack  # noqa: E402

W, H = 512, 384


def frame_pairs(root, limit):
    """Consecutive pairs from a directory of time-ordered images.

    Anything works as long as the two frames are close in time; nuScenes sweeps
    are what the shipped build used because they are 12 Hz and automotive.
    """
    import cv2

    files = sorted(glob.glob(os.path.join(root, "**", "*.jpg"), recursive=True) +
                   glob.glob(os.path.join(root, "**", "*.png"), recursive=True))
    out = []
    for a, b in zip(files, files[1:]):
        if len(out) >= limit:
            break
        i0, i1 = cv2.imread(a), cv2.imread(b)
        if i0 is None or i1 is None:
            continue
        out.append((cv2.resize(i0, (W, H)).transpose(2, 0, 1).astype(np.float32),
                    cv2.resize(i1, (W, H)).transpose(2, 0, 1).astype(np.float32),
                    os.path.basename(a)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--sweeps", required=True,
                    help="dir of time-ordered frames (nuScenes sweeps/CAM_FRONT works)")
    ap.add_argument("--limit", type=int, default=60)
    a = ap.parse_args()

    cfg = load_config(a.config)
    if len(cfg.inputs) != 2:
        sys.exit("this model has two inputs; the config declares %d" % len(cfg.inputs))

    pairs = frame_pairs(a.sweeps, a.limit)
    if not pairs:
        sys.exit("no frames found under %s" % a.sweeps)
    print("[calib] %d pairs, %dx%d, BGR 0-255 (the /255 is inside the graph)"
          % (len(pairs), H, W))
    srcs = [{"index": i, "source": n} for i, (_, _, n) in enumerate(pairs)]
    pack(cfg.inputs[0], [p[0] for p in pairs], fmt="npy", sources=srcs)
    pack(cfg.inputs[1], [p[1] for p in pairs], fmt="npy", sources=srcs)


if __name__ == "__main__":
    main()
