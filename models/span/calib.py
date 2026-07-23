#!/usr/bin/env python3
"""[2/3] PTQ calibration data for SPAN x4 -- 128x128 RGB tiles in [0,1].

RECONSTRUCTED, not salvaged. The original span conversion kept no calibration
script; this one is rebuilt by analogy from the sibling super-resolution model
(Real-ESRGAN Compact) and from span_ref.py's preprocessing, both of which feed
the network RGB tiles scaled by 1/255. If you need the exact activation ranges of
the shipped `.hbm`, note they cannot be reproduced from here -- the original
calibration set was never recorded (the manifest written beside the output is the
mechanism that stops that happening again).

Why the tile mix matters. A super-res model is fed two different input domains
depending on the caller:

  * NATIVE crops    -- a real image cropped at native resolution: sensor detail,
                       sharp edges, full high-frequency content.
  * DOWNSCALED crops -- a crop that was 4x-downscaled first: soft, band-limited,
                       the "enlarge a small image" case.

Their activation statistics differ, so this script alternates the two (native on
even samples, 4x-down on odd), exactly as the sibling model's calib did.
Calibrating on only one leaves the other's ranges unrepresented.

The input is declared `featuremap` / `no_preprocess`, so calib_pack.pack applies
NO normalisation -- it writes what it is handed. We therefore hand it the model's
own domain: RGB, CHW, float32, already divided by 255. Routing through
common/calib_pack.py is still mandatory: it is the one entry point that writes a
calibration set (and its manifest), so a rebuild can prove what it used.

Usage:
    python calib.py --config config.yaml \\
        --images '/path/to/imgs/*.png' '/path/to/more/*.jpg' --limit 60
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


def make_tile(img_bgr, tile, downscale, rng):
    """One RGB [0,1] CHW tile. downscale=True -> 4x INTER_AREA first."""
    img = img_bgr
    if downscale:
        h, w = img.shape[:2]
        img = cv2.resize(img, (max(1, w // 4), max(1, h // 4)),
                         interpolation=cv2.INTER_AREA)
    h, w = img.shape[:2]
    if h < tile or w < tile:
        img = cv2.resize(img, (max(tile, w), max(tile, h)),
                         interpolation=cv2.INTER_LINEAR)
        h, w = img.shape[:2]
    y = int(rng.integers(0, h - tile + 1))
    x = int(rng.integers(0, w - tile + 1))
    crop = img[y:y + tile, x:x + tile, ::-1]            # BGR -> RGB
    return np.ascontiguousarray(crop.transpose(2, 0, 1).astype(np.float32) / 255.0)


def collect_files(patterns):
    files = []
    for pat in patterns:
        hits = (sorted(glob.glob(os.path.join(pat, "*"))) if os.path.isdir(pat)
                else sorted(glob.glob(pat)))
        files += [f for f in hits
                  if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))]
    return files


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="config.yaml")
    ap.add_argument("--images", nargs="+", required=True,
                    help="image dirs or globs; mix scenes for range coverage")
    ap.add_argument("--tile", type=int, default=128)
    ap.add_argument("--limit", type=int, default=60, help="max tiles (default 60)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    cfg = load_config(a.config)
    if len(cfg.inputs) != 1:
        sys.exit(f"{a.config}: expected 1 input (lr), got {len(cfg.inputs)}")
    spec = cfg.inputs[0]

    files = collect_files(a.images)
    if not files:
        sys.exit(f"no images found under {a.images!r}")
    files = files[: a.limit]

    rng = np.random.default_rng(a.seed)
    tiles, srcs = [], []
    for i, f in enumerate(files):
        img = cv2.imread(f)
        if img is None:
            continue
        downscale = (i % 2 == 1)     # alternate native / 4x-down domains
        tiles.append(make_tile(img, a.tile, downscale, rng))
        srcs.append({"index": len(tiles) - 1, "source": os.path.basename(f),
                     "domain": "downscaled" if downscale else "native"})

    print(f"[calib] {len(tiles)} tiles, {a.tile}x{a.tile}, "
          f"native+4x-down mix, RGB [0,1]")
    pack(spec, tiles, fmt="npy", sources=srcs)


if __name__ == "__main__":
    main()
