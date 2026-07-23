#!/usr/bin/env python3
"""[2/3] PTQ calibration set for Real-ESRGAN Compact x4 (+ an optional golden).

The tile is an arbitrary tensor to the model (`featuremap` / `no_preprocess`),
so nothing is normalised at compile time. The net was trained on RGB in [0,1],
so this script divides by 255 itself; the write still routes through
common/calib_pack.py because the invariant is "there is no second way to write
calibration data", not "there is normalisation to apply here". With
no_preprocess, pack() passes the [0,1] tiles through untouched.

Domain the calibration set has to cover — this model is fed TWO input domains:

  * NATIVE crops   — a real image cropped without rescaling: sharp edges, sensor
                     detail, the case where the caller is sharpening a large
                     image.
  * DOWNSCALED crops — a crop taken after a 4x INTER_AREA downscale: soft,
                     band-limited, the case where the caller is enlarging a
                     small image.

Half the set is each. Calibrating on only one leaves the other's activation
ranges unrepresented, and the quantised model saturates on the domain it never
saw. Domain match beats sample count (20-100 tiles is plenty), so draw the tiles
from images that look like the deployment content.

Usage:
    python calib.py --config config.yaml \\
                    --images '/path/to/imgs/*.png' --limit 60
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


def tile(img_bgr, t, downscale, rng):
    """One HxW=t RGB tile in [0,1], CHW float32.

    downscale=True first shrinks the image 4x (INTER_AREA) to emulate the
    enlarge-a-small-image domain; False keeps native resolution. Undersized
    images are bilinear-enlarged to at least the tile so the crop always fits.
    """
    img = img_bgr
    if downscale:
        h, w = img.shape[:2]
        img = cv2.resize(img, (w // 4, h // 4), interpolation=cv2.INTER_AREA)
    h, w = img.shape[:2]
    if h < t or w < t:
        img = cv2.resize(img, (max(t, w), max(t, h)), interpolation=cv2.INTER_LINEAR)
        h, w = img.shape[:2]
    y = int(rng.integers(0, h - t + 1))
    x = int(rng.integers(0, w - t + 1))
    crop = img[y:y + t, x:x + t, ::-1]                 # BGR -> RGB
    return np.ascontiguousarray(crop.transpose(2, 0, 1).astype(np.float32) / 255.0)


def collect_images(pat, limit):
    files = sorted(glob.glob(pat) if glob.has_magic(pat)
                   else glob.glob(os.path.join(pat, "*")))
    files = [f for f in files
             if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))]
    if limit:
        files = files[:limit]
    if not files:
        sys.exit(f"no images found under {pat!r}")
    return files


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="config.yaml (selects the tile)")
    ap.add_argument("--images", required=True,
                    help="directory or glob of source images")
    ap.add_argument("--limit", type=int, default=60, help="max tiles (0 = all)")
    ap.add_argument("--seed", type=int, default=0)
    # optional fp32 golden, for verify_cosine layer C (single isolated tile, no
    # blending) -- see README "Running it".
    ap.add_argument("--golden", action="store_true")
    ap.add_argument("--onnx", help="exported ONNX, for the golden")
    ap.add_argument("--golden-image", help="an HR image; its 4x-downscale is the LR input")
    a = ap.parse_args()

    cfg = load_config(a.config)
    if len(cfg.inputs) != 1:
        sys.exit(f"{a.config}: expected 1 input (lr), got {len(cfg.inputs)}")
    spec = cfg.inputs[0]
    t = spec.h or spec.w
    if not t:
        sys.exit(f"{a.config}: could not read the tile size from input_shape")

    files = collect_images(a.images, a.limit)
    rng = np.random.default_rng(a.seed)
    tiles, srcs = [], []
    for i, p in enumerate(files):
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        if img is None:
            continue
        down = (i % 2 == 1)                            # alternate native / downscaled
        tiles.append(tile(img, t, down, rng))
        srcs.append({"index": len(tiles) - 1, "source": os.path.basename(p),
                     "domain": "downscaled" if down else "native"})

    print(f"[calib] {len(tiles)} tiles, {t}x{t}, "
          f"{sum(s['domain'] == 'native' for s in srcs)} native / "
          f"{sum(s['domain'] == 'downscaled' for s in srcs)} downscaled")
    pack(spec, tiles, fmt="npy", sources=srcs)

    if a.golden:
        import onnxruntime as ort
        if not a.onnx or not a.golden_image:
            sys.exit("--golden needs --onnx and --golden-image")
        hr = cv2.imread(a.golden_image, cv2.IMREAD_COLOR)
        h, w = (hr.shape[0] // 4) * 4, (hr.shape[1] // 4) * 4
        hr = hr[:h, :w]
        lr = cv2.resize(hr, (w // 4, h // 4), interpolation=cv2.INTER_AREA)
        # The LR frame may be shorter than a tile; replicate-pad to the tile
        # exactly as the on-board tiler does, then take one isolated tile.
        lp = cv2.copyMakeBorder(lr, 0, max(0, t - lr.shape[0]),
                                0, max(0, t - lr.shape[1]), cv2.BORDER_REPLICATE)
        ti = lp[:t, :t, ::-1].astype(np.float32) / 255.0
        inp = np.ascontiguousarray(ti.transpose(2, 0, 1)[None])
        out = ort.InferenceSession(
            a.onnx, providers=["CPUExecutionProvider"]).run(None, {"lr": inp})[0]
        out_dir = os.path.dirname(os.path.abspath(a.config))
        np.save(os.path.join(out_dir, "val_lr.npy"), inp)
        np.save(os.path.join(out_dir, "val_sr_golden.npy"), out)
        print(f"[golden] single-tile fp32 sr range [{out.min():.3f}, {out.max():.3f}] "
              f"-> {out_dir}/val_*.npy")


if __name__ == "__main__":
    main()
