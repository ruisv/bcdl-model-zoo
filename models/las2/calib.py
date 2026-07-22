#!/usr/bin/env python3
"""[2/3] Generate PTQ calibration data for LAS2 (+ an optional fp32 golden).

Two rules this model punishes you for breaking:

RULE 1 — calibration, validation and deployment must use the SAME --mode and
the same resolution.
  * resize : scale the whole image to HxW. This also scales the disparity range
             by (source width / W), so far-field disparity goes sub-pixel and
             depth (z = fx*B/d) amplifies the error by 1/d. Measured 10-33%
             far-field depth error on ZED 2K resized to 640 wide.
  * crop   : centre-crop HxW without scaling. Preserves the disparity scale, so
             depth noise stays tiny (measured P99 1.18%). Costs field of view,
             and needs a source image at least HxW.
  Neither mode is better. Calibrating one way and deploying the other mismatches
  the disparity range, saturates the BPU quantisation, and the disparity
  collapses. This script refuses to write into a config whose cal_data_dir does
  not name the mode, which is the cheapest available guard against that.

RULE 2 — domain match beats sample count. Images from the actual camera win:
  5 same-domain ZED pairs gave EPE 0.12px, and adding 54 Middlebury pairs made
  it WORSE (0.26px). Do not pad the set with out-of-domain public data.

Writing goes through common/calib_pack.py so normalisation is taken from the
same config.yaml hb_compile reads. LAS2 declares `no_preprocess`, so nothing is
rescaled here — but it routes through the one entry point regardless, because
the invariant is "there is no second way to write calibration data".

Usage:
    python calib.py --config config.yaml --mode crop \\
                    --sbs-dir /path/to/side_by_side_pngs --limit 30
"""
import os

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import argparse
import glob
import sys

import numpy as np
import imageio.v2 as imageio
import cv2

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "common"))
from calib_pack import load_config, pack  # noqa: E402


def fit(img_rgb, hw, mode):
    """Bring an image to HxW. resize = scale whole image; crop = centre crop."""
    img = img_rgb[..., :3]
    if mode == "crop":
        h, w = img.shape[:2]
        if h < hw[0] or w < hw[1]:
            sys.exit(f"crop needs source ({h}x{w}) >= target ({hw[0]}x{hw[1]}); "
                     f"use --mode resize instead")
        t, l = (h - hw[0]) // 2, (w - hw[1]) // 2
        return img[t:t + hw[0], l:l + hw[1]]
    return cv2.resize(img, (hw[1], hw[0]))


def to_chw(img_rgb, hw, mode):
    return np.ascontiguousarray(fit(img_rgb, hw, mode).transpose(2, 0, 1)).astype(np.float32)


def split_sbs(path):
    img = imageio.imread(path)[..., :3]
    w = img.shape[1] - img.shape[1] % 2
    return img[:, : w // 2], img[:, w // 2:]


def collect_pairs(a):
    if a.sbs_dir:
        # A glob, not just a directory: a capture folder usually holds more than
        # the stereo set (figures, gifs, other rigs), and mixing those in breaks
        # RULE 2 — or, for crop, simply fails on the first undersized image.
        pat = a.sbs_dir if glob.has_magic(a.sbs_dir) else os.path.join(a.sbs_dir, "*")
        files = [p for p in sorted(glob.glob(pat))
                 if p.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))]
        pairs = [("sbs", p, None) for p in files]
    elif a.left_dir and a.right_dir:
        ls = sorted(glob.glob(os.path.join(a.left_dir, "*")))
        rs = sorted(glob.glob(os.path.join(a.right_dir, "*")))
        pairs = [("lr", l, r) for l, r in zip(ls, rs)]
    else:
        sys.exit("need --sbs-dir, or both --left-dir and --right-dir")
    if a.limit:
        pairs = pairs[: a.limit]
    if not pairs:
        sys.exit("no stereo pairs found")
    return pairs


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="config.yaml or config_resize.yaml")
    ap.add_argument("--mode", required=True, choices=["resize", "crop"])
    ap.add_argument("--sbs-dir", help="directory of side-by-side stereo images")
    ap.add_argument("--left-dir")
    ap.add_argument("--right-dir")
    ap.add_argument("--hw", nargs=2, type=int, default=[480, 640], metavar=("H", "W"))
    ap.add_argument("--limit", type=int, default=0, help="max pairs (0 = all)")
    # optional fp32 golden, for verify_cosine layer C
    ap.add_argument("--golden", action="store_true")
    ap.add_argument("--repo", help="LiteAnyStereo checkout")
    ap.add_argument("--ckpt")
    ap.add_argument("--size", default="m")
    ap.add_argument("--golden-sbs")
    ap.add_argument("--golden-left")
    ap.add_argument("--golden-right")
    ap.add_argument("--max-disp", type=int, default=192)
    a = ap.parse_args()

    cfg = load_config(a.config)
    if len(cfg.inputs) != 2:
        sys.exit(f"{a.config}: expected 2 inputs (left,right), got {len(cfg.inputs)}")
    left_spec, right_spec = cfg.inputs

    # RULE 1 guard: the config and the mode must agree. Getting these out of
    # step is the failure this model is most likely to hit, and it is silent.
    for spec in (left_spec, right_spec):
        if a.mode not in (spec.cal_dir or ""):
            sys.exit(
                f"--mode {a.mode} but {spec.name} writes to {spec.cal_dir!r}, "
                f"which does not name that mode. Calibrating in one letterbox "
                f"mode and deploying in the other collapses the disparity. Use "
                f"config.yaml for crop and config_resize.yaml for resize."
            )

    pairs = collect_pairs(a)
    lefts, rights = [], []
    for kind, p0, p1 in pairs:
        l, r = (split_sbs(p0) if kind == "sbs"
                else (imageio.imread(p0), imageio.imread(p1)))
        lefts.append(to_chw(l, a.hw, a.mode))
        rights.append(to_chw(r, a.hw, a.mode))

    srcs = [{"index": i, "source": os.path.basename(p0), "mode": a.mode}
            for i, (_, p0, _) in enumerate(pairs)]
    print(f"[calib] {len(pairs)} pairs, mode={a.mode}, {a.hw[0]}x{a.hw[1]}")
    pack(left_spec, lefts, fmt="npy", sources=srcs)
    pack(right_spec, rights, fmt="npy", sources=srcs)

    if a.golden:
        import torch
        here = os.path.dirname(os.path.abspath(__file__))
        sys.path.insert(0, here)
        import shifts
        sys.path.insert(0, a.repo)
        import core.liteanystereov2 as v2
        from core.models import build_model, load_model_weights
        shifts.patch(v2)

        gl, gr = (split_sbs(a.golden_sbs) if a.golden_sbs
                  else (imageio.imread(a.golden_left), imageio.imread(a.golden_right)))
        lt = torch.tensor(to_chw(gl, a.hw, a.mode)[None])
        rt = torch.tensor(to_chw(gr, a.hw, a.mode)[None])
        ck = a.ckpt or os.path.join(a.repo, "checkpoints", f"LAS2_{a.size.upper()}.pth")
        m = build_model("las2", fnet_pretrained=False, model_size=a.size,
                        max_disp=a.max_disp)
        load_model_weights(m, torch.load(ck, map_location="cpu"), strict=True)
        m = m.eval()
        with torch.no_grad():
            d = m(lt, rt, max_disp=a.max_disp, test_mode=True).float().numpy().reshape(a.hw)
        out = os.path.dirname(os.path.abspath(a.config))
        np.save(os.path.join(out, "val_left.npy"), to_chw(gl, a.hw, a.mode)[None])
        np.save(os.path.join(out, "val_right.npy"), to_chw(gr, a.hw, a.mode)[None])
        np.save(os.path.join(out, "val_disp_golden.npy"), d)
        print(f"[golden] fp32 disp [{d.min():.2f},{d.max():.2f}] "
              f"mean {d.mean():.2f} -> {out}/val_*.npy")


if __name__ == "__main__":
    main()
