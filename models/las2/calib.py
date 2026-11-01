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

RULE 2 — domain match beats sample count. Images from the actual camera win;
  do not pad the set with out-of-domain public data. (The figures once quoted
  for this — 5 ZED pairs 0.12px vs +54 Middlebury pairs 0.26px — were measured
  while this graph's output nodes saw only one sample in eight, see
  common/calib_pack.py, so they compared one pair against eight there.)

RULE 3 — the calibration set must span the disparity range you deploy into.
  The int16 threshold on the output is set by the largest disparity the pairs
  contain, and anything above it is clipped without error. The first shipped
  builds were calibrated on scenes with no near field and clipped at 131 px
  (crop) and 47.7 px (resize) out of the model's 192. --extend-range fixes this
  without leaving the domain: it adds a copy of each pair with the right view
  shifted left by k px, which raises every disparity by exactly k and changes
  nothing else. It needs the float model (--repo) to measure each pair. --pool
  adds a few real large-disparity pairs on top; they are out of domain, so keep
  them few (RULE 2).

Writing goes through common/calib_pack.py so normalisation is taken from the
same config.yaml hb_compile reads. LAS2 declares `no_preprocess`, so nothing is
rescaled here — but it routes through the one entry point regardless, because
the invariant is "there is no second way to write calibration data".

Usage:
    python calib.py --config config.yaml --mode crop \\
                    --sbs-dir /path/to/side_by_side_pngs --limit 30 \\
                    --extend-range --repo /path/to/LiteAnyStereo
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
from calib_pack import CALIB_REPEAT, load_config, pack  # noqa: E402


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


def shift_left(chw, k):
    """Move a CHW image k px left, replicating the last column. Applied to the
    right view, every disparity grows by exactly k."""
    if k <= 0:
        return chw
    out = np.empty_like(chw)
    out[..., :-k] = chw[..., k:]
    out[..., -k:] = chw[..., -1:]
    return out


def load_model(a):
    """Float LAS2 with the unfold rewrite, for --extend-range / --pool / --golden."""
    import torch
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, here)
    import shifts
    sys.path.insert(0, a.repo)
    import core.liteanystereov2 as v2
    from core.models import build_model, load_model_weights
    shifts.patch(v2)
    ck = a.ckpt or os.path.join(a.repo, "checkpoints", f"LAS2_{a.size.upper()}.pth")
    m = build_model("las2", fnet_pretrained=False, model_size=a.size, max_disp=a.max_disp)
    load_model_weights(m, torch.load(ck, map_location="cpu"), strict=True)
    m = m.eval()

    def run(l_chw, r_chw):
        with torch.no_grad():
            d = m(torch.tensor(l_chw[None]), torch.tensor(r_chw[None]),
                  max_disp=a.max_disp, test_mode=True)
        return d.float().numpy().reshape(l_chw.shape[1:])
    return run


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
    # RULE 3: calibration range
    ap.add_argument("--extend-range", action="store_true",
                    help="add a right-shifted copy of each pair reaching --target px")
    ap.add_argument("--target", type=float, default=185.0,
                    help="disparity (p99.9) the shifted copies are pushed to")
    ap.add_argument("--pool", nargs="*", default=[],
                    help="dirs of scene dirs (im0.png/im1.png) with large disparity")
    ap.add_argument("--pool-min", type=float, default=140.0,
                    help="keep pool pairs whose p99.9 disparity is at least this")
    ap.add_argument("--pool-max-n", type=int, default=6)
    a = ap.parse_args()
    if (a.extend_range or a.pool or a.golden) and not a.repo:
        sys.exit("--extend-range / --pool / --golden need --repo (the float model)")

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
    run = load_model(a) if (a.extend_range or a.pool or a.golden) else None
    lefts, rights, srcs = [], [], []

    def add(l_chw, r_chw, src):
        srcs.append(dict(src, index=len(lefts), mode=a.mode))
        lefts.append(l_chw)
        rights.append(r_chw)

    for kind, p0, p1 in pairs:
        l, r = (split_sbs(p0) if kind == "sbs"
                else (imageio.imread(p0), imageio.imread(p1)))
        lc, rc = to_chw(l, a.hw, a.mode), to_chw(r, a.hw, a.mode)
        add(lc, rc, {"source": os.path.basename(p0)})
        if a.extend_range:
            top = float(np.percentile(run(lc, rc), 99.9))
            k = int(a.target - top)
            if k > 0:
                add(lc, shift_left(rc, k), {"source": os.path.basename(p0), "shift_px": k})
                print(f"[range] {os.path.basename(p0)}: p99.9 {top:.1f} px, +{k} copy")

    picked = []
    for d in sorted(x for pd in a.pool for x in glob.glob(os.path.join(pd, "*"))):
        f0, f1 = os.path.join(d, "im0.png"), os.path.join(d, "im1.png")
        if not os.path.isfile(f1):
            continue
        l, r = imageio.imread(f0), imageio.imread(f1)
        if a.mode == "crop" and (l.shape[0] < a.hw[0] or l.shape[1] < a.hw[1]):
            continue
        lc, rc = to_chw(l, a.hw, a.mode), to_chw(r, a.hw, a.mode)
        top = float(np.percentile(run(lc, rc), 99.9))
        if a.pool_min <= top <= a.max_disp:
            picked.append((top, d, lc, rc))
    for top, d, lc, rc in sorted(picked, key=lambda x: -x[0])[:a.pool_max_n]:
        add(lc, rc, {"source": os.path.basename(d), "pool": True})
        print(f"[pool] {os.path.basename(d)}: p99.9 {top:.1f} px")

    print(f"[calib] {len(lefts)} pairs ({len(pairs)} source), mode={a.mode}, "
          f"{a.hw[0]}x{a.hw[1]}")
    # LAS2 runs at batch 8 but its output Resize keeps only the first sample of
    # each batch once the compiler folds its size; repeating each pair makes
    # every batch one pair (common/calib_pack.py, README trap 4).
    pack(left_spec, lefts, fmt="npy", sources=srcs, repeat=CALIB_REPEAT)
    pack(right_spec, rights, fmt="npy", sources=srcs, repeat=CALIB_REPEAT)

    if a.golden:
        gl, gr = (split_sbs(a.golden_sbs) if a.golden_sbs
                  else (imageio.imread(a.golden_left), imageio.imread(a.golden_right)))
        d = run(to_chw(gl, a.hw, a.mode), to_chw(gr, a.hw, a.mode))
        out = os.path.dirname(os.path.abspath(a.config))
        np.save(os.path.join(out, "val_left.npy"), to_chw(gl, a.hw, a.mode)[None])
        np.save(os.path.join(out, "val_right.npy"), to_chw(gr, a.hw, a.mode)[None])
        np.save(os.path.join(out, "val_disp_golden.npy"), d)
        print(f"[golden] fp32 disp [{d.min():.2f},{d.max():.2f}] "
              f"mean {d.mean():.2f} -> {out}/val_*.npy")


if __name__ == "__main__":
    main()
