#!/usr/bin/env python3
"""[2/3] PTQ calibration data for LingBot-Depth (+ the optional fp32 golden).

**This file is the single definition of the model's host-side preprocessing.**
The `.hbm` holds only static tensor ops; the resize, the normalisation and the
log remap all happen outside it, which means the deployed runtime and this
script have to agree exactly or the quantiser was fitted to a distribution the
board never produces. BCDL mirrors this in `src/tasks/depth_refine.cc` and its
test suite asserts the two agree to within cv2's uint8 rounding.

The preprocessing, both inputs:

  image     : BGR -> RGB, AREA-average resize to (rows*14, cols*14), /255,
              ImageNet mean/std, planar CHW.
  depth_log : nearest resize to the same grid, readings <= 0.01 m (and
              non-finite ones) forced to 0, natural log of the rest. Zero
              therefore means "no reading" -- upstream's own convention, which
              collides with a genuine 1 m reading; the model separates them
              using the RGB tokens.

AREA, not bilinear: the reference implementation resizes with an antialiased
bilinear filter that neither OpenCV nor a dependency-free C++ resize reproduces
exactly. Measured against it, area averaging costs 6.1e-4 mean absolute relative
depth error and plain bilinear costs 9.1e-4.

RULE, shared with every model here: domain match beats sample count. The shipped
build used the eight scenes upstream bundles plus horizontal mirrors. If you are
deploying against a specific depth source, recalibrate on that source.

Writing goes through common/calib_pack.py so normalisation comes from the same
config.yaml hb_compile reads. This model declares `no_preprocess`, so nothing is
rescaled there -- but it routes through the one entry point regardless, because
the invariant is "there is no second way to write calibration data".

Usage:
    python calib.py --config config.yaml --scenes /path/to/lingbot-depth/examples \\
                    --flip --golden --repo /path/to/lingbot-depth --ckpt model_v05.pt
"""
import os

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import argparse
import glob
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "common"))
from calib_pack import load_config, pack  # noqa: E402

IMAGE_MEAN = np.array([0.485, 0.456, 0.406], np.float32).reshape(3, 1, 1)
IMAGE_STD = np.array([0.229, 0.224, 0.225], np.float32).reshape(3, 1, 1)
MIN_VALID_DEPTH = 0.01


def preprocess_rgb(bgr, h14, w14):
    """BGR uint8 HxWx3 -> normalised planar float32 CHW."""
    small = cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), (w14, h14),
                       interpolation=cv2.INTER_AREA)
    x = small.astype(np.float32).transpose(2, 0, 1) / 255.0
    return (x - IMAGE_MEAN) / IMAGE_STD


def preprocess_depth(raw_m, h14, w14):
    """Metric depth HxW float32 -> log depth 1xHxW, 0 where there is no reading."""
    d = cv2.resize(raw_m, (w14, h14), interpolation=cv2.INTER_NEAREST)
    d = np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)
    valid = d > MIN_VALID_DEPTH
    out = np.zeros_like(d)
    out[valid] = np.log(d[valid])
    return out[None]


def load_scene(d, depth_scale):
    rgb = sorted(glob.glob(os.path.join(d, "rgb.*")))
    if not rgb:
        return None
    bgr = cv2.imread(rgb[0])
    raw = cv2.imread(os.path.join(d, "raw_depth.png"), cv2.IMREAD_UNCHANGED)
    if bgr is None or raw is None:
        return None
    raw = np.nan_to_num(raw.astype(np.float32) / depth_scale, nan=0.0, posinf=0.0,
                        neginf=0.0)
    return bgr, raw


def write_golden(args, out_dir, h14, w14, img, dep):
    """Run the fp32 graph on the held-out pair, for verify_cosine layer C."""
    import torch
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, args.repo)
    from export import StaticMDM, patch_batch1_blocks
    from mdm.model.v2 import MDMModel
    from mdm.model.dinov2_rgbd.models.vision_transformer import DinoVisionTransformer

    patch_batch1_blocks(DinoVisionTransformer)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MDMModel.from_pretrained(args.ckpt).to(device).eval()
    aspect = args.width / args.height
    w = StaticMDM(model, h14 // 14, w14 // 14, aspect, args.height, args.width,
                  args.repo).to(device).eval()
    with torch.no_grad():
        outs = w(torch.from_numpy(img[None]).to(device),
                 torch.from_numpy(dep[None]).to(device))
    np.save(os.path.join(out_dir, "val_image.npy"), img[None])
    np.save(os.path.join(out_dir, "val_depth_log.npy"), dep[None])
    depth = outs[0][0, 0].float().cpu().numpy()
    np.save(os.path.join(out_dir, "val_depth_ref.npy"), depth)
    print(f"golden depth {depth.shape} range [{depth.min():.3f}, {depth.max():.3f}] m")
    if len(outs) > 1:
        mask = outs[1][0, 0].float().cpu().numpy()
        np.save(os.path.join(out_dir, "val_mask_ref.npy"), mask)
        print(f"golden mask  kept {(mask > 0).mean()*100:.1f}%")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--scenes", required=True,
                    help="directory of scene subdirs, each with rgb.* and raw_depth.png")
    ap.add_argument("--depth-scale", type=float, default=1000.0,
                    help="divisor taking the raw depth image to metres (16-bit mm)")
    ap.add_argument("--flip", action="store_true",
                    help="also emit horizontal mirrors (same statistics, 2x the set)")
    ap.add_argument("--limit", type=int, default=0, help="max scenes (0 = all)")
    ap.add_argument("--hold-out", default="3",
                    help="scene name kept out of calibration as the golden sample")
    # optional fp32 golden, for verify_cosine layer C
    ap.add_argument("--golden", action="store_true")
    ap.add_argument("--repo", help="lingbot-depth checkout (with --golden)")
    ap.add_argument("--ckpt", help="model.pt (with --golden)")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    a = ap.parse_args()

    cfg = load_config(a.config)
    if len(cfg.inputs) != 2:
        sys.exit(f"{a.config}: expected 2 inputs (image, depth_log), got {len(cfg.inputs)}")
    img_spec, dep_spec = cfg.inputs
    if img_spec.name != "image" or dep_spec.name != "depth_log":
        sys.exit(f"{a.config}: expected inputs named image;depth_log, got "
                 f"{img_spec.name};{dep_spec.name}")
    h14, w14 = img_spec.h, img_spec.w
    if not h14 or not w14:
        sys.exit(f"{a.config}: could not read the encoder grid from input_shape")
    if (h14 % 14) or (w14 % 14):
        sys.exit(f"encoder input {h14}x{w14} is not a multiple of the 14px patch")
    print(f"encoder grid {h14//14}x{w14//14} -> input {h14}x{w14}")

    scenes = [d for d in sorted(glob.glob(os.path.join(a.scenes, "*"))) if os.path.isdir(d)]
    if a.limit:
        scenes = scenes[:a.limit]
    if not scenes:
        sys.exit(f"no scene directories under {a.scenes}")

    images, depths, held = [], [], None
    for d in scenes:
        loaded = load_scene(d, a.depth_scale)
        if loaded is None:
            print(f"  skipping {os.path.basename(d)}: missing rgb or raw_depth")
            continue
        bgr, raw = loaded
        variants = [(bgr, raw)]
        if a.flip:
            variants.append((cv2.flip(bgr, 1), cv2.flip(raw, 1)))
        for k, (b, r) in enumerate(variants):
            img = preprocess_rgb(b, h14, w14)
            dep = preprocess_depth(r, h14, w14)
            if os.path.basename(d) == a.hold_out and k == 0:
                held = (img, dep)
                print(f"holding out scene {os.path.basename(d)} as the golden sample")
                continue
            images.append(img)
            depths.append(dep)

    if not images:
        sys.exit("every scene was held out or unreadable")
    pack(img_spec, images)
    pack(dep_spec, depths)
    print(f"wrote {len(images)} calibration pairs")

    if a.golden:
        if not (a.repo and a.ckpt):
            sys.exit("--golden needs --repo and --ckpt")
        if held is None:
            sys.exit(f"--hold-out {a.hold_out!r} matched no scene, so there is no "
                     f"golden pair to run")
        out_dir = os.path.dirname(os.path.abspath(img_spec.cal_dir.rstrip("/")))
        write_golden(a, out_dir, h14, w14, *held)


if __name__ == "__main__":
    main()
