#!/usr/bin/env python3
"""Write PTQ calibration data for hb_compile — the only supported way to do it.

Calibration data must be pre-normalised by hand. When `cal_data_type` is
float32, the compiler's `norm_type` / `mean_value` / `scale_value` are applied
to the *runtime* input path only; they are **not** applied to the calibration
data you hand it. Feed raw 0-255 pixels to a config that declares a mean and
scale and the activation statistics are gathered on a distribution the deployed
model never sees. The input thresholds come out wrong, and:

    the model compiles without a single warning, loads, runs at full speed,
    and decodes to garbage.

That is the whole reason this file exists. The failure has no loud signal, so
the defence cannot be a warning in a document — it has to be that there is no
other way to produce a calibration set.

Which is why normalisation parameters are **read from the same config.yaml that
hb_compile reads**, and cannot be passed on the command line. Previously the
mean/scale lived in the yaml while normalisation happened in a separate ad-hoc
script; nothing tied the two together, so they drifted. Here they cannot: if you
change mean_value in the yaml, the calibration data changes with it.

Usage:
    python calib_pack.py --config models/<name>/config.yaml \\
                         --images /path/to/images --limit 64

Writes to the `cal_data_dir` named by the config, so the compile step needs no
matching argument either.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys

import numpy as np
import yaml

# cv2 is imported lazily inside the packing path: --self-test needs only numpy
# and yaml, and the toolchain env does not necessarily carry OpenCV.


def _parse_triplet(v, name):
    """hb_compile accepts these as a space-separated string or a bare scalar."""
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return np.array([float(v)] * 3, dtype=np.float32)
    parts = str(v).split()
    if len(parts) == 1:
        return np.array([float(parts[0])] * 3, dtype=np.float32)
    if len(parts) != 3:
        raise SystemExit(f"{name}: expected 1 or 3 values, got {len(parts)}: {v!r}")
    return np.array([float(p) for p in parts], dtype=np.float32)


def normalize(chw: np.ndarray, norm_type: str, mean, scale) -> np.ndarray:
    """Apply exactly what the runtime will apply. `chw` is float32 CHW, 0-255.

    Kept as a standalone function so it can be tested against a known-good
    calibration set (see --self-test).
    """
    out = chw.astype(np.float32, copy=True)
    if norm_type in (None, "", "no_preprocess"):
        return out
    if norm_type == "data_scale":
        if scale is None:
            raise SystemExit("norm_type=data_scale requires scale_value")
        return out * scale.reshape(3, 1, 1)
    if norm_type == "data_mean":
        if mean is None:
            raise SystemExit("norm_type=data_mean requires mean_value")
        return out - mean.reshape(3, 1, 1)
    if norm_type == "data_mean_and_scale":
        if mean is None or scale is None:
            raise SystemExit(
                "norm_type=data_mean_and_scale requires mean_value and scale_value"
            )
        return (out - mean.reshape(3, 1, 1)) * scale.reshape(3, 1, 1)
    raise SystemExit(
        f"unsupported norm_type {norm_type!r}. If the model genuinely needs "
        f"something else, add it here — do not pre-normalise elsewhere."
    )


def load_config(path):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    ip = cfg.get("input_parameters") or {}
    cp = cfg.get("calibration_parameters") or {}

    shape = ip.get("input_shape")
    if not shape:
        raise SystemExit(f"{path}: input_parameters.input_shape is required")
    dims = [int(d) for d in str(shape).lower().split("x")]
    if len(dims) != 4:
        raise SystemExit(f"{path}: expected a 4-D input_shape, got {shape!r}")

    layout = (ip.get("input_layout_train") or "NCHW").upper()
    if layout == "NCHW":
        _, c, h, w = dims
    elif layout == "NHWC":
        _, h, w, c = dims
    else:
        raise SystemExit(f"{path}: unsupported input_layout_train {layout!r}")
    if c != 3:
        raise SystemExit(f"{path}: only 3-channel inputs are handled, got C={c}")

    return {
        "h": h,
        "w": w,
        "layout": layout,
        "colour": (ip.get("input_type_train") or "rgb").lower(),
        "norm_type": ip.get("norm_type"),
        "mean": _parse_triplet(ip.get("mean_value"), "mean_value"),
        "scale": _parse_triplet(ip.get("scale_value"), "scale_value"),
        "cal_dir": cp.get("cal_data_dir"),
        "cal_dtype": (cp.get("cal_data_type") or "float32").lower(),
    }


def self_test(cfg, raw_dir, ref_dir):
    """Prove the normalisation matches a known-good set, byte for byte.

    Reads raw (un-normalised) calibration files and the reference normalised
    ones produced by the original conversion, and checks this module reproduces
    the reference. This is what makes the tool trustworthy on models whose
    calibration set we cannot regenerate from source images.
    """
    raws = sorted(glob.glob(os.path.join(raw_dir, "*.bin")))
    refs = sorted(glob.glob(os.path.join(ref_dir, "*.bin")))
    if not raws or len(raws) != len(refs):
        raise SystemExit(f"self-test: {len(raws)} raw vs {len(refs)} reference files")
    shape = (3, cfg["h"], cfg["w"]) if cfg["layout"] == "NCHW" else (cfg["h"], cfg["w"], 3)
    worst = 0.0
    for r, e in zip(raws, refs):
        a = np.fromfile(r, dtype=np.float32).reshape(shape)
        want = np.fromfile(e, dtype=np.float32).reshape(shape)
        if cfg["layout"] == "NHWC":
            got = normalize(a.transpose(2, 0, 1), cfg["norm_type"], cfg["mean"],
                            cfg["scale"]).transpose(1, 2, 0)
        else:
            got = normalize(a, cfg["norm_type"], cfg["mean"], cfg["scale"])
        worst = max(worst, float(np.abs(got - want).max()))
    print(f"self-test: {len(raws)} files, max abs deviation {worst:.3e}")
    if worst > 1e-4:
        raise SystemExit("self-test FAILED — normalisation does not match reference")
    print("self-test PASSED")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="the model's hb_compile config.yaml")
    ap.add_argument("--images", help="directory or glob of source images")
    ap.add_argument("--out", help="override cal_data_dir from the config")
    ap.add_argument("--limit", type=int, default=64,
                    help="how many images to pack (default 64)")
    ap.add_argument("--self-test", nargs=2, metavar=("RAW_DIR", "REF_DIR"),
                    help="verify normalisation against a known-good calibration set")
    args = ap.parse_args()

    cfg = load_config(args.config)

    if args.self_test:
        self_test(cfg, *args.self_test)
        return

    if not args.images:
        ap.error("--images is required unless --self-test is given")

    import cv2  # noqa: E402  (see the note beside the numpy import)

    out_dir = args.out or cfg["cal_dir"]
    if not out_dir:
        raise SystemExit("no cal_data_dir in the config and no --out given")
    # cal_data_dir is written as a container path (/ws/...); map it back to the
    # model directory when running outside the container.
    if out_dir.startswith("/ws/") and not os.path.isdir("/ws"):
        out_dir = os.path.join(os.path.dirname(os.path.abspath(args.config)),
                               os.path.basename(out_dir.rstrip("/")))
    os.makedirs(out_dir, exist_ok=True)

    pats = args.images
    files = sorted(glob.glob(os.path.join(pats, "*")) if os.path.isdir(pats)
                   else glob.glob(pats))
    files = [f for f in files
             if os.path.splitext(f)[1].lower() in
             (".jpg", ".jpeg", ".png", ".bmp", ".webp")][: args.limit]
    if not files:
        raise SystemExit(f"no images found under {pats!r}")

    manifest = {
        "config": os.path.basename(args.config),
        "norm_type": cfg["norm_type"],
        "mean_value": None if cfg["mean"] is None else cfg["mean"].tolist(),
        "scale_value": None if cfg["scale"] is None else cfg["scale"].tolist(),
        "input_shape": f"1x3x{cfg['h']}x{cfg['w']}",
        "layout": cfg["layout"],
        "colour": cfg["colour"],
        "count": len(files),
        "sources": [],
    }

    for i, path in enumerate(files):
        img = cv2.imread(path, cv2.IMREAD_COLOR)  # BGR HWC uint8
        if img is None:
            raise SystemExit(f"could not read {path}")
        img = cv2.resize(img, (cfg["w"], cfg["h"]), interpolation=cv2.INTER_LINEAR)
        if cfg["colour"] == "rgb":
            img = img[:, :, ::-1]
        elif cfg["colour"] != "bgr":
            raise SystemExit(f"unsupported input_type_train {cfg['colour']!r}")

        chw = np.ascontiguousarray(img.transpose(2, 0, 1)).astype(np.float32)
        chw = normalize(chw, cfg["norm_type"], cfg["mean"], cfg["scale"])
        arr = chw if cfg["layout"] == "NCHW" else chw.transpose(1, 2, 0)

        if cfg["cal_dtype"] == "float32":
            arr = arr.astype(np.float32)
        elif cfg["cal_dtype"] == "uint8":
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        else:
            raise SystemExit(f"unsupported cal_data_type {cfg['cal_dtype']!r}")

        dst = os.path.join(out_dir, f"{i:03d}.bin")
        arr.tofile(dst)
        manifest["sources"].append({
            "index": i,
            "source": os.path.basename(path),
            "sha256": hashlib.sha256(open(path, "rb").read()).hexdigest()[:16],
        })

    # The manifest is how a rebuild proves it used the same images as the build
    # the accuracy numbers came from — calibration data is not committed.
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    a = np.fromfile(os.path.join(out_dir, "000.bin"), dtype=np.float32)
    print(f"wrote {len(files)} files to {out_dir}  dtype={cfg['cal_dtype']}")
    print(f"norm_type={cfg['norm_type']}  range [{a.min():.4f}, {a.max():.4f}]")
    if cfg["norm_type"] not in (None, "", "no_preprocess") and a.max() > 64:
        print("WARNING: values look un-normalised despite a norm_type being set.",
              file=sys.stderr)


if __name__ == "__main__":
    main()
