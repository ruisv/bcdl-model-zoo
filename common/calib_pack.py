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

Two ways to use it:

  CLI — single image input, sourced from a directory:
      python calib_pack.py --config models/<name>/config.yaml \\
                           --images /path/to/images --limit 64

  Library — anything else. Models whose calibration needs real geometry (stereo
  pairs, letterbox modes, crops driven by a detector) own that logic in their
  own calib.py and call pack() with the arrays they built. The geometry is
  model-specific and does not belong in here; the normalisation invariant does,
  and stays enforced either way:

      from calib_pack import load_config, pack
      cfg = load_config("config.yaml")
      pack(cfg.inputs[0], left_arrays)     # CHW float32, 0-255 for images
      pack(cfg.inputs[1], right_arrays)

hb_compile takes multi-input models as `;`-separated lists in every one of
input_name / input_type_train / norm_type / cal_data_dir, and those lists must
line up positionally. Mismatched lengths are rejected here rather than at
compile time, where the error does not name the field.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys
from dataclasses import dataclass, field
from typing import Iterable, Optional

import numpy as np
import yaml

# cv2 is imported lazily inside the CLI path: --self-test and the library API
# need only numpy and yaml, and the toolchain env does not carry OpenCV.

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def _split(v, n, name):
    """hb_compile encodes per-input values as a `;`-separated list."""
    if v is None:
        return [None] * n
    parts = [p.strip() for p in str(v).split(";")]
    if len(parts) == 1 and n > 1:
        return parts * n
    if len(parts) != n:
        raise SystemExit(
            f"{name}: {len(parts)} value(s) for {n} input(s). Every per-input "
            f"field must line up positionally with input_name."
        )
    return parts


def _parse_triplet(v, name):
    """Accepted as a space-separated string or a bare scalar."""
    if v is None or v == "":
        return None
    if isinstance(v, (int, float)):
        return np.array([float(v)] * 3, dtype=np.float32)
    parts = str(v).split()
    if len(parts) == 1:
        return np.array([float(parts[0])] * 3, dtype=np.float32)
    if len(parts) != 3:
        raise SystemExit(f"{name}: expected 1 or 3 values, got {len(parts)}: {v!r}")
    return np.array([float(p) for p in parts], dtype=np.float32)


@dataclass
class InputSpec:
    """One model input: how to normalise it and where its calibration data goes."""
    name: str
    norm_type: Optional[str]
    colour: str
    cal_dir: Optional[str]
    mean: Optional[np.ndarray] = None
    scale: Optional[np.ndarray] = None
    h: Optional[int] = None
    w: Optional[int] = None
    layout: str = "NCHW"
    dtype: str = "float32"
    fmt: str = "npy"   # OE 3.7.0 asks for npy; bin is legacy

    @property
    def is_image(self) -> bool:
        # `featuremap` inputs are arbitrary tensors: no colour order, and the
        # caller supplies them directly rather than reading images from disk.
        return self.colour in ("rgb", "bgr")


@dataclass
class CalibConfig:
    inputs: list = field(default_factory=list)
    path: str = ""


def normalize(chw: np.ndarray, spec: InputSpec) -> np.ndarray:
    """Apply exactly what the runtime will apply. `chw` is float32, CHW.

    Standalone so it can be checked against a known-good calibration set
    (see --self-test).
    """
    out = chw.astype(np.float32, copy=True)
    nt = spec.norm_type
    if nt in (None, "", "no_preprocess"):
        return out
    c = out.shape[0]
    if nt == "data_scale":
        if spec.scale is None:
            raise SystemExit(f"{spec.name}: norm_type=data_scale requires scale_value")
        return out * spec.scale[:c].reshape(c, 1, 1)
    if nt == "data_mean":
        if spec.mean is None:
            raise SystemExit(f"{spec.name}: norm_type=data_mean requires mean_value")
        return out - spec.mean[:c].reshape(c, 1, 1)
    if nt == "data_mean_and_scale":
        if spec.mean is None or spec.scale is None:
            raise SystemExit(
                f"{spec.name}: norm_type=data_mean_and_scale requires both "
                f"mean_value and scale_value"
            )
        return (out - spec.mean[:c].reshape(c, 1, 1)) * spec.scale[:c].reshape(c, 1, 1)
    raise SystemExit(
        f"{spec.name}: unsupported norm_type {nt!r}. If the model genuinely "
        f"needs something else, add it here — do not pre-normalise elsewhere."
    )


def _resolve_dir(d: Optional[str], config_path: str) -> Optional[str]:
    """cal_data_dir is written as a container path; map it back outside one.

    compile.sh mounts the model directory at /ws, so `/ws/<rest>` is
    `<model dir>/<rest>`. The whole suffix has to survive: collapsing it to a
    basename merges `/ws/cal_crop/left` and `/ws/cal_resize/left` into the same
    directory, which is precisely the crop/resize mix-up this model punishes.
    """
    if not d:
        return d
    if d.startswith("/ws/") and not os.path.isdir("/ws"):
        return os.path.join(os.path.dirname(os.path.abspath(config_path)),
                            *d[len("/ws/"):].rstrip("/").split("/"))
    return d


def load_config(path: str) -> CalibConfig:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    ip = cfg.get("input_parameters") or {}
    cp = cfg.get("calibration_parameters") or {}

    names = [n.strip() for n in str(ip.get("input_name") or "input").split(";")]
    n = len(names)

    norms = _split(ip.get("norm_type"), n, "norm_type")
    colours = _split(ip.get("input_type_train") or "rgb", n, "input_type_train")
    means = _split(ip.get("mean_value"), n, "mean_value")
    scales = _split(ip.get("scale_value"), n, "scale_value")
    shapes = _split(ip.get("input_shape"), n, "input_shape")
    layouts = _split(ip.get("input_layout_train") or "NCHW", n, "input_layout_train")
    dirs = _split(cp.get("cal_data_dir"), n, "cal_data_dir")
    # Per-input like every other field. hb_compile rejects a scalar here on a
    # multi-input model ("Num of cal_data_type given: 1 is not equal to input
    # num 2"), so it cannot be treated as a single value.
    #
    # It is also deprecated as of OE 3.7.0, which asks for npy calibration data
    # instead; prefer omitting it entirely on new models. Legacy configs that
    # still carry it (with .bin data) keep working.
    dtypes = _split(cp.get("cal_data_type"), n, "cal_data_type")

    specs = []
    for i in range(n):
        h = w = None
        layout = (layouts[i] or "NCHW").upper()
        if shapes[i]:
            dims = [int(d) for d in str(shapes[i]).lower().split("x")]
            if len(dims) == 4:
                if layout == "NCHW":
                    _, _, h, w = dims
                elif layout == "NHWC":
                    _, h, w, _ = dims
                else:
                    raise SystemExit(f"unsupported input_layout_train {layout!r}")
        specs.append(InputSpec(
            name=names[i],
            norm_type=norms[i],
            colour=(colours[i] or "rgb").lower(),
            cal_dir=_resolve_dir(dirs[i], path),
            mean=_parse_triplet(means[i], "mean_value"),
            scale=_parse_triplet(scales[i], "scale_value"),
            h=h, w=w, layout=layout,
            dtype=(dtypes[i] or "float32").lower(),
        ))
    return CalibConfig(inputs=specs, path=path)


def pack(spec: InputSpec, arrays: Iterable[np.ndarray], out_dir: str = None,
         fmt: str = None, sources: list = None) -> str:
    """Normalise and write one input's calibration set.

    `arrays` yields CHW float32 — 0-255 for image inputs, whatever the model
    expects for featuremap inputs. Normalisation comes from `spec` (i.e. from
    config.yaml) and cannot be overridden here; that is the point of the module.
    """
    out_dir = out_dir or spec.cal_dir
    if not out_dir:
        raise SystemExit(f"{spec.name}: no cal_data_dir in config and no out_dir given")
    fmt = fmt or spec.fmt
    os.makedirs(out_dir, exist_ok=True)

    count = 0
    lo = hi = None
    for i, arr in enumerate(arrays):
        a = np.asarray(arr, dtype=np.float32)
        if a.ndim == 4:
            if a.shape[0] != 1:
                raise SystemExit(f"{spec.name}: expected batch 1, got {a.shape}")
            a = a[0]
        if a.ndim != 3:
            raise SystemExit(f"{spec.name}: expected CHW or 1CHW, got {a.shape}")

        a = normalize(a, spec)
        out = a if spec.layout == "NCHW" else a.transpose(1, 2, 0)

        if spec.dtype == "float32":
            out = out.astype(np.float32)
        elif spec.dtype == "uint8":
            out = np.clip(out, 0, 255).astype(np.uint8)
        else:
            raise SystemExit(f"unsupported cal_data_type {spec.dtype!r}")

        if fmt == "npy":
            np.save(os.path.join(out_dir, f"{i:03d}.npy"), out[None])
        else:
            out.tofile(os.path.join(out_dir, f"{i:03d}.bin"))

        lo = out.min() if lo is None else min(lo, out.min())
        hi = out.max() if hi is None else max(hi, out.max())
        count += 1

    if count == 0:
        raise SystemExit(f"{spec.name}: no calibration samples were written")

    # The manifest is how a rebuild proves it used the same inputs as the build
    # the accuracy numbers came from — calibration data is not committed.
    manifest = {
        "input_name": spec.name,
        "norm_type": spec.norm_type,
        "mean_value": None if spec.mean is None else spec.mean.tolist(),
        "scale_value": None if spec.scale is None else spec.scale.tolist(),
        "layout": spec.layout,
        "dtype": spec.dtype,
        "format": fmt,
        "count": count,
        "value_range": [float(lo), float(hi)],
        "sources": sources or [],
    }
    # NOT inside out_dir: hb_compile treats every file in cal_data_dir as a
    # calibration sample and fails trying to reshape the manifest into an input
    # ("cannot reshape array of size 196 into shape (1,3,480,640)").
    manifest_path = os.path.join(os.path.dirname(out_dir.rstrip("/")),
                                 f"{os.path.basename(out_dir.rstrip('/'))}.manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"  {spec.name}: {count} x {fmt}  norm={spec.norm_type}  "
          f"range [{lo:.4f}, {hi:.4f}]  -> {out_dir}")
    if spec.norm_type not in (None, "", "no_preprocess") and hi > 64:
        print(f"  WARNING: {spec.name} looks un-normalised despite norm_type="
              f"{spec.norm_type}.", file=sys.stderr)
    return out_dir


def _read_images(files, spec: InputSpec):
    import cv2
    for path in files:
        img = cv2.imread(path, cv2.IMREAD_COLOR)  # BGR HWC uint8
        if img is None:
            raise SystemExit(f"could not read {path}")
        if spec.h and spec.w:
            img = cv2.resize(img, (spec.w, spec.h), interpolation=cv2.INTER_LINEAR)
        if spec.colour == "rgb":
            img = img[:, :, ::-1]
        yield np.ascontiguousarray(img.transpose(2, 0, 1)).astype(np.float32)


def self_test(cfg: CalibConfig, raw_dir: str, ref_dir: str):
    """Prove normalisation matches a known-good set, byte for byte.

    Reads raw (un-normalised) calibration files and the reference normalised
    ones from the original conversion, and checks this module reproduces them.
    This is what makes the tool trustworthy for models whose calibration set
    cannot be regenerated from source images.
    """
    spec = cfg.inputs[0]
    raws = sorted(glob.glob(os.path.join(raw_dir, "*.bin")))
    refs = sorted(glob.glob(os.path.join(ref_dir, "*.bin")))
    if not raws or len(raws) != len(refs):
        raise SystemExit(f"self-test: {len(raws)} raw vs {len(refs)} reference files")
    shape = ((3, spec.h, spec.w) if spec.layout == "NCHW"
             else (spec.h, spec.w, 3))
    worst = 0.0
    for r, e in zip(raws, refs):
        a = np.fromfile(r, dtype=np.float32).reshape(shape)
        want = np.fromfile(e, dtype=np.float32).reshape(shape)
        if spec.layout == "NHWC":
            got = normalize(a.transpose(2, 0, 1), spec).transpose(1, 2, 0)
        else:
            got = normalize(a, spec)
        worst = max(worst, float(np.abs(got - want).max()))
    print(f"self-test: {len(raws)} files, max abs deviation {worst:.3e}")
    if worst > 1e-4:
        raise SystemExit("self-test FAILED — normalisation does not match reference")
    print("self-test PASSED")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="the model's hb_compile config.yaml")
    ap.add_argument("--images", help="directory or glob of source images")
    ap.add_argument("--out", help="override cal_data_dir from the config")
    ap.add_argument("--limit", type=int, default=64, help="images to pack (default 64)")
    ap.add_argument("--format", choices=["npy", "bin"], default="npy",
                    help="npy is what the toolchain now asks for; bin is legacy")
    ap.add_argument("--self-test", nargs=2, metavar=("RAW_DIR", "REF_DIR"),
                    help="verify normalisation against a known-good calibration set")
    args = ap.parse_args()

    cfg = load_config(args.config)

    if args.self_test:
        self_test(cfg, *args.self_test)
        return

    if not args.images:
        ap.error("--images is required unless --self-test is given")
    if len(cfg.inputs) != 1:
        ap.error(
            f"this model has {len(cfg.inputs)} inputs "
            f"({', '.join(i.name for i in cfg.inputs)}); the CLI handles the "
            f"single-image-input case only. Multi-input models pair their "
            f"samples with model-specific logic, so write the model's calib.py "
            f"against the library API — see the module docstring."
        )
    spec = cfg.inputs[0]
    if not spec.is_image:
        ap.error(f"input_type_train={spec.colour!r} is not an image input; "
                 f"use the library API from the model's calib.py")

    pats = args.images
    files = sorted(glob.glob(os.path.join(pats, "*")) if os.path.isdir(pats)
                   else glob.glob(pats))
    files = [f for f in files
             if os.path.splitext(f)[1].lower() in IMAGE_EXTS][: args.limit]
    if not files:
        raise SystemExit(f"no images found under {pats!r}")

    sources = [{"index": i, "source": os.path.basename(f),
                "sha256": hashlib.sha256(open(f, "rb").read()).hexdigest()[:16]}
               for i, f in enumerate(files)]
    print(f"packing {len(files)} images from {pats}")
    pack(spec, _read_images(files, spec), out_dir=args.out, fmt=args.format,
         sources=sources)


if __name__ == "__main__":
    main()
