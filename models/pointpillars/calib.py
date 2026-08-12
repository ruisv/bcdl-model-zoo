#!/usr/bin/env python3
"""[2/3] PTQ calibration data for both PointPillars graphs, from real sweeps.

    python calib.py --config config_vfe.yaml    --velodyne /path/to/kitti/velodyne --limit 8
    python calib.py --config config_bbhead.yaml --velodyne /path/to/kitti/velodyne \\
                    --vfe-onnx vfe_conv.onnx --limit 8

Neither input is pixels, so there is nothing to normalise — both are
`no_preprocess` featuremaps — but each is a *derived* tensor, and deriving it
differently from the runtime is the same failure as feeding un-normalised images:
the model compiles clean and the activation statistics were gathered on a
distribution the board never sees.

  * the VFE's input is the 10-channel expansion `(1, 10, P, 32)`, which the
    runtime computes with `bcdl::pillarPointFeatures()`. `_expand()` below is the
    same arithmetic; if you change one, change both.
  * the backbone's input is the scatter canvas `(1, 64, gy, gx)`, which only
    exists downstream of the VFE. So this script RUNS the exported float
    `vfe_conv.onnx` to produce it. Calibrating the backbone on anything else —
    random data, or the published VFE's output — describes a distribution the
    deployed stack does not produce.

**Sample count is bounded by size here, not by taste.** One VFE sample is 15 MB
and one canvas sample is 53 MB, so a 64-frame set is 4.3 GB. Eight frames of real
KITTI is what this recipe recommends, and the shipped build used **one**, which
is thin — the measured accuracy held up (0.9988 against the float reference), but
if a rebuild degrades, the sample count is the first thing to raise.

Points must be in the frame the model was trained in. KITTI `velodyne/*.bin` is
`(N, 4)` float32 x, y, z, reflectance in the *lidar* frame and PointPillars was
trained on it directly, so no extrinsic is applied. Feeding a different sensor's
cloud without transforming it first is undetectable downstream.
"""
import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "common"))
from calib_pack import load_config, pack  # noqa: E402

# The KITTI configuration the published models and BCDL's VoxelConfig share.
RANGE = (0.0, -39.68, -3.0, 69.12, 39.68, 1.0)      # xmin ymin zmin xmax ymax zmax
PILLAR = (0.16, 0.16)
MAX_POINTS, MAX_PILLARS = 32, 11546


def voxelize(points):
    """Point cloud -> (voxels, num_points, coords), the pillar encoding.

    coords is (batch, z, y, x); z is always 0 and exists because the same slot
    holds a voxel index in 3-D encoders.
    """
    xmin, ymin, zmin, xmax, ymax, zmax = RANGE
    gx = int(round((xmax - xmin) / PILLAR[0]))
    gy = int(round((ymax - ymin) / PILLAR[1]))
    p = points[(points[:, 0] >= xmin) & (points[:, 0] < xmax) &
               (points[:, 1] >= ymin) & (points[:, 1] < ymax) &
               (points[:, 2] >= zmin) & (points[:, 2] < zmax)]
    cx = ((p[:, 0] - xmin) / PILLAR[0]).astype(np.int32)
    cy = ((p[:, 1] - ymin) / PILLAR[1]).astype(np.int32)
    cell = cy * gx + cx

    voxels = np.zeros((MAX_PILLARS, MAX_POINTS, 4), np.float32)
    npts = np.zeros(MAX_PILLARS, np.float32)
    coords = np.zeros((MAX_PILLARS, 4), np.float32)
    slot_of, used = {}, 0
    for i in range(len(p)):
        c = int(cell[i])
        slot = slot_of.get(c)
        if slot is None:
            if used >= MAX_PILLARS:
                continue                       # the cap is real; a dense sweep loses columns
            slot = used
            slot_of[c] = slot
            coords[slot] = (0, 0, cy[i], cx[i])
            used += 1
        n = int(npts[slot])
        if n >= MAX_POINTS:
            continue
        voxels[slot, n] = p[i]
        npts[slot] = n + 1
    return voxels, npts, coords, used


def _expand(voxels, npts, coords):
    """The 10 per-point features, as (10, P, 32). Mirrors pillarPointFeatures()."""
    xmin, ymin, zmin, _, _, zmax = RANGE
    pz = zmax - zmin
    v = voxels.astype(np.float32)
    n = np.maximum(npts, 1.0)[:, None, None]
    mean = v[:, :, :3].sum(1, keepdims=True) / n
    cluster = v[:, :, :3] - mean
    centre = np.empty_like(v[:, :, :3])
    centre[:, :, 0] = v[:, :, 0] - (coords[:, 3:4] * PILLAR[0] + PILLAR[0] / 2 + xmin)
    centre[:, :, 1] = v[:, :, 1] - (coords[:, 2:3] * PILLAR[1] + PILLAR[1] / 2 + ymin)
    centre[:, :, 2] = v[:, :, 2] - (coords[:, 1:2] * pz + pz / 2 + zmin)
    feats = np.concatenate([v, cluster, centre], axis=2)
    # Unfilled slots must be zero BEFORE the MLP: they go through it too, and the
    # max over points would otherwise pick up whatever the bias maps them to.
    live = (np.arange(MAX_POINTS)[None, :] < npts.astype(np.int64)[:, None])[:, :, None]
    return (feats * live).transpose(2, 0, 1)


def _scatter(feats, coords, used):
    """(P, 64) pillar features -> the dense (64, gy, gx) BEV canvas."""
    xmin, ymin, _, xmax, ymax, _ = RANGE
    gx = int(round((xmax - xmin) / PILLAR[0]))
    gy = int(round((ymax - ymin) / PILLAR[1]))
    canvas = np.zeros((feats.shape[1], gy * gx), np.float32)
    idx = (coords[:used, 2] * gx + coords[:used, 3]).astype(np.int64)
    canvas[:, idx] = feats[:used].T
    return canvas.reshape(-1, gy, gx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--velodyne", required=True, help="dir of KITTI velodyne .bin sweeps")
    ap.add_argument("--vfe-onnx", default="vfe_conv.onnx",
                    help="float VFE, needed only for the backbone's calibration")
    ap.add_argument("--limit", type=int, default=8)
    a = ap.parse_args()

    cfg = load_config(a.config)
    spec = cfg.inputs[0]
    want_canvas = spec.name == "spatial_features"

    files = sorted(glob.glob(os.path.join(a.velodyne, "*.bin")))[:a.limit]
    if not files:
        sys.exit(f"no .bin sweeps under {a.velodyne}")

    sess = None
    if want_canvas:
        import onnxruntime as ort
        sess = ort.InferenceSession(a.vfe_onnx, providers=["CPUExecutionProvider"])

    arrays, srcs = [], []
    for f in files:
        pts = np.fromfile(f, np.float32).reshape(-1, 4)
        voxels, npts, coords, used = voxelize(pts)
        x = _expand(voxels, npts, coords)                     # (10, P, 32)
        if want_canvas:
            out = sess.run(None, {sess.get_inputs()[0].name: x[None]})[0]
            arrays.append(_scatter(out[0, :, :, 0].T, coords, used))
        else:
            arrays.append(x)
        srcs.append({"index": len(arrays) - 1, "source": os.path.basename(f),
                     "pillars": used, "points": int(npts.sum())})
        print("  %-16s %6d points -> %5d pillars" % (os.path.basename(f), len(pts), used))

    print("[calib] %d sweeps -> %s %s" % (len(arrays), spec.name, arrays[0].shape))
    pack(spec, arrays, fmt="npy", sources=srcs)


if __name__ == "__main__":
    main()
