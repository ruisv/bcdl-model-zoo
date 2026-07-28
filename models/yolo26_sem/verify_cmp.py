#!/usr/bin/env python3
"""Accuracy verification step 2 (run on the S100P board). Feeds the SAME
NV12 input verify_gen_ref.py used through the compiled .hbm, and computes
layer B (execution consistency, expect bit-identical) and layer C
(end-to-end, board .hbm vs float onnx).

Usage:
    python verify_cmp.py <hbm_path> <ref_dir>
"""
import glob
import os
import sys

import numpy as np
from hbm_runtime import HB_HBMRuntime


def cos(a, b):
    a = a.astype("float64").ravel()
    b = b.astype("float64").ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def main():
    hbm = sys.argv[1]
    ref = sys.argv[2] if len(sys.argv) > 2 else (os.path.dirname(hbm) or ".")

    m = HB_HBMRuntime(hbm)
    name = m.model_names[0]
    feed = {
        "images_y": np.load(os.path.join(ref, "in_y.npy")),
        "images_uv": np.load(os.path.join(ref, "in_uv.npy")),
    }
    res = m.run(feed)[name]
    outs = [np.asarray(v) for v in (res.values() if hasattr(res, "items") else res)]

    onnx_files = sorted(glob.glob(os.path.join(ref, "out_onnx__*.npy")))
    bc_files = sorted(glob.glob(os.path.join(ref, "out_bc__*.npy")))

    for i, oh in enumerate(outs):
        print(f"== output[{i}] shape={oh.shape}")
        if i < len(onnx_files):
            oo = np.load(onnx_files[i])
            print(f"   [layer C] cosine(hbm, onnx) = {cos(oh, oo):.6f}")
        if i < len(bc_files):
            ob = np.load(bc_files[i])
            d = float(np.abs(oh.astype("float64").ravel()
                              - ob.astype("float64").ravel()).max())
            print(f"   [layer B] cosine(hbm, bc)   = {cos(oh, ob):.6f}   "
                  f"max|diff| = {d}   exact={bool(np.array_equal(oh, ob))}")


if __name__ == "__main__":
    main()
