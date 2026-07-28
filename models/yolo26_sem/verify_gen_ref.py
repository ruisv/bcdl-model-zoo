#!/usr/bin/env python3
"""Accuracy verification step 1b (run inside the OE docker container on the
convert host -- it carries no OpenCV, so verify_prep.py must run first,
outside the container, to produce the input .npy files this reads).

Loads the NV12 + RGB float inputs verify_prep.py wrote and runs both the
float ONNX and the quantised .bc, saving each output and printing layer A
(quantisation fidelity, .bc vs onnx). See ../../docs (project
docs/10-accuracy-verification.md, private) or
[[accuracy-verification-method]] for the three-layer method this belongs to.

Usage (cwd = this model's directory, inside the OE docker):
    python3 verify_gen_ref.py --ref ./out_verify --model-dir ./out
"""
import argparse
import glob
import os
import sys

import numpy as np
from horizon_tc_ui.hb_runtime import HBRuntime


def cos(a, b):
    a = a.astype("float64").ravel()
    b = b.astype("float64").ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="./out_verify",
                     help="dir with in_y.npy/in_uv.npy/in_rgb01_chw.npy from verify_prep.py")
    ap.add_argument("--model-dir", default="./out",
                     help="hb_compile working_dir with *_original_float_model.onnx "
                          "and *_quantized_model.bc")
    args = ap.parse_args()

    onnx_path = glob.glob(os.path.join(args.model_dir, "*_original_float_model.onnx"))
    if not onnx_path:
        sys.exit(f"no *_original_float_model.onnx under {args.model_dir}")
    bc_path = glob.glob(os.path.join(args.model_dir, "*_quantized_model.bc"))
    if not bc_path:
        sys.exit(f"no *_quantized_model.bc under {args.model_dir}")

    # ---- float ONNX reference ----
    so = HBRuntime(model=onnx_path[0])
    chw = np.load(os.path.join(args.ref, "in_rgb01_chw.npy"))
    oo = [np.asarray(x) for x in so.run(so.output_names, {so.input_names[0]: chw})]
    for n, a in zip(so.output_names, oo):
        np.save(os.path.join(args.ref, f"out_onnx__{n.replace('/', '_')}.npy"), a)
    print("onnx:", onnx_path[0], "outputs:", [a.shape for a in oo])

    # ---- quantised .bc reference ----
    sb = HBRuntime(model=bc_path[0])
    y = np.load(os.path.join(args.ref, "in_y.npy")).reshape(sb.input_shapes[0]).astype(np.uint8)
    uv = np.load(os.path.join(args.ref, "in_uv.npy")).reshape(sb.input_shapes[1]).astype(np.uint8)
    feed = dict(zip(sb.input_names, [y, uv]))
    ob = [np.asarray(x) for x in sb.run(sb.output_names, feed)]
    for n, a in zip(sb.output_names, ob):
        np.save(os.path.join(args.ref, f"out_bc__{n.replace('/', '_')}.npy"), a)
    print("bc:", bc_path[0], "inputs:", sb.input_names, sb.input_shapes)

    for i, (a, b) in enumerate(zip(ob, oo)):
        print(f"[layer A] output[{i}] cosine(bc, onnx) = {cos(a, b):.6f}")


if __name__ == "__main__":
    main()
