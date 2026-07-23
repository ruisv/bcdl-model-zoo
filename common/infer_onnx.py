#!/usr/bin/env python3
"""Run an ONNX model on fixed inputs — the host reference for layer C.

Runs on the host in an env with onnxruntime. Feeds the same .npy inputs the
board .hbm will get, and saves each output as an .npy for verify_cosine.py.

    python infer_onnx.py --onnx model.onnx \\
        --input left:left.npy --input right:right.npy --out-prefix onnx
"""
from __future__ import annotations

import argparse

import numpy as np
import onnxruntime as ort


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--input", action="append", required=True,
                    help="name:path.npy, one per model input")
    ap.add_argument("--out-prefix", default="onnx")
    args = ap.parse_args()

    sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    want = [i.name for i in sess.get_inputs()]

    feed = {}
    for spec in args.input:
        name, path = spec.split(":", 1)
        feed[name] = np.load(path).astype(np.float32)
    if set(feed) != set(want):
        raise SystemExit(f"inputs {list(feed)} do not match model inputs {want}")

    outs = sess.run(None, feed)
    for i, v in enumerate(outs):
        a = np.ascontiguousarray(np.asarray(v))
        np.save(f"{args.out_prefix}.{i}.npy", a)
        print(f"  {args.out_prefix}.{i}.npy  shape={a.shape} dtype={a.dtype}")


if __name__ == "__main__":
    main()
