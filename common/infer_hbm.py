#!/usr/bin/env python3
"""Run a .hbm model on fixed inputs — ON THE BOARD, the model under test.

Runs on the S100-series board (hbm_runtime env). Feeds the same .npy inputs the
host references got, and saves each output as an .npy to compare with
verify_cosine.py: bit-identical against the .bc (layer B), > 0.99 cosine against
the ONNX (layer C).

    python infer_hbm.py --hbm model.hbm \\
        --input left:left.npy --input right:right.npy --out-prefix hbm
"""
from __future__ import annotations

import argparse

import numpy as np
from hbm_runtime import HB_HBMRuntime


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hbm", required=True)
    ap.add_argument("--input", action="append", required=True,
                    help="name:path.npy, one per model input")
    ap.add_argument("--out-prefix", default="hbm")
    args = ap.parse_args()

    feed = {}
    for spec in args.input:
        name, path = spec.split(":", 1)
        feed[name] = np.ascontiguousarray(np.load(path).astype(np.float32))

    model = HB_HBMRuntime(args.hbm)
    result = model.run(feed)

    # hbm_runtime returns {model_name: {output_name: array}}. There is one model
    # per .hbm here, so unwrap the outer model-name layer, then take the outputs
    # in declared order.
    if isinstance(result, dict):
        inner = next(iter(result.values()))
        outs = list(inner.values()) if isinstance(inner, dict) else [inner]
    elif isinstance(result, (list, tuple)):
        outs = list(result)
    else:
        outs = [result]

    for i, v in enumerate(outs):
        a = np.ascontiguousarray(np.asarray(v))
        np.save(f"{args.out_prefix}.{i}.npy", a)
        print(f"  {args.out_prefix}.{i}.npy  shape={a.shape} dtype={a.dtype}")


if __name__ == "__main__":
    main()
