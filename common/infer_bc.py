#!/usr/bin/env python3
"""Run a quantized .bc model on fixed inputs — the host reference for layer B.

Runs INSIDE the OpenExplorer container (hbdk4 lives only there). Loads the
`<prefix>_quantized_model.bc` that hb_compile emits, feeds the .npy inputs, and
saves each output as an .npy. verify_cosine.py then compares this against the
board's .hbm output, which must be bit-identical (layer B).

Inputs are matched to the model's declared input names, in order. Pass one
--input per model input, in signature order:

    python infer_bc.py --bc model_quantized_model.bc \\
        --input left:left.npy --input right:right.npy --out-prefix bc

writes bc.0.npy, bc.1.npy, ... one per output.
"""
from __future__ import annotations

import argparse

import numpy as np
import hbdk4.compiler as hbdk


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bc", required=True)
    ap.add_argument("--input", action="append", required=True,
                    help="name:path.npy, one per model input (signature order)")
    ap.add_argument("--out-prefix", default="bc")
    args = ap.parse_args()

    module = hbdk.load(args.bc)
    func = module[0]
    want = [a.name for a in func.inputs]

    supplied = {}
    order = []
    for spec in args.input:
        name, path = spec.split(":", 1)
        supplied[name] = np.load(path).astype(np.float32)
        order.append(name)

    if set(supplied) != set(want):
        raise SystemExit(f"inputs {order} do not match model inputs {want}")
    # Feed in the model's own order, not the order given on the command line.
    arrays = [supplied[n] for n in want]

    out = func(*arrays)
    if isinstance(out, dict):
        out = list(out.values())
    elif not isinstance(out, (list, tuple)):
        out = [out]

    for i, v in enumerate(out):
        a = np.ascontiguousarray(np.asarray(v))
        np.save(f"{args.out_prefix}.{i}.npy", a)
        print(f"  {args.out_prefix}.{i}.npy  shape={a.shape} dtype={a.dtype}")


if __name__ == "__main__":
    main()
