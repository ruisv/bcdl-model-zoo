#!/usr/bin/env python3
"""Three-layer accuracy check for a converted model, gated on expected.json.

The three layers isolate *where* a regression came from, which matters because
they fail for different reasons and have different fixes:

  A  quantisation fidelity   host .bc     vs ONNX        cosine > 0.99
  B  execution consistency   board .hbm   vs host .bc    bit-identical
  C  end-to-end              board .hbm   vs ONNX        cosine > 0.99

A alone tells you the PTQ was sound. B alone tells you the board runtime agrees
with the host simulation — it is expected to be *exact*, so any drift at B is a
runtime or stride problem, never a quantisation one. C is the one users feel.
Running only C leaves you unable to say which half broke.

Layer B deserves its strictness: an input-stride bug (a row-padded tensor fed as
if it were packed) produces plausible-looking output and moved one model's
cosine to 0.015 while every compile-time check stayed green. B catches that
class immediately, because correct plumbing is bit-exact, not merely close.

This tool does not run inference — the three environments are mutually
exclusive (ONNX on the host, .bc inside the toolchain container, .hbm on the
board). Each side saves its outputs as .npy and this compares them.

    python verify_cosine.py --layer B --ref host_bc.npy --test board_hbm.npy \\
                            --expected models/<name>/expected.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

# Layer B compares two executions of the same compiled graph, so anything short
# of exact equality is a defect rather than a tolerance question.
DEFAULT_THRESHOLDS = {"A": 0.99, "B": 1.0, "C": 0.99}


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        # Two all-zero tensors are consistent; one zero and one not is a total
        # failure. Reporting 1.0 for the second case would hide it.
        return 1.0 if na == nb else 0.0
    return float(np.dot(a, b) / (na * nb))


def load(path: str) -> np.ndarray:
    if not os.path.exists(path):
        raise SystemExit(f"missing tensor file: {path}")
    if path.endswith(".npy"):
        return np.load(path)
    raise SystemExit(f"expected a .npy file, got {path!r}")


def compare(ref: np.ndarray, test: np.ndarray, layer: str, threshold: float):
    if ref.shape != test.shape:
        # A shape mismatch is usually the wrong build rather than a bad one —
        # say so, because the cosine number below would be meaningless.
        raise SystemExit(
            f"shape mismatch: ref {ref.shape} vs test {test.shape}. "
            f"This is normally the wrong build or the wrong output index, "
            f"not an accuracy problem."
        )

    cos = cosine(ref, test)
    max_abs = float(np.abs(ref.astype(np.float64) - test.astype(np.float64)).max())
    exact = bool(np.array_equal(ref, test))

    print(f"layer {layer}: cosine={cos:.6f}  max_abs_err={max_abs:.6e}  exact={exact}")

    if layer == "B":
        ok = exact
        if not ok:
            print(
                "\nlayer B is not bit-identical. The board and the host ran the "
                "same compiled graph, so this is a plumbing fault, not a "
                "quantisation one. Check the input stride first: a row-padded "
                "buffer fed as packed reproduces exactly this signature.",
                file=sys.stderr,
            )
    else:
        ok = cos >= threshold
        if not ok:
            print(
                f"\ncosine {cos:.6f} is below the {threshold} gate for layer "
                f"{layer}.",
                file=sys.stderr,
            )
    return ok, {"cosine": cos, "max_abs_err": max_abs, "exact": exact}


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--layer", required=True, choices=["A", "B", "C"])
    ap.add_argument("--ref", required=True, help="reference tensor (.npy)")
    ap.add_argument("--test", required=True, help="tensor under test (.npy)")
    ap.add_argument("--expected", help="model's expected.json, for thresholds")
    ap.add_argument("--update", action="store_true",
                    help="write the measured values back into expected.json")
    args = ap.parse_args()

    exp = {}
    if args.expected and os.path.exists(args.expected):
        with open(args.expected) as f:
            exp = json.load(f)

    threshold = (exp.get("cosine") or {}).get(args.layer,
                                              DEFAULT_THRESHOLDS[args.layer])

    ref, test = load(args.ref), load(args.test)
    ok, measured = compare(ref, test, args.layer, threshold)

    task = exp.get("task_metric")
    if task and args.layer == "C":
        # Cosine is not sufficient wherever a task metric exists. OSNet is the
        # worked example: int8 PTQ returns well-formed unit vectors that pass a
        # cosine check while Market-1501 Rank-1 falls from 85% to 51%.
        print(
            f"\nNOTE: this model declares a task metric "
            f"({task.get('name')} >= {task.get('threshold')}). Cosine alone "
            f"does not accept it — measure the task metric separately.",
        )

    if args.update and args.expected:
        exp.setdefault("measured", {})[args.layer] = measured
        with open(args.expected, "w") as f:
            json.dump(exp, f, indent=2)
        print(f"updated {args.expected}")

    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
