#!/usr/bin/env python3
"""[1/3] Prepare the two face ONNX graphs from an insightface model pack.

This is NOT a torch export. Both graphs already exist, pre-exported, inside the
insightface `buffalo_l` pack (the same pack `insightface`'s FaceAnalysis app
downloads). The recipe is *locate + tiny edit*, and stating that plainly is the
point of this file:

    detection    det_10g.onnx   ->  scrfd_10g_640.onnx    (SCRFD-10G)
    recognition  w600k_r50.onnx ->  arcface_r50_112.onnx  (ArcFace R50)

- SCRFD (det_10g): the upstream graph is dynamic in H/W. The only work is
  pinning the input to a static 1x3x640x640 and letting the shapes resolve, so
  the BPU compiler sees a fully static graph. Nine raw per-stride outputs
  (score/bbox/kps x strides 8/16/32) are emitted unchanged; BCDL decodes them.
- ArcFace (w600k_r50): already 112x112. The edit is *just* fixing the input to a
  static 1x3x112x112 (upstream leaves the batch axis symbolic). It is a handful
  of bytes on one dim -- arcface_r50_112.onnx and w600k_r50.onnx are the same
  graph. main() prints the byte delta to make that provenance auditable.

Both inputs are named `input.1` upstream; the configs key off that name.
HBDK 4.x rejects ONNX IR > 9, so the IR is capped after any rewrite.

Usage:
    python export.py --buffalo /path/to/buffalo_l --task det
    python export.py --buffalo /path/to/buffalo_l --task rec
    python export.py --buffalo /path/to/buffalo_l --task both
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

import onnx

# Upstream file names inside the buffalo_l / antelope pack, and the static input
# each task is compiled at. `input.1` is the upstream input name for both.
SRC = {"det": "det_10g.onnx", "rec": "w600k_r50.onnx"}
OUT = {"det": "scrfd_10g_640.onnx", "rec": "arcface_r50_112.onnx"}
HW = {"det": (640, 640), "rec": (112, 112)}
INPUT_NAME = "input.1"


def cap_ir(path: str) -> None:
    """HBDK 4.x rejects IR > 9. IR9 covers up to opset 20, so this is a header
    change, not an operator downgrade. onnxslim preserves the source IR, so it
    has to be done after slimming."""
    m = onnx.load(path)
    if m.ir_version > 9:
        m.ir_version = 9
        onnx.save(m, path)
        print(f"[export] capped IR version -> 9 ({os.path.basename(path)})")


def static_shapes(path: str) -> dict:
    m = onnx.load(path)
    out = {}
    for group in (m.graph.input, m.graph.output):
        for v in group:
            out[v.name] = [
                (d.dim_value if d.dim_value else (d.dim_param or "?"))
                for d in v.type.tensor_type.shape.dim
            ]
    return out


def assert_static(path: str) -> dict:
    shapes = static_shapes(path)
    for name, dims in shapes.items():
        if any(isinstance(d, str) for d in dims):
            raise SystemExit(
                f"{name} is still dynamic after editing: {dims}. The BPU "
                f"compiler needs fully static shapes."
            )
    return shapes


def export_det(src: str, out: str) -> None:
    """SCRFD-10G: pin the dynamic H/W to 640x640 and resolve the graph.

    onnxslim fixes the input shape and constant-folds the derived per-stride
    output shapes to static -- the same slim step ppocr_v6 uses. No head surgery:
    det_10g already emits the raw score/bbox/kps tensors BCDL decodes.
    """
    h, w = HW["det"]
    print(f"[export] det: {src} -> {out} ({h}x{w})")
    subprocess.run(
        ["onnxslim", src, out, "--input-shapes", f"{INPUT_NAME}:1,3,{h},{w}"],
        check=True, stdout=subprocess.DEVNULL,
    )
    cap_ir(out)
    print(f"[export] det shapes {assert_static(out)}")


def export_rec(src: str, out: str) -> None:
    """ArcFace R50: the tiny edit -- fix `input.1` to a static 1x3x112x112.

    Upstream w600k_r50 is 112x112 already; only the batch axis is left symbolic.
    We rewrite exactly that one input's dims and touch nothing else, so the
    output is w600k_r50 with a few bytes changed. The byte delta is printed as
    proof of provenance.
    """
    h, w = HW["rec"]
    print(f"[export] rec: {src} -> {out} ({h}x{w})")
    m = onnx.load(src)

    inp = next((i for i in m.graph.input if i.name == INPUT_NAME), None)
    if inp is None:
        names = [i.name for i in m.graph.input]
        raise SystemExit(f"{src}: no input named {INPUT_NAME!r}; found {names}. "
                         f"This is not the expected w600k_r50 graph.")
    dims = inp.type.tensor_type.shape.dim
    if len(dims) != 4:
        raise SystemExit(f"{src}: expected a 4-D input, got {len(dims)} dims")
    for d, v in zip(dims, (1, 3, h, w)):
        d.ClearField("dim_param")
        d.dim_value = v

    onnx.save(m, out)
    cap_ir(out)

    src_bytes = os.path.getsize(src)
    out_bytes = os.path.getsize(out)
    print(f"[export] rec shapes {assert_static(out)}")
    print(f"[export] provenance: w600k_r50={src_bytes} B, arcface_r50_112={out_bytes} B, "
          f"delta={out_bytes - src_bytes:+d} B (same graph, static-input edit only)")


def run_task(task: str, buffalo: str) -> None:
    src = os.path.join(buffalo, SRC[task])
    if not os.path.exists(src):
        raise SystemExit(
            f"missing upstream ONNX: {src}\n"
            f"Point --buffalo at the extracted insightface buffalo_l pack "
            f"(it holds {SRC['det']} and {SRC['rec']})."
        )
    (export_det if task == "det" else export_rec)(src, OUT[task])


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--buffalo", required=True,
                    help="extracted insightface buffalo_l pack directory")
    ap.add_argument("--task", default="both", choices=["det", "rec", "both"])
    a = ap.parse_args()

    tasks = ["det", "rec"] if a.task == "both" else [a.task]
    for t in tasks:
        run_task(t, a.buffalo)


if __name__ == "__main__":
    main()
