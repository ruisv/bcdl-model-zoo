#!/usr/bin/env python3
"""[1/3] Produce the BPU-friendly YOLOP ONNX by CUTTING the anchor decode.

This is the whole recipe for this model, and the reason the shipped build is
called `_cut`. YOLOP's published export (`yolop-640-640.onnx`, downloaded
directly from the upstream release — it is byte-identical to hustvl/YOLOP's
release asset, there is no re-export step) bakes the detection anchor decode into
the graph with a `ScatterND`. That graph compiles to a `.hbm` **without a single
error**, loads, and runs — but the objectness and class columns of the detection
output are **never written**: the scatter that was supposed to fill them does not
survive quantised lowering, so every box comes back with zero confidence and the
detector appears to see nothing.

The fix is not a calibration or precision knob. It is to CUT the graph before the
decode and emit the **three raw detection heads** (the per-scale conv outputs)
plus the two segmentation maps, and do the anchor decode on the CPU in BCDL,
where it is a handful of cheap ops on already-small tensors. This is the general
move for a head whose decode does not quantise: keep the convolutional trunk on
the BPU, move the postprocess arithmetic to the host.

YOLOP has three outputs on three tasks:
  * detection (vehicles): the Detect module, three scales at strides 8/16/32,
    each a conv of shape [1, 3*(5+nc), H, W]. nc=1 here, so 18 channels:
    [1,18,80,80], [1,18,40,40], [1,18,20,20]. These raw head tensors are what we
    keep — NOT the decoded `det_out`.
  * drivable-area segmentation: [1,2,640,640]
  * lane-line segmentation:     [1,2,640,640]

The cut is done with onnx.utils.extract_model, taking the graph input and the
five tensors above as the new outputs. The three raw-head tensors are discovered
by their shape signature (channels 3*(5+nc), the three feature strides) so this
survives a re-download of the release ONNX; pass --det-heads to override if a
future release renames them.

Usage:
    python export.py --onnx yolop-640-640.onnx --out yolop_cut.onnx
    # download the parent first, e.g. from the upstream YOLOP release:
    #   https://github.com/hustvl/YOLOP  (weights/yolop-640-640.onnx)
"""

import argparse
import sys
from pathlib import Path

import onnx
from onnx import shape_inference

NC = 1  # YOLOP detects one class (vehicle); 5 = x,y,w,h,obj -> 3*(5+nc) channels


def _value_shapes(model):
    """name -> [dims] for every tensor the shape inference could resolve."""
    out = {}
    for vi in list(model.graph.value_info) + list(model.graph.output) + list(model.graph.input):
        dims = [d.dim_value if (d.HasField("dim_value")) else None
                for d in vi.type.tensor_type.shape.dim]
        out[vi.name] = dims
    return out


def find_det_heads(model):
    """The three raw detection-head conv outputs, ordered large->small stride.

    A YOLO detection head's last conv per scale has channel count 3*(5+nc) and a
    spatial size of input/{8,16,32}. We match on the channel count (unambiguous:
    18 is not a shape any other layer here produces) and sort by area so the
    strides come out in the canonical 80/40/20 order.
    """
    want_c = 3 * (5 + NC)
    shapes = _value_shapes(model)
    heads = []
    for node in model.graph.node:
        if node.op_type != "Conv":
            continue
        for o in node.output:
            dims = shapes.get(o)
            if dims and len(dims) == 4 and dims[1] == want_c:
                heads.append((o, dims[2] or 0))
    heads = sorted(heads, key=lambda t: -t[1])  # largest feature map first
    names = [h[0] for h in heads]
    if len(names) != 3:
        sys.exit(
            f"expected 3 detection-head convs with {want_c} channels, found "
            f"{len(names)}: {names}. Pass --det-heads a,b,c to name them "
            f"explicitly (read them off the graph in netron)."
        )
    return names


def find_seg_outputs(model):
    """The two segmentation outputs, [1,2,640,640]. They are already graph
    outputs in the upstream ONNX; keep them by name."""
    segs = [o.name for o in model.graph.output
            if [d.dim_value for d in o.type.tensor_type.shape.dim][1:2] == [2]]
    if len(segs) != 2:
        sys.exit(f"expected 2 segmentation outputs [1,2,H,W], found {len(segs)}: {segs}")
    return segs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", type=Path, required=True, help="yolop-640-640.onnx (upstream)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--det-heads", help="comma-separated override for the 3 head tensors")
    args = ap.parse_args()

    model = shape_inference.infer_shapes(onnx.load(str(args.onnx)))

    det = (args.det_heads.split(",") if args.det_heads
           else find_det_heads(model))
    seg = find_seg_outputs(model)
    keep = det + seg
    inputs = [i.name for i in model.graph.input]
    print(f"input:  {inputs}")
    print(f"det heads (raw, decode CUT): {det}")
    print(f"seg outputs (kept): {seg}")

    # extract_model rebuilds the graph with exactly these outputs, dropping the
    # ScatterND decode tail and everything else only it fed.
    onnx.utils.extract_model(str(args.onnx), str(args.out),
                             input_names=inputs, output_names=keep)

    cut = onnx.load(str(args.out))
    got = [o.name for o in cut.graph.output]
    assert got == keep, f"output set mismatch: {got} != {keep}"
    print(f"[export] OK -> {args.out}  ({len(cut.graph.node)} nodes, "
          f"{len(got)} outputs: 3 raw det heads + 2 seg). "
          f"Anchor decode is now BCDL's job.")


if __name__ == "__main__":
    main()
