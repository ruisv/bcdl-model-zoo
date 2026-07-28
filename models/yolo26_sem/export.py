#!/usr/bin/env python3
"""[1/3] YOLO26n-sem -> dense per-pixel logit ONNX for the BPU.

Run on the convert host (torch + ultralytics >= 8.4.x, which is the release
that added the `semantic` task and its `-sem` checkpoints).

WHY THIS EXPORT NEEDS NO CUT (unlike detect/seg/pose/obb)
----------------------------------------------------------
Every other YOLO26 head in this repo (see ../yolo26/, ../yoloe/) needs its
forward replaced before export, because the stock head bakes NMS, anchor
math or a DFL integral into the graph. `SemanticSegment` does not: at
inference it is a plain classifier conv fused with an `F.interpolate(...,
mode="bilinear")` back up to input resolution -- no boxes, no instances, no
decode to cut. `model.export(format="onnx")` already produces exactly the
graph BCDL wants: one dense-logit output, shape [1, nc, H, W] (channels
first), full input resolution.

Verified once against the export this recipe was written from: op profile is
Conv/Sigmoid/Mul (backbone) + one Softmax/MatMul/Transpose attention block
(the same neck stage YOLO26's detector already runs on this BPU/march) +
three Resize nodes (two "nearest" FPN upsamples, identical to every other
yolo26 head; one "linear" for the final 8x upsample, the same op PIDNet-S
already validates on this stack -- see ../pidnet/). Max tensor rank 4, no
Unfold/GridSample. No operator rewrite needed.

Normalisation is NOT baked here, same convention as ../yolo26/: the graph
expects RGB already scaled to [0,1]; the /255 (and nv12->rgb) is applied by
the compiler via config.yaml, and by calib.py on the calibration set.

BCDL's existing `bcdl::Segmenter`/`decodeSeg` (used for DeepLabV3+ and
PIDNet-S) already consumes this exact shape -- [1,C,H,W] logits,
channels_first, argmax on the CPU -- so no new C++ decode path is needed
either. This recipe is purely: get the ONNX, quantise it right.

Usage:
    python export.py --weights yolo26n-sem.pt --output yolo26n_sem_bpu.onnx --imgsz 640
"""
import argparse
import os
import shutil

import onnx
from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="yolo26n-sem.pt", help="YOLO26n-sem checkpoint")
    ap.add_argument("--output", default="yolo26n_sem_bpu.onnx")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--opset", type=int, default=19)
    a = ap.parse_args()

    model = YOLO(a.weights)
    if model.task != "semantic":
        raise SystemExit(f"{a.weights}: task={model.task!r}, expected 'semantic' "
                          f"(-sem checkpoint)")
    head = model.model.model[-1]
    for attr in ("nc", "nl", "classifier"):
        assert hasattr(head, attr), f"head is missing {attr!r}; not a SemanticSegment head?"
    print("head:", type(head).__name__, "nc:", head.nc, "nl:", head.nl)

    ep = model.export(format="onnx", imgsz=a.imgsz, dynamic=False,
                      opset=a.opset, simplify=False)
    if ep and ep != a.output:
        d = os.path.dirname(a.output)
        if d:
            os.makedirs(d, exist_ok=True)
        shutil.move(ep, a.output)
        ep = a.output

    om = onnx.load(ep)
    out = om.graph.output[0]
    dims = [d.dim_value for d in out.type.tensor_type.shape.dim]
    print(f"output {out.name!r}: shape {dims} (expect [1, nc, {a.imgsz}, {a.imgsz}])")
    if dims != [1, head.nc, a.imgsz, a.imgsz]:
        print("WARNING: output shape does not match the expected dense full-res "
              "logit map -- check the ultralytics version's SemanticSegment.forward.")

    # HBDK 4.x rejects ONNX IR > 9; opset 19 already exports at IR9 here, but cap
    # defensively in case a newer exporter bumps it (see ../yolo26/export.py).
    if om.ir_version > 9:
        om.ir_version = 9
        onnx.save(om, ep)
        print("[export] capped IR version -> 9")

    print("EXPORTED:", ep)


if __name__ == "__main__":
    main()
