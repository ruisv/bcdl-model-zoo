#!/usr/bin/env python3
"""[1/3] YOLO26s -> raw split-head DETECTION ONNX for the BPU.

Run on the convert host (torch + ultralytics). Single RGB input, six outputs:
three scales x [cls, box], each permuted to NHWC. BCDL's YoloLtrbDetector
consumes it unchanged; it auto-detects the DFL reg_max from the box channel
count (4 * reg_max = 64 -> reg_max 16).

WHY the head is cut (this is the whole recipe)
----------------------------------------------
The stock Detect head, after its cv2 (box) / cv3 (cls) convs, DECODES: DFL
softmax+integral over reg_max, anchor-grid generation (meshgrid/arange), stride
multiply, dist2bbox, a class sigmoid, concat across scales, and NMS. Those are
reshape/gather/reduce/meshgrid ops -- a mix of poorly-quantised and non-BPU-native
operators, and they bake a fixed anchor layout into the compiled binary.
`bpu_forward` replaces the head's forward so the graph stops at the raw per-scale
conv outputs; BCDL does the DFL integral, anchor math, sigmoid and NMS on the CPU.
The BPU then runs pure convolution (clean int8) and one .hbm layout serves.

NHWC on the way out (`.permute(0,2,3,1)`) matches the BPU's native output layout
and the split-head decoder, so no transpose is inserted.

Normalisation is NOT baked here: the graph expects RGB already scaled to [0,1];
the /255 (and nv12->rgb) is applied by the compiler via config, and by calib.py
on the calibration set. See config.yaml.

This is the same cut used for YOLOE (models/yoloe/export_det.py) minus the
open-vocabulary text-embedding fuse -- YOLO26 has a fixed class set, so the
classifier convs already emit class logits.

Usage:
    python export.py --weights yolo26s.pt --output yolo26s_det_bpu.onnx --imgsz 640
"""
import argparse
import os
import shutil

import onnx
from ultralytics import YOLO


def bpu_forward(self, x):
    """Raw split head: per scale, [cls, box] as NHWC. No DFL, anchors or NMS."""
    res = []
    for i in range(self.nl):
        scores = self.cv3[i](x[i]).permute(0, 2, 3, 1)   # [B,H,W,nc]
        bboxes = self.cv2[i](x[i]).permute(0, 2, 3, 1)   # [B,H,W,4*reg_max]  DFL
        res.append(scores)
        res.append(bboxes)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="yolo26s.pt", help="YOLO26s detection checkpoint")
    ap.add_argument("--output", default="yolo26s_det_bpu.onnx")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--opset", type=int, default=19)
    a = ap.parse_args()

    model = YOLO(a.weights)
    m = model.model
    m.eval()

    head = m.model[-1]
    # Guard against a wrong checkpoint / head layout: the raw cut needs the DFL
    # split head (cv2 = box, cv3 = cls). A silently-partial match would export a
    # graph that decodes to nothing.
    for attr in ("nl", "cv2", "cv3", "reg_max", "nc"):
        assert hasattr(head, attr), f"head is missing {attr!r}; not a DFL Detect head?"
    print("head:", type(head).__name__, "nc:", head.nc,
          "reg_max:", head.reg_max, "nl:", head.nl)

    type(head).forward = bpu_forward

    ep = model.export(format="onnx", imgsz=a.imgsz, dynamic=False,
                      opset=a.opset, simplify=False)
    if ep and ep != a.output:
        d = os.path.dirname(a.output)
        if d:
            os.makedirs(d, exist_ok=True)
        shutil.move(ep, a.output)
        ep = a.output

    # HBDK 4.x rejects ONNX IR > 9. opset 19 exports at IR10, and IR9 already
    # covers up to opset 20, so this is a header cap, not an operator downgrade.
    om = onnx.load(ep)
    if om.ir_version > 9:
        om.ir_version = 9
        onnx.save(om, ep)
        print("[export] capped IR version -> 9")

    print("EXPORTED:", ep)


if __name__ == "__main__":
    main()
