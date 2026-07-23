#!/usr/bin/env python3
"""[1/3] YOLOE-11s-seg (open-vocab) -> fixed COCO-80 BPU split SEG ONNX.

Same head surgery as export_det.py, plus the instance-mask branch. Single RGB
input, TEN outputs: three scales x [cls, box, mask_coef] (NHWC) + one proto.

    per scale i:  cv3[i](x) -> cls   [B,H,W,80]   (text pe fused into the conv)
                  cv2[i](x) -> box   [B,H,W,64]   (DFL, 4 * reg_max)
                  cv5[i](x) -> coef  [B,H,W,32]   (mask coefficients, nm=32)
    once:         proto(x0) -> proto [B,160,160,32]  (prototype masks at stride 4)

WHY the head is rewritten -- read export_det.py's header first; the three
reasons (freeze the vocabulary via fuse(), cut the decode and keep only the
convs, emit NHWC) apply identically here. The seg branch adds two things the
graph keeps as raw convolution and bcdl assembles on the CPU:

  * cv5 mask COEFFICIENTS (32 per anchor) -- kept raw, permuted NHWC.
  * proto: the Proto module's 32 prototype masks at stride 4 (160x160 for a
    640 input) -- kept as a conv output, permuted NHWC.

The mask assembly that the stock head would fold into the graph -- coef @ proto,
sigmoid, then crop to each decoded box -- is elementwise/matmul work that bcdl
does on the CPU after NMS, so only the per-detection masks are ever built. This
keeps the compiled graph pure convolution (BPU-clean) and avoids materialising a
full 32 x 160 x 160 mask stack on the BPU.

Normalisation is NOT baked (RGB in [0,1] expected; compiler + calib.py apply the
/255 and nv12->rgb). See config_seg.yaml.

Usage:
    python export_seg.py --weights yoloe-11s-seg.pt \\
        --output yoloe_11s_coco80_seg_bpu.onnx --imgsz 640
"""
import argparse
import os
import shutil

import onnx
import torch
from ultralytics import YOLOE

COCO80 = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]


def bpu_seg_forward(self, x):
    """Raw split head: per scale [cls, box, coef] NHWC, plus one NHWC proto."""
    res = []
    for i in range(self.nl):
        f = x[i]
        res.append(self.cv3[i](f).permute(0, 2, 3, 1))   # cls (fused)  -> [B,H,W,80]
        res.append(self.cv2[i](f).permute(0, 2, 3, 1))   # box (DFL)    -> [B,H,W,64]
        res.append(self.cv5[i](f).permute(0, 2, 3, 1))   # mask coef    -> [B,H,W,32]
    proto = self.proto(x[0])
    res.append(proto.permute(0, 2, 3, 1))                # proto        -> [B,mh,mw,32]
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="yoloe-11s-seg.pt")
    ap.add_argument("--output", default="yoloe_11s_coco80_seg_bpu.onnx")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--opset", type=int, default=19)
    a = ap.parse_args()

    model = YOLOE(a.weights)
    # inference_mode(False): fuse() writes into parameters, which a pure
    # inference-mode tensor forbids; clone the embeddings so the fold is safe.
    with torch.inference_mode(False), torch.no_grad():
        pe = model.get_text_pe(COCO80).clone()
        model.set_classes(COCO80, pe)
        m = model.model
        m.eval()
        head = m.model[-1]
        tpe = head.get_tpe(pe)
        if tpe is not None:
            tpe = tpe.clone()
        if not getattr(head, "is_fused", False):
            head.fuse(tpe)
    print("head:", type(head).__name__, "nc:", head.nc, "reg_max:", head.reg_max,
          "nm:", head.nm, "is_fused:", head.is_fused)

    type(head).forward = bpu_seg_forward

    ep = model.export(format="onnx", imgsz=a.imgsz, dynamic=False,
                      opset=a.opset, simplify=False)
    if ep and ep != a.output:
        d = os.path.dirname(a.output)
        if d:
            os.makedirs(d, exist_ok=True)
        shutil.move(ep, a.output)
        ep = a.output

    # HBDK 4.x rejects ONNX IR > 9; opset 19 exports at IR10. Cap the header.
    om = onnx.load(ep)
    if om.ir_version > 9:
        om.ir_version = 9
        onnx.save(om, ep)
        print("[export] capped IR version -> 9")

    print("EXPORTED:", ep)


if __name__ == "__main__":
    main()
