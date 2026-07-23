#!/usr/bin/env python3
"""[1/3] YOLOE-11s (open-vocab) -> fixed COCO-80 BPU split-head DETECTION ONNX.

Run on the convert host (torch + ultralytics). Single RGB input, six outputs:
three scales x [cls, box], each permuted to NHWC. bcdl's YoloLtrbDetector
consumes it unchanged; it auto-detects the DFL reg_max from the box channel
count (4 * reg_max = 64 -> reg_max 16).

WHY the head is rewritten (this is the whole point of the recipe)
-----------------------------------------------------------------
1. Freeze the vocabulary. YOLOE is *open-vocabulary*: its classifier compares
   image features to CLIP text embeddings through a contrastive head
   (BNContrastiveHead), so the class set is a runtime tensor, not baked weights.
   A static .hbm cannot take a runtime text-embedding matmul. So we pin ONE
   vocabulary: `get_text_pe(COCO80)` computes the 80 text embeddings once,
   `set_classes` installs them, and `head.fuse()` folds them into the cv3 conv.
   After fusing, `cv3[i](x)` emits 80-channel class LOGITS directly (the
   contrastive head collapses to Identity). To change the vocabulary you re-run
   this export with a different class list; true runtime prompting is gone by
   construction.

2. Cut the decode off the graph, keep only the convolutions. The stock head,
   after the cv2/cv3 convs, DECODES: DFL softmax+integral over reg_max, anchor
   grid generation (meshgrid/arange), stride multiply, dist2bbox, a class
   sigmoid, concat across scales, and finally NMS. Those are reshape/gather/
   reduce/meshgrid ops -- cheap on the CPU, but a mix of poorly quantised and
   non-BPU-native operators, and they force a fixed anchor layout into the
   compiled binary. `bpu_forward` replaces `head.forward` so the graph stops at
   the raw per-scale conv outputs. bcdl does the DFL integral, anchor math,
   sigmoid and NMS on the CPU. Result: the BPU runs pure convolution (clean
   int8/int16), and one .hbm layout serves.

3. NHWC on the way out. `.permute(0,2,3,1)` matches the BPU's native output
   layout and bcdl's split-head decoder, so no transpose is inserted.

Normalisation is NOT baked here. The exported graph expects RGB already scaled
to [0,1]; the /255 (and nv12->rgb) is applied by the compiler via config, and
by calib.py on the calibration set. See config_det.yaml.

Usage:
    python export_det.py --weights yoloe-11s-seg.pt \\
        --output yoloe_11s_coco80_det_bpu.onnx --imgsz 640
"""
import argparse
import os
import shutil

import onnx
import torch
from ultralytics import YOLOE
from ultralytics.nn.modules import YOLOEDetect, YOLOESegment

# The fixed vocabulary baked into the classifier. 80 COCO classes, upstream
# order -- the decoder's class indices are exactly these positions.
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


def bpu_forward(self, x):
    """Raw split head: per scale, [cls, box] as NHWC. No DFL, anchors or NMS."""
    res = []
    for i in range(self.nl):
        scores = self.cv3[i](x[i]).permute(0, 2, 3, 1)   # [B,H,W,nc]  fused text pe
        bboxes = self.cv2[i](x[i]).permute(0, 2, 3, 1)   # [B,H,W,4*reg_max]  DFL
        res.append(scores)
        res.append(bboxes)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="yoloe-11s-seg.pt",
                    help="YOLOE-11s checkpoint (the seg weights carry the det head)")
    ap.add_argument("--output", default="yoloe_11s_coco80_det_bpu.onnx")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--opset", type=int, default=19)
    a = ap.parse_args()

    model = YOLOE(a.weights)
    # Bake COCO-80 text-prompt embeddings into the classifier.
    pe = model.get_text_pe(COCO80)
    model.set_classes(COCO80, pe)
    m = model.model
    m.eval()

    # Fuse the text embeddings into cv3 so cv3[i](x) -> nc logits directly.
    head = m.model[-1]
    if (isinstance(head, (YOLOEDetect, YOLOESegment))
            and hasattr(head, "fuse") and not getattr(head, "is_fused", False)):
        try:
            head.fuse(head.get_tpe(pe))
        except Exception as e:  # noqa: BLE001 -- report and continue; export still validates below
            print("fuse() note:", str(e)[:120])
    print("head:", type(head).__name__, "nc:", head.nc,
          "reg_max:", head.reg_max, "is_fused:", getattr(head, "is_fused", None))

    # Detection-only split head: monkeypatch the concrete head class forward.
    # (Drops the YOLOESegment mask/proto branch -- see export_seg.py for that.)
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
