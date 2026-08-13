#!/usr/bin/env python3
"""[1/3] Export GLASS to a static ONNX graph at 288x288, batch 1.

    python export.py --ckpt /path/to/bottle/ckpt.pth --repo /path/to/GLASS
    onnxslim glass_bottle.onnx glass_bottle_slim.onnx

**What is actually being exported matters more than how.** GLASS's backbone is a
FROZEN ImageNet WideResNet-50. Per product, only two small heads are trained —
`pre_projection` and `discriminator`, about 16 MB. So "retrain for your own
part" is a realistic thing to ask of a user, and the model shipped with a
library can only ever be an EXAMPLE of one product, not a general detector.

The one operator worth worrying about is `torch.nn.Unfold` (the 3x3 neighbourhood
aggregation), which the BPU does not support. It does not survive to the graph:
torch lowers it into standard ops during export. After slimming the graph is 136
nodes and carries **no red-flag operator at all** — only Conv, Relu, Pool,
Reshape, Transpose, Gemm, MatMul and Sigmoid.

Upstream's exporter defaults to a dynamic batch; this fixes batch 1 and the
input resolution so the graph is static.
"""
import argparse
import sys

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="a trained GLASS checkpoint (one product)")
    ap.add_argument("--repo", default=".", help="a GLASS checkout, for the model class")
    ap.add_argument("--size", type=int, default=288)
    ap.add_argument("--out", default="glass_bottle.onnx")
    a = ap.parse_args()

    sys.path.insert(0, a.repo)
    from glass import GLASS                                   # noqa: E402

    model = GLASS(device="cpu")
    state = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    # Only the two per-product heads are in the checkpoint; the WideResNet-50
    # backbone comes from torchvision's ImageNet weights and is frozen.
    model.load_state_dict(state, strict=False)
    model.eval()

    x = torch.zeros(1, 3, a.size, a.size, dtype=torch.float32)
    torch.onnx.export(model, x, a.out, input_names=["input"], output_names=["anomaly_map"],
                      opset_version=17, do_constant_folding=True, dynamo=False)
    print("wrote", a.out)
    print("now run:  onnxslim %s %s" % (a.out, a.out.replace(".onnx", "_slim.onnx")))
    print("expect 136 nodes and no unsupported operator; Unfold is lowered by the exporter")


if __name__ == "__main__":
    main()
