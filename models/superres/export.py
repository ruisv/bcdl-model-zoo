#!/usr/bin/env python3
"""[1/3] Export realesr-general-x4v3 (SRVGGNetCompact) -> ONNX at a fixed tile.

Real-ESRGAN Compact x4. Runs on the convert host in a torch env.

The architecture is written out here rather than pulled from basicsr, so it has
to be proved right: `load_state_dict(strict=True)` is the proof — every
parameter name and shape must line up, and `num_feat` / `num_conv` are read off
the checkpoint instead of assumed. Get the width or depth wrong and strict load
fails loudly here rather than compiling into a silently wrong graph.

The net is fully convolutional, so the ONNX is exported at ONE fixed tile size
(the input shape is baked in). The compiled `.hbm` is mostly instruction stream
and scales with tile AREA, not weights — so the tile size is the single most
important knob here. See README.md; the shipped build is the 128 tile.

Usage:
    python export.py --weights /path/to/realesr-general-x4v3.pth --tile 128 \\
                     --out realesr_general_x4v3_128.onnx
"""
import argparse
import collections
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SRVGGNetCompact(nn.Module):
    """Conv/PReLU stack -> PixelShuffle, plus a NEAREST-upsampled skip: the net
    only learns the residual over a nearest-neighbour enlargement."""

    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4):
        super().__init__()
        self.upscale = upscale
        self.body = nn.ModuleList()
        self.body.append(nn.Conv2d(num_in_ch, num_feat, 3, 1, 1))
        self.body.append(nn.PReLU(num_parameters=num_feat))
        for _ in range(num_conv):
            self.body.append(nn.Conv2d(num_feat, num_feat, 3, 1, 1))
            self.body.append(nn.PReLU(num_parameters=num_feat))
        self.body.append(nn.Conv2d(num_feat, num_out_ch * upscale * upscale, 3, 1, 1))
        self.upsampler = nn.PixelShuffle(upscale)

    def forward(self, x):
        out = x
        for layer in self.body:
            out = layer(out)
        out = self.upsampler(out)
        return out + F.interpolate(x, scale_factor=self.upscale, mode="nearest")


def main():
    ap = argparse.ArgumentParser(description="Export Real-ESRGAN Compact x4 to ONNX")
    ap.add_argument("--weights", required=True,
                    help="realesr-general-x4v3.pth (xinntao/Real-ESRGAN)")
    ap.add_argument("--tile", type=int, default=128,
                    help="input tile size (square). Shipped build is 128; the "
                         "compiled size scales with tile AREA")
    ap.add_argument("--scale", type=int, default=4)
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--ir", type=int, default=9,
                    help="cap to IR9 (HBDK 4.x rejects IR>9)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    sd = torch.load(a.weights, map_location="cpu", weights_only=True)
    sd = sd.get("params", sd)

    # Read the width and depth off the checkpoint rather than hardcoding them.
    num_feat = sd["body.0.weight"].shape[0]
    # Count CONVS by weight rank: a PReLU's parameter is also called "weight", so
    # keying on the name alone counts every activation as a layer.
    n_conv_layers = sum(1 for k, v in sd.items()
                        if k.startswith("body.") and k.endswith(".weight") and v.ndim == 4)
    num_conv = n_conv_layers - 2            # minus the first and the last conv
    print("checkpoint: num_feat=%d num_conv=%d" % (num_feat, num_conv))

    net = SRVGGNetCompact(num_feat=num_feat, num_conv=num_conv, upscale=a.scale).eval()
    net.load_state_dict(sd, strict=True)
    print("state_dict loaded strict=True (architecture matches the checkpoint)")

    ops = collections.Counter(type(m).__name__ for m in net.modules())
    print("modules:", {k: v for k, v in ops.items()
                       if k in ("Conv2d", "PReLU", "PixelShuffle")})

    x = torch.rand(1, 3, a.tile, a.tile)
    with torch.no_grad():
        ref = net(x)
    print("forward: %s -> %s  range [%.3f, %.3f]"
          % (list(x.shape), list(ref.shape), ref.min(), ref.max()))

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    torch.onnx.export(net, x, a.out, input_names=["lr"], output_names=["sr"],
                      opset_version=a.opset, dynamo=False)

    import onnx  # noqa: E402

    m = onnx.load(a.out)
    # HBDK 4.x rejects IR>9; recent exporters emit IR10. Capping the header does
    # not downgrade any operator (IR9 covers up to opset 20).
    m.ir_version = a.ir
    onnx.save(m, a.out)
    print("ops:", dict(collections.Counter(n.op_type for n in m.graph.node)))

    def shape(v):
        return [d.dim_value if d.HasField("dim_value") else (d.dim_param or "?")
                for d in v.type.tensor_type.shape.dim]

    print("in :", [(v.name, shape(v)) for v in m.graph.input])
    print("out:", [(v.name, shape(v)) for v in m.graph.output])

    try:
        import onnxruntime as ort  # noqa: E402
        o = ort.InferenceSession(a.out, providers=["CPUExecutionProvider"]).run(
            None, {"lr": x.numpy()})[0]
        print("onnx vs torch max|diff| %.3e" % np.abs(ref.numpy() - o).max())
    except ImportError:
        print("onnxruntime not installed; skipped the onnx-vs-torch check")
    print("[export] OK ->", a.out, "(tile %dx%d, x%d, opset%d, IR%d)"
          % (a.tile, a.tile, a.scale, a.opset, a.ir))


if __name__ == "__main__":
    main()
