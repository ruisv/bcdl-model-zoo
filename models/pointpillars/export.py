#!/usr/bin/env python3
"""[1/3] Export PointPillars from an OpenPCDet checkpoint as TWO compilable graphs.

    python export.py --ckpt pointpillar_7728.pth            # both graphs
    python export.py --ckpt ... --which vfe --pillars 11546

Neither graph is the upstream one, and both departures are the point of this
recipe.

**1. The VFE is re-expressed as a 1x1 convolution.** Upstream's PillarVFE takes
`(pillars, points, 4)` and expands each point into 10 features inside the graph.
That shape is a huge pillar axis against a 4-wide channel axis — the worst case
for a convolution engine — and it is layout-bound, not compute-bound: the
published build spends **290 ms** on a 4->64 MLP of about 95 MFLOP, i.e. roughly
0.3 GFLOPS.

The same arithmetic over `(1, 10, P, 32)` is a 1x1 convolution (the per-point
MLP becomes a channel convolution) followed by a `(1, 32)` max-pool (the max over
points becomes a pooling window). Measured on S100P: **1.62 ms, a 179x
difference**, for a graph that is arithmetically identical. The 10-channel
expansion moves to the host, where it is 1.1 ms of C++ (`calib.py` has the numpy
version; BCDL ships `pillarPointFeatures()`).

**2. The backbone and the anchor head are exported as one graph, cut before the
decode.** Upstream's `AnchorHeadSingle` bakes anchor decoding in with
`ScatterND`, which — as with YOLOP in this repo — compiles without a single
warning into a model whose outputs are never written. Cut at the three
convolutions (`cls` 18, `box` 42, `dir` 12) it becomes three 1x1 convolutions
that may as well ride along with the backbone: one Engine and one round trip
instead of two, 113.7 ms -> 7.20 ms.

Both modules are rebuilt from the checkpoint's own key layout rather than from a
config, so they cannot drift from the weights they load: a missing or extra layer
surfaces as a `load_state_dict` error instead of as a quietly different network.

The BN folding in the VFE and the `eps=1e-3` on every BatchNorm here are
upstream's values, not defaults — `nn.BatchNorm2d` defaults to `1e-5`, and using
it shifts every activation slightly with no symptom a compile could show.
"""
import argparse
import os

import torch
import torch.nn as nn


# --- VFE: linear + BN + ReLU + max over points, as conv + pool ---------------

class VfeConv(nn.Module):
    def __init__(self, sd, max_points=32):
        super().__init__()
        w = sd["vfe.pfn_layers.0.linear.weight"]              # (64, 10)
        g = sd["vfe.pfn_layers.0.norm.weight"]
        b = sd["vfe.pfn_layers.0.norm.bias"]
        rm = sd["vfe.pfn_layers.0.norm.running_mean"]
        rv = sd["vfe.pfn_layers.0.norm.running_var"]
        eps = 1e-3                                            # upstream's, not the default
        # Fold BN into the 1x1 convolution: same arithmetic, one layer instead of two.
        scale = g / torch.sqrt(rv + eps)
        self.conv = nn.Conv2d(w.shape[1], w.shape[0], 1, bias=True)
        with torch.no_grad():
            self.conv.weight.copy_((w * scale[:, None]).reshape(w.shape[0], w.shape[1], 1, 1))
            self.conv.bias.copy_(b - rm * scale)
        self.pool = nn.MaxPool2d((1, max_points))

    def forward(self, x):                     # (1, 10, P, 32)
        return self.pool(torch.relu(self.conv(x)))            # (1, 64, P, 1)


# --- 2-D backbone + anchor head, cut before the decode -----------------------

def build_block(sd, prefix):
    """One down-sampling stage: ZeroPad, strided conv, then N same-size convs."""
    idx = sorted({int(k[len(prefix) + 1:].split(".")[0])
                  for k in sd if k.startswith(prefix + ".")})
    conv_idx = [i for i in idx
                if f"{prefix}.{i}.weight" in sd and sd[f"{prefix}.{i}.weight"].dim() == 4]
    layers = [nn.ZeroPad2d(1)]
    for j, i in enumerate(conv_idx):
        w = sd[f"{prefix}.{i}.weight"]
        conv = nn.Conv2d(w.shape[1], w.shape[0], w.shape[2],
                         stride=2 if j == 0 else 1, padding=0 if j == 0 else 1, bias=False)
        conv.weight.data.copy_(w)
        bn = nn.BatchNorm2d(w.shape[0], eps=1e-3, momentum=0.01)
        for a in ("weight", "bias", "running_mean", "running_var"):
            getattr(bn, a).data.copy_(sd[f"{prefix}.{i + 1}.{a}"])
        layers += [conv, bn, nn.ReLU()]
    return nn.Sequential(*layers)


def build_deblock(sd, prefix, stride):
    w = sd[f"{prefix}.0.weight"]
    deconv = nn.ConvTranspose2d(w.shape[0], w.shape[1], w.shape[2], stride=stride, bias=False)
    deconv.weight.data.copy_(w)
    bn = nn.BatchNorm2d(w.shape[1], eps=1e-3, momentum=0.01)
    for a in ("weight", "bias", "running_mean", "running_var"):
        getattr(bn, a).data.copy_(sd[f"{prefix}.1.{a}"])
    return nn.Sequential(deconv, bn, nn.ReLU())


class BackboneHead(nn.Module):
    def __init__(self, sd):
        super().__init__()
        self.blocks = nn.ModuleList(
            [build_block(sd, f"backbone_2d.blocks.{i}") for i in range(3)])
        self.deblocks = nn.ModuleList(
            [build_deblock(sd, f"backbone_2d.deblocks.{i}", s)
             for i, s in enumerate((1, 2, 4))])
        self.cls = nn.Conv2d(384, 18, 1)
        self.box = nn.Conv2d(384, 42, 1)
        self.dir = nn.Conv2d(384, 12, 1)
        for m, name in ((self.cls, "conv_cls"), (self.box, "conv_box"),
                        (self.dir, "conv_dir_cls")):
            m.weight.data.copy_(sd[f"dense_head.{name}.weight"])
            m.bias.data.copy_(sd[f"dense_head.{name}.bias"])

    def forward(self, x):
        ups = []
        for block, deblock in zip(self.blocks, self.deblocks):
            x = block(x)
            ups.append(deblock(x))
        f = torch.cat(ups, dim=1)
        # Raw head tensors only. The anchor decode stays on the host, because the
        # published graph's ScatterND version compiles clean and writes nothing.
        return self.cls(f), self.box(f), self.dir(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="pointpillar_7728.pth",
                    help="OpenPCDet KITTI PointPillars checkpoint")
    ap.add_argument("--which", choices=("vfe", "bbhead", "both"), default="both")
    ap.add_argument("--pillars", type=int, default=11546,
                    help="max_pillars; must match the runtime VoxelConfig")
    ap.add_argument("--points", type=int, default=32, help="max points per pillar")
    ap.add_argument("--grid", type=int, nargs=2, default=(496, 432),
                    help="BEV canvas as gridY gridX")
    ap.add_argument("--out-dir", default=".")
    a = ap.parse_args()

    sd = torch.load(a.ckpt, map_location="cpu", weights_only=False)["model_state"]

    if a.which in ("vfe", "both"):
        model = VfeConv(sd, a.points).eval()
        x = torch.zeros(1, 10, a.pillars, a.points)
        out = os.path.join(a.out_dir, "vfe_conv.onnx")
        torch.onnx.export(model, x, out, input_names=["pillar_features_10ch"],
                          output_names=["pillar_features"], opset_version=17,
                          do_constant_folding=True, dynamo=False)
        print("wrote %s  %.1f MB" % (out, os.path.getsize(out) / 1e6))

    if a.which in ("bbhead", "both"):
        model = BackboneHead(sd).eval()
        gy, gx = a.grid
        x = torch.zeros(1, 64, gy, gx)
        with torch.no_grad():
            c, b, d = model(x)
        print("head shapes:", tuple(c.shape), tuple(b.shape), tuple(d.shape))
        # The geometry is the check that the rebuilt stack has the right strides:
        # 496x432 at stride 2 is 248x216, and 18/42/12 channels are 2 rotations x
        # 3 classes x (1 score, 7 box residuals, 2 direction bins).
        assert tuple(c.shape[1:]) == (18, gy // 2, gx // 2), \
            "head geometry does not match the published one"
        out = os.path.join(a.out_dir, "pp_backbone_head.onnx")
        torch.onnx.export(model, x, out, input_names=["spatial_features"],
                          output_names=["cls", "box", "dir"], opset_version=17,
                          do_constant_folding=True, dynamo=False)
        print("wrote %s  %.1f MB" % (out, os.path.getsize(out) / 1e6))


if __name__ == "__main__":
    main()
