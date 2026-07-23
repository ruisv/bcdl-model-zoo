#!/usr/bin/env python3
"""[1/3] Export SPAN x4 (ch48) -> BPU-friendly ONNX at a fixed tile.

Upstream: SPAN (hongyuanyu/SPAN), Apache-2.0. Run on the convert host in a torch
env; the arch is read straight out of the checkout so basicsr need not be
installed.

Three things this model forces you to *determine* rather than assume:

  1. The OUTPUT convention. SPAN's forward normalises its input with
     `(x - mean) * img_range` but never undoes it, and the repo ships no SPAN
     inference script or config that says what the raw output means. So both
     candidate conventions are scored against a real ground-truth HR image and
     the one that reconstructs it wins. Guess wrong and the model runs at full
     speed and returns a plausibly-shaped image with shifted colour/levels.

  2. Conv3XC's reparameterisation. Each block fuses a 1x1 -> 3x3 -> 1x1 branch
     plus a 1x1 skip into a single 3x3 at eval. The fusion is exact, but it is
     recomputed INSIDE forward(), so tracing the export would capture that weight
     arithmetic as graph operators. It is run once and forward() is replaced with
     a plain conv, after checking the replacement is numerically identical.

  3. That the exported ONNX matches torch.

The model input is RGB in [0,1] (pixels / 255). It is declared `featuremap` /
`no_preprocess` to the toolchain, so calibration tiles must be handed to the BPU
in exactly that [0,1] domain -- see calib.py.

Usage:
    python export.py --repo /path/to/SPAN --ckpt /path/to/spanx4_ch48.pth \\
        --ref-hr /path/to/hr.png --ref-lr /path/to/lr.png \\
        --tile 128 --out spanx4_ch48_128.onnx
"""
import argparse
import collections
import os
import re
import sys
import types

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# DIV2K RGB mean SPAN normalises with; used to undo it if that convention wins.
MEAN = torch.tensor([0.4488, 0.4371, 0.4040]).view(1, 3, 1, 1)


def load_arch(repo):
    """Import SPAN, Conv3XC from the checkout without dragging in basicsr."""
    src = open(os.path.join(repo, "basicsr", "archs", "span_arch.py")).read()
    src = re.sub(r"^from basicsr.*$", "", src, flags=re.M)
    src = re.sub(r"^@ARCH_REGISTRY\.register\(\)$", "", src, flags=re.M)
    src = src.split('if __name__ == "__main__"')[0]
    mod = types.ModuleType("span_arch")
    mod.__dict__.update({"torch": torch, "nn": nn, "F": F})
    exec(compile(src, "span_arch.py", "exec"), mod.__dict__)
    return mod.SPAN, mod.Conv3XC


def psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return 10 * np.log10(255.0 ** 2 / mse) if mse > 0 else float("inf")


def to_bgr(t):
    a = t[0].numpy().transpose(1, 2, 0)[:, :, ::-1]
    return (np.clip(a, 0, 1) * 255).round().astype(np.uint8)


class SpanBpu(nn.Module):
    """SPAN with the output convention folded in, so the model returns [0,1]."""

    def __init__(self, net, denorm):
        super().__init__()
        self.net = net
        self.denorm = denorm

    def forward(self, x):
        y = self.net(x)
        if self.denorm:
            y = y / 255.0 + MEAN.to(y.dtype)
        return y


def main():
    ap = argparse.ArgumentParser(description="Export SPAN x4 (ch48) to BPU ONNX")
    ap.add_argument("--repo", required=True, help="SPAN (hongyuanyu/SPAN) checkout")
    ap.add_argument("--ckpt", required=True, help="spanx4_ch48.pth")
    ap.add_argument("--ref-hr", required=True,
                    help="ground-truth HR image, to score the output convention")
    ap.add_argument("--ref-lr", required=True,
                    help="the 4x-downscaled LR of --ref-hr (see span_ref logic)")
    ap.add_argument("--tile", type=int, default=128, help="square input tile")
    ap.add_argument("--channels", type=int, default=48)
    ap.add_argument("--scale", type=int, default=4)
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--ir", type=int, default=9, help="cap IR (HBDK 4.x rejects >9)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    SPAN, Conv3XC = load_arch(a.repo)

    net = SPAN(3, 3, feature_channels=a.channels, upscale=a.scale)
    sd = torch.load(a.ckpt, map_location="cpu", weights_only=True)
    sd = sd.get("params", sd)
    missing, unexpected = net.load_state_dict(sd, strict=False)
    missing = [k for k in missing if "eval_conv" not in k]   # derived, not stored
    assert not missing and not unexpected, (missing, unexpected)
    net.eval()
    print("state_dict loaded (only derived eval_conv weights were absent)")

    # --- 1. output convention ------------------------------------------------
    hr = cv2.imread(a.ref_hr)
    lr = cv2.imread(a.ref_lr)
    if hr is None or lr is None:
        sys.exit("could not read --ref-hr / --ref-lr")
    x = torch.from_numpy(np.ascontiguousarray(
        lr[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0))
    # Pad up to a whole number of tiles so one forward covers the frame.
    T = a.tile
    ph, pw = -(-x.shape[2] // T) * T, -(-x.shape[3] // T) * T
    xp = F.pad(x, (0, pw - x.shape[3], 0, ph - x.shape[2]), mode="replicate")
    with torch.no_grad():
        raw = net(xp)[:, :, :x.shape[2] * a.scale, :x.shape[3] * a.scale]
    print("raw output range [%.3f, %.3f] mean %.3f" % (raw.min(), raw.max(), raw.mean()))

    cands = {
        "raw (already [0,1])": raw,
        "raw/255 + mean": raw / 255.0 + MEAN,
    }
    scores = {k: psnr(to_bgr(v), hr) for k, v in cands.items()}
    for k, v in scores.items():
        print("  %-22s PSNR vs ground truth %.2f dB" % (k, v))
    best = max(scores, key=scores.get)
    print("  -> output convention:", best)
    assert scores[best] > 20, "neither convention reconstructs the image; investigate"

    # --- 2. fuse Conv3XC once, then stop recomputing it inside forward -------
    probe = torch.randn(1, a.channels, 16, 16)
    sample = [m for m in net.modules() if isinstance(m, Conv3XC)][1]
    with torch.no_grad():
        before = sample(probe)
    for m in net.modules():
        if isinstance(m, Conv3XC):
            m.update_params()
    Conv3XC.forward = lambda self, x: self.eval_conv(x)
    with torch.no_grad():
        after = sample(probe)
    print("Conv3XC fused-forward vs original eval forward: max|diff| %.3e"
          % (before - after).abs().max())
    assert (before - after).abs().max() < 1e-5

    bpu = SpanBpu(net, denorm=(best != "raw (already [0,1])")).eval()

    # --- 3. export -----------------------------------------------------------
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    torch.onnx.export(bpu, torch.rand(1, 3, T, T), a.out,
                      input_names=["lr"], output_names=["sr"],
                      opset_version=a.opset, dynamo=False)

    import onnx           # noqa: E402
    import onnxruntime as ort   # noqa: E402

    m = onnx.load(a.out)
    m.ir_version = a.ir   # HBDK 4.x rejects IR > 9; header change, not a downgrade
    onnx.save(m, a.out)
    print("ops:", dict(collections.Counter(n.op_type for n in m.graph.node)))

    t = torch.rand(1, 3, T, T)
    with torch.no_grad():
        ref = bpu(t)
    got = ort.InferenceSession(a.out, providers=["CPUExecutionProvider"]).run(
        None, {"lr": t.numpy()})[0]
    print("onnx vs torch max|diff| %.3e" % np.abs(ref.numpy() - got).max())
    print(f"[export] OK -> {a.out}  (SPAN x4 ch{a.channels} {T}x{T}, opset{a.opset}, IR{a.ir})")


if __name__ == "__main__":
    main()
