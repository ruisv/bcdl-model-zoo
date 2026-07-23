#!/usr/bin/env python3
"""[1/3] Export PIDNet-S to a "no-norm" opset-19 ONNX at 1024x2048.

TWO things this export must get right, and both are load-bearing for the build
that works (`_v3`):

1. NO NORMALISATION IN THE GRAPH ("nonorm"). The exported graph takes RAW 0-255
   pixels and does no mean/scale itself. All normalisation is declared in
   config.yaml (`data_mean_and_scale`, ImageNet mean/std) and applied by the
   runtime to the nv12 input — and, crucially, by calib.py to the calibration
   data through the same config. Baking normalisation into the graph instead
   (or doing a partial job, like an earlier build that only applied 1/255 with
   no mean subtraction) puts the model on the wrong input distribution. That
   still compiles cleanly; see calib.py and README for how it fails.

2. OPSET 19. PIDNet's bilinear upsampling / Resize configuration needs a recent
   opset to lower cleanly on the toolchain; opset 19 is what the shipped build
   used. Lower opsets export but can drop or approximate the Resize.

Semantic segmentation, Cityscapes 19 classes, single seg-logit output. BCDL takes
the argmax (per-class channel -> label map).

RECONSTRUCTED: the original export script for this model was not recovered. This
reproduces the METHOD from the surviving config (`pidnet_op19_nonorm.onnx`, input
name `image`, 1x3x1024x2048, data_mean_and_scale) and the upstream PIDNet-S
inference forward. Verify the output resolution against your PIDNet checkout — some
PIDNet variants emit at 1/8 and rely on the consumer to upsample; this exports the
inference head as the checkpoint defines it.

Usage:
    python export.py --repo /path/to/PIDNet --weights PIDNet_S_Cityscapes_val.pt \\
                     --out pidnet_op19_nonorm.onnx
"""

import argparse
import sys
from pathlib import Path

import torch


def load_pidnet(repo: str, weights: str):
    """Build PIDNet-S from an upstream checkout and load the checkpoint.

    Kept tolerant of the two checkpoint conventions PIDNet ships (a bare
    state_dict, or one under 'state_dict' with a 'model.' prefix), because a
    silently-partial load produces a half-random network that still exports.
    """
    sys.path.insert(0, repo)
    from models.pidnet import get_pred_model  # upstream models/pidnet.py

    model = get_pred_model(name="pidnet_s", num_classes=19)
    ckpt = torch.load(weights, map_location="cpu")
    state = ckpt.get("state_dict", ckpt)
    state = {k[6:] if k.startswith("model.") else k: v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    # PIDNet's auxiliary training heads are absent at inference; tolerate those,
    # but a large missing set means the wrong architecture.
    if len(missing) > 8:
        sys.exit(f"suspicious load: {len(missing)} missing keys — wrong arch/checkpoint?")
    print(f"loaded PIDNet-S (missing={len(missing)}, unexpected={len(unexpected)})")
    return model.eval()


class SegHead(torch.nn.Module):
    """Expose a single seg-logit output and take RAW pixels (no in-graph norm).

    PIDNet's forward returns extra branches in training; at inference we want the
    one seg output. The input is raw 0-255 NCHW — normalisation is the config's
    job, so nothing is subtracted or scaled here.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, image):
        out = self.model(image)
        return out[0] if isinstance(out, (list, tuple)) else out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="PIDNet checkout (XuJiacong/PIDNet)")
    ap.add_argument("--weights", required=True, help="PIDNet_S_Cityscapes_*.pt")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--hw", nargs=2, type=int, default=[1024, 2048], metavar=("H", "W"))
    ap.add_argument("--opset", type=int, default=19)
    args = ap.parse_args()

    model = load_pidnet(args.repo, args.weights)
    net = SegHead(model).eval()
    H, W = args.hw
    dummy = torch.zeros(1, 3, H, W)

    with torch.no_grad():
        y = net(dummy)
    print(f"seg output shape: {tuple(y.shape)}  (expect [1,19,*,*])")

    torch.onnx.export(
        net, dummy, str(args.out),
        input_names=["image"], output_names=["seg"],
        opset_version=args.opset, do_constant_folding=True,
    )
    print(f"[export] OK -> {args.out}  (opset{args.opset}, raw-pixel input, "
          f"{H}x{W}). Normalisation is config.yaml's job, not the graph's.")


if __name__ == "__main__":
    main()
