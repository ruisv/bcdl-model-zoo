#!/usr/bin/env python3
"""[1/4] Export a torchreid OSNet person-ReID model to a fixed-shape ONNX graph.

This ONNX is used three ways downstream: it is the FP32 *teacher* the QAT student
distills toward (qat.py), the float baseline the Market-1501 metric scores against
(market.py), and the input to the rejected int8 PTQ build (calib.py + config.yaml).

The runtime wants ONE thing from this model: a 512-d appearance embedding per
person crop. So we export the eval-mode forward (which returns the pooled
feature, not the classifier logits) at a FIXED 1x3x256x128, because the board's
graphs are static anyway and torchreid's own export is known to produce wrong
outputs for dynamic batch > 1 (deep-person-reid issue #585).

Three variants can be exported. They differ only in how much instance
normalization they carry, which turned out to be the axis that decides whether
int8 survives at all (measured, see README):
  osnet_ain_x1_0 - IN inside the residual blocks; best float cross-domain, and
                   the ONLY one worth shipping. This is the default.
  osnet_ibn_x1_0 - IN paired with BN through the early stages
  osnet_x1_0     - BatchNorm only

Normalization (ImageNet mean/std) is left OUT of the graph: the crops are fed as
already-normalized float NCHW, matching how the other crop-input models in this
project are compiled (featuremap in, no_preprocess).

Usage:
    python export.py --weights osnet_ain_x1_0_msmt17.pth \\
                     --out osnet_ain_x1_0_256x128.onnx
"""

import argparse
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import osnet  # noqa: E402  torchreid/models/osnet.py, vendored beside this file
import osnet_ain  # noqa: E402  torchreid/models/osnet_ain.py, vendored

BUILDERS = {
    "osnet_ain_x1_0": osnet_ain.osnet_ain_x1_0,
    "osnet_ibn_x1_0": osnet.osnet_ibn_x1_0,
    "osnet_x1_0": osnet.osnet_x1_0,
}


class EmbeddingHead(torch.nn.Module):
    """The eval-mode forward, re-expressed to emit a 4-D [N,512,1,1] embedding.

    WHY NOT JUST FLATTEN. torchreid ends with `view(N,-1)` + `nn.Linear`, i.e. a
    2-D output. The compiler's internal graph keeps the tensor 4-D and appends a
    Reshape to restore that declared 2-D shape — with the batch size baked into a
    constant. Calibration wants to run batches of 8, hits that constant, and
    silently falls back to feeding samples ONE AT A TIME, which is an 8x longer
    calibration for nothing. (Declaring the ONNX batch axis dynamic does not help:
    hb_compile pins the input shape from its config, and the constant survives.)
    Emitting 4-D removes the reshape entirely, so batch-8 calibration works.

    Nothing downstream cares: the board reads product(shape) = 512 floats either
    way, which is what BCDL's ImageEmbedder does.

    The rewrite is Linear -> 1x1 Conv and BatchNorm1d -> BatchNorm2d, both exact
    re-indexings of the same weights, and main() asserts the two forms agree
    before anything is exported.

    This class is imported by qat.py and deploy.py too, so the FP32 teacher, the
    QAT student and the compiled model are all the same graph by construction.
    """

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model
        self.fc = None
        if model.fc is None:
            return

        linear, bn, act = model.fc[0], model.fc[1], model.fc[2]
        conv = torch.nn.Conv2d(linear.in_features, linear.out_features, 1, bias=True)
        conv.weight.data.copy_(linear.weight.data.view(linear.out_features,
                                                       linear.in_features, 1, 1))
        conv.bias.data.copy_(linear.bias.data)

        bn2d = torch.nn.BatchNorm2d(bn.num_features, eps=bn.eps, momentum=bn.momentum,
                                    affine=bn.affine,
                                    track_running_stats=bn.track_running_stats)
        bn2d.load_state_dict(bn.state_dict())

        self.fc = torch.nn.Sequential(conv, bn2d, act).eval()

    def forward(self, x):
        v = self.model.global_avgpool(self.model.featuremaps(x))  # [N,C,1,1]
        return self.fc(v) if self.fc is not None else v


def load(arch: str, weights: Path) -> torch.nn.Module:
    """Build `arch` and load `weights` into it with strict=True.

    The checkpoint's classifier width tells us how many identities it was
    trained on, so the model is built to match and the load can stay strict —
    a silently-partial load is exactly how you end up exporting a half-random
    network that still produces plausible-looking vectors.
    """
    ckpt = torch.load(weights, map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict", ckpt)
    state = {k[7:] if k.startswith("module.") else k: v for k, v in state.items()}

    num_classes = state["classifier.weight"].shape[0]
    model = BUILDERS[arch](num_classes=num_classes, loss="softmax", pretrained=False)
    model.load_state_dict(state, strict=True)
    model.eval()
    print(f"{arch}: loaded {weights.name} strict=True, num_classes={num_classes}")
    return model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=sorted(BUILDERS), default="osnet_ain_x1_0")
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--height", type=int, default=256)
    ap.add_argument("--width", type=int, default=128)
    args = ap.parse_args()

    model = load(args.arch, args.weights)
    net = EmbeddingHead(model).eval()
    dummy = torch.randn(1, 3, args.height, args.width)

    with torch.no_grad():
        feat = model(dummy)                      # [1, 512]
        wrapped = net(dummy)                     # [1, 512, 1, 1]
    print(f"eval forward -> {tuple(feat.shape)} / rewritten {tuple(wrapped.shape)}")
    assert feat.ndim == 2 and feat.shape[0] == 1, f"unexpected feature shape {feat.shape}"
    assert wrapped.shape == (1, feat.shape[1], 1, 1), f"expected 4-D, got {wrapped.shape}"
    # The Linear->Conv1x1 rewrite must be an identity, not an approximation.
    delta = (feat - wrapped.flatten(1)).abs().max().item()
    print(f"  rewrite vs original forward: max|diff|={delta:.3e}")
    assert delta < 1e-5, f"EmbeddingHead diverges from the model's forward: {delta}"
    # ...and it must accept a batch, which is the whole reason for the rewrite.
    with torch.no_grad():
        assert net(torch.randn(4, 3, args.height, args.width)).shape == (4, 512, 1, 1)

    torch.onnx.export(
        net,
        dummy,
        str(args.out),
        input_names=["images"],
        output_names=["embedding"],
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
        dynamic_axes={"images": {0: "batch"}, "embedding": {0: "batch"}},
    )

    # Prove the exported graph still computes the same thing as the torch model.
    import numpy as np
    import onnxruntime as ort

    sess = ort.InferenceSession(str(args.out), providers=["CPUExecutionProvider"])
    for i in range(3):
        x = torch.randn(1, 3, args.height, args.width)
        with torch.no_grad():
            ref = model(x).numpy()
        got = sess.run(None, {"images": x.numpy()})[0].reshape(ref.shape)
        diff = float(np.abs(ref - got).max())
        # Judge on RELATIVE error, not absolute: the feature is a post-ReLU
        # activation whose scale is model-dependent, so a fixed absolute
        # tolerance would flag ordinary float32 accumulation noise on one
        # variant and wave through a real divergence on another.
        rel = diff / float(np.abs(ref).max() + 1e-12)
        cos = float(
            (ref * got).sum() / (np.linalg.norm(ref) * np.linalg.norm(got) + 1e-12)
        )
        print(f"  onnx vs torch [{i}]: max|diff|={diff:.3e} rel={rel:.3e} cosine={cos:.8f}")
        assert rel < 1e-3 and cos > 0.9999, f"ONNX diverges from torch: rel={rel} cos={cos}"

    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
