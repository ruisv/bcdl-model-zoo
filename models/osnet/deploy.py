#!/usr/bin/env python3
"""[4/4] QAT OSNet -> .hbm, following the toolchain's OWN export flow.

THE FLOW (from hat/utils/hbdk4/hbir_exporter.py in horizon_torch_samples, which
ships inside the OpenExplorer container):

    set_fake_quantize(model, FakeQuantState.VALIDATION)
    qat_hbir       = export(model, example)              # export the QAT model
    quantized_hbir = hbdk4.compiler.convert(qat_hbir, march)
    hbdk4.compiler.compile(quantized_hbir, path, march)

The load-bearing detail: **`convert_fx` is not part of this at all.** The
fake-quant -> real-quant conversion happens at the HBIR level, not the PyTorch
level. Exporting the QAT model directly produces a graph still carrying
`qnt.const_fake_quant` nodes, which hbdk marks `illegal` — that is EXPECTED, and
`convert` is what resolves them. It is not a broken graph; do not try to "fix" it
in PyTorch.

DEAD ENDS, recorded so nobody re-walks them. Converting in PyTorch with
`convert_fx` instead marches straight into the plugin's quantized-mean gap:
quantized InstanceNorm AND AdaptiveAvgPool2d both reduce over `dim=(2,3)`, and
the plugin's quantized `mean` has no multi-dim path (its typeguard rejects the
tuple; silence that and the implementation does `scale * x.shape[dim]` on the
tuple and dies; reduce sequentially instead and the int8 intermediate has no
inferable dtype). All avoidable by not using convert_fx. The general lesson,
learned the hard way here: when the vendor ships a reference exporter, copy it —
do not patch your way down the stack from the first error message.

The QuantStub-fronted graph has 399 fake-quant nodes and passes prepare in one
go: a pure CNN, unlike attention stacks, has no operator-coverage surprises.

Runs inside the OpenExplorer GPU container (needs horizon_plugin_pytorch + hbdk4):
    ./compile.sh --gpu 2 python deploy.py --weights osnet_ain_x1_0_msmt17.pth \\
        --qat qat_osnet_ain.pth --crops crops/ \\
        --out out/osnet_ain_qat_nashm_256x128.hbm
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from export import EmbeddingHead, load  # noqa: E402

from horizon_plugin_pytorch.march import March, set_march  # noqa: E402
from horizon_plugin_pytorch.quantization import (  # noqa: E402
    FakeQuantState,
    QuantStub,
    prepare,
    set_fake_quantize,
)
from horizon_plugin_pytorch.quantization.qconfig_template import (  # noqa: E402
    default_qat_qconfig_setter,
)

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
H, W = 256, 128


class QatNet(torch.nn.Module):
    def __init__(self, head):
        super().__init__()
        self.quant = QuantStub()
        self.head = head

    def forward(self, x):
        return self.head(self.quant(x))


def preprocess(path: Path) -> np.ndarray:
    img = cv2.resize(cv2.imread(str(path)), (W, H), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return ((rgb - MEAN) / STD).transpose(2, 0, 1)


def deq(t):
    return t.dequantize() if hasattr(t, "dequantize") else t.as_subclass(torch.Tensor)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="osnet_ain_x1_0")
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--qat", type=Path, required=True)
    ap.add_argument("--crops", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--jobs", type=int, default=32)
    args = ap.parse_args()

    set_march(March.NASH_M)

    teacher = EmbeddingHead(load(args.arch, args.weights)).eval()
    model = prepare(QatNet(EmbeddingHead(load(args.arch, args.weights))).eval(),
                    example_inputs=(torch.zeros(1, 3, H, W),),
                    qconfig_setter=(default_qat_qconfig_setter,))
    ckpt = torch.load(args.qat, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    set_fake_quantize(model, FakeQuantState.VALIDATION)
    print(f"loaded QAT checkpoint (training cosine {ckpt.get('cosine', float('nan')):.4f})")

    # The number the board should reproduce, measured on real crops. The QAT
    # (fake-quant) forward here matches the board within ~0.003 cosine, so this
    # is a faithful preview, not just a sanity print.
    files = sorted(args.crops.glob("*.jpg"))[:256]
    x = torch.from_numpy(np.stack([preprocess(p) for p in files])).float()
    with torch.no_grad():
        ref = teacher(x).flatten(1)
        got = deq(model(x)).flatten(1)
    cos = F.cosine_similarity(got, ref)
    print(f"QAT vs FP32 on {len(files)} crops: mean {cos.mean():.4f} min {cos.min():.4f}")

    from hbdk4.compiler import compile as hbdk_compile
    from hbdk4.compiler import convert, save, statistics
    from horizon_plugin_pytorch.quantization.hbdk4 import export

    qat_hbir = export(model, torch.zeros(1, 3, H, W), input_names=["images"],
                      output_names=["embedding"])
    print("exported QAT hbir")

    quantized_hbir = convert(qat_hbir, "nash-m")
    print("converted hbir to quantized (the illegal const_fake_quant nodes are gone)")
    try:
        statistics(quantized_hbir)
    except Exception as e:  # noqa: BLE001 - informational only
        print(f"(statistics unavailable: {type(e).__name__})")

    # Save the .bc alongside so verify_cosine layer A/B has the host reference.
    save(quantized_hbir, str(args.out.with_suffix(".bc")))
    hbdk_compile(quantized_hbir, str(args.out), march="nash-m", opt=2,
                 jobs=args.jobs, progress_bar=False)
    print(f"compiled -> {args.out} ({args.out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
