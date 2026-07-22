#!/usr/bin/env python3
"""[1/3] 导出 LAS2 → BPU 友好 ONNX(含 unfold 移位重写 + IR 降版)。

在转换主机上运行(conda env: 含 torch + timm; 必须禁用 dynamo,见下)。
LAS2 模型内部对部分函数加了 @torch.compile,本机若无 triton 会报错,
故脚本顶部强制 TORCHDYNAMO_DISABLE=1(纯 eager,不影响导出结果)。

用法:
    python export_onnx.py --repo /path/to/LiteAnyStereo --size m --hw 480 640 \
        --out out/las2_m_480x640.onnx
"""
import os
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")   # 必须在 import torch 之前
import sys
import argparse
import torch
import onnx
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import shifts


def main():
    ap = argparse.ArgumentParser(description="Export LAS2 to BPU-friendly ONNX")
    ap.add_argument("--repo", required=True, help="LiteAnyStereo 仓库路径")
    ap.add_argument("--size", default="m", choices=["s", "m", "l", "h"])
    ap.add_argument("--ckpt", default=None, help="权重路径(默认 repo/checkpoints/LAS2_<SIZE>.pth)")
    ap.add_argument("--hw", nargs=2, type=int, default=[480, 640], metavar=("H", "W"))
    ap.add_argument("--max-disp", type=int, default=192)
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--ir", type=int, default=9, help="降到 IR9(D-Robotics 工具链上限)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    sys.path.insert(0, a.repo)
    import core.liteanystereov2 as v2
    from core.models import build_model, load_model_weights
    shifts.patch(v2)   # 关键:替换两个 unfold 算子

    ckpt = a.ckpt or os.path.join(a.repo, "checkpoints", f"LAS2_{a.size.upper()}.pth")
    model = build_model("las2", fnet_pretrained=False, model_size=a.size, max_disp=a.max_disp)
    load_model_weights(model, torch.load(ckpt, map_location="cpu"), strict=True)
    model = model.eval()

    H, W = a.hw
    z0 = torch.zeros(1, 3, H, W)
    z1 = torch.zeros(1, 3, H, W)

    class Wrap(torch.nn.Module):
        """固定 max_disp / test_mode,只暴露 (left, right) 两个输入。"""
        def __init__(self, m, md):
            super().__init__(); self.m = m; self.md = md
        def forward(self, left, right):
            return self.m(left, right, max_disp=self.md, test_mode=True)

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    torch.no_grad().__enter__()
    torch.onnx.export(
        Wrap(model, a.max_disp).eval(), (z0, z1), a.out,
        opset_version=a.opset, input_names=["left", "right"], output_names=["disp"],
    )
    m = onnx.load(a.out)
    m.ir_version = a.ir
    onnx.save(m, a.out)
    print(f"[export] OK -> {a.out}  (LAS2-{a.size.upper()} {H}x{W}, opset{a.opset}, IR{a.ir})")


if __name__ == "__main__":
    main()
