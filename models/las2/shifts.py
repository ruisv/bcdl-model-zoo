"""LAS2 两个 unfold 算子的 BPU 友好移位重写(pad + slice),与原版数值等价(EPE=1e-6)。

为什么需要:torch.onnx 无法导出 `aten::unfold`(报 "input size not accessible"),
而 LAS2 的 `build_correlation_volume` 和 `context_upsample` 都用了 unfold。
这里用「逐位移 + 切片」的等价写法替换,全部是 BPU 原语(Pad/Slice/Mul/ReduceMean/Concat)。

用法:
    import core.liteanystereov2 as v2
    import shifts; shifts.patch(v2)        # 必须在构建/前向之前 monkeypatch
"""
import torch
import torch.nn.functional as F


def build_correlation_volume_shift(left, right, max_disp):
    """相关代价体: cost[d] = mean_C( left * shift_right_by_d(right) )。

    等价于原版 `(left_volume * right_volume).mean(1)`(right 沿视差维右移 d 列,左侧补零)。
    输出 [B, max_disp, H, W]。
    """
    B, C, H, W = left.shape
    outs = []
    for d in range(max_disp):
        rs = right if d == 0 else F.pad(right, (d, 0, 0, 0))[..., :W]   # right(x - d)
        outs.append((left * rs).mean(dim=1, keepdim=True))             # [B,1,H,W]
    return torch.cat(outs, dim=1).contiguous()                         # [B,D,H,W]


def context_upsample_shift(depth_low, up_weights):
    """convex 上采样: 3x3 邻域用 pad+slice 取代 unfold(3,1,1),再最近邻放大 4x 加权求和。"""
    b, c, h, w = depth_low.shape                                       # c == 1
    p = F.pad(depth_low, (1, 1, 1, 1))
    nbrs = [p[..., i:i + h, j:j + w] for i in range(3) for j in range(3)]  # 9 x [b,1,h,w]
    du = F.interpolate(torch.cat(nbrs, dim=1), (h * 4, w * 4), mode='nearest')
    return torch.sum(du * up_weights, dim=1, keepdim=True)


def patch(v2_module):
    """monkeypatch LAS2 的 liteanystereov2 模块(名字已被 from-import 绑定到该命名空间)。"""
    v2_module.build_correlation_volume = build_correlation_volume_shift
    v2_module.context_upsample = context_upsample_shift
