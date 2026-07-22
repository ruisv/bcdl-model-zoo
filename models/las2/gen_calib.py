#!/usr/bin/env python3
"""[2/3] 生成 PTQ 校准数据(+ 可选 fp32 金标准用于验证)。

⚠️ 铁律 1 — 预处理一致: 校准/验证/部署必须用同一种 --mode(crop 或 resize) + 同分辨率。
   * resize: 整图缩放到 HxW。会按 (原宽/W) 压缩视差范围 → 远处视差变亚像素 →
             深度 z=fx*B/d 被 1/d 放大成大噪点 (实测 ZED 2K resize 到 640宽, 远处深度误差 10~33%)。
   * crop  : 从原图中心裁 HxW (不缩放)。保持原始视差尺度 → 深度噪点极小 (实测 P99 1.18%)。
             代价是视野变窄。**原图分辨率需 ≥ HxW。** 适合用 ROI 看中远距离。
   两种 mode 没有绝对优劣, 但校准必须和部署用同一种, 否则视差范围不匹配 → BPU 量化饱和、视差崩。

⚠️ 铁律 2 — domain match: 校准用你相机现场采的图最好(10~30 张); 混入异域公开数据反而掉点
   (实测: 5 张同域 ZED → EPE 0.12px; 加 54 张 Middlebury → 0.26px 变差)。

输入二选一:
  --sbs-dir   DIR    左右拼接(side-by-side)图目录, 如 ZED Explorer 输出
  --left-dir L --right-dir R   左/右分目录(文件名排序配对)

输出: <out-dir>/left/000.npy ... 和 <out-dir>/right/000.npy ...(raw RGB, 1x3xHxW float32)

可选金标准(验证用, 需 torch+仓库):
  --golden --repo <LiteAnyStereo> --size m --golden-sbs <one_pair.png>
  → 在 <out-dir> 写 val_left.npy / val_right.npy / val_disp_golden.npy
"""
import os
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
import sys
import glob
import argparse
import numpy as np
import cv2
import imageio.v2 as imageio


def fit(img_rgb, hw, mode):
    """把图按 mode 处理成 HxW。resize=整图缩放; crop=中心裁切(不缩放,保视差尺度)。"""
    img = img_rgb[..., :3]
    if mode == "crop":
        h, w = img.shape[:2]
        if h < hw[0] or w < hw[1]:
            sys.exit(f"crop 模式需原图({h}x{w}) >= 目标({hw[0]}x{hw[1]}); 否则改用 --mode resize")
        t, l = (h - hw[0]) // 2, (w - hw[1]) // 2
        return img[t:t + hw[0], l:l + hw[1]]
    return cv2.resize(img, (hw[1], hw[0]))                    # resize (W,H)


def to_nchw(img_rgb, hw, mode):
    x = fit(img_rgb, hw, mode)
    return np.ascontiguousarray(x.transpose(2, 0, 1)[None].astype(np.float32))


def split_sbs(path):
    img = imageio.imread(path)[..., :3]
    W = img.shape[1] - img.shape[1] % 2
    return img[:, : W // 2], img[:, W // 2:]


def collect_pairs(a):
    pairs = []
    if a.sbs_dir:
        for p in sorted(glob.glob(os.path.join(a.sbs_dir, "*"))):
            if p.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                pairs.append(("sbs", p, None))
    elif a.left_dir and a.right_dir:
        ls = sorted(glob.glob(os.path.join(a.left_dir, "*")))
        rs = sorted(glob.glob(os.path.join(a.right_dir, "*")))
        pairs = [("lr", l, r) for l, r in zip(ls, rs)]
    else:
        sys.exit("需要 --sbs-dir 或 (--left-dir 且 --right-dir)")
    if a.limit:
        pairs = pairs[: a.limit]
    return pairs


def main():
    ap = argparse.ArgumentParser(description="Generate PTQ calibration data for LAS2")
    ap.add_argument("--sbs-dir")
    ap.add_argument("--left-dir"); ap.add_argument("--right-dir")
    ap.add_argument("--hw", nargs=2, type=int, default=[480, 640], metavar=("H", "W"))
    ap.add_argument("--mode", default="resize", choices=["resize", "crop"],
                    help="resize=整图缩放(保视野压视差) | crop=中心裁切(保视差尺度,视野窄)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--limit", type=int, default=0, help="最多用几对(0=全部)")
    # golden
    ap.add_argument("--golden", action="store_true")
    ap.add_argument("--repo"); ap.add_argument("--size", default="m")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--golden-sbs", help="生成金标准用的一对 side-by-side 图")
    ap.add_argument("--golden-left"); ap.add_argument("--golden-right")
    ap.add_argument("--max-disp", type=int, default=192)
    a = ap.parse_args()

    cl = os.path.join(a.out_dir, "left"); cr = os.path.join(a.out_dir, "right")
    os.makedirs(cl, exist_ok=True); os.makedirs(cr, exist_ok=True)
    pairs = collect_pairs(a)
    for i, (kind, p0, p1) in enumerate(pairs):
        l, r = split_sbs(p0) if kind == "sbs" else (imageio.imread(p0), imageio.imread(p1))
        np.save(os.path.join(cl, f"{i:03d}.npy"), to_nchw(l, a.hw, a.mode))
        np.save(os.path.join(cr, f"{i:03d}.npy"), to_nchw(r, a.hw, a.mode))
    print(f"[calib] wrote {len(pairs)} pairs -> {cl} , {cr}  (1x3x{a.hw[0]}x{a.hw[1]} raw RGB, mode={a.mode})")

    if a.golden:
        import torch
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import shifts
        sys.path.insert(0, a.repo)
        import core.liteanystereov2 as v2
        from core.models import build_model, load_model_weights
        shifts.patch(v2)
        if a.golden_sbs:
            gl, gr = split_sbs(a.golden_sbs)
        else:
            gl, gr = imageio.imread(a.golden_left), imageio.imread(a.golden_right)
        lt = torch.tensor(to_nchw(gl, a.hw, a.mode)); rt = torch.tensor(to_nchw(gr, a.hw, a.mode))
        ck = a.ckpt or os.path.join(a.repo, "checkpoints", f"LAS2_{a.size.upper()}.pth")
        m = build_model("las2", fnet_pretrained=False, model_size=a.size, max_disp=a.max_disp)
        load_model_weights(m, torch.load(ck, map_location="cpu"), strict=True); m = m.eval()
        with torch.no_grad():
            d = m(lt, rt, max_disp=a.max_disp, test_mode=True).float().numpy().reshape(a.hw)
        np.save(os.path.join(a.out_dir, "val_left.npy"), to_nchw(gl, a.hw, a.mode))
        np.save(os.path.join(a.out_dir, "val_right.npy"), to_nchw(gr, a.hw, a.mode))
        np.save(os.path.join(a.out_dir, "val_disp_golden.npy"), d)
        print(f"[golden] fp32 disp [{d.min():.2f},{d.max():.2f}] mean {d.mean():.2f} -> {a.out_dir}/val_*.npy")


if __name__ == "__main__":
    main()
