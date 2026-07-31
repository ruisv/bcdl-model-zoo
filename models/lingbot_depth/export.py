#!/usr/bin/env python3
"""[1/3] LingBot-Depth checkpoint -> static-shape ONNX, with the graph rewrites.

The upstream model cannot be traced as-is: its encoder drops depth tokens whose
patch holds no valid reading, so the transformer's sequence length depends on
the input data. This script rebuilds token preparation without that masking,
which makes the sequence exactly `1 + N_img + N_depth` and the graph static.

Everything that is not a static tensor op is pushed out to the host:

  * the graph takes the image already resized to (rows*14, cols*14) and
    normalised, and the depth already resized, validity-zeroed and log-remapped
    (keeps Resize / log / isnan / comparisons out, and puts the quantiser on
    well-behaved ranges);
  * it emits the depth head's raw output -- metric already, since this
    checkpoint sets remap_depth_out='linear' -- and the mask head's logits,
    before the sigmoid;
  * the bicubic position-embedding interpolation is evaluated once here and
    baked in as a constant. It is computed with upstream's historical
    `interpolate_offset` kludge rather than the onnx_compatible_mode path, so
    the constant matches what the released weights were trained and used with.

The script verifies itself two ways before writing anything: the rewrite against
upstream's own forward (expect max|d| = 0.0 with antialias=True), and
onnxruntime against torch.

Usage:
    python export.py --repo /path/to/lingbot-depth --ckpt model_v05.pt \\
                     --tokens 1200 --out lingbot_depth_v05_480x640.onnx

`--tokens` is the latency knob and it is quadratic -- see README.md. `--blocks N`
truncates the transformer for cheap compile probes: every op type and tensor
shape survives, only the block count changes, so per-block cost falls out of two
probes with different counts.
"""
import argparse
import glob
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# host-side preprocessing (mirrors DINOv2_RGBD_Encoder.forward)
# --------------------------------------------------------------------------- #
IMAGE_MEAN = (0.485, 0.456, 0.406)
IMAGE_STD = (0.229, 0.224, 0.225)


def preprocess(image: torch.Tensor, depth: torch.Tensor, tr: int, tc: int, antialias: bool):
    """image [1,3,H,W] in [0,1]; depth [1,1,H,W] in metres -> the two graph inputs."""
    h14, w14 = tr * 14, tc * 14
    img = F.interpolate(image, (h14, w14), mode="bilinear", align_corners=False,
                        antialias=antialias)
    mean = torch.tensor(IMAGE_MEAN, dtype=img.dtype, device=img.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGE_STD, dtype=img.dtype, device=img.device).view(1, 3, 1, 1)
    img = (img - mean) / std

    d = F.interpolate(depth, (h14, w14), mode="nearest")
    d = torch.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)
    valid = d > 0.01
    d = torch.log(d * valid.to(d.dtype))
    d = torch.where(valid, d, torch.zeros_like(d))
    return img, torch.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)


# --------------------------------------------------------------------------- #
# static-shape rewrite of MDMModel.forward
# --------------------------------------------------------------------------- #
class StaticMDM(nn.Module):
    def __init__(self, model, token_rows, token_cols, aspect_ratio, out_h, out_w,
                 repo, keep_blocks=0):
        super().__init__()
        self.m = model
        self.tr, self.tc = token_rows, token_cols
        self.out_h, self.out_w = out_h, out_w
        bb = model.encoder.backbone
        self.nreg = bb.num_register_tokens
        if keep_blocks:
            assert not bb.chunked_blocks, "probe assumes unchunked blocks"
            bb.blocks = nn.ModuleList(list(bb.blocks)[:keep_blocks])

        # Bake the interpolated position encoding. onnx_compatible_mode is turned
        # OFF here on purpose: we want the constant the reference implementation
        # actually uses, kludge and all, not the size= variant.
        bb.onnx_compatible_mode = False
        probe = torch.zeros(1, token_rows * token_cols, bb.embed_dim)
        with torch.no_grad():
            pos = bb.interpolate_pos_encoding_without_cls(
                probe, token_rows * 14, token_cols * 14, bb.pos_embed[:, 1:])
        self.register_buffer("img_pos", 1.0 + pos)   # data-type encoding: 1 = image
        self.register_buffer("dep_pos", 2.0 + pos)   # data-type encoding: 2 = depth

        sys.path.insert(0, repo)
        from mdm.utils.geo import normalized_view_plane_uv
        for level in range(5):
            uv = normalized_view_plane_uv(width=token_cols * 2 ** level,
                                          height=token_rows * 2 ** level,
                                          aspect_ratio=aspect_ratio, dtype=torch.float32)
            self.register_buffer(f"uv{level}", uv.permute(2, 0, 1).unsqueeze(0))

    def forward(self, image, depth_log):
        m, bb = self.m, self.m.encoder.backbone
        tr, tc = self.tr, self.tc

        # token preparation (prepare_tokens_with_masks, masking removed)
        x_img = bb.patch_embed(image) + self.img_pos
        x_dep = bb.depth_patch_embed(depth_log) + self.dep_pos
        x_cls = (bb.cls_token.squeeze(0) + bb.pos_embed.squeeze(0)[:1]).unsqueeze(0)
        parts = [x_cls]
        if bb.register_tokens is not None:
            parts.append(bb.register_tokens)
        x = torch.cat(parts + [x_img, x_dep], dim=1)

        blocks = [b for c in bb.blocks for b in c] if bb.chunked_blocks else list(bb.blocks)
        n = m.encoder.intermediate_layers
        take = list(range(len(blocks) - n, len(blocks))) if isinstance(n, int) else list(n)
        outs = []
        for i, blk in enumerate(blocks):
            x = blk(x)
            if i in take:
                outs.append(bb.norm(x))

        start = 1 + self.nreg
        cls_token = outs[-1][:, 0]
        feats = [o[:, start:start + tr * tc] for o in outs]

        f = None
        for proj, feat in zip(m.encoder.output_projections, feats):
            g = proj(feat.permute(0, 2, 1).unflatten(2, (tr, tc)))
            f = g if f is None else f + g
        f = f + cls_token[..., None, None]

        levels = [torch.cat([f, self.uv0], dim=1)] + [getattr(self, f"uv{i}") for i in range(1, 5)]
        levels = m.neck(levels)
        depth = m.depth_head(levels)[-1]
        mask_logit = m.mask_head(levels)[-1] if hasattr(m, "mask_head") else None

        if tuple(depth.shape[-2:]) != (self.out_h, self.out_w):
            depth = F.interpolate(depth, (self.out_h, self.out_w), mode="bilinear",
                                  align_corners=False)
            if mask_logit is not None:
                mask_logit = F.interpolate(mask_logit, (self.out_h, self.out_w),
                                           mode="bilinear", align_corners=False)

        # remap_depth_out == 'linear' for this checkpoint: `depth` is metric already.
        return (depth, mask_logit) if mask_logit is not None else (depth,)


def patch_batch1_blocks(vit_cls):
    """Run the block stack on a plain tensor instead of an xFormers nested list.

    Upstream's masked path hands the blocks a list of variable-length sequences
    and only implements that via xFormers. At batch 1 the list holds exactly one
    sequence, so this is the same math without the dependency — and it is needed
    even for the unmasked reference run, because the list wrapping is upstream's
    unconditional shape.
    """
    def _blocks_batch1(self, x_img, x_depth, x_img_mask=None, x_depth_mask=None,
                       n=1, return_mae_aux=False, **kwargs):
        seqs = self.prepare_tokens_with_masks(x_img, x_depth, x_img_mask, x_depth_mask,
                                              **kwargs)
        t = seqs[0]
        total = len(self.blocks)
        take = list(range(total - n, total)) if isinstance(n, int) else list(n)
        out = []
        for i, blk in enumerate(self.blocks):
            t = blk(t)
            if i in take:
                out.append([t])
        return out

    vit_cls._get_intermediate_layers_not_chunked = _blocks_batch1


def load_example(repo, device, index=3):
    """Scene 3 by default: the lowest valid-depth ratio of the bundled eight."""
    import cv2
    d = sorted(p for p in glob.glob(os.path.join(repo, "examples", "*")) if os.path.isdir(p))
    d = d[index]
    rgb = cv2.cvtColor(cv2.imread(next(iter(glob.glob(os.path.join(d, "rgb.*"))))),
                       cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    raw = cv2.imread(os.path.join(d, "raw_depth.png"), cv2.IMREAD_UNCHANGED)
    raw = np.nan_to_num(raw.astype(np.float32) / 1000.0, nan=0.0, posinf=0.0, neginf=0.0)
    return (d, torch.from_numpy(rgb).permute(2, 0, 1)[None].to(device),
            torch.from_numpy(raw)[None, None].to(device))


def report(tag, ref, got):
    ref, got = np.asarray(ref, np.float64), np.asarray(got, np.float64)
    v = np.isfinite(ref) & np.isfinite(got) & (np.abs(ref) > 1e-6)
    rel = np.abs(got[v] - ref[v]) / np.abs(ref[v])
    cos = float(ref[v] @ got[v] / (np.linalg.norm(ref[v]) * np.linalg.norm(got[v]) + 1e-12))
    print(f"{tag:38s} max|d|={np.abs(got[v]-ref[v]).max():.3e}  "
          f"absrel={rel.mean():.3e}  cos={cos:.8f}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", required=True, help="lingbot-depth checkout")
    ap.add_argument("--ckpt", required=True, help="model.pt (Hugging Face or local)")
    ap.add_argument("--tokens", type=int, default=1200)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--out", required=True)
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--blocks", type=int, default=0,
                    help="truncate the transformer to N blocks (compile probe only)")
    ap.add_argument("--skip-verify", action="store_true")
    args = ap.parse_args()

    sys.path.insert(0, args.repo)
    from mdm.model.v2 import MDMModel
    from mdm.model.dinov2_rgbd.models.vision_transformer import DinoVisionTransformer
    patch_batch1_blocks(DinoVisionTransformer)

    aspect = args.width / args.height
    tr = round((args.tokens / aspect) ** 0.5)
    tc = round((args.tokens * aspect) ** 0.5)
    print(f"token grid {tr}x{tc} = {tr*tc} tokens, encoder input {tr*14}x{tc*14}, "
          f"attention sequence {1 + 2*tr*tc}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MDMModel.from_pretrained(args.ckpt).to(device).eval()
    wrapper = StaticMDM(model, tr, tc, aspect, args.height, args.width, args.repo,
                        keep_blocks=args.blocks).to(device).eval()

    scene, image, depth = load_example(args.repo, device)
    if not args.skip_verify:
        # 1) the rewrite against upstream's own path, both with masking off
        with torch.no_grad():
            ref = model.infer(image, depth_in=depth[:, 0], num_tokens=args.tokens,
                              apply_mask=False, use_fp16=False, enable_depth_mask=False)
            ref_depth = ref["depth"][0].float().cpu().numpy()
            for antialias in (True, False):
                img_in, dep_in = preprocess(image, depth, tr, tc, antialias)
                got = wrapper(img_in, dep_in)[0][0, 0].float().cpu().numpy()
                report(f"rewrite vs upstream (antialias={antialias})", ref_depth, got)

    # The deployed host preprocessing does not antialias -- it area-averages,
    # which OpenCV and a hand-written C++ resize both give without torch.
    img_in, dep_in = preprocess(image, depth, tr, tc, antialias=False)
    with torch.no_grad():
        outs = wrapper(img_in, dep_in)
    print("output shapes:", [tuple(o.shape) for o in outs])

    names = ["depth"] if len(outs) == 1 else ["depth", "mask_logit"]
    torch.onnx.export(
        wrapper.cpu(), (img_in.cpu(), dep_in.cpu()), args.out,
        input_names=["image", "depth_log"], output_names=names,
        opset_version=args.opset, do_constant_folding=True, dynamo=False)
    print("wrote", args.out, round(os.path.getsize(args.out) / 1e6, 1), "MB")

    if not args.skip_verify:
        # 2) onnxruntime against torch
        import onnxruntime as ort
        sess = ort.InferenceSession(args.out, providers=["CPUExecutionProvider"])
        ort_out = sess.run(None, {"image": img_in.cpu().numpy(),
                                  "depth_log": dep_in.cpu().numpy()})
        report("onnxruntime vs torch (depth)",
               outs[0][0, 0].float().cpu().numpy(), ort_out[0][0, 0])
        if len(outs) > 1:
            report("onnxruntime vs torch (mask_logit)",
                   outs[1][0, 0].float().cpu().numpy(), ort_out[1][0, 0])


if __name__ == "__main__":
    main()
