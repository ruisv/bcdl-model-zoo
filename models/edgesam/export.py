#!/usr/bin/env python3
"""[1/3] Export EdgeSAM -> BPU-friendly ONNX: one encoder + two fixed-prompt decoders.

Upstream: chongzhou96/EdgeSAM ("EdgeSAM: Prompt-In-the-Loop Distillation for
On-Device Deployment of SAM"). EdgeSAM keeps SAM's promptable-segmentation split
but replaces the ViT-H image encoder with a lightweight RepViT backbone distilled
from SAM, so the whole thing runs on an edge device. The prompt encoder and the
two-way mask decoder are SAM's, unchanged in shape (256x64x64 image embedding).

This script drives the UPSTREAM EdgeSAM export (from a `--repo` checkout) and cuts
it into three static graphs. The split and the fixed prompt shapes ARE the BPU
adaptation — everything else is the upstream model:

  edge_sam_encoder_bpu.onnx   raw_image [1,3,1024,1024]  -> image_embeddings [1,256,64,64]
  decoder_sp1.onnx            (image_embeddings; point_coords [1,1,2]; point_labels [1,1])
  decoder_bp2.onnx            (image_embeddings; point_coords [1,2,2]; point_labels [1,2])

Why split encoder / decoder at all:
  The encoder is the expensive part and depends only on the image, so it runs
  ONCE per frame. The decoder is tiny and runs once PER PROMPT (every click/box).
  A single fused graph would re-run the encoder on every prompt. Splitting lets
  BCDL cache the embedding and pay only the cheap decoder per interaction.

Why TWO decoders (sp1 vs bp2):
  SAM's reference ONNX decoder takes a DYNAMIC number of prompt points plus
  mask_input / has_mask_input / orig_im_size. A BPU graph is STATIC: the number of
  prompt points is baked into the shapes at export time. So each prompt arity is a
  separate compiled decoder:
    sp1 = 1 foreground point                       -> point_coords [1,1,2], labels [1,1]
    bp2 = a box, encoded as its 2 corners          -> point_coords [1,2,2], labels [1,2]
  SAM point-label convention: 1 = foreground, 0 = background, 2 = box top-left,
  3 = box bottom-right, -1 = padding. sp1 uses label {1}; bp2 uses labels {2,3}.
  We also drop the dynamic dense-prompt path (mask_input/has_mask_input) and the
  orig_im_size resize — the low-res mask upsample to the original frame is CPU
  post-processing in BCDL, not in the graph.

Encoder input `raw_image` + `no_preprocess` (see config_encoder.yaml):
  SAM normalises with pixel_mean/pixel_std. This export BAKES that normalisation
  into the encoder graph (a Sub/Div at the front), which is why the input is named
  `raw_image` and the compiler config declares `no_preprocess` — the board hands
  the graph raw pixels and the graph normalises internally. The resize-longest-
  side-to-1024 + pad-to-square letterbox stays on the CPU: it is per-image and
  aspect-dependent, so it cannot be a static graph. Pass --no-bake-norm if your
  upstream export leaves normalisation on the CPU instead; then the board must feed
  an already-normalised tensor and calib.py must be run with --cpu-normalize.

Run on the convert host, in a torch env that can import the EdgeSAM `--repo`.

Usage:
    python export.py --repo /path/to/EdgeSAM \\
                     --ckpt /path/to/EdgeSAM/weights/edge_sam.pth \\
                     --out-dir .
"""
import argparse
import os
import sys

import torch


# SAM preprocessing constants (RGB, 0-255 domain), img size 1024, embed 256x64x64.
PIXEL_MEAN = [123.675, 116.28, 103.53]
PIXEL_STD = [58.395, 57.12, 57.375]
IMG_SIZE = 1024


def cap_ir(path, ir=9):
    """HBDK 4.x rejects IR > 9. Header change, not an operator downgrade."""
    import onnx
    m = onnx.load(path)
    m.ir_version = ir
    onnx.save(m, path)


class EncoderWrap(torch.nn.Module):
    """Image encoder with SAM normalisation optionally baked in.

    forward(raw_image[1,3,1024,1024]) -> image_embeddings[1,256,64,64].
    When bake_norm, the graph subtracts pixel_mean and divides by pixel_std so the
    board can feed raw pixels (hence `raw_image` + `no_preprocess`).
    """

    def __init__(self, image_encoder, bake_norm=True):
        super().__init__()
        self.image_encoder = image_encoder
        self.bake_norm = bake_norm
        self.register_buffer("mean", torch.tensor(PIXEL_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(PIXEL_STD).view(1, 3, 1, 1))

    def forward(self, raw_image):
        x = raw_image
        if self.bake_norm:
            x = (x - self.mean) / self.std
        return self.image_encoder(x)


class DecoderWrap(torch.nn.Module):
    """SAM prompt encoder + mask decoder at a FIXED prompt arity.

    Mirrors the upstream SamOnnxModel path but with a static point count and
    without the dynamic mask_input / has_mask_input / orig_im_size inputs. Outputs
    the low-res mask logits and the per-mask score; the upsample to the original
    frame is done on the CPU in BCDL.

    forward(image_embeddings[1,256,64,64],
            point_coords[1,N,2], point_labels[1,N]) -> (scores[1,K], masks[1,K,256,256])
    """

    def __init__(self, prompt_encoder, mask_decoder, npoints):
        super().__init__()
        self.prompt_encoder = prompt_encoder
        self.mask_decoder = mask_decoder
        self.npoints = npoints

    def forward(self, image_embeddings, point_coords, point_labels):
        # Sparse prompt embeddings from the (static) point set. This follows SAM's
        # prompt encoder: positional encoding of the coords + per-label embeddings.
        points = (point_coords, point_labels)
        sparse_emb, dense_emb = self.prompt_encoder(
            points=points, boxes=None, masks=None)

        low_res_masks, scores = self.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_emb,
            dense_prompt_embeddings=dense_emb,
            multimask_output=False,
        )
        return scores, low_res_masks


def build_model(repo, ckpt):
    """Import EdgeSAM from the checkout and load its weights.

    Upstream API (EdgeSAM registry). Kept behind this one function so the exact
    entry point is easy to adjust to the checkout in hand — the terms of the
    upstream repo are NOT reviewed here (licence tier D); confirm before use.
    """
    sys.path.insert(0, repo)
    from edge_sam import sam_model_registry  # noqa: E402
    model = sam_model_registry["edge_sam"](checkpoint=ckpt)
    return model.eval()


def export_encoder(model, out_dir, bake_norm, opset, ir):
    path = os.path.join(out_dir, "edge_sam_encoder_bpu.onnx")
    wrap = EncoderWrap(model.image_encoder, bake_norm=bake_norm).eval()
    dummy = torch.zeros(1, 3, IMG_SIZE, IMG_SIZE)  # raw pixels, 0-255 domain
    with torch.no_grad():
        torch.onnx.export(
            wrap, dummy, path,
            input_names=["raw_image"], output_names=["image_embeddings"],
            opset_version=opset)
    cap_ir(path, ir)
    print(f"[export] encoder -> {path}  (bake_norm={bake_norm})")
    return path


def export_decoder(model, out_dir, name, npoints, label_values, opset, ir):
    path = os.path.join(out_dir, name)
    wrap = DecoderWrap(model.prompt_encoder, model.mask_decoder, npoints).eval()
    emb = torch.zeros(1, 256, 64, 64)
    coords = torch.zeros(1, npoints, 2)
    labels = torch.tensor([label_values], dtype=torch.float32)  # [1, npoints]
    with torch.no_grad():
        torch.onnx.export(
            wrap, (emb, coords, labels), path,
            input_names=["image_embeddings", "point_coords", "point_labels"],
            output_names=["scores", "masks"],
            opset_version=opset)
    cap_ir(path, ir)
    print(f"[export] decoder -> {path}  (npoints={npoints}, labels={label_values})")
    return path


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", required=True, help="EdgeSAM checkout")
    ap.add_argument("--ckpt", required=True, help="EdgeSAM weights (.pth)")
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--no-bake-norm", action="store_true",
                    help="leave SAM normalisation on the CPU (board feeds a "
                         "normalised tensor; run calib.py with --cpu-normalize)")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--ir", type=int, default=9, help="cap IR; HBDK 4.x rejects IR>9")
    a = ap.parse_args()

    os.makedirs(os.path.abspath(a.out_dir), exist_ok=True)
    model = build_model(a.repo, a.ckpt)

    export_encoder(model, a.out_dir, not a.no_bake_norm, a.opset, a.ir)
    # sp1: one foreground point (label 1). bp2: a box's two corners (labels 2,3).
    export_decoder(model, a.out_dir, "decoder_sp1.onnx", 1, [1.0], a.opset, a.ir)
    export_decoder(model, a.out_dir, "decoder_bp2.onnx", 2, [2.0, 3.0], a.opset, a.ir)
    print("[export] done. Now run calib.py for the encoder and each decoder build.")


if __name__ == "__main__":
    main()
