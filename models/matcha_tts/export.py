#!/usr/bin/env python3
"""[1/3] Export Matcha-TTS zh-en as two fixed-shape graphs for the BPU.

    python export.py --ckpt pytorch_model_sherpa.bin --matcha-src /path/to/matcha-tts-0.0.7.2 \\
                     --ids test_ids.json --reference-onnx /path/to/model-steps-3.onnx
    onnxslim matcha_enc_L128.onnx matcha_enc_L128_slim.onnx
    onnxslim matcha_dec_T512.onnx matcha_dec_T512_slim.onnx

Writes

  matcha_enc_L128.onnx : x_emb [1, 192, 128], x_mask [1, 1, 128] -> mu_x [1, 80, 128], logw [1, 1, 128]
  matcha_dec_T512.onnx : mu_y [1, 80, 512], y_mask [1, 1, 512], z [1, 80, 512] -> mel [1, 80, 512]
  matcha_zh_en_embedding.npy : the token embedding table [2190, 192], ALREADY scaled by sqrt(192)

The decoder has the three Euler steps (t = 0, 1/3, 2/3) and the mel
de-normalisation compiled in. What stays on the CPU: the embedding lookup,
the durations `ceil(exp(logw)) * length_scale`, length regulation (repeat each
mu_x column by its duration to get mu_y), and the starting noise z.

Three rewrites, each of which the stock model needs before a fixed-length graph
is the same model:

**1. Masked GroupNorm in the decoder.** Matcha runs each sentence at its own
length, rounded up to a multiple of 4 (the U-Net halves time twice).
`nn.GroupNorm` normalises over the whole time axis, and the decoder's conv
outputs in the padding are not zero (bias, plus spill from the last valid
frames), so padded to 512 its statistics depend on HOW MUCH padding there is.
The rewrite takes the statistics only over the frames the unpadded model would
have.

**2. Attention bias instead of the 0/1 mask.** Matcha hands its 0/1 float mask
straight to `scaled_dot_product_attention`, which ADDS a float mask to the
scores: valid keys get +1, padded keys +0. Padding is down-weighted, not
excluded — and at the model's own length there are at most 3 padded frames, at
512 there are hundreds. The rewrite keeps the +1/+0 over the frames the
unpadded model has and adds -100 past them.

Neither of those cares what the padding CONTAINS, only how much there is. So the
obvious check — "is the output sensitive to the padded values?" — passes, while
the stock export padded to 512 is off by up to ~10 in log-mel. `--ids` measures
both, against an unpadded run of the untouched model.

**3. The embedding lookup comes out of the graph.** Token ids cannot be a float
graph input: the compiler quantises them like any activation (it picked a
threshold of 127 for a 2190-entry vocabulary) and the encoder's duration output
dropped to cosine 0.83. The table ships as an .npy and the CPU does the lookup;
the encoder then calibrates to 0.99999.

A smaller trap, in this file's own code: the two masks above are upsampled from
the pooled mask with expand + reshape. Written with `repeat_interleave` twice in
one graph, torch.onnx tracing froze the attention's Reshape at a sequence
length of 2.

Upstream is the PyPI `matcha-tts==0.0.7.2` source tree (pass the extracted
sdist as --matcha-src; it is not pip-installed). Its `matcha.utils` package
imports lightning/hydra and diffusers 0.25.0 imports `cached_download`, which
current huggingface_hub no longer has; both are shimmed in memory below, so the
source tree is used unmodified. Extra deps: einops, conformer==0.3.2,
diffusers==0.25.0 (and Pillow, which diffusers imports).
"""
import argparse
import json
import logging
import os
import sys
import types
from types import SimpleNamespace as NS

import numpy as np
import torch

STEPS = 3
VOCAB = 2190


def shim_imports(src):
    """Use the matcha-tts source tree as shipped, without its training stack.

    `matcha/utils/__init__.py` pulls in lightning, hydra and rich for training
    utilities; the inference modules need only `get_pylogger`. Registering a
    bare `matcha.utils` package (pointing at the real directory, so
    `matcha.utils.model` still loads from source) avoids all of it.
    """
    sys.path.insert(0, src)
    import huggingface_hub
    if not hasattr(huggingface_hub, "cached_download"):    # removed in hub 0.26
        huggingface_hub.cached_download = huggingface_hub.hf_hub_download

    import matcha
    get_pylogger = lambda name=__name__: logging.getLogger(name)  # noqa: E731
    utils = types.ModuleType("matcha.utils")
    utils.__path__ = [os.path.join(src, "matcha", "utils")]
    utils.get_pylogger = get_pylogger
    pylogger = types.ModuleType("matcha.utils.pylogger")
    pylogger.get_pylogger = get_pylogger
    sys.modules["matcha.utils"] = utils
    sys.modules["matcha.utils.pylogger"] = pylogger
    matcha.utils = utils


def build(ckpt):
    """The zh-en hyper-parameters, rebuilt around the published state dict."""
    from matcha.models.components.flow_matching import CFM
    from matcha.models.components.text_encoder import TextEncoder

    enc_params = NS(n_feats=80, n_channels=192, filter_channels=768, filter_channels_dp=256,
                    n_heads=2, n_layers=6, kernel_size=3, p_dropout=0.1, spk_emb_dim=64,
                    n_spks=1, prenet=True)
    dp_params = NS(filter_channels_dp=256, kernel_size=3, p_dropout=0.1)
    dec_params = dict(channels=[256, 256], dropout=0.05, attention_head_dim=64, n_blocks=1,
                      num_mid_blocks=2, num_heads=2, act_fn="snakebeta")
    cfm_params = NS(solver="euler", sigma_min=1e-4)

    class Matcha(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = TextEncoder("RoPE Encoder", enc_params, dp_params, VOCAB, 1, 64)
            self.decoder = CFM(160, 80, cfm_params, dec_params, 1, 64)
            self.register_buffer("mel_mean", torch.tensor(0.0))
            self.register_buffer("mel_std", torch.tensor(1.0))

    m = Matcha()
    sd = torch.load(ckpt, map_location="cpu")
    sd = {k[len("model."):] if k.startswith("model.") else k: v for k, v in sd.items()}
    missing, unexpected = m.load_state_dict(sd, strict=False)
    if missing or unexpected:
        sys.exit("checkpoint does not match the zh-en architecture:\n  missing %s\n  unexpected %s"
                 % (missing, unexpected))
    return m.eval()


# --- rewrites 1 and 2 -------------------------------------------------------

def upsample_mask(mask, k):
    """1 over every frame of a k-frame block that holds any valid frame.

    At full rate k = 4, so this marks frames [0, ceil4(F)) — exactly the frames
    the unpadded model runs on. After one halving k = 2, after two k = 1.
    expand + reshape, not repeat_interleave: see the module docstring.
    """
    if k <= 1:
        return mask
    # Block1D sees [B, 1, T]; the transformer blocks see [B, T].
    p = torch.nn.functional.max_pool1d(mask if mask.dim() == 3 else mask.unsqueeze(1), k, k)
    return p.unsqueeze(-1).expand(-1, -1, -1, k).reshape(mask.shape)


def masked_group_norm(h, nm, gn):
    """GroupNorm whose statistics cover only the frames where nm == 1."""
    B, C, Tl = h.shape
    G = gn.num_groups
    hg = h.reshape(B, G, C // G, Tl)
    w = nm.reshape(B, 1, 1, Tl)
    cnt = (C // G) * w.sum(dim=(2, 3), keepdim=True)
    mean = (hg * w).sum(dim=(2, 3), keepdim=True) / cnt
    var = (((hg - mean) * w) ** 2).sum(dim=(2, 3), keepdim=True) / cnt
    hn = ((hg - mean) / torch.sqrt(var + gn.eps)).reshape(B, C, Tl)
    return hn * gn.weight.view(1, C, 1) + gn.bias.view(1, C, 1)


def rewrite_decoder(model, T):
    """Patch every Block1D and BasicTransformerBlock in place.

    Each block records its U-Net level on the first (eager) call from the mask
    length it sees, so run the decoder once before tracing.
    """
    from matcha.models.components.decoder import Block1D
    from matcha.models.components.transformer import BasicTransformerBlock

    def level(self, mask):
        if getattr(self, "_k", None) is None:
            self._k = 4 * int(mask.shape[-1]) // T   # 4 at full rate, 2, then 1
        return self._k

    def block1d_fwd(self, x, mask):
        conv, gn, act = self.block
        h = conv(x * mask)
        nm = upsample_mask(mask, level(self, mask))
        return act(masked_group_norm(h, nm, gn)) * mask

    def tblock_fwd(self, hidden_states, attention_mask=None, timestep=None, **kw):
        nm = upsample_mask(attention_mask, level(self, attention_mask))
        bias = attention_mask + (nm - 1.0) * 100.0
        return self._orig_forward(hidden_states, attention_mask=bias, timestep=timestep, **kw)

    for mod in model.modules():
        if isinstance(mod, Block1D):
            mod.forward = types.MethodType(block1d_fwd, mod)
        elif isinstance(mod, BasicTransformerBlock):
            mod._orig_forward = mod.forward
            mod.forward = types.MethodType(tblock_fwd, mod)


# --- the two graphs ---------------------------------------------------------

class Enc(torch.nn.Module):
    """TextEncoder.forward from the embedding on: takes scaled embeddings, not ids."""

    def __init__(self, e):
        super().__init__()
        self.e = e

    def forward(self, x_emb, x_mask):
        e = self.e
        x = e.prenet(x_emb, x_mask) if e.prenet is not None else x_emb
        x = e.encoder(x, x_mask)
        mu = e.proj_m(x) * x_mask
        logw = e.proj_w(torch.detach(x), x_mask)
        return mu, logw


class Dec(torch.nn.Module):
    """Three Euler steps of the flow-matching ODE, then mel de-normalisation."""

    def __init__(self, cfm, mean, std):
        super().__init__()
        self.est, self.mean, self.std = cfm.estimator, float(mean), float(std)

    def forward(self, mu, y_mask, z):
        x = z
        dt = 1.0 / STEPS
        for k in range(STEPS):
            t = torch.full((1,), k * dt)
            x = x + dt * self.est(x, y_mask, mu, t, None, None)
        return (x * self.std + self.mean) * y_mask


def embed(table, ids, L):
    """[1, 192, L] scaled embeddings, zero past the sentence."""
    x = np.zeros((1, table.shape[1], L), np.float32)
    x[0, :, :len(ids)] = table[np.asarray(ids)].T
    return x


def regulate(mu_x, logw, n, length_scale=1.0):
    """Durations and length regulation exactly as Matcha (and BCDL) do them."""
    d = np.ceil(np.exp(logw[0, 0, :n]) * length_scale).astype(int)
    return np.repeat(mu_x[0, :, :n], d, axis=1), int(d.sum())


@torch.no_grad()
def run_estimator(est, mu, mask, z, mean, std):
    x = z
    for k in range(STEPS):
        x = x + (1 / STEPS) * est(x, mask, mu, torch.full((1,), k / STEPS), None, None)
    return x * std + mean


def float_check(ref, enc_path, dec_path, table, ids_file, L, T, limit):
    """Padded fixed-shape ONNX pair vs the UNPADDED untouched model, same noise.

    Also runs the untouched model padded to T — the stock fixed-length export —
    to show what rewrites 1 and 2 are for.
    """
    import onnxruntime as ort
    E = ort.InferenceSession(enc_path, providers=["CPUExecutionProvider"])
    D = ort.InferenceSession(dec_path, providers=["CPUExecutionProvider"])
    est, mean, std = ref.decoder.estimator, ref.mel_mean, ref.mel_std
    rng = np.random.default_rng(0)
    worst = stock = 0.0
    n_sent = 0
    for sents in json.load(open(ids_file))[:limit]:
        for ids in sents:
            n = len(ids)
            xm = np.zeros((1, 1, L), np.float32)
            xm[..., :n] = 1
            mu_x, logw = E.run(None, {"x_emb": embed(table, ids, L), "x_mask": xm})
            mu_y, F = regulate(mu_x, logw, n)
            if F > T:
                continue
            z = rng.standard_normal((1, 80, T)).astype(np.float32)
            a = np.zeros((1, 80, T), np.float32)
            a[0, :, :F] = mu_y
            ym = np.zeros((1, 1, T), np.float32)
            ym[..., :F] = 1
            mel = D.run(None, {"mu_y": a, "y_mask": ym, "z": z})[0][0, :, :F]

            Fp = (F + 3) // 4 * 4                   # what the model itself runs at
            want = run_estimator(est, torch.tensor(a[:, :, :Fp]), torch.tensor(ym[:, :, :Fp]),
                                 torch.tensor(z[:, :, :Fp]), mean, std)[0, :, :F].numpy()
            padded = run_estimator(est, torch.tensor(a), torch.tensor(ym), torch.tensor(z),
                                   mean, std)[0, :, :F].numpy()
            worst = max(worst, float(np.abs(mel - want).max()))
            stock = max(stock, float(np.abs(padded - want).max()))
            n_sent += 1
    print("[check] %d sentences, unpadded untouched model as reference (max |diff| log-mel):"
          % n_sent)
    print("[check]   stock decoder padded to %d : %.3g" % (T, stock))
    print("[check]   exported ONNX pair          : %.2e" % worst)
    return worst


def reference_check(m, onnx_path, ids_file, limit):
    """The rebuilt torch model vs the published ONNX at noise 0 (deterministic)."""
    import onnxruntime as ort
    from matcha.utils.model import generate_path, sequence_mask
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    worst = 0.0
    for sents in json.load(open(ids_file))[:limit]:
        for ids in sents:
            with torch.no_grad():
                x, xl = torch.tensor([ids]), torch.tensor([len(ids)])
                mu_x, logw, x_mask = m.encoder(x, xl, None)
                w_ceil = torch.ceil(torch.exp(logw) * x_mask)
                y_len = torch.clamp_min(w_ceil.sum([1, 2]), 1).long()
                F = int(y_len.max())
                y_mask = sequence_mask(y_len, (F + 3) // 4 * 4).unsqueeze(1).to(x_mask.dtype)
                attn = generate_path(w_ceil.squeeze(1),
                                     (x_mask.unsqueeze(-1) * y_mask.unsqueeze(2)).squeeze(1))
                mu_y = torch.matmul(attn.transpose(1, 2), mu_x.transpose(1, 2)).transpose(1, 2)
                y = m.decoder(mu_y, y_mask, STEPS, 0.0, None)
                mine = (y * m.mel_std + m.mel_mean)[0, :, :F].numpy()
            ref = sess.run(None, {"x": np.array([ids], np.int64),
                                  "x_length": np.array([len(ids)], np.int64),
                                  "noise_scale": np.array([0.0], np.float32),
                                  "length_scale": np.array([1.0], np.float32)})[0][0]
            if mine.shape != ref.shape:
                sys.exit("frame count differs from the published ONNX: %s vs %s"
                         % (mine.shape, ref.shape))
            worst = max(worst, float(np.abs(mine - ref).max()))
    print("[check] rebuilt torch vs published ONNX (noise 0): max |diff| %.2e" % worst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="pytorch_model_sherpa.bin",
                    help="from the ModelScope repo; the *_sherpa one matches model-steps-3.onnx")
    ap.add_argument("--matcha-src", required=True,
                    help="extracted matcha-tts-0.0.7.2 sdist (the directory holding matcha/)")
    ap.add_argument("--deps", help="optional dir with einops/conformer/diffusers, put on sys.path")
    ap.add_argument("--L", type=int, default=128, help="encoder tokens per sentence")
    ap.add_argument("--T", type=int, default=512, help="decoder frames (multiple of 4)")
    ap.add_argument("--ids", help="token ids JSON from frontend_ids.py, for the float check")
    ap.add_argument("--check-texts", type=int, default=20)
    ap.add_argument("--reference-onnx",
                    help="the published model-steps-3.onnx, to check the rebuild against")
    a = ap.parse_args()
    if a.T % 4:
        sys.exit("--T must be a multiple of 4: the U-Net halves time twice")

    if a.deps:
        sys.path.insert(0, a.deps)
    shim_imports(a.matcha_src)
    ref = build(a.ckpt)      # untouched: the reference every check compares against
    mx = build(a.ckpt)       # the copy that is rewritten and exported
    rewrite_decoder(mx, a.T)

    table = (mx.encoder.emb.weight * (mx.encoder.n_channels ** 0.5)).detach().float().numpy()
    np.save("matcha_zh_en_embedding.npy", np.ascontiguousarray(table))
    print("wrote matcha_zh_en_embedding.npy", table.shape)

    enc = Enc(mx.encoder).eval()
    dec = Dec(mx.decoder, mx.mel_mean, mx.mel_std).eval()

    with torch.no_grad():    # the wrapper must be TextEncoder.forward from the embedding on
        from matcha.utils.model import sequence_mask
        ids = [703, 420, 4, 1078, 1599, 316, 536, 5]
        xm = sequence_mask(torch.tensor([len(ids)]), len(ids)).unsqueeze(1).float()
        mu0, lw0, _ = ref.encoder(torch.tensor([ids]), torch.tensor([len(ids)]), None)
        mu1, lw1 = enc(torch.tensor(embed(table, ids, len(ids))), xm)
        d = max(float((mu0 - mu1).abs().max()), float((lw0 - lw1).abs().max()))
        print("[check] encoder wrapper vs TextEncoder: max |diff| %.2e" % d)

    enc_path, dec_path = "matcha_enc_L%d.onnx" % a.L, "matcha_dec_T%d.onnx" % a.T
    torch.onnx.export(enc, (torch.zeros(1, table.shape[1], a.L), torch.ones(1, 1, a.L)),
                      enc_path, input_names=["x_emb", "x_mask"], output_names=["mu_x", "logw"],
                      opset_version=17, do_constant_folding=True, dynamo=False)
    mu, ym, z = torch.zeros(1, 80, a.T), torch.ones(1, 1, a.T), torch.zeros(1, 80, a.T)
    with torch.no_grad():
        dec(mu, ym, z)       # eager pass: every block records its U-Net level
    torch.onnx.export(dec, (mu, ym, z), dec_path, input_names=["mu_y", "y_mask", "z"],
                      output_names=["mel"], opset_version=17, do_constant_folding=True,
                      dynamo=False)
    print("wrote", enc_path, dec_path)

    if a.reference_onnx and a.ids:
        reference_check(ref, a.reference_onnx, a.ids, a.check_texts)
    if a.ids:
        float_check(ref, enc_path, dec_path, table, a.ids, a.L, a.T, a.check_texts)
    print("now run:  onnxslim %s %s" % (enc_path, enc_path.replace(".onnx", "_slim.onnx")))
    print("          onnxslim %s %s" % (dec_path, dec_path.replace(".onnx", "_slim.onnx")))


if __name__ == "__main__":
    main()
