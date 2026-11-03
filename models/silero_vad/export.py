#!/usr/bin/env python3
"""[1/3] Export Silero VAD v6.2.3 (16 kHz) as one fixed-shape streaming step.

    pip install silero-vad==6.2.3
    python export.py --check-wav /path/to/any_16k.wav
    onnxslim silero_vad_16k_mag.onnx silero_vad_16k_mag_slim.onnx

**The weights come from the published `silero_vad_op18_ifless.onnx`**, which the
pip package installs under `silero_vad/data/` (found automatically, or pass
`--upstream`). The same package also ships `silero_vad_16k.safetensors`, and it
is a trap: it holds **older weights** — loaded into this graph they move the
speech probability by up to 0.52 against the v6.2.3 ONNX and JIT models. Do not
use it.

**The graph starts at the magnitude spectrum, not at audio.** Silero VAD opens
with an STFT written as a strided convolution, then `sqrt(re² + im²)`. Compiled
on the BPU, the squared terms span ~5000; int16 resolves that range in steps of
~0.15, every quiet frequency bin rounds away, and the LSTM state drifts (output
cosine 0.989 per step). Because the state is fed back every 32 ms the error
compounds, and the speech probability sags to 0.16-0.5 in the middle of a
sentence. So the STFT runs on the CPU (`frontend()` below, which is exactly what
the runtime does) and the BPU graph is:

    mag [1,129,4] -> 4 x (Conv1d + ReLU) -> LSTM cell (h, c [1,128]) -> 1x1 conv
                  -> sigmoid -> prob [1,1], h_out [1,128], c_out [1,128]

`--whole` exports the rejected graph with the STFT inside (input `x [1,1,640]`)
so the failure can be reproduced; it is not what BCDL loads.

**The CPU front end**, per 512-sample chunk at 16 kHz: prepend the previous
chunk's last 64 samples (zeros after a reset), reflect-pad 64 samples on the
right (edge sample excluded) -> 640 samples; 4 frames of 256, hop 128, periodic
Hann window, |rfft| -> 129 bins x 4 frames.

`--check-wav` runs the exported graph and the upstream ONNX side by side over a
recording, chunk by chunk with the state carried, and prints the largest
probability difference; if the `silero_vad` package imports, the official
`load_silero_vad()` model is compared too.
"""
import argparse
import os

import numpy as np

SR = 16000
CHUNK = 512
CONTEXT = 64
N_FFT, HOP, BINS, FRAMES = 256, 128, 129, 4
HANN = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(N_FFT) / N_FFT)   # periodic


def frontend(chunk, ctx):
    """One chunk -> (padded window [640], magnitude [129, 4], next context).

    The CPU half of the model. calib.py imports this, so calibration and export
    cannot disagree about it; it matches BCDL's runtime implementation.
    """
    x = np.concatenate([ctx, chunk]).astype(np.float32)               # 576
    x = np.concatenate([x, x[-2:-CONTEXT - 2:-1]])                     # 640, reflect
    fr = np.stack([x[i * HOP:i * HOP + N_FFT] for i in range(FRAMES)]).astype(np.float64)
    mag = np.abs(np.fft.rfft(fr * HANN, axis=1)).T.astype(np.float32)  # [129, 4]
    return x, mag, chunk[-CONTEXT:].astype(np.float32)


def find_upstream():
    try:
        import silero_vad
    except ImportError:
        raise SystemExit("silero_vad is not installed; pip install silero-vad==6.2.3 "
                         "or pass --upstream /path/to/silero_vad_op18_ifless.onnx")
    p = os.path.join(os.path.dirname(silero_vad.__file__), "data",
                     "silero_vad_op18_ifless.onnx")
    if not os.path.exists(p):
        raise SystemExit(f"{p} not found; pass --upstream")
    return p


def load_onnx_weights(path):
    """The 16 kHz weights out of the published ONNX's initializers.

    NOT the package's safetensors file: that one holds older weights.
    """
    import onnx
    import torch

    W = {i.name: torch.from_numpy(onnx.numpy_helper.to_array(i).copy())
         for i in onnx.load(path).graph.initializer}
    sd = {"stft": W["model.stft.forward_basis_buffer"]}               # [258, 1, 256]
    for i in range(4):
        sd[f"conv{i}.weight"] = W[f"model.encoder.{i}.reparam_conv.weight"]
        sd[f"conv{i}.bias"] = W[f"model.encoder.{i}.reparam_conv.bias"]
    for a in ("weight_ih", "weight_hh", "bias_ih", "bias_hh"):
        sd[a] = W["model.decoder.rnn." + a]
    sd["final.weight"] = W["model.decoder.decoder.2.weight"]
    sd["final.bias"] = W["model.decoder.decoder.2.bias"]
    return sd


def build(sd, whole):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class SileroStep(nn.Module):
        """One 32 ms step. `whole` puts the STFT inside (the rejected build)."""

        def __init__(self):
            super().__init__()
            self.whole = whole
            self.stft = nn.Conv1d(1, 2 * BINS, N_FFT, stride=HOP, bias=False)
            self.convs = nn.ModuleList([
                nn.Conv1d(129, 128, 3, 1, 1), nn.Conv1d(128, 64, 3, 2, 1),
                nn.Conv1d(64, 64, 3, 2, 1), nn.Conv1d(64, 128, 3, 1, 1)])
            self.ih = nn.Linear(128, 512)
            self.hh = nn.Linear(128, 512)
            self.final = nn.Conv1d(128, 1, 1)
            self.stft.weight.data = sd["stft"]
            for i, c in enumerate(self.convs):
                c.weight.data = sd[f"conv{i}.weight"]
                c.bias.data = sd[f"conv{i}.bias"]
            self.ih.weight.data, self.ih.bias.data = sd["weight_ih"], sd["bias_ih"]
            self.hh.weight.data, self.hh.bias.data = sd["weight_hh"], sd["bias_hh"]
            self.final.weight.data = sd["final.weight"]
            self.final.bias.data = sd["final.bias"]

        def forward(self, m, h, c):
            if self.whole:                       # m is audio [1, 1, 640]
                s = self.stft(m)
                re, im = s[:, :BINS], s[:, BINS:]
                m = torch.sqrt(re * re + im * im)
            for conv in self.convs:
                m = F.relu(conv(m))              # -> [1, 128, 1]
            g = self.ih(m[:, :, 0]) + self.hh(h)
            i, f, gg, o = g.chunk(4, dim=1)      # PyTorch LSTMCell gate order
            c2 = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(gg)
            h2 = torch.sigmoid(o) * torch.tanh(c2)
            p = torch.sigmoid(self.final(F.relu(h2).unsqueeze(2)))
            return p.reshape(1, 1), h2, c2

    return SileroStep().eval()


def check(onnx_path, upstream, wav, whole):
    """Max |prob difference| against the upstream ONNX and the official package."""
    import onnxruntime as ort
    import soundfile as sf

    a, sr = sf.read(wav, dtype="float32", always_2d=True)
    a = a.mean(1)
    if sr != SR:
        raise SystemExit(f"{wav}: {sr} Hz; this check needs 16 kHz audio")
    ours = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    ref = ort.InferenceSession(upstream, providers=["CPUExecutionProvider"])
    try:
        import torch
        from silero_vad import load_silero_vad
        official = load_silero_vad(onnx=False)
        official.reset_states()
    except ImportError:
        official = None

    h = np.zeros((1, 128), np.float32)
    c = np.zeros((1, 128), np.float32)
    st = np.zeros((2, 1, 128), np.float32)
    ctx = np.zeros(CONTEXT, np.float32)
    d_ref = d_off = 0.0
    n = len(a) // CHUNK
    for k in range(n):
        ch = a[k * CHUNK:(k + 1) * CHUNK]
        x, mag, nctx = frontend(ch, ctx)
        feed = x.reshape(1, 1, -1) if whole else mag[None]
        p, h, c = ours.run(None, {ours.get_inputs()[0].name: feed, "h": h, "c": c})
        pr, st = ref.run(None, {"input": np.concatenate([ctx, ch])[None],
                                "state": st, "sr": np.array(SR, np.int64)})
        d_ref = max(d_ref, abs(float(p.ravel()[0]) - float(pr.ravel()[0])))
        if official is not None:
            po = official(torch.from_numpy(ch.copy()), SR).item()
            d_off = max(d_off, abs(float(p.ravel()[0]) - po))
        ctx = nctx
    print(f"[check] {n} chunks of {os.path.basename(wav)}")
    print(f"[check] max |p(export) - p(upstream ONNX)|      = {d_ref:.2e}")
    if official is not None:
        print(f"[check] max |p(export) - p(silero_vad package)| = {d_off:.2e}")
    return d_ref


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--upstream", help="silero_vad_op18_ifless.onnx (default: from the "
                                       "installed silero_vad package)")
    ap.add_argument("--out", help="default silero_vad_16k_mag.onnx "
                                  "(silero_vad_16k_t640.onnx with --whole)")
    ap.add_argument("--whole", action="store_true",
                    help="put the STFT inside the graph: the REJECTED build, for reproduction")
    ap.add_argument("--check-wav", help="a 16 kHz recording to compare against upstream on")
    a = ap.parse_args()

    import onnx
    import torch

    upstream = a.upstream or find_upstream()
    out = a.out or ("silero_vad_16k_t640.onnx" if a.whole else "silero_vad_16k_mag.onnx")
    m = build(load_onnx_weights(upstream), a.whole)
    first = torch.zeros(1, 1, 640) if a.whole else torch.zeros(1, BINS, FRAMES)
    torch.onnx.export(m, (first, torch.zeros(1, 128), torch.zeros(1, 128)), out,
                      input_names=["x" if a.whole else "mag", "h", "c"],
                      output_names=["prob", "h_out", "c_out"],
                      opset_version=17, do_constant_folding=True, dynamo=False)
    # HBDK 4.x rejects IR > 9; opset 17 is well inside IR 9.
    g = onnx.load(out)
    if g.ir_version > 9:
        g.ir_version = 9
        onnx.save(g, out)
    print("wrote", out)
    if a.check_wav:
        check(out, upstream, a.check_wav, a.whole)
    print("now run:  onnxslim %s %s" % (out, out.replace(".onnx", "_slim.onnx")))


if __name__ == "__main__":
    main()
