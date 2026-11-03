# Silero VAD v6.2.3 — voice activity detection, 16 kHz

One 32 ms chunk of audio in, a speech probability out, with the LSTM state
carried from call to call. Upstream:
[snakers4/silero-vad](https://github.com/snakers4/silero-vad), v6.2.3 — the
weights are taken from the `silero_vad_op18_ifless.onnx` that
`pip install silero-vad==6.2.3` installs.

| build | BCDL consumes | on-board |
|---|---|---|
| `silero_vad_16k_int16_nashm.hbm` | one streaming step: magnitude spectrum + state in, probability + state out | **~0.28 ms BPU**, 0.39 ms per chunk end to end (1.2% of real time) |

The graph is tiny (~1.0 MB compiled). Everything interesting about this recipe is
**where the graph starts** and **what it is calibrated on**.

## What goes wrong: the STFT on the BPU

Silero VAD opens with a short-time Fourier transform written as a strided
convolution (258 output channels: 129 real, 129 imaginary), then
`sqrt(re² + im²)`. Exported as published, the whole thing compiles cleanly at
int16, and it is wrong:

| node | quantised cosine |
|---|---|
| `Mul` (re²) | 0.9969 |
| **`Mul_1` (im²)** | **0.0** |
| `h_out` (model output) | 0.9891 |
| `c_out` (model output) | 0.9948 |
| `prob` (model output) | 1.0 |

The squared terms span ~5000 (the `Sqrt` input threshold is 5025); int16
resolves that range in steps of about 0.15, and every quiet frequency bin rounds
to zero. One step of the model loses only a little — but the state is fed back
every 32 ms, so the error compounds, and **on the board the speech probability
sagged to 0.16–0.5 in the middle of sentences**.

Two things in that table deserve a second look:

- **`prob` reads 1.0 for the broken build.** It is a single number, and the
  cosine of a one-element vector is 1 whatever its value. The compiler's output
  cosine is blind on exactly the output that matters. Read the state outputs.
- **A state cosine of 0.989 is fatal here and would be fine almost anywhere
  else.** A recurrent graph feeds its output back as input 31 times a second; a
  per-step cosine is not an end-to-end number.

**The fix is to move the STFT to the CPU**, where it is a 256-point FFT on four
frames — trivial. The shipped graph starts at the magnitude spectrum, and its
state outputs come out of the compiler at cosine 0.9998.

(Every cosine in this section is the compiler's closing table, which — measured
below — is computed on **calibration sample 0 alone**. The board result is what
condemns the whole-graph build; the table only shows where it breaks.)

## The CPU front end (and the input contract)

Per 512-sample chunk at 16 kHz — this is `frontend()` in `export.py`, and what
BCDL's `SileroVad` does in C++:

1. Prepend the previous chunk's last **64** samples (zeros after a reset) → 576.
2. Reflect-pad **64** samples on the right, edge sample excluded → 640.
3. Four frames of **256**, hop **128**, **periodic** Hann window, `|rfft|`.
4. → `mag [1, 129, 4]` (bins × frames).

Inputs `mag [1,129,4]`, `h [1,128]`, `c [1,128]`; outputs `prob [1,1]`,
`h_out [1,128]`, `c_out [1,128]`. Zero `h`/`c` and zero context after a reset.

## Exporting it: take the weights from the ONNX, not the safetensors

The pip package ships two sets of weights, and they are **not the same model**.
`silero_vad_16k.safetensors` holds **older weights**: loaded into this graph,
they move the speech probability by ~0.5 against the v6.2.3 ONNX and JIT models
on the same audio. `export.py` reads the initializers of
`silero_vad_op18_ifless.onnx` — the release everyone actually runs.

The graph is re-expressed in torch (4 × Conv1d+ReLU, one LSTM cell, a 1×1 conv,
sigmoid) and exported at a fixed shape, so there is no control flow and no
dynamic dimension. `--check-wav` runs the result against the upstream ONNX and
the official package chunk by chunk with the state carried; on a 30 s noisy
recording (989 chunks) the largest probability difference is **1.0e-6** against
the upstream ONNX and **1.7e-6** against `load_silero_vad()`.

`export.py --whole` exports the rejected graph (STFT inside, `x [1,1,640]`), so
the failure above can be reproduced. It matches the upstream ONNX exactly
(max difference 0) — the graph is right; the quantisation is what breaks it.

## Calibration: real streaming states

A sample here is **a spectrum plus the LSTM state that chunk actually arrives
with**. Zero states describe only the first chunk after a reset; every other call
sees a state the network produced itself. `calib.py` runs the float graph over
whole recordings and snapshots `(mag, h, c)` every 37th chunk, then keeps every
third — three positionally-matched directories.

The audio is what a VAD is for — noisy speech, and noise alone:

- ~30 s sessions of **AISHELL-1** utterances ([OpenSLR 33](https://www.openslr.org/33/)),
  trimmed to their speech and separated by 0.3–3 s of silence, mixed with
  **MUSAN** noise ([OpenSLR 17](https://www.openslr.org/17/), the `noise/` part)
  at **10 dB and 0 dB** SNR over the speech;
- three minutes of MUSAN noise alone, at the level of a 10 dB session's
  background.

**~245 samples is enough.** All 733 snapshots took ~20 minutes to calibrate
(under the default method search) and changed nothing measurable.

**Pin the calibration method.** With `calibration_type: "default"`, hb_compile
searches between max-percentile and kl and keeps one. On the shipped calibration
set it kept max-percentile 0.99995; on this recipe's own draw of the same kind of
audio it kept **kl**, which put the spectrum input's threshold at **1.8** —
speech reaches 13 and more, so every loud bin saturates. That build compiles
cleanly; on held-out streams its 99th-percentile probability error against float
is 0.48, against 0.19 for the same data calibrated with max-percentile.
`config.yaml` pins `max` / `0.99995`, which reproduces the shipped build's
quantisation thresholds exactly when given the shipped calibration set, and
compiles in about 1.3 minutes instead of 10.

Which clips a 245-sample set happens to draw still moves the input range — the
shipped set's largest spectrum value is 70.9, this recipe's draw tops out
at 13.6 — and with the method pinned that did not matter on held-out audio (see
below).

## The compiler's output cosine is one sample

The closing table hb_compile prints (`h_out`, `c_out` cosine) is computed on
**calibration sample 0 only**. Measured: running the compiled `.bc` and the float
ONNX on every calibration sample, sample 0 reproduces the printed figures to three
decimals, and the rest of the set does not.

On this recipe's draw, sample 0 is a near-silent chunk (largest spectrum value
0.001), and the table reads **h_out 0.9249 / c_out 0.9598** — a number that looks
like a broken build. Over all 247 samples the median is **0.99980 / 0.99999**,
and the shipped build, whose sample 0 happened to be speech, has a worst sample
of 0.936 hiding behind its printed 0.9998. Neither number is the gate. For a
recurrent model the gate is a **streaming** run: carry the state through whole
held-out recordings and compare the probabilities.

## Why not int8

int8 compiles, is **only 3% faster** on a graph this small, and its state
outputs drop to cosine 0.958 / 0.975 in the compiler's table. On the board it
calls **5.5 s of every minute of pure noise** speech. A VAD that fires on noise wakes the recogniser for nothing;
int16 calls none.

## Measured on the board (from BCDL)

100 noisy sessions (30 s each, AISHELL-1 utterances 0.3–3 s apart, MUSAN noise at
10 and 0 dB) and 24 minutes of noise alone, BCDL's default segmenter (threshold
0.5, 250 ms to open, 300 ms to close):

| | frame F1, 10 / 0 dB | utterances missed, 10 / 0 dB | end of speech declared after (median, 10 dB) | false speech on 24 min of noise |
|---|---|---|---|---|
| float (ONNX) | 0.929 / 0.870 | 2.3% / 11.5% | 0.36 s | 0 |
| **int16, BPU (shipped)** | **0.924 / 0.863** | **3.4% / 13.2%** | **0.36 s** | **0** |

BPU time ~0.28 ms per call (`hrt_model_exec`); 0.39 ms per chunk end to end
including the CPU STFT and Python.

## Running it

```bash
pip install silero-vad==6.2.3 onnxruntime onnxslim soundfile
python export.py --check-wav /path/to/any_16k.wav
onnxslim silero_vad_16k_mag.onnx silero_vad_16k_mag_slim.onnx
python calib.py --config config.yaml --onnx silero_vad_16k_mag_slim.onnx \
                --aishell /path/to/data_aishell/wav --musan /path/to/musan/noise
./compile.sh --gpu 0
```

AISHELL-1 ships each speaker as a `.tar.gz` under `data_aishell/wav/`; untar them
first. Both datasets are 16 kHz, which `calib.py` requires.

## Verification

This recipe was re-run end to end on the convert host (OE 3.7.0): export,
calibration from AISHELL-1 + MUSAN, compile. What was checked, and how:

**Export.** `export.py` from the pip package's `silero_vad_op18_ifless.onnx`,
then `onnxslim`, gives a graph with **byte-identical weights and the same 30-node
operator sequence** as the shipped ONNX; streamed over 12 held-out recordings
(13 736 chunks, state carried) the two agree to **0.0** in probability and state.
Against upstream: max probability difference **1.0e-6** (upstream ONNX) and
**1.7e-6** (`silero_vad` package), 989 chunks.

**Calibration + compile.** 23 recordings (14.5 min) → 247 samples; compile
succeeds (~1.3 min on an RTX 4090), every BPU node `si16`, `.hbm`
1 006 536 bytes — the same size as the shipped one. The graph cannot run at batch
8 (*"Reset batch_size=1"*), so every sample counts. Compiler table: prob 1.0,
h_out 0.9249, c_out 0.9598 — sample 0 only, see above.

**Pinned config against the shipped calibration set.** `config.yaml` compiled
with the shipped build's own 245 samples gives a `quant_info.json` identical to
the shipped build's, and the same streaming results to every printed digit.

**Streaming, `.bc` vs float ONNX**, on 12 held-out recordings (5 sessions at
10 dB, 5 at 0 dB, 2 min of noise alone at the 0 dB level; 13 736 chunks):

| build | mean \|Δp\| | p99 \|Δp\| | same decision at 0.5 |
|---|---|---|---|
| shipped | 0.0146 | 0.240 | 98.42% |
| **this recipe** | **0.0134** | **0.189** | **98.88%** |
| this recipe, `calibration_type: default` (kl) | 0.0189 | 0.480 | 98.01% |

The held-out recordings are different mixtures from the calibration set, but are
built from the same utterance and noise pools.

**Not re-run:** the rebuilt `.hbm` was **not** run on the board. The on-board
numbers above (task metrics and latency) are BCDL's measurements of the shipped
build; the rebuild matches it in size, method and thresholds-on-the-same-data,
and does slightly better than it on the host streaming check. The `--whole` and
int8 builds were not recompiled here; their figures are from the original
conversion.

## Licence

**MIT** for the code and for the published weights (the ONNX release this build
is converted from). The training data is not disclosed, so there is no data
chain to inherit a restriction from. **Redistributable, including
commercially.**

The calibration data (AISHELL-1, Apache-2.0; MUSAN, CC BY 4.0 for the noise
part) only sets quantisation thresholds and is not distributed with the `.hbm`.
