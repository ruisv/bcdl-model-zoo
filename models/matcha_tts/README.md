# Matcha-TTS zh-en — speech synthesis acoustic model, two graphs

Chinese (and some English) text in, an 80-band mel spectrogram out; a vocoder
(Vocos 16 kHz, published with it) turns the mel into 16 kHz audio. Upstream:
[Matcha-TTS](https://github.com/shivammehta25/Matcha-TTS) (code), checkpoint
`pytorch_model_sherpa.bin` from ModelScope
[`dengcunqin/matcha_tts_zh_en_20251010`](https://modelscope.cn/models/dengcunqin/matcha_tts_zh_en_20251010).

Matcha is non-autoregressive, which is what makes it a BPU model at all: one
encoder pass, length regulation, one decoder pass. The split:

| stage | where | what |
|---|---|---|
| text → token ids | CPU | sherpa-onnx's lexicon front end (BCDL `MatchaFrontend` reproduces it id for id) |
| ids → embeddings | CPU | lookup in `matcha_zh_en_embedding.npy` — see trap 3 |
| `matcha_zh_en_enc_l128_int16_nashm.hbm` | BPU | text encoder + duration predictor, 128 tokens → `mu_x`, `logw` |
| durations, length regulation | CPU | `ceil(exp(logw)) * length_scale` frames per token; repeat `mu_x` columns into `mu_y` |
| `matcha_zh_en_dec_t512_int16_nashm.hbm` | BPU | U-Net decoder, 512 frames (8.2 s), **three Euler flow-matching steps compiled in** → mel |

| build | BCDL consumes | on-board |
|---|---|---|
| `matcha_zh_en_enc_l128_int16_nashm.hbm` | `x_emb` [1,192,128] + `x_mask` [1,1,128] → `mu_x` [1,80,128], `logw` [1,1,128] | **1.2 ms** |
| `matcha_zh_en_dec_t512_int16_nashm.hbm` | `mu_y`, `y_mask`, `z` [1,80,512] → `mel` [1,80,512] | **8.2 ms** |

With Vocos on the BPU too, a sentence takes **26 ms end to end** on average
(real-time factor 0.007–0.009). The runtime also needs
`matcha_zh_en_embedding.npy` (built by `export.py`) and `matcha_zh_en_tokens.txt`
/ `matcha_zh_en_lexicon.txt` — the `tokens.txt` and `lexicon.txt` of the
sherpa-onnx packaging of this model (`matcha-icefall-zh-en`), unchanged.

> [!NOTE]
> **Redistributable per the model card, with one thing worth checking.** The
> card declares Apache-2.0; the training data is not disclosed. See
> [Licence](#licence).

## Trap 1: padded to a fixed length, the stock model is a different model

The BPU needs one fixed shape; Matcha runs each sentence at its own length
(rounded up to a multiple of 4, because the U-Net halves time twice). Padding
the decoder to 512 frames changes its output in two places:

- **GroupNorm normalises over the whole time axis, padding included.** The
  conv outputs in the padding are not zero — bias, plus spill from the last
  valid frames — so the statistics depend on how much padding there is.
- **The attention mask is a 0/1 float that `scaled_dot_product_attention`
  *adds* to the scores.** Valid keys get +1, padded keys +0: padding is
  down-weighted, not excluded. At the model's own length there are at most 3
  padded frames; at 512 there are hundreds.

**Neither depends on what the padding contains — only on how much of it there
is.** So the obvious check, "is the output sensitive to the padded values?",
passes. Measured against an unpadded run of the untouched model, same noise, on
the 29 sentences of the first 20 test texts:

| | max \|Δ\| log-mel |
|---|---|
| stock decoder, padded to 512 | **9.96** |
| exported graph (masked GroupNorm + attention bias) | **8.5e-5** |

`export.py` replaces GroupNorm with one whose statistics cover exactly the
frames the unpadded model has (`ceil4(F)`), and turns the mask into a bias that
keeps the +1/+0 over those frames and adds −100 past them. It runs both rows of
that table every time it exports.

## Trap 2: `repeat_interleave` twice in one graph traces wrong

The two rewrites mark frames with a mask pooled and upsampled per U-Net level.
Written with `repeat_interleave` twice in one graph, `torch.onnx` tracing froze
the attention's `Reshape` at a sequence length of 2. `expand` + `reshape` traces
correctly. If you touch the mask code, rerun the float check — the eager model
is fine either way, only the exported graph breaks.

## Trap 3: token ids cannot be a graph input

Exported as published, the encoder takes token ids. Fed as a float input, the
compiler quantises them like any activation — it chose a threshold of **127 for
a 2190-entry vocabulary** — and the duration output `logw` came out at **cosine
0.83**. The fix is to take the lookup out of the graph: the encoder takes
embeddings, already scaled by √192, and the table ships beside it as
`matcha_zh_en_embedding.npy` ([2190, 192] float32, 1.7 MB). Encoder outputs then
calibrate to 0.999995 / 0.999997.

## int8 compiles, and the decoder is unintelligible

Scored by round-tripping 60 sentences through an ASR model (FireRedASR2) and
measuring the character error rate:

| build | CER | BPU |
|---|---|---|
| float, CPU (sherpa-onnx) | 1.61% | — |
| **int16 encoder + int16 decoder (shipped)** | **1.73%** | 1.2 + 8.2 ms |
| int8 encoder | 2.85% | 0.85 ms |
| int8 decoder | **99.4%** | 4.8 ms |

The int8 decoder squashes the mel peaks from ~5.2 to ~3.0; 1378 of its 2052
nodes sit under 0.95 per-layer cosine, with attention and feed-forward outputs
collapsing to zero or negative. Three Euler steps feed the estimator its own
output, so the error compounds — mixed precision would have to keep most of it
at int16. **Two automatic speech-quality predictors (UTMOS, SQUIM) still scored
that build well.** They judge whether audio sounds like speech, not whether it
says the right thing; an intelligibility metric is the one that catches this.

The int8 encoder's extra errors are all swallowed or doubled characters:
durations are `ceil(exp(logw))`, so a small `logw` error moves a whole frame.

**Even at int16, durations do not match float frame for frame**, and that is
fine. On 19 test sentences (147 tokens) on the board, 30 token durations differ
from the float encoder's, in two groups:

- **tokens whose float duration sits exactly on an integer.** Many land on
  2.00 (also 6.00, 12.0). A `logw` error of 0.002 is enough to make `ceil` go
  one frame either way.
- **long tokens: pauses and punctuation, 8–58 frames.** Here a 1–2% `logw`
  error (median |Δlogw| over all tokens is 0.005) is a frame or two. The two
  biggest were +2 frames on a 16-frame token and −3 on a 55-frame one.

Each sentence's total length moved by at most 2 frames. The round-trip CER
above is the evidence that this is inaudible. Compare mels per token segment,
not frame for frame; a frame-aligned diff makes an accurate build look broken.

## Calibration

`calib.py` builds both sets in one pass from the same 61 sentences, so the five
positional directories cannot drift apart:

- **The decoder is calibrated on the encoder's real output**: the float
  encoder's `mu_x`, length-regulated on the CPU exactly as at run time, padded
  to 512 with `y_mask` marking the valid frames.
- **`z` is N(0, 1)**, the run-time noise at `noise_scale` 1.0 (sherpa-onnx's
  and BCDL's default), seeded.
- **The token ids come from sherpa-onnx's own front end** (`frontend_ids.py`),
  not a hand-written one. Its sentence splitting, number normalisation and
  punctuation tokens are what the model is fed; a character-by-character
  front end without punctuation scored 4.2% CER against 2.0% in a comparison
  on the same sentences.

`calib_texts.txt` is 40 short robot replies — disjoint from the 60 sentences the
build was scored on. hb_compile tries batch 8 on both graphs, fails on a
batch-1 `Reshape`, and drops to batch 1 (*"Reset batch_size=1"*), so every
sample counts and `repeat` is not needed.

## Running it

```bash
# upstream: the matcha-tts 0.0.7.2 sdist from PyPI, extracted, NOT installed
tar xzf matcha-tts-0.0.7.2.tar.gz
pip install --target deps --no-deps einops conformer==0.3.2 diffusers==0.25.0 pillow
# + torch, onnxruntime, sherpa-onnx in the environment

python frontend_ids.py --sherpa-dir /path/to/matcha-icefall-zh-en \
    --vocoder /path/to/vocos-16khz-univ.onnx --texts calib_texts.txt --out calib_ids.json
python export.py --ckpt /path/to/pytorch_model_sherpa.bin \
    --matcha-src matcha-tts-0.0.7.2 --deps deps --ids calib_ids.json \
    --reference-onnx /path/to/matcha-icefall-zh-en/model-steps-3.onnx
onnxslim matcha_enc_L128.onnx matcha_enc_L128_slim.onnx
onnxslim matcha_dec_T512.onnx matcha_dec_T512_slim.onnx
python calib.py --ids calib_ids.json
./compile.sh --config config_encoder.yaml --gpu 1
./compile.sh --config config_decoder.yaml --gpu 1
```

`export.py` shims two things in memory rather than patching the source:
`matcha/utils/__init__.py` imports lightning and hydra for training utilities
the inference modules never use, and diffusers 0.25.0 imports
`huggingface_hub.cached_download`, which current releases removed. Use
`pytorch_model_sherpa.bin`, not `pytorch_model.bin` — the ModelScope repo has
both, and the `_sherpa` one is what `model-steps-3.onnx` was exported from
(`--reference-onnx` checks it: max |Δ| 5.7e-4 at noise 0).

The decoder is the slow compile — about 20 minutes calibrating and 50 compiling.

## Verified: the recipe rebuilds the shipped build

This recipe was re-run end to end in a fresh directory: the upstream checkpoint,
the PyPI sdist (byte-identical to the one the shipped build used), the in-memory
shims, freshly installed dependencies.

| step | result |
|---|---|
| `frontend_ids.py` | token ids identical to the original build's, for all 61 calibration and 94 test sentences |
| `export.py` + `onnxslim` | both graphs: max \|Δ\| **0.0** against the shipped build's ONNX on the 94 test sentences |
| embedding table | byte-identical to the shipped `.npy` |
| float check | 8.54e-5 after the rewrites, 9.96 without; rebuild vs `model-steps-3.onnx` 5.7e-4 |
| `calib.py` | all five directories value-identical to the shipped calibration set |
| encoder compile | same thresholds (`quant_info.json` byte-identical), output cosine 0.999995 / 0.999997 |
| decoder compile | same 2098 thresholds (202 nodes renamed by an earlier export run), output cosine 0.999759 |
| **on the board** | both rebuilt `.hbm` against the shipped ones on 19 sentences: max \|Δ\| **0**. Against float ONNX: encoder ≥ 0.99989, decoder mel ≥ 0.99928 |

The `.hbm` files are not byte-identical (the decoder is 19 KB smaller), but
their outputs are, so the CER and latency figures above, measured on the shipped
builds, hold for the rebuild too.

## What is not here

One encoder length (128 tokens) and one decoder length (512 frames, 8.2 s). The
front end splits text into sentences, so 128 tokens is generous; a sentence
whose durations sum past 512 frames needs a longer decoder build, and hbm size
and latency scale with the frame count. The `model-steps-{2,4,5,6}` variants
were not built — the step count is compiled into the decoder graph.

## Licence

The ModelScope model card declares **Apache-2.0**; the Matcha-TTS code is
**MIT**.

**The training data is not disclosed.** The card says the model is trained on
mixed Chinese–English data, fine-tuned from the same author's earlier zh-en
model, which was trained with icefall's `baker_zh` recipe. Whether that recipe's
Baker corpus (BZNSYP, **non-commercial**) is in the data is not stated. The
card's licence makes the weights redistributable on its face; check the data
question before a commercial build.

`tokens.txt` and `lexicon.txt` come from the sherpa-onnx packaging of the same
model.
