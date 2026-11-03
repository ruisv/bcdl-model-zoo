# sherpa-onnx KWS zipformer2 (zh-en, 3M) — open-vocabulary wake words, 16 kHz

A streaming transducer: 160 ms of audio in per call, four 40 ms encoder frames
out, and a keyword search that matches them against wake words given as token
lines (`n ǐ h ǎo x iǎo ch ē @你好小车`) — a new wake word is a line of text, not
a new model. Upstream: the k2-fsa sherpa-onnx release archive
[`sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20.tar.bz2`](https://github.com/k2-fsa/sherpa-onnx/releases/download/kws-models/sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20.tar.bz2)
(GitHub release tag `kws-models`), files `*-epoch-13-avg-2-chunk-8-left-64.onnx`.

Only the encoder goes to the BPU. The decoder, joiner and keyword search run on
the CPU — per hypothesis they are an embedding lookup, a tiny grouped conv and
three small matmuls, not worth a round trip.

| artefact | what it is | on-board |
|---|---|---|
| `kws_zipformer_zh_en_3m_c16_int16_nashm.hbm` | the streaming encoder, one 160 ms chunk per call | **2.0 ms BPU per chunk**; whole spotter at real-time factor **0.023** |
| `kws_zipformer_zh_en_3m_decoder_joiner.npy` | decoder + joiner weights as one [1166, 321] float32 table | CPU |
| `kws_zipformer_zh_en_3m_tokens.txt` | the archive's `tokens.txt`, 263 tokens | — |

All three come out of this directory; BCDL's `KeywordSpotter` takes them in that
order.

> [!IMPORTANT]
> **Recipe only — no binary.** sherpa-onnx is Apache-2.0, but the model is
> published as a release archive with no model-specific licence, and nothing
> says what the zh-en model was trained on. See [Licence](#licence).

## What goes wrong

One of these builds a model that compiles, loads and runs at full speed and is
wrong; two stop the compiler; two are runtime mistakes that produce confident
output from the right model. And one number looks like a failure and is not.

**Swoosh's softplus, quantised as written: encoder cosine -0.21.** The encoder's
Swoosh activations compute `log(1 + exp(s))`. `exp(s)` spans many orders of
magnitude, and no fixed-point scale holds that: the `Exp` quantised to cosine 0
and took the whole `encoder_out` to **-0.21** against float. Written as
`relu(s) + log(1 + exp(-|s|))` — the same function — `exp` stays in (0, 1] and
`log` in [0, log 2]. This is the rewrite that made int16 mean anything.

**Depthwise convolutions whose padding is longer than their input: "invalid
storage".** Ten of the conv modules have kernel 15 padded by 7, and deep in the
encoder the time axis is only 1–4 frames long. The code generator refuses with
"invalid storage" and does not say which node; it was found by cutting the graph
after node K, compiling each cut with `calibration_type: "skip"` (no calibration
data needed), and bisecting K. On L frames only the 2L−1 centre taps
ever meet data — the rest multiply padding zeros — so `export.py` crops the
kernel to those taps and pads by L−1: the same convolution.

**hb_compile segfaults during optimisation.** One of hmct's optimiser checks
builds throwaway onnxruntime sessions, and onnxruntime crashes on one of this
graph's subgraphs: the log ends at "Start to optimize the onnx model." and the
container exits with 139. It still does on `export.py`'s final graph (re-checked
for this recipe). `compile.sh` therefore runs `hbc.py`, which turns that one
check off; see its docstring.

**Features scaled to int16 range.** sherpa-onnx feeds kaldi fbank with samples
in **[-1, 1]**, not multiplied by 32768, plus a Povey window, `high_freq = -400`
and `snip_edges = false`. A port that scaled by 32768 agreed with sherpa-onnx on
37 of 87 clips — and nothing about its output looked broken.

**The cache counter at the wrong rate.** The upstream encoder takes
`processed_lens`, the number of frames already in its attention cache, and the
count is at the **embedding** rate: +8 per 160 ms chunk, not the +4 encoder
frames the chunk produces. Counting +4 still runs and drives `encoder_out` off by
up to 34. In this build the counter has become `pad_mask` (below), and the same
rule applies to whoever computes it.

**Whole-array cosine 0.915 is not a failure.** Against float, the on-board
`encoder_out` for an AISHELL-1 utterance has per-frame cosine **0.995 mean,
0.973 worst**; over the whole array it is 0.915, and the compiler reports 0.959.
The gap is magnitude: some frames come out smaller rather than pointing
elsewhere, and the joiner's `tanh` and the softmax after it do not notice. The
task numbers below are the gate — recall matching float, no false wake-ups —
not a cosine.

## Measured on the board (from BCDL)

Wake words 你好小车 and 小车小车, 202 clips each clean and with MUSAN noise
mixed in (9 synthetic voices); false wake-ups on AISHELL-1 speech, spoken
commands and 2.3 h of noisy sessions; 350 confusable phrases (你好小菜,
校车校车, ...). Default threshold 0.25, boost 1.0:

| | wake words found, clean / noisy | false wake-ups | confusable phrases that fire, clean / noisy |
|---|---|---|---|
| float (sherpa-onnx) | 96.5% / 87.1% | 0 in 4.1 h | 5.4% / 4.6% |
| **int16, BPU** | **97.5% / 89.1%** | **0 in 4.4 h** | **5.7% / 4.0%** |

The encoder takes 2.0 ms of BPU time per 160 ms chunk (the compiler estimates
1.19 ms); with the CPU decoding, the spotter runs at a real-time factor of 0.023.

## Exporting it: four rewrites on upstream's ONNX

There is no torch step: upstream publishes the streaming encoder as ONNX, and
`export.py` edits that graph. Use the `chunk-8-left-64` files. The 8 counts
frames at the embedding rate; in fbank frames it is a 16-frame, 160 ms chunk —
the graph's own `decode_chunk_len: 16`, and the `c16` in the build name. The
archive's `chunk-16` files are a different (320 ms) model.

Step 0 is onnxsim with the batch dimension fixed to 1 (1212 nodes). Then:

| # | upstream | becomes | why |
|---|---|---|---|
| 1 | int64 `processed_lens` → boolean mask built in-graph, used by 18 `Where` | float input `pad_mask` [1, 72]; each `Where(m, c, x)` → `x·(1−m) + c·m` | the BPU takes no int64 and no boolean select. Exact for a 0/1 mask |
| 2 | Swoosh `log(1 + exp(s))` (34) and its inf guard (30) | `relu(s) + log(1 + exp(−|s|))` | the quantisation failure above |
| 3 | 6 relative-position `GatherElements` with constant indices, each wrapped in a Reshape pair | per-query `Slice`s of the 4-D score tensor, concatenated; the Reshapes go | each index row is a contiguous run whose offset depends only on the query position |
| 4 | 10 depthwise `Conv` k15 p7 on 1–4 frames | kernel cropped to the 2L−1 centre taps, pad L−1 | "invalid storage" above |

A final onnxsim pass leaves 1321 nodes with 39 inputs and 38 outputs. Every
rewrite is exact in real arithmetic; in float the rewritten encoder is within
**1.5e-4** of upstream, and the detections are identical (see
[Verification](#verification)).

The decoder and joiner are not exported as graphs: their weights are packed
into one table, `[2V + 2D, D + 1]` with V = 263 tokens and D = 320, rows
embedding / grouped-conv taps / decoder projection / joiner output. The exact
layout is in `export.py` (`decoder_joiner_table`) and BCDL checks the shape
against the token list and the encoder's output width.

## What the runtime feeds it

The calibration data has to look like this, so it is the contract:

- **Features**: 80-bin kaldi fbank, samples in [-1, 1], Povey window, low_freq
  20, high_freq -400, dither 0, `snip_edges = false`. BCDL reproduces
  `snip_edges = false` by prepending a 120-sample reflection of the stream's
  start, which gives the same frames.
- **`x`** [1, 29, 80]: 29 fbank frames per call, the window advancing by 16.
- **`pad_mask`** [1, 72]: 64 cache slots then the 8 frames of this chunk, 1 for
  a slot not yet filled: `[1] * (64 − q) + [0] * (q + 8)`, `q = min(processed, 64)`,
  `processed` += 8 per call and reset with the stream.
- **37 caches** in, a `new_<name>` for each out, fed back on the next call;
  zeros after a reset.
- **`encoder_out`** [1, 4, 320] out: four 40 ms frames for the search.

`kws_ref.py` is this pipeline in Python, including a port of sherpa-onnx's
keyword search (Aho-Corasick context graph, strict mode, modified beam search
with 4 paths); BCDL's C++ follows the same code line by line.

## Calibration: real streaming states

39 inputs, and 37 of them are caches the previous call wrote. **Their
distribution exists only inside a stream**, so `calib.py` runs the float graph
over real audio exactly as the runtime does and records every input of a call:
every chunk of the first 8 of each clip (where `pad_mask` is still partly 1 —
the state after every reset) and then every 5th chunk, with clips cut at 20 s.
Calibrating on zeros or random caches was not tried; it would be calibrating
inputs the runtime never produces.

The shipped build used 39 clips — 12 wake-word clips clean and 12 noisy, 8
confusable phrases, 6 noisy-speech sessions (AISHELL-1 speech with MUSAN noise
at 10 and 0 dB SNR) and one noise-only recording — which gives 576 states, of
which every 4th was kept: **144**. Any 16 kHz audio works; what matters is that
it sounds like the deployment. Public sources: AISHELL-1
([OpenSLR 33](https://www.openslr.org/33/)) and MUSAN
([OpenSLR 17](https://www.openslr.org/17/)); record or synthesise your own wake
word.

**Calibration is the slow part:** hb_compile spends 15–33 s per state on this
graph. 144 states took 50–62 minutes and the whole compile 75–95 on the two builds
so far; all 576 would take hours.

## Running it

```bash
pip install onnx onnxsim onnxruntime kaldi-native-fbank soundfile pyyaml sherpa-onnx

wget https://github.com/k2-fsa/sherpa-onnx/releases/download/kws-models/sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20.tar.bz2
tar xf sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20.tar.bz2
M=sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20

python export.py --model-dir $M          # encoder ONNX + decoder_joiner.npy + tokens.txt

# keywords.txt: one wake word per line, e.g.  n ǐ h ǎo x iǎo ch ē @你好小车
python check_float.py --model-dir $M --keywords keywords.txt \
    --audio /path/to/wakeword_clips --audio /path/to/other_speech

python calib.py --config config.yaml \
    --audio /path/to/wakeword_clips:24 --audio /path/to/confusables:8 \
    --audio /path/to/noisy_speech:6 --audio /path/to/noise:1
./compile.sh --gpu 0                     # an Ampere or newer card
```

`export.py` needs onnxsim; the other scripts need onnxruntime,
kaldi-native-fbank, soundfile and pyyaml, and `check_float.py` needs
sherpa-onnx as the reference. The `.hbm` lands in `out/`; ship it with the
`.npy` and `tokens.txt` from this directory.

## Verification

Re-run for this recipe on the convert host, from the upstream archive's ONNX
(OpenExplorer 3.7.0):

| step | result |
|---|---|
| `export.py` against the graph the shipped `.hbm` was compiled from | same 1321 nodes, same operator counts, same 425 initializers; `encoder_out` and all 37 caches **bit-identical** (max difference 0) over 512 streaming chunks of 12 clips |
| decoder/joiner table and `tokens.txt` | **byte-identical** to the shipped files |
| `check_float.py`, 87 clips (26 wake-word clean, 26 noisy, 35 confusables) | table vs upstream decoder 7.6e-6, joiner 3.1e-5; encoder vs upstream **1.51e-4**; detections **87/87 identical to sherpa-onnx** (which fired on 50) |
| `calib.py` on the shipped build's 39 clips, in the same order | 576 states → 144, equal to the shipped calibration set within 7.3e-5 (that set came from an earlier revision of the rewrites, equal in float but not bit for bit) |
| plain `hb_compile`, `calibration_type: "skip"` | dies as described above (exit 139) — `hbc.py` is still needed |
| `compile.sh`, all 144 states, all-int16 | **compiled**: 62 min of calibration, 95 min in all; 13,288,088 bytes against the shipped 13,282,784; compiler estimate 1.19 ms, the same; compiler-reported `encoder_out` cosine 0.949 against the shipped build's 0.959 |
| both `.bc` streamed on the host over the same 87 clips, caches fed back | per-frame cosine against float **0.9926** (rebuild) / **0.9929** (shipped); detections identical to float on **86/87** each — the shipped build misses one noisy wake word that float finds, the rebuild finds one that float misses |

The rebuild is not bit-identical to the shipped `.hbm` — the 7e-5 difference in
calibration inputs moves some thresholds — and is equivalent to it on everything
measured here. **The rebuilt `.hbm` was not run on the board**; the board
numbers above are the shipped build's.

One trap when streaming a `.bc` on the host: the arrays the hbdk call returns
are overwritten by the next call. Copy every output (the caches especially)
before feeding them back, or the comparison collapses to a per-frame cosine of
about 0.56 and looks like a broken build.

## What is not here

- **The `chunk-16` (320 ms) model** from the same archive was not built. The
  rewrites find their targets structurally, but nothing beyond `chunk-8` was
  run.
- **int8** was not built; int16 already matched float on the task metrics at
  2.0 ms per chunk.
- **Layer B** (board `.hbm` against host `.bc`) was not recorded for the shipped
  build; it was judged on the task metrics above and the per-frame cosine.

## Licence

sherpa-onnx — the code, the export scripts and the keyword search this build
reproduces — is **Apache-2.0**. The model itself is published by the k2-fsa
project as a GitHub release asset (tag `kws-models`) **with no model-specific
licence**.

**The data chain is unknown.** The release does not say what the zh-en model was
trained on. Its Chinese-only sibling is trained on WenetSpeech; whether that, or
which English corpus, is in the zh-en model is not stated, so there is no data
licence to check the weights against.

So this sits with the other **unresolved** rows: **recipe only, do not
redistribute the `.hbm` yet.** Silence is not permission. The recipe is the part
that matters: the same rewrites should carry over to a zipformer2 streaming KWS
model you train on data you can account for, though only this one has been run.

The calibration audio (AISHELL-1, Apache-2.0; MUSAN, CC BY 4.0) only sets
quantisation thresholds and is not distributed with the `.hbm`.
