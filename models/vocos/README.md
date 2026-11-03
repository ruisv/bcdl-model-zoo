# Vocos 16 kHz — the vocoder for Matcha-TTS zh-en

Mel spectrogram in, waveform out. Vocos predicts a magnitude and a phase for
every STFT bin and leaves the waveform to an inverse STFT, so the network is a
short stack of 1-D ConvNeXt blocks with no upsampling — an easy graph for the
BPU, once one thing is moved off it. Upstream: `vocos-16khz-univ.onnx`, published
alongside Matcha-TTS zh-en on
[ModelScope](https://modelscope.cn/models/dengcunqin/matcha_tts_zh_en_20251010)
and mirrored in the [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx)
`vocoder-models` release (md5 `2b43a6235c158cb70480334c48c1c8b9`).

| build | BCDL consumes | on-board |
|---|---|---|
| `vocos_16k_t256_int16_nashm.hbm` | log-magnitude + phase, `[1, 513, 256]` each; exp / cos / sin and the iSTFT on the CPU | **1.3 ms BPU per 256-frame window**; real-time factor 0.0026 with the iSTFT |

The compiled model is **17.4 MB** for 80 mels × 256 frames (4.1 s of audio).
Longer input is chunked by the runtime.

## The trap: the phase, not the network

The published graph ends in `clip(exp(log_mag), 100)`, `cos(phase)` and
`sin(phase)`. Compiled as published, those last two are what breaks: the
compiler lowers `cos`/`sin` to a **quantised range reduction**, and the raw
phase they are applied to spans about **±42 rad**. The phase output comes back
at cosine **0.58** against float at int8 and **0.97–0.99** at int16. Neither is
silent — 0.97 on a phase is audible.

So `export.py` **cuts the graph at the head's linear output**. The BPU returns
log-magnitude and raw phase; `exp`, the clip at 100, `cos`/`sin` and the iSTFT
run on the CPU (BCDL's `VocosVocoder`). Raw phase is a plain linear output at
that point, and int16 resolves ±42 rad to about 1e-3 rad. What the cut costs:
nothing on the BPU, 2.5 ms of `exp`/`cos`/`sin` on the CPU per 3 s of audio.

The cut graph followed by the CPU tail reproduces the published graph to float
round-off — max |diff| 4e-4 on magnitudes up to 100, 2e-4 on cos/sin, measured by
`export.py --check` on 94 sentences.

## int16, and why mixed precision is not the compromise

Scored against the float model on 60 sentences, over the time–frequency cells
**within 40 dB of each utterance's peak** (the ones you can hear): the
log-spectral distance, and the share of those cells more than 3 dB off.

| build | log-spectral distance | cells off by > 3 dB | round-trip CER | BPU per window |
|---|---|---|---|---|
| float, chunked as on the board | 0.01 dB | 0.0% | 1.98% | — |
| int8 | 5.38 dB | 49.8–50.3% | 2.10% | 0.76 ms |
| int16 on the last ConvNeXt block + final norm + head only | 4.54 dB | 40.5% | — | 0.81 ms |
| **int16 (shipped)** | **0.54 dB** | **0.3–0.4%** | 2.48% | 1.3 ms |

**CER does not see a noise floor.** The int8 build is still intelligible — the
recogniser barely notices — but half of what you hear is more than 3 dB off,
which is a noise floor under every word. Round-trip CER is the right gate for an
acoustic model; for a vocoder it is nearly blind. (The int16 build's extra CER
is one homophone, 它 → 他, in 60 sentences.)

**The error is not in the tail.** The obvious mixed-precision guess — int16 on
the last block, the final norm and the head, where the outputs are formed —
recovers almost nothing (4.54 dB against int8's 5.38). The error accumulates
over all eight blocks, so there is no small set of layers to promote. And there
is little to win anyway: the BPU is not where the time goes. Per 3 s of audio
on the board it is 1.3 ms BPU, 2.5 ms `exp`/`cos`/`sin` and 4.1 ms iSTFT, so
int8 would save about 0.5 ms of 7.9.

Size is no argument either: int8 is 15.8 MB against int16's 17.4 MB. The `.hbm`
is mostly instruction stream, not weights.

## Chunking is exact, so calibrate on the chunks

The model is compiled for a fixed 256-frame window. The runtime cuts longer
input into windows that **overlap by 32 frames on each side** and keeps each
window's middle 192. Every output frame depends on at most **27** input frames
either way (nine kernel-7 convolutions), so with 32 of context the seams are
the same computation as a whole-utterance pass — on the board, bit-identical.
In float, `export.py --check` measures chunked against whole at max |diff|
4e-4 on interior frames (round-off) and 107 dB SNR on the audio away from the
ends.

The only approximation is at the two ends of the utterance, where frames past
the edge are filled with **−16** instead of the network's own zero padding.
Silence in these mels sits near −22; −16 measured closest to the
whole-utterance output at the edges (audio SNR over the whole utterance: min 46,
median 56 dB).

`calib.py` calibrates on **exactly these windows**: 40 short robot-reply
sentences → 61 sentences after sherpa-onnx splits them → 65 windows, edges and
padding included.

## Calibration mels: the float Matcha, through sherpa-onnx's front end

The vocoder should be calibrated on what feeds it in deployment: mels from the
Matcha-TTS zh-en acoustic model. `calib.py` gets them from the **float** Matcha
as sherpa-onnx packages it (`matcha-icefall-zh-en`), so it needs `sherpa_onnx`
and `onnxruntime` and nothing compiled. Two details decide whether those mels
are the right ones:

- **sherpa-onnx's own front end, not a hand-written one.** A front end that maps
  characters through the lexicon and drops punctuation produces mels with no
  pauses, and scored 4.2% round-trip CER against 2.0% for the real front end.
  sherpa-onnx does not expose its token ids, so `calib.py` reads them from its
  debug log, one sentence at a time, and runs the acoustic model on them itself.
  That log line is not an API; the script fails loudly if it disappears.
- **noise_scale 1.0, length_scale 1.0**, sherpa-onnx's defaults for this model,
  so the mels carry the variance the runtime will see.

Matcha's decoder draws its starting noise with an **unseeded**
`RandomNormalLike`, so the published graph never produces the same mel twice.
`calib.py --seed` (default 0) seeds an in-memory copy, which makes the recipe
repeatable (a seeded rerun is bit-identical). The shipped build was calibrated on
an unseeded draw of the same 40 sentences: same sentence split, same frame
counts, mels that differ by the noise (mean |diff| 0.33 log-mel), and the same
value range (−22.9 to 5.6).

## Running it

On the convert host, in an env with `onnx`, `onnxslim`, `onnxruntime` and
`sherpa_onnx` (`torch` optional, only to cross-check the iSTFT):

```bash
# upstream: vocos-16khz-univ.onnx (sherpa-onnx vocoder-models release) and the
# matcha-icefall-zh-en directory (sherpa-onnx tts-models release)
python export.py --onnx vocos-16khz-univ.onnx --out vocos16k_t256_head.onnx
python calib.py --config config.yaml --matcha /path/to/matcha-icefall-zh-en \
                --vocoder vocos-16khz-univ.onnx --save-mels mels_calib
python export.py --onnx vocos-16khz-univ.onnx --check mels_calib   # optional
./compile.sh --gpu 0          # an Ampere-or-newer card; ~10 min
```

`export.py --check DIR` takes any directory of `[80, frames]` mel `.npy` files;
use held-out sentences rather than the calibration ones for a real check.

## Rebuilding it

This recipe was re-run end to end in a fresh directory. `export.py` produces a
head graph **numerically identical** to the one the shipped `.hbm` was compiled
from (max |diff| 0 on every window of 94 held-out sentences). The calibration
set is a seeded redraw (above), so the rebuild is not byte-identical: 17.6 MB
against 17.4 MB, same 1.02 ms compiler latency estimate. Scored on the host —
the quantised `.bc` in place of the board `.hbm`, 94 held-out sentences, the
audible-cell metric above:

| | log-spectral distance | cells off by > 3 dB | hb_compile output cosine (log_mag / phase) |
|---|---|---|---|
| shipped build | 0.54 dB | 0.37% | 0.99999 / 0.99989 |
| this recipe's rebuild | 0.53 dB | 0.35% | 0.999991 / 0.99985 |

## What is not here

The rebuild was checked on the host, not taken to the board; the on-board
numbers are the shipped build's. No window length other than 256 frames was
compiled — size and latency scale with the window, and the runtime reads the
length from the model. The 22 kHz and 24 kHz Vocos models are different
checkpoints and were not tried.

Calibration does not depend on a compiled Matcha: the vocoder is calibrated on
the float acoustic model's mels. In deployment it is fed by the int16 BPU build
of Matcha instead; the shipped vocoder was scored in that full chain on the
board (round-trip CER 1.73% on 60 sentences, against 1.61% for float
sherpa-onnx), not re-calibrated on its output.

## Licence

`vocos-16khz-univ` is published with Matcha-TTS zh-en by the same author, and
the **model card declares Apache-2.0**. The Vocos architecture and code are
**MIT**. Both allow redistribution, including commercially.

**The one thing worth checking is the training data, which is not disclosed.**
Neither the vocoder nor the acoustic model it is paired with says what it was
trained on — the Matcha card says only "mixed Chinese-English data". That is a
gap, not a known restriction; look at it before a commercial build.
