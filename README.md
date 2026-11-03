# bcdl-model-zoo

ONNX → `.hbm` conversion recipes for the D-Robotics RDK **S100 / S100P / S600**
BPU, for the models that [BCDL](https://github.com/ruisv/bcdl) can decode.

BCDL is the on-board runtime: it loads a compiled `.hbm` and runs inference,
pre/post-processing, codecs and pipelines. **It does not convert models.** This
repo is the other half — it runs on an x86 host with a GPU and the D-Robotics
OpenExplorer toolchain, and produces the `.hbm` files BCDL consumes.

## What this repo ships

**Recipes**: the ONNX export (including graph surgery), the calibration data
generator, the `hb_compile` config, and the acceptance thresholds — enough to
reproduce any model from its upstream checkpoint. The compiled `.hbm` files
themselves are large and not committed (`.gitignore` excludes `*.hbm`); the
recipe rebuilds them.

**The model weights and the `.hbm` compiled from them follow their own upstream
licences.** A compiled `.hbm` is a derivative of upstream weights, so check the
model's licence before redistributing or shipping commercially — some are
copyleft (AGPL) and some are non-commercial.

**The licence on the repository is usually not the licence on the weights**, and
that gap is where this goes wrong. A checkpoint inherits restrictions the code
licence never mentions — from the **weights it was fine-tuned from** (ViTPose
starts at MAE, which is CC-BY-NC) and from the **dataset it was trained on**
(Cityscapes bars commercial use of models trained on it, explicitly). The table
below is grouped by what you may actually do with a compiled `.hbm`, not by the
badge on the repo. Rows marked *unresolved* are ones no upstream source answers:
silence is not permission.

| what you may do with the `.hbm` | models |
|---|---|
| **Redistribute, including commercially** | PP-OCRv6/v5 det·cls·rec (Apache-2.0 on code and every model card) · XFeat (weights committed in the Apache-2.0 repo) · Real-ESRGAN Compact (BSD-3, weights released by the copyright holder) · LingBot-Depth (Apache-2.0 on code and weights) · Depth-Anything-V2 **Small** · Silero VAD v6.2.3 (MIT on code and the published weights) |
| **Redistribute, but non-commercial** | GLASS — MIT code and weights, but the checkpoint is trained on **MVTec AD (CC BY-NC-SA 4.0)**, so a redistributed build carries the same terms including **share-alike**. Does not reach a deployment retrained on your own parts · ViTPose-S — Apache-2.0 code, but the checkpoint is fine-tuned from **MAE (CC-BY-NC-4.0)** and trained on COCO-WholeBody · PIDNet-S — MIT code, but **Cityscapes** permits distributing trained models and bars commercial use of them in the same breath · YOLOP — MIT code and in-repo weights, but **BDD100K** limits commercial use to member organizations |
| **Redistributable, with one thing worth checking** | Matcha-TTS zh-en — the model card declares Apache-2.0, but the training data is not disclosed and the author's earlier zh-en model was trained with a recipe built on the Baker corpus (BZNSYP, non-commercial); whether it is in this one is not stated · Vocos 16 kHz — Apache-2.0 card, training data undisclosed · LAS2 — MIT on code and weights, but the release was distilled from FoundationStereo, whose NVIDIA licence is research-only. That is upstream's compliance question rather than a term passed down · SPAN — Apache-2.0 by project statement, but the checkpoints ship from a separate file host with no terms attached to the artefact |
| **Recipe only — copyleft** | YOLO26 / YOLOv8 / YOLOE (Ultralytics) — **AGPL-3.0**; commercial use needs Ultralytics' Enterprise licence |
| **Recipe only — non-commercial weights** | SCRFD / ArcFace (insightface) · Depth-Anything-V2 **Base and up**; for a commercial build see the alternatives noted per model · PointPillars — Apache-2.0 code (OpenPCDet) but the checkpoint is **KITTI**-trained, and KITTI is CC BY-NC-SA |
| **Unresolved — do not redistribute yet** | sherpa-onnx KWS zipformer zh-en 3M — sherpa-onnx is Apache-2.0, but the model ships as a release archive with no model-specific licence and nothing says what the zh-en model was trained on (its Chinese-only sibling is WenetSpeech) · NeuFlow v2 — Apache-2.0 code with the weights committed in the same repo, but trained on **FlyingThings3D**, whose terms are research-only and **silent about models trained on the data**. Silence is not permission · EdgeSAM (terms not reviewed) · OSNet — MIT code, but every ReID-useful checkpoint is trained on MSMT17 / Market-1501 under academic-only agreements and no upstream source says whether that reaches the weights |

## Prebuilt `.hbm` on Hugging Face

The recipes above are the reproducible half. For the models whose **weights** may
be redistributed, the compiled artefact is published too, so a user with a board
and no x86 host can skip all of this:

| repo | recipe here | licence |
|---|---|---|
| [`ruisv/bcdl-ppocr-v6`](https://huggingface.co/ruisv/bcdl-ppocr-v6) | [`ppocr_v6`](models/ppocr_v6/) | Apache-2.0 |
| [`ruisv/bcdl-ppocr-v5`](https://huggingface.co/ruisv/bcdl-ppocr-v5) | [`ppocr_v5`](models/ppocr_v5/) | Apache-2.0 |
| [`ruisv/bcdl-las2-stereo`](https://huggingface.co/ruisv/bcdl-las2-stereo) | [`las2`](models/las2/) | MIT |
| [`ruisv/bcdl-lingbot-depth`](https://huggingface.co/ruisv/bcdl-lingbot-depth) | [`lingbot_depth`](models/lingbot_depth/) | Apache-2.0 |
| [`ruisv/bcdl-superres`](https://huggingface.co/ruisv/bcdl-superres) | [`superres`](models/superres/) + [`span`](models/span/) | BSD-3 / Apache-2.0 |
| [`ruisv/bcdl-xfeat`](https://huggingface.co/ruisv/bcdl-xfeat) | [`xfeat`](models/xfeat/) | Apache-2.0 |
| [`ruisv/bcdl-glass-bottle`](https://huggingface.co/ruisv/bcdl-glass-bottle) | [`glass`](models/glass/) | **CC BY-NC-SA 4.0** |
| [`ruisv/bcdl-vitpose-wholebody`](https://huggingface.co/ruisv/bcdl-vitpose-wholebody) | [`vitpose`](models/vitpose/) | **non-commercial** |
| [`ruisv/bcdl-pidnet`](https://huggingface.co/ruisv/bcdl-pidnet) | [`pidnet`](models/pidnet/) | **non-commercial** |
| [`ruisv/bcdl-yolop`](https://huggingface.co/ruisv/bcdl-yolop) | [`yolop`](models/yolop/) | **non-commercial** |

BCDL's `scripts/fetch_models.sh` pulls these automatically.

**The speech models are not hosted yet** — wake word
([`kws_zipformer`](models/kws_zipformer/)), VAD ([`silero_vad`](models/silero_vad/)),
and the TTS pair ([`matcha_tts`](models/matcha_tts/) + [`vocos`](models/vocos/)).
Build them here, then point BCDL's fetch script at the output directories and
it installs them under the names BCDL expects:

```bash
BCDL_SRC_KWS=/path/to/kws_build BCDL_SRC_VAD=/path/to/vad_build \
BCDL_SRC_MATCHA=/path/to/matcha_build BCDL_SRC_VOCOS=/path/to/vocos_build \
  scripts/fetch_models.sh
```

Each recipe's README lists exactly which files it produces. The speech
recogniser BCDL uses (Paraformer) needs no recipe: it is D-Robotics' own
prebuilt build, and the fetch script downloads it.

**The rows missing from that table are the point of the licence table above**:
the Ultralytics builds are AGPL, the face models and PointPillars carry
non-commercial *weights*, and NeuFlow v2, EdgeSAM and OSNet are unresolved. Those
stay recipe-only — which is exactly what this repo is for.

## Layout

```
common/
  calib_pack.py     # the ONLY way calibration .npy files get written
  verify_cosine.py  # three-layer cosine check (A/B/C, see below)
models/<name>/
  README.md         # what goes wrong on this model, and which build is correct
  export.py         # upstream checkpoint -> ONNX, including graph rewrites
  calib.py          # regenerate calibration data from a public dataset
  config.yaml       # hb_compile configuration
  expected.json     # acceptance thresholds + measured on-board reference
```

## Two rules that exist because they were learned the hard way

**1. Calibration data must be pre-normalised, and only `calib_pack.py` writes
it.** When `cal_data_type` is float32, the compiler's `norm_type` does **not**
apply to the calibration data. Get this wrong and the input thresholds come out
wrong — and the model still compiles without a single warning, then decodes to
garbage. This is not a footnote; it is why PIDNet needed three attempts.

**2. Every model's README documents the *wrong* build, not just the right one.**
The failure mode across nearly all of these models is that the bad build
**compiles cleanly**. A document that only describes the correct path offers no
protection. So each README leads with what the sibling build does to you.

The same reason drives the naming rule inherited from BCDL: **a file's name
matches the model name inside it.** Suffixes like `_cut`, `_v3`, `_aligned`,
`_qat` are *which build this is*, not decoration.

## Verification

A recipe is accepted when `expected.json` is met and measured:

- **A — quantisation fidelity**: `.bc` vs ONNX, cosine > 0.99
- **B — execution consistency**: on-board `.hbm` vs host `.bc`, bit-identical (cosine = 1.0)
- **C — end-to-end**: on-board `.hbm` vs ONNX on real images, cosine > 0.99

**Cosine alone is not sufficient and must not be the only gate where a task
metric exists.** OSNet is the worked example: int8 PTQ produces well-formed unit
vectors that look fine to a cosine check, while Market-1501 Rank-1 collapses
from 85% to 51%. Any model with a measurable downstream metric declares that
metric in `expected.json` too.

## Status

Every converted model in BCDL's catalogue now has a recipe. **A recipe being
present does not mean it has been re-run** — eleven are `complete`, the rest
reconstruct the method faithfully from the original conversion work (`recipe`).
Of the `complete` ones, seven were board-verified end to end from the recipe;
the four speech recipes were re-run from upstream to a compiled model and
checked against the shipped, board-measured builds — Matcha-TTS on the board
itself, the other three on the host. Each model's README states exactly how far it got, and
`expected.json` distinguishes gates from measurements — where no board numbers
survived, the measured fields are `null` on purpose, never invented.

| model | recipe | verified to |
|---|---|---|
| [`las2`](models/las2/) — stereo disparity | complete | **full three-layer close-out on the board**: A 0.999954, B bit-identical, C 0.9998. Rebuild differs from the shipped binary (calibration set) |
| [`ppocr_v6`](models/ppocr_v6/) — OCR det + rec | complete | compiled and **board-verified**; det layer B bit-identical. Recognition in **four builds** (320/960 × int8/int16); int16 halves the character error at both widths and 960 int16 decodes long lines near-perfectly (79-char sentence exact). Preprocessing corrected to aspect-preserving pad |
| [`osnet`](models/osnet/) — person ReID | complete | **the first task-metric recipe**: int8 PTQ collapses (Rank-1 51% vs 85%) so the shipped build is **QAT self-distillation** — no labels, ~20 min, Rank-1 recovered to 84.6%. Board-validated in the original M10 work (cosine 0.9799, 0.82 ms/crop). Includes a reproducible int8-collapse path |
| [`yolop`](models/yolop/) — panoptic driving | recipe | the `_cut` surgery: the uncut graph bakes the anchor decode via `ScatterND`, which compiles clean and **never writes the objectness/class columns** (silent, total detector failure). `export.py` cuts before the decode and emits the 3 raw heads; BCDL decodes. Not re-run through the toolchain here |
| [`pidnet`](models/pidnet/) — real-time seg | recipe | the `_v3` pre-normalisation trap: un-normalised calibration data compiles clean and segments to noise. `calib.py` routes through `calib_pack` so calibration matches the runtime distribution. Export reconstructed (opset-19, no-norm); calibration is the faithful, load-bearing part |
| [`superres`](models/superres/) — Real-ESRGAN ×4 | recipe | scripts intact. Documents the tile-**area** size trap (~37 MB @128 vs ~148 MB @256 for identical per-pixel quality). No board numbers survived — measured fields left null |
| [`span`](models/span/) — SPAN ×4 | recipe | scripts intact except calibration (reconstructed by analogy). The fidelity half of the keep-both super-res pair (~1/6 of Compact's size). Measured fields null |
| [`xfeat`](models/xfeat/) — sparse features | recipe | scripts intact, two numerically-checked graph rewrites (InstanceNorm → CPU preprocessing; `_unfold2d` → SpaceToDepth). Single-channel featuremap input. Measured fields null |
| [`yoloe`](models/yoloe/) — open-vocab det + seg | recipe | scripts intact. Head surgery freezes the open vocabulary (CLIP text embeddings folded into `cv3`) and emits raw heads; DFL/anchor/mask decode on the CPU. **AGPL-3.0 — recipe only, no redistributable binary.** Config reconstructed |
| [`vitpose`](models/vitpose/) — whole-body pose | recipe | 133-keypoint ViTPose-S. Export reconstructed (only prep + reference survived). Documents the ViT-on-BPU LayerNorm-precision concern (let the compiler auto-promote to int16). Measured fields null |
| [`yolo26`](models/yolo26/) — detection (main) | recipe | BCDL's main detector. Same DFL split-head cut as YOLOE det, minus the open-vocab fuse. Highest-risk reconstruction: the compile config was never saved, so `config.yaml` is reconstructed and flagged to verify. **AGPL-3.0 — recipe only, no binary** |
| [`ppocr_v5`](models/ppocr_v5/) — OCR det + rec + cls | recipe | kept for the **angle classifier v6 lacks** (v5 det/rec are a fallback). The export rot is the write-up: the surviving PaddleOCR script pointed at a `PP-OCRv4_server_seal_det` (wrong model *and* version) with the shape-fix commented out. Dictionary count asserted against the ONNX output width |
| [`face`](models/face/) — SCRFD det + ArcFace rec | recipe | both ONNX come from the insightface `buffalo_l` pack (locate + tiny edit, not a torch export). The `_aligned` calibration is the shipped build; int8/int16 recognition builds. **insightface weights non-commercial — recipe only** |
| [`edgesam`](models/edgesam/) — promptable seg | recipe | encoder + two fixed-prompt decoders (sp1 / bp2). The multi-input decoder calibration feeds real encoder embeddings + sampled prompts. Upstream terms unreviewed — recipe only |
| [`pointpillars`](models/pointpillars/) — lidar 3-D detection | complete | compiled from this recipe and **board-verified**. The published VFE spends 290 ms on a 95 MFLOP MLP — about 0.3 GFLOPS, i.e. **layout-bound, not compute-bound**: `(11546, 32, 4)` is a huge pillar axis against a 4-wide channel axis. Re-expressed as a 1×1 convolution over `(1, 10, P, 32)` it is **1.62 ms, a 179× difference for identical arithmetic**; with the backbone and cut head fused, BPU total goes 403.7 → **8.82 ms**. The published head bakes its anchor decode in with `ScatterND` and compiles clean into a model that never writes its outputs (the YOLOP failure again). **int8 beats int16 here**, which is not the usual direction. Also the clearest **calibration-size** measurement in this repo: a single-frame set costs 4 of 18 boxes and two π heading flips, and lowering the threshold does not recover them because the *ranking* degrades — eight sweeps fix it at identical latency. **KITTI weights — recipe only, no binary** |
| [`lingbot_depth`](models/lingbot_depth/) — RGB-D depth refinement | complete | compiled and **board-verified** (C 0.989, absrel 2.92%, mask IoU 0.959). The upstream graph is not compilable as written — it drops depth tokens by validity, so the sequence length depends on the data; the rewrite is **bit-identical** to upstream with that masking off. int8 collapses the output's dynamic range (233% absrel) and the per-layer report **predicted it**. **DDR-bound at 1457 ms**, with a measured token-budget/accuracy curve for trading it down |
| [`glass`](models/glass/) — industrial anomaly detection | complete | compiled and **board-verified** on MVTec-AD `bottle` with pixel ground truth: image and pixel AUROC both **0.9905, matching float — at an output cosine of 0.9409**. AUROC is rank-based, so quantisation noise that moves every score without reordering them costs nothing. Read against `neuflow` below, where 0.9954 hid an unusable model. Backbone is a **frozen** WideResNet-50 with two ~16 MB per-product heads, so "retrain for your part" is realistic. **MVTec AD is CC BY-NC-SA — redistributable, non-commercial, share-alike** |
| [`neuflow`](models/neuflow/) — optical flow | complete | compiled and **board-verified** with ground truth *synthesised* (shift an image by a known number of pixels). int8 becomes a **flat ~1.9 px noise floor** — 2 px and 16 px displacements return the same error — while the compiler reports **0.9954**; int16 gets back to 0.055–0.081 px. Export needs three things copied from upstream's `infer.py` or the graph is silently wrong. **FlyingThings3D is research-only and says nothing about trained models — recipe only, no binary** |
| [`kws_zipformer`](models/kws_zipformer/) — wake words | complete | sherpa-onnx's open-vocabulary zipformer2 KWS (zh-en, 3M); only the streaming encoder goes to the BPU. **Four rewrites** before it compiles or means anything: the int64 `processed_lens` becomes a float mask; Swoosh's `log(1+exp(s))` quantised as written takes the encoder to **cosine −0.21**; six relative-position `GatherElements` become slices; ten depthwise convs padded longer than their input stop the code generator. One optimizer pass crashes onnxruntime on the graph and is switched off by a wrapper. Rebuild: encoder bit-identical to the shipped ONNX, detections match sherpa-onnx in float on 87/87 and float on 86/87 after int16 (as the shipped build). **Licence unresolved — recipe only** |
| [`silero_vad`](models/silero_vad/) — voice activity | complete | the STFT has to come out of the graph: squared on the BPU, the power spans ~5000 and int16 loses every quiet bin. **The default calibration search is the second trap**: on a fresh set it picked KL and clipped the spectrum input at 1.8 against speech that reaches 13 — compiles clean, twice the worst-case error; the config pins `max`. The package's `safetensors` holds older weights than its ONNX. Also shows that **`hb_compile`'s printed output cosine is computed on calibration sample 0 alone** |
| [`matcha_tts`](models/matcha_tts/) — TTS acoustic model | complete | rebuilt and **board-verified: output identical to the shipped builds** (max diff 0 on 19 sentences). A fixed-length export of stock Matcha is not the model: GroupNorm and an *added* 0/1 attention mask both depend on how much padding there is (log-mel off by up to **9.96**); rewritten to 8.5e-5. Token ids cannot be a graph input (the compiler quantised them to a range of 127 for 2190 ids); the embedding lookup moves to the CPU. int8 decoder: **99% CER** while UTMOS still scores it well |
| [`vocos`](models/vocos/) — TTS vocoder | complete | cut at the head's log-magnitude and phase: compiled as published, the quantised `cos`/`sin` take the phase to cosine 0.58 (int8) / 0.97–0.99 (int16). Rebuild matches the shipped ONNX exactly and scores **0.53 dB** audible-cell spectral distance (shipped 0.54). int8 is 5.4 dB and mixed precision 4.5 dB — the error builds up over all eight blocks |

Every model here is one BCDL has a decoder for and has run on the board. Not in
this repo, by design: the **download-only** entries in BCDL's catalogue (Depth-
Anything-V2, SigLIP, DeepLabV3+, YOLOv8, the YOLO26n zoo builds, D-Robotics'
Paraformer speech recogniser) — their "recipe"
is a URL, so they live in BCDL's `fetch_models.sh`, not here — and the
exploratory conversions with no BCDL decoder (dinov3, rf-detr, rt-detr, rtmpose,
mono3d), which would ship a recipe with no end-to-end path to validate it.

## Getting the compiled binaries

The compiled `.hbm` are hosted on Hugging Face — see
[the table above](#prebuilt-hbm-on-hugging-face). They are **not** committed
here; this repo stays the recipes.

The rule that governs what may be hosted, and it is stricter than the badge on
an upstream repo:

- **Redistributable, including commercially** — hosted.
- **Redistributable but non-commercial** — hosted, with the restriction in a
  banner at the top of the model card rather than a footnote. A restriction that
  exists only in the weights' provenance is exactly the kind a reader misses.
- **Copyleft weights, non-commercial weights we may not redistribute, or
  unresolved** — **recipe only**. Build it yourself from weights you are
  licensed to use.

Each model's `expected.json` records the finding *and which link in the chain*
it comes from — the code licence, the pretrained weights it was fine-tuned from,
or the data it was trained on — plus the date it was reviewed. That last field
matters: these are judgements about somebody else's terms at a point in time,
not facts that stay true by themselves.

## Licence

The recipes and scripts in this repo are MIT. **The models are not ours** — each
one carries its upstream licence, recorded in its model directory. Check it
before you redistribute or ship commercially.
