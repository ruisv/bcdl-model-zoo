# SCRFD-10G + ArcFace R50 — face detection & recognition

Two sub-models that make one face pipeline, like `ppocr_v6` is det + rec:

- **SCRFD-10G** — a fast, accurate face detector. One 640×640 image in, faces
  out as boxes **plus 5 landmarks** (eyes, nose, mouth corners). The landmarks
  are the hinge between the two models.
- **ArcFace R50** — a face-recognition backbone. One **aligned** 112×112 face
  chip in, a **512-d embedding** out; two faces are the same identity when their
  L2-normalised embeddings have high cosine similarity.

Both come from insightface's **`buffalo_l`** pack — the same pack the
`insightface` app downloads. This recipe does **not** train or torch-export
anything: the two ONNX already exist inside the pack, and `export.py` just
locates them and applies a tiny static-shape edit (see [Running it](#running-it)).

| build | config | BCDL consumes |
|---|---|---|
| `scrfd_10g_nashm_640x640_nv12.hbm` | `config_det.yaml` | face boxes + 5 landmarks (NV12 in, int8) |
| `arcface_r50_aligned_nashm_112x112.hbm` | `config_rec.yaml` | 512-d embedding, int8 |
| `arcface_r50_aligned_int16_nashm_112x112.hbm` | `config_rec_int16.yaml` | 512-d embedding, **int16 (recommended)** |

## Licence: non-commercial weights — recipe only

**This is the first thing to know.** insightface's *code* is MIT, but the
pretrained **`buffalo_l` weights** (SCRFD-10G and the ArcFace R50 / glint360k
model) are released for **academic / non-commercial research only**. A compiled
`.hbm` is a derivative of those weights and carries the same restriction, so this
directory ships the **recipe, not a redistributable binary**, and you must not
use the resulting models in a commercial product.

**For a commercial build, change the weights:** detection → **YuNet** (OpenCV
Zoo, Apache-2.0); recognition → a permissively-licensed backbone. Those are
different graphs and are out of scope here.

## The traps / which build is wrong

**1. ArcFace must be calibrated on ALIGNED chips — that is what `_aligned`
means.** ArcFace never sees a detector box. It sees a 112×112 chip produced by a
5-point **similarity transform** that warps the eyes/nose/mouth onto ArcFace's
canonical template (insightface's `norm_crop`, driven by the landmarks SCRFD
emits). The shipped recogniser is calibrated on those aligned chips. Calibrating
on **centre-crops of boxes instead** gives a different, silently-worse file: it
compiles, loads, and runs identically, and the embedding quietly degrades because
the calibration distribution no longer matches what the runtime feeds. `calib.py`
takes a directory of already-aligned chips and refuses to fabricate them from
boxes for exactly this reason. (A salvaged config for the centre-crop variant
exists; it is deliberately **not** carried forward — see `expected.json`
`rejected_builds`.)

**2. int8 vs int16 for the recognition embedding — cosine will lie to you.**
This is the same trap `osnet` documents for ReID and `ppocr` for CTC: a
recognition tower that has genuinely lost fidelity **still emits a well-formed
512-d unit vector**, so its cosine against the float reference can look
acceptable while face **verification** accuracy drops. The number that tells the
truth is a **task metric** — verification ROC (TAR @ fixed FAR) on a labelled
pair set — not the per-embedding cosine. That is why the int16 build
(`config_rec_int16.yaml`, `set_all_nodes_int16`) exists: it recovers embedding
fidelity, and — as on `las2`/`ppocr` — being one contiguous fixed-point stream
with no int8↔int16 requant it is often *faster* than the mixed default too.
**Judge int8 vs int16 on the verification metric on the board, then ship int16 if
it wins.** Detection stays int8 — a plain CNN with no requant problem.

**3. The weights are non-commercial** (see the licence section above). This is a
"which build is wrong for your use" just as much as the other two: for a product,
every build here is the wrong one, and the fix is swapping the weights, not the
bit width.

## Running it

`export.py` needs the extracted **`buffalo_l`** pack (it holds `det_10g.onnx` and
`w600k_r50.onnx` alongside the models this recipe does not use).

```bash
BUFFALO=<dir of the extracted insightface buffalo_l pack>

# [1/3] locate + tiny static-shape edit -> two ONNX graphs
python export.py --buffalo $BUFFALO --task both
#   det_10g.onnx   -> scrfd_10g_640.onnx    (1x3x640x640, onnxslim-resolved)
#   w600k_r50.onnx -> arcface_r50_112.onnx  (input dims fixed to 1x3x112x112;
#                     export prints the byte delta -- it is the same graph)

# [2/3] calibration.
#   det: face-containing scenes (config applies (x-127.5)/128)
#   rec: ALIGNED 112x112 chips from insightface norm_crop, NOT centre-crops
python calib.py --config config_det.yaml     --task det --images <face scenes>
python calib.py --config config_rec.yaml     --task rec --aligned-dir <112 chips>

# [3/3] compile (Ampere-or-newer GPU; --gpu is an nvidia-smi index)
./compile.sh --config config_det.yaml       --gpu <N>
./compile.sh --config config_rec.yaml       --gpu <N>   # int8
./compile.sh --config config_rec_int16.yaml --gpu <N>   # int16 (recommended)
```

Produce the aligned chips the way deployment does: run insightface's
`FaceAnalysis` (or SCRFD directly) to get `face.kps`, then `norm_crop` to the
112×112 template. That is the same distribution ArcFace sees at inference, and
matching it is the whole point of trap 1.

`jobs` lives in the config, not on the command line (`hb_compile` has no `--jobs`).
Both models target `nash-m` (S100/S100P).

## Status

**Recipe reconstructed from salvaged `hb_compile` configs; not re-run.** The
detection and recognition YAMLs (input names, shapes, norm, prefixes, march)
are faithful to salvage; the SCRFD normalisation is `mean 127.5 / scale
0.0078125` (= 1/128). `export.py` and `calib.py` are reconstructed to match — the
load-bearing parts are the **provenance** (locate + static-shape edit, not a
torch export) and the **aligned-chip calibration**.

- **export** — `det` is onnxslim-resolved to a static `1×3×640×640`; `arcface`
  is `w600k_r50` with only its input dims fixed to `1×3×112×112`. IR capped to 9.
- **calibration** — routed through `common/calib_pack.py`; det normalised by the
  config, rec normalised in `calib.py` (featuremap/no_preprocess).
- **normalisation to confirm** — the shipped ArcFace divisor is taken as
  **/128** (to match SCRFD's `scale_value`); insightface's own recognition
  wrapper uses **/127.5**. The runtime's face preprocessing is the source of
  truth; `--rec-std` overrides it. This is the one value most worth checking
  against the deployment code before trusting the build.
- **not measured** — no `.hbm` sizes, latency, cosine, or verification numbers
  survived. `expected.json` leaves them **null** rather than guessing, and
  declares the verification task metric that a real acceptance run must fill in.
