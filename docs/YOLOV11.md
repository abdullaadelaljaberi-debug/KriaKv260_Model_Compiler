# YOLOv11 on KV260 — User Guide

This document covers everything needed to take a stock Ultralytics YOLOv11
model and deploy it to the Kria KV260 DPU through this pipeline.

It's longer than the YOLOv5 documentation because YOLOv11 needs
architectural modifications that YOLOv5 doesn't. Once you understand the
additional training step, the compile and deploy stages are identical to
other supported families.

Two variants are validated end-to-end:

- **`yolov11n`** — small, fast (~26 FPS end-to-end on the KV260). Used as
  the v0.8 baseline.
- **`yolov11s`** — larger (3.7× parameters), slower (~17 FPS) but
  significantly more robust to int8 quantization noise. Added in v0.10 for
  the capacity-vs-quantization study; see "Capacity vs quantization" below.

---

## TL;DR (the three commands)

```bash
# 1. Train a DPU-friendly YOLOv11 on your dataset (once per dataset).
#    Substitute yolo11n.pt with yolo11s.pt for the larger variant.
python3 scripts/host/_train_yolov11.py \
    --weights yolo11n.pt \
    --data    my_dataset/data.yaml \
    --output  data/weights/my_model_dpu.pt \
    --epochs  50 --batch 16

# 2. Compile to xmodel (standard pipeline). Use yolov11n or yolov11s.
NUM_CLASSES=<N> bash scripts/host/02_compile.sh yolov11 yolov11n \
    data/weights/my_model_dpu.pt data/calib/

# 3. Sync to Kria
bash scripts/host/03_sync_to_kria.sh ubuntu@<kria-ip> yolov11n
```

End state: `/home/ubuntu/xmodels_vai35/yolov11n/yolov11n_kv260.xmodel`
on the board, loadable via PYNQ-DPU's `DpuOverlay.load_model()`, deployable
via the standard `ModelRunner` in `lpr_pipeline.deploy.runner`.

---

## Why training is a separate step

Most supported models in this pipeline (YOLOv5n/s, YOLOX-nano/tiny) compile
to a single DPU subgraph out of the box because their architectures use
only DPU-supported operations.

YOLOv11 doesn't. Stock YOLOv11n contains two structural patterns that the
KV260 DPU can't accelerate cleanly:

1. **The C2PSA attention block** uses `torch.matmul`, `torch.softmax`,
   `Tensor.chunk`, `Tensor.split`, and various reshapes inside an attention
   computation. The DPU has no hardware for these — they fall back to CPU,
   creating multiple CPU/DPU boundary subgraphs.

2. **The Detect head's `cv3` classification branch** uses depthwise
   convolution (`DWConv`) blocks. These hit a shape constraint of the
   DPUCZDX8G_ISA1 architecture (KV260 B4096) that fragments the cv3
   computation into one CPU subgraph per scale.

Without intervention, a stock YOLOv11n compile produces 17 subgraphs.
PYNQ-DPU's `DpuOverlay.load_model()` asserts that the xmodel has exactly
one DPU subgraph (with output dequantization tolerated), so the model is
unusable on that runtime.

This pipeline fixes both problems by replacing the offending blocks with
DPU-friendly equivalents:

| Stock YOLOv11 block | DPU-friendly replacement | Located in |
|---|---|---|
| `C2PSA` (attention) | `C2PSA_DPU` (HardSigmoid gating, conv-based channel selection) | `lpr_pipeline/c2psa_dpu.py` |
| `DWConv` (in cv3) | plain `Conv` | applied via monkey-patch in `lpr_pipeline/detect_dpu.py` |

The replacements are **not mathematically equivalent** to the originals.
`C2PSA_DPU` replaces softmax-based attention with element-wise gating;
plain `Conv` is a different operation than `DWConv + Conv`. You cannot
load a stock-trained YOLOv11n `.pt`, apply the replacements, and expect
good results — the new operations need to be retrained. The training
helper script (`_train_yolov11.py`) applies the monkey-patches before
Ultralytics builds the model, so the substitutions happen at YAML parse
time and the training loop optimizes the new operations.

---

## Architecture decisions: why each change

This section explains *why* each modification was chosen, for the
thesis-defense audience. The list below covers both YOLOv5 and YOLOv11
because the two families share most DPU adaptations; only the
attention-block surgery is unique to YOLOv11.

### Shared decisions (apply to YOLOv5 and YOLOv11)

**SiLU → LeakyReLU(0.1015625).** The KV260 DPU natively accelerates
ReLU, ReLU6, and LeakyReLU with a *single* fixed negative slope of
`13/128 = 0.1015625`. SiLU/Swish, GELU, Mish, and custom activations
fall back to CPU, fragmenting the graph into many small subgraphs that
fail to load via `DpuOverlay.load_model()`. The slope `0.1015625` is
chosen exactly (not approximated as `0.1`) because the Vitis-AI
quantizer would otherwise auto-correct it during quantization, and the
correction introduces float-vs-int8 drift that degrades accuracy.

**Stripped Detect head.** The Detect heads in both YOLOv5u and YOLOv11
include inline post-processing: DFL projection (softmax + dot product
over the reg_max=16 dimension), anchor-grid construction, sigmoid for
class scores, and NMS. Implementing these inside the DPU subgraph would
require operations the DPU can't accelerate (softmax over a large
dimension, broadcasting over precomputed anchor tables). Stripping the
head and emitting raw conv outputs lets the DPU produce its three
multi-scale tensors at full hardware speed; the decode runs on the CPU
in ~0.8 ms per frame, which is negligible compared to the DPU's ~8 ms
on yolov5n or ~38 ms on yolov11n.

**NHWC input wrapper.** The DPU's native data layout is NHWC, but
PyTorch tensors are NCHW. A no-op `Permute` wrapper at the model
boundary lets the rest of the pipeline keep PyTorch's convention while
the DPU sees its preferred layout. Without this, NNDCT inserts the
permute internally as a CPU op, adding a tiny but unnecessary subgraph
at every model run.

### YOLOv11-specific decisions (don't apply to YOLOv5)

**C2PSA → C2PSA_DPU.** Stock C2PSA is a self-attention block: it
computes Q·Kᵀ (a `torch.matmul` over the spatial-token dimension),
applies softmax, then dots against V. None of those operations are in
the DPU's op set. Three replacement choices were considered:

| Option | Pros | Cons | Choice |
|---|---|---|---|
| Remove C2PSA entirely | Simplest; matches YOLOv11 - "no PSA" variant | Loses the accuracy gain that motivates choosing v11 over v5u | No |
| Implement multi-subgraph compile preserving original C2PSA | Mathematically exact | Forces CPU↔DPU boundaries at every attention block; ~2-4× slower; can't use PYNQ-DPU's loader | No |
| Replace with DPU-friendly approximation | Single DPU subgraph; near-native speed | Not equivalent — must retrain | **Yes** |

`C2PSA_DPU` uses HardSigmoid (a piecewise-linear approximation of
sigmoid that the DPU has dedicated hardware for) over a 1×1 conv
projection to produce a per-position gating tensor, then element-wise
multiplies the value tensor by the gate. This preserves attention's
*role* (selective amplification of important positions) while using
only operations the DPU can accelerate. The quantization argument is
also favorable: HardSigmoid has a bounded output range (0–1) so
quantization scales are easy to determine; softmax produces values that
are very small for most positions and very large for a few, which is
worst-case behavior for uniform int8 quantization.

**DWConv → Conv in the Detect head's cv3 branch.** Depthwise
convolutions (`groups=channels`) hit a shape constraint of the
DPUCZDX8G_ISA1 architecture for the cv3 branch's per-scale convs,
forcing the entire branch to CPU. Replacing each `DWConv(C, k=3,
groups=C)` followed by `Conv(C, 64, k=1)` with a single `Conv(C, 64,
k=3)` keeps the receptive field identical, costs ~1M additional
parameters across the three scales, and stays entirely on the DPU.
Re-training is required because the parameter shapes change.

**`tensor.repeat` → `torch.cat`.** Inside the attention block, the
stock implementation does `attn_score.repeat(1, n, 1, 1)` to expand
across the channel dimension. The XIR op set doesn't include the
`nndct_repeat` op, so this would force a CPU subgraph. For `repeat
factor = 2` (the case in YOLOv11n), `torch.cat([x, x], dim=1)` is
mathematically equivalent and is a native XIR operation. The
`C2PSA_DPU` block uses `cat` exclusively.

### Why YOLOv11n was chosen as the eggs baseline (and why YOLOv11s was added later)

YOLOv11n was chosen as the eggs detection baseline for three reasons:

1. **Footprint fits B4096.** Even after surgery (~3.6M params), it
   compiles to a single 352-op DPU subgraph well within the KV260's
   B4096 DPU budget.
2. **It's the smallest current Ultralytics variant.** Provides a
   meaningful contrast to the older YOLOv5n in the same pipeline.
3. **Full feature set.** Has C3k2 (the parameter-efficient C3 successor)
   and C2PSA (the attention block we adapted) — both of which justify
   choosing v11 over v5u.

YOLOv11s was added in v0.10 *after* deploying v0.8 revealed an
unexpected problem: yolov11n's int8 quantized model produced thousands
of false positives on industrial test imagery despite the PyTorch float
version being perfect. The "capacity vs quantization" experiment below
documents that finding and the resolution.

---

## The seven architectural modifications (operator-level reference)

For thesis / reference, here's the complete list of operations replaced.
All of these happen automatically when you use `_train_yolov11.py` and
the `yolov11` compile family — you don't apply them by hand.

### Training-time modifications (applied via monkey-patch in `_train_yolov11.py`)

1. **`Attention.matmul` → element-wise multiplication.** The stock
   attention computes `q @ k.transpose(-2, -1)` (batched matmul over
   token dimension). `AttentionDPU` replaces this with `q * k` per
   spatial position. Loses cross-token information; recovers via
   retraining.

2. **`Attention.softmax` → `HardSigmoid`.** The stock attention applies
   softmax over the token (anchor) dimension. The DPU has no softmax op.
   `AttentionDPU` uses HardSigmoid as a per-position activation. The
   normalization semantics change but the gating role is preserved.

3. **`Attention.chunk`/`split` → three identity-initialized 1×1
   convolutions.** The stock attention does `qkv.split([nh_kd, nh_kd, dim],
   dim=1)` to extract q, k, v from the concatenated qkv tensor. The DPU
   handles convolutions natively; `conv_q`, `conv_k`, `conv_v` are
   one-hot-initialized so the freshly-constructed module starts identical
   to the original.

4. **`Tensor.repeat` (in attention channel expansion) → `torch.cat`.**
   `attn_score.repeat(1, n, 1, 1)` produces a `nndct_repeat` op which
   is not in XIR's op set and forces a CPU subgraph. `torch.cat` is a
   native XIR operation. For repeat factor 2 (the case in YOLOv11n/s):
   `cat([attn_score, attn_score], dim=1)` is mathematically equivalent.

5. **Stock `SiLU` activations → `LeakyReLU(0.1015625)`.** This is shared
   with all other DPU-deployable YOLOs in this pipeline. Applied during
   compile (not training) by `lpr_pipeline.compile.yolov5._swap_silu_to_leakyrelu`.
   The slope 0.1015625 = 13/128 is the DPU's only supported negative
   slope; using it exactly avoids the quantizer's auto-correction drift.

6. **`HardSigmoid` instead of `SiLU` inside `C2PSA_DPU`.** All convs
   inside the replaced attention block use HardSigmoid (DPU-native
   piecewise-linear approximation of sigmoid).

7. **Detect head `DWConv` → plain `Conv`.** YOLOv11's `Detect.cv3` uses
   `Sequential(DWConv(C, k=3, groups=C), Conv(C, 64, k=1))` per stage,
   per scale. We replace this with a single `Conv(C, 64, k=3)` per stage,
   per scale. Adds ~1M parameters but eliminates the depthwise-conv2d
   CPU subgraphs.

### Compile-time transformations (applied automatically by `lpr_pipeline/compile/yolov11.py`)

The yolov11 compile path inherits the entire yolov5 compile flow. So
the following also apply:

- `SiLU → LeakyReLU(13/128)` swap on all backbone/neck activations
- Detect head's inline NMS/decode stripped (raw conv outputs only)
- NHWC permute wrapper for DPU input layout

---

## Result: what you get

For a YOLOv11n trained on the eggs dataset (single class, 933 train / 32
val images) with default `_train_yolov11.py` settings:

| Metric | yolov11n | yolov11s |
|---|---:|---:|
| Trained-model params | 3,600,083 | 13,479,891 |
| Trained-model layers (fused) | 117 | ~155 |
| mAP@0.5 on val | 0.995 | 0.995 |
| mAP@0.5:0.95 on val | 0.97 | 0.915 |
| Quantized xmodel size | 4.7 MB | 15 MB |
| DPU fingerprint | 0x101000056010407 | 0x101000056010407 |
| Subgraph count | 5 (1 USER + 1 DPU/352 ops + 3 CPU fix2float) | 5 (same shape) |
| DPU latency on KV260 | ~38 ms | ~58 ms |
| End-to-end FPS (synthetic) | ~26 | ~17 |
| Loadable via `DpuOverlay.load_model()` | yes | yes |

Both compile to a single DPU subgraph carrying the entire backbone, neck,
modified attention, and detection conv layers. The three output
`fix2float` CPU operations are standard int8→float32 dequantization
boundaries that every quantized model has.

---

## Detailed workflow

### Step 1: produce a DPU-friendly trained model

```bash
python3 scripts/host/_train_yolov11.py \
    --weights yolo11n.pt \           # or yolo11s.pt for the larger variant
    --data    my_dataset/data.yaml \
    --output  data/weights/my_model_dpu.pt \
    --epochs  50 \
    --batch   16 \
    --imgsz   640
```

What the script does internally:

1. Applies the C2PSA → C2PSA_DPU monkey-patch
2. Applies the DWConv → Conv monkey-patch
3. Detects GPU (works without one but is much slower)
4. Resolves dataset paths to absolute form (handles Roboflow's `../` convention)
5. Constructs `YOLO(args.weights)` — Ultralytics' trainer will rebuild
   the model from YAML during `setup_model()`, and the rebuild uses the
   patched namespaces
6. Trains for `--epochs` epochs
7. Locates the resulting `best.pt` (Ultralytics moves it around) and
   copies it to `--output`
8. Verifies the saved model has zero `DWConv` modules and layer 10 is `C2PSA_DPU`

The verification step is important — if it reports `✗ WARNING: monkey-patches may not have held`,
something went wrong (Ultralytics version drift, broken Python path, etc).
Don't proceed with a model that fails verification.

GPU is strongly recommended. On an RTX A2000 8GB:
- yolov11n, 50 epochs at batch 16 on a ~900-image single-class dataset:
  ~12 minutes
- yolov11s, same configuration: ~27 minutes

CPU-only training takes 5+ hours per variant.

### Step 2: compile to xmodel

```bash
# yolov11n
NUM_CLASSES=1 bash scripts/host/02_compile.sh yolov11 yolov11n \
    data/weights/yolo11n_eggs_dpu.pt data/calib_v2_hardneg/

# yolov11s (same shape, just swap the variant + weights paths)
NUM_CLASSES=1 bash scripts/host/02_compile.sh yolov11 yolov11s \
    data/weights/yolo11s_eggs_dpu.pt data/calib_v2_hardneg/
```

This is the standard pipeline. The yolov11 compile path adds defensive
imports of `lpr_pipeline.c2psa_dpu` and `lpr_pipeline.detect_dpu` so the
unpickled model can resolve the C2PSA_DPU class inside the Vitis-AI
container. Everything else (SiLU swap, head stripping, NHWC wrap,
quantize, compile) is shared with the yolov5 path.

A critical detail: **calibration data must match deployment domain**.
Using LPR calibration images for an egg model (or vice versa) causes
the int8 model to see false positives or missed detections. Always
calibrate with images that match your deployment data distribution. The
simplest approach is to put a representative sample of your training
images in `data/calib/`. For hard-negative training (see below), include
the hard-negative images in the calibration mix too — or *don't*, see
the "calibration set composition" warning at the end of this document.

Compile time depends on whether you have the GPU VAI image:

- GPU: ~1 minute end-to-end
- CPU: ~5 minutes end-to-end (still acceptable; result is bit-identical to GPU)

### Step 3: sync to Kria

```bash
bash scripts/host/03_sync_to_kria.sh ubuntu@10.42.0.189 yolov11n   # or yolov11s
```

This copies the xmodel to
`/home/ubuntu/xmodels_vai35/<variant>/<variant>_kv260.xmodel` on the
Kria. The Kria-side scripts (`run_benchmark.sh`, `run_live.sh`) look
there.

### Step 4: deploy

Multiple options on the Kria:

```bash
# Live demo (yolov11n or yolov11s)
sudo bash scripts/kria/run_live.sh yolov11n
sudo bash scripts/kria/run_live.sh yolov11s

# Or use ModelRunner directly from Python
python3 -c "
from pynq_dpu import DpuOverlay
from lpr_pipeline.shared.models import get_spec
from lpr_pipeline.deploy.runner import ModelRunner

overlay = DpuOverlay('dpu.bit')
runner = ModelRunner(
    get_spec('yolov11s'),
    '/home/ubuntu/xmodels_vai35/yolov11s/yolov11s_kv260.xmodel',
    overlay,
)
runner.warmup(n=3)
# ... runner.infer(frame) returns (detections, timings)
"
```

The `ModelRunner` automatically dispatches to `decode_yolov11()` (which
delegates to `decode_yolov5u()`) and to `Preprocessor("yolov11", 640)`
based on the spec's `family` field. Both decoders use the same DFL math
because YOLOv5u and YOLOv11 share the Detect head structure.

The `run_live.sh` script launches `notebooks/eggs/05_deploy_visual.ipynb`
for both yolov11n and yolov11s.

---

## Capacity vs architecture — what int8 quantization actually depends on

This section documents the v0.10 / v0.11 experimental investigation
that motivated adding yolov11s and later yolov5s_eggs to the registry.
The headline finding evolved through the investigation; we present the
final conclusion and the data that led to it.

### Setup

After v0.8 deployed yolov11n successfully on the eggs dataset,
evaluation on a held-out **57-image industrial test set** (real
conveyor footage with no eggs present — any detection is a false
positive) revealed an unexpected pattern:

- **PyTorch float `.pt` model** produced zero detections on industrial
  imagery (max raw confidence ~0.0006 — the model correctly suppresses
  egg-like activations on background)
- **DPU int8 xmodel** produced thousands of false positives at conf=0.85,
  with mean confidence ≈ 0.96 (saturated)

The float-to-int8 gap was a confidence inflation of roughly 1,600×:
near-zero pre-sigmoid activations on background regions were quantized
in a way that saturated the output sigmoid.

Two early interventions were tried and quantified:

**Intervention 1 — hard-negative training.** Augmenting the eggs
dataset with 402 labeled-empty industrial images brought the float
model to **0 FPs** at conf=0.85 — but the int8 model's FP count was
**unchanged**, demonstrating the bottleneck was the int8 quantization
itself, not training data.

**Intervention 2 — mixed calibration.** Including hard-negative images
in the calibration set was tested on the assumption that the
quantizer would benefit from seeing them. Counterintuitively this
**increased** int8 FPs (4,220 → 6,609 at conf=0.85). The reason is the
DPU's per-tensor (not per-channel) activation scale constraint: mixing
in-domain and out-of-domain images widens the activation range the
per-tensor scales must cover, making each int8 bucket coarser and
amplifying per-layer quantization noise.

With training-data fixes failing, the next hypothesis was **model
capacity**: that quantization noise scales inversely with parameter
count, and a bigger model would have more weight redundancy to absorb
the noise. yolov11s was added to test this and succeeded at reducing
industrial FPs (2,172 vs yolov11n's 6,609 at conf=0.85). The v0.10
release initially framed this as "capacity is the dominant lever."

That conclusion turned out to be incomplete. To test whether the
benefit came from capacity *or* from something specific to the YOLOv11
architecture, we trained a third model: **yolov5s** on the same
augmented eggs dataset at the same imgsz=640. yolov5s sits between
yolov11n (3.6M params) and yolov11s (13.5M params) at 9.1M params —
**2.5× yolov11n's capacity**. If capacity alone explained yolov11s's
int8 robustness, yolov5s should land between them on FP count.

### Experimental design

Three architecturally distinct models, trained on identical data,
evaluated identically:

| Model | Params | Architecture features |
|---|---:|---|
| yolov11n | 3,600,083 | C2PSA_DPU attention + DPU-friendly conv head |
| yolov5s_eggs | 9,122,579 | Pure-conv YOLOv5u; no attention; no DPU surgery needed |
| yolov11s | 13,479,891 | C2PSA_DPU + DPU-friendly conv head; 3.7× yolov11n capacity |

**Identical training pipeline:**
- 50 epochs, batch 16, imgsz=640
- Eggs + hardneg dataset: 1,335 train images (933 egg-positive + 402
  hard-negative), 32 validation images with 1,749 ground-truth eggs
- Final float mAP@0.5 = 0.995 for all three (architectural differences
  invisible at float)

**Identical compile pipeline:**
- NNDCT PTQ quantization
- In-domain (eggs-only) calibration set
- vai_c_xir compilation for KV260 B4096 (DPUCZDX8G_ISA1 fingerprint
  `0x101000056010407`)

**Two complementary evaluations:**

1. **Eggs validation set** (32 images, 1,749 ground-truth eggs) — measures
   detection quality on positive-class imagery using IoU≥0.5 matching for
   TPs / FPs / FNs.
2. **Industrial test set** (57 negative-class images) — measures
   background false-positive rate. Any detection is a FP.

Together these give a complete picture: test 1 catches "model lost real
eggs"; test 2 catches "model invented eggs in noise." Either alone tells
an incomplete story.

### Float baselines (all three models)

All three models are effectively perfect at float precision:

| Model | Industrial FPs @ 0.85 | Eggs P / R / F1 @ 0.85 |
|---|---:|:---|
| yolov11n float | 0 | 0.9994 / 0.9971 / 0.9983 |
| yolov5s_eggs float | 0 | **1.0000** / 0.9983 / **0.9991** |
| yolov11s float | 0 | 0.9994 / 0.9966 / 0.9980 |

The 1,749 ground-truth eggs are detected with > 99.6% recall and > 99.9%
precision across all three architectures. **Architectural differences
are invisible at float precision.** The training pipeline works.

### Int8 results — the full picture

All numbers below measured on KV260 B4096 at VAI 3.5 / DPUCZDX8G_ISA1.

**Eggs validation set (32 images, 1,749 ground-truth eggs, conf ≥ 0.85, IoU ≥ 0.5):**

| Model | Params | TPs | FPs | FNs | Precision | Recall | F1 | F1 drop vs float |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| yolov11n | 3.6M | 1,449 | 160 | 300 | 0.9006 | 0.8285 | **0.863** | −13.5 |
| **yolov5s_eggs** | **9.1M** | **1,133** | **485** | **616** | **0.7002** | **0.6478** | **0.673** | **−32.6** |
| yolov11s | 13.5M | 1,525 | 348 | 224 | 0.8142 | **0.8719** | **0.842** | −15.6 |

**Industrial test set (57 negative images, any detection is FP):**

| Model | FPs @ 0.50 | FPs @ 0.70 | FPs @ 0.85 | Mean confidence |
|---|---:|---:|---:|---:|
| yolov11n | 7,685 | 7,493 | 6,609 | 0.96 |
| **yolov5s_eggs** | 6,769 | 6,565 | 5,690 | **0.98** |
| yolov11s | 4,826 | 3,443 | **2,172** | 0.93 |

**Throughput on the Kria (synthetic input, 100 iterations):**

| Variant | DPU mean latency | End-to-end FPS |
|---|---:|---:|
| yolov11n | 38.8 ms | 25.8 |
| yolov5s_eggs | 49.0 ms | 20.4 |
| yolov11s | 58.2 ms | 17.2 |

### Finding 1: capacity alone does not explain int8 robustness

If capacity were the dominant factor, **yolov5s at 9.1M parameters
(2.5× yolov11n's capacity) should outperform yolov11n on int8 metrics.**
It doesn't. It produces:

- **Worse eggs F1** than yolov11n (0.673 vs 0.863)
- **More eggs FPs** than yolov11n (485 vs 160)
- **More eggs FNs** than yolov11n (616 vs 300)
- **The highest mean prediction confidence (0.98)** — most saturated of
  the three int8 models

Despite having 2.5× the parameters, yolov5s's int8 deployment is
strictly worse than yolov11n's on the same task. The simple
capacity hypothesis is falsified.

### Finding 2: architecture-family matters more than capacity

The YOLOv11 family (with the C2PSA_DPU attention block and the DPU-
friendly conv head) resists int8 quantization noticeably better than
the YOLOv5u family on this task:

| Family / size class | F1 drop float → int8 |
|---|---:|
| YOLOv11 small (3.6M, yolov11n) | −13.5 |
| YOLOv5  medium (9.1M, yolov5s_eggs) | **−32.6** |
| YOLOv11 large (13.5M, yolov11s) | −15.6 |

Both YOLOv11 variants — at very different capacities — degrade by
about the same amount (~14-16 F1 points). The mid-capacity YOLOv5s
degrades **more than twice as much** (32.6 points).

The *architecture family* is the dominant first-order factor; capacity
is a second-order modifier within a family.

### Finding 3: capacity within a family still matters

Comparing yolov11n vs yolov11s (same architecture, different capacity):

- yolov11s industrial FPs at 0.85: **2,172 vs yolov11n's 6,609** (−67%)
- yolov11s eggs recall: **87.2% vs yolov11n's 82.9%** (+4.3 points)
- yolov11s eggs F1: 0.842 vs yolov11n's 0.863 (−2.1 points, due to
  more FPs on eggs imagery)

Within YOLOv11, more capacity = strictly fewer background FPs and
slightly better recall on real eggs, traded for marginally lower
eggs-image precision and 33% lower throughput. The intra-family
scaling matches the original "capacity gives weight redundancy"
story — but only *within* the YOLOv11 family, not across families.

### Mechanism (interpretation)

The plausible mechanism for v11 outperforming v5 at int8 is the
C2PSA_DPU attention block's gating action. HardSigmoid-gated activations
have bounded output (0,1) which produces tighter activation scales
under per-tensor quantization. YOLOv5u's pure-conv stack passes
features through without gating, producing wider activation
distributions that spread the per-tensor scales further — making each
int8 bucket coarser and more error-prone for fine-grained
discrimination.

Indirect evidence supporting this:

- yolov5s int8 produces **`RuntimeWarning: overflow encountered in
  exp`** inside the YOLOv5u decoder's sigmoid (saturation past
  representable float range)
- yolov5s mean prediction confidence (0.98) is highest of the three —
  most saturated
- yolov11s mean confidence (0.93) is lowest — least saturated

The architectural design choice that justifies the cost of the
yolov11 surgery (the C2PSA_DPU replacement) appears to pay back as
int8 quantization robustness, not just deployability.

### Recommendation

For DPU int8 deployment of fine-grained single-class discrimination on
the Kria KV260:

1. **Choose YOLOv11 over YOLOv5u** if int8 quality matters. The
   architecture surgery (C2PSA_DPU, DWConv→Conv) is worth its cost.
2. **Within YOLOv11, choose capacity per your throughput budget**:
   yolov11s if you can afford 17 FPS and need low background FP rate;
   yolov11n if you need 25+ FPS and can tolerate higher FPs.
3. **Don't assume more capacity in another family solves int8
   problems** — it doesn't, as yolov5s demonstrates.

### Caveats

- **Single dataset, single class.** Generalization to multi-class
  detection or different scenes hasn't been tested.
- **Single DPU hardware target** (DPUCZDX8G_ISA1, B4096). Newer
  Versal AI Edge DPUs with per-channel scales may behave differently.
- **yolov11s float is not strictly better than yolov11n float.**
  yolov11n has higher mAP@0.5:0.95 (0.97 vs 0.915) on the eggs valid
  set. The choice of yolov11s for int8 deployment trades a small
  float-precision penalty for substantial int8 robustness.
- **Hard-negative training is necessary but not sufficient.** All
  three models needed hard-negative training to reach 0 float FPs;
  none reached 0 int8 FPs at conf=0.85.

---

## Hard-negative training workflow

For industrial deployments where background false positives matter,
augmenting the training set with hard-negative images is the standard
intervention. This section documents the procedure used for the v0.10
eggs deployment.

### What counts as a hard-negative image

An image that contains no instances of the target class but is visually
similar to environments where the target class would normally appear.
For the eggs deployment: conveyor belts, packaging machinery, plastic
baskets, cardboard cartons — anything that triggered false positives
on real industrial footage.

The labels for hard-negative images are **empty** (zero objects).
Ultralytics YOLOv11 handles empty labels correctly; the loss penalizes
any positive predictions on these images.

### Dataset assembly

The eggs+hardneg dataset used for v0.10:

```
data/datasets/eggs_hardneg/
├── train/
│   ├── images/         (933 eggs + 402 hardneg = 1335 images)
│   └── labels/         (933 eggs labels + 402 empty .txt files)
├── valid/              (32 eggs validation, unchanged)
└── data.yaml           (single class "egg", nc=1)
```

The 402 hard-negative images came from Roboflow's "Production Line
Package Tracking v8i" public dataset. Selection criteria:

1. Conveyor belts / industrial backgrounds matching the deployment scene
2. No egg-shaped objects in frame
3. Resolution close to the eggs training set

After merging, the empty `.txt` label files (one per hard-negative
image) ensure Ultralytics' dataloader includes them in training without
counting them toward any class.

### Training command

Identical to a normal yolov11 training run; just point at the augmented
dataset's `data.yaml`:

```bash
python3 scripts/host/_train_yolov11.py \
    --weights yolo11n.pt \
    --data    data/datasets/eggs_hardneg/data.yaml \
    --output  data/weights/yolo11n_eggs_dpu.pt \
    --epochs  50 --batch 16 --imgsz 640
```

### Calibration set composition warning

The temptation is to also include hard-negative images in the
calibration set (so the quantizer "knows about them"). For DPU int8 on
this DPU hardware, **this typically makes false positives worse**, not
better. The reason is the per-tensor activation scale constraint
(documented in "Capacity vs quantization" above): mixing domains
widens scale ranges and increases per-layer noise.

Empirical observation from v0.10:
- Calibration with 600 eggs-only images → 4,220 FPs @ 0.85
- Calibration with 300 eggs + 300 hardneg → 6,609 FPs @ 0.85 (+57%)

The safer default is to calibrate with in-domain (eggs-only) images
even if the training set is augmented. If you must mix, evaluate
side-by-side before deploying.

---

## ONNX deployment path — investigated, not deployable

This section documents an alternative deployment path that was
investigated in v0.10 and found not to be usable for this YOLOv11
architecture in the current toolchain. Included for thesis completeness;
if you're not trying to use the ONNX path, skip this section.

### Motivation

The default compile flow uses Vitis-AI's NNDCT (PyTorch-native) PTQ.
Vitis-AI 3.5 also ships `vai_q_onnx`, an ONNX-based PTQ path with two
theoretical advantages:

1. **Per-channel weight quantization** (NNDCT is per-tensor only)
2. **VitisQuantFormat options** (FixNeuron vs QDQ for different
   downstream compilers)

Per-channel weight quantization in particular can reduce quantization
error on layers with high weight variance — exactly the issue motivating
the v0.10 false-positive investigation.

### Scripts

Two helpers were added in v0.10:

| Script | Role |
|---|---|
| `scripts/host/_export_onnx_yolov11.py` (+`.sh`) | PyTorch → clean FP32 ONNX (~14 MB) |
| `scripts/host/_quantize_onnx_yolov11.py` (+`.sh`) | ONNX → int8 → xmodel via `vai_q_onnx` |

The export script applies the same monkey-patches as `_train_yolov11.py`
plus the compile-time transformations (SiLU swap, NHWC wrap, head
stripping), then validates the exported ONNX against the PyTorch reference
to <1e-4 maximum absolute difference. **This step works reliably and is
reusable** for any future ONNX-based deployment work.

### Investigation outcome

`vai_q_onnx.quantize_static()` fails on the YOLOv11 graph in
`align_concat`, an internal refinement pass:

```
TypeError: '<' not supported between instances of 'NoneType' and 'int'
  at: pass_align_concat
```

The failure persists regardless of:
- `per_channel` setting (True or False)
- `quant_format` (FixNeuron or QDQ)
- `N_CALIB` value (50, 200, 600)
- with or without `optimize_model=True`

Additionally, even if the PTQ step succeeded, the DPU hardware uses
per-tensor activation scales — which negates the per-channel
quantization advantage that motivated the ONNX path in the first place.

### Decision

The ONNX path is not deployable for this YOLOv11 architecture in
`vai_q_onnx 1.14.0`. Scripts retained in-tree because:

1. The export step is independently useful (ONNX is a portable
   intermediate)
2. A future Vitis-AI release may fix the `align_concat` issue
3. The graph itself is cross-validated, so anyone exploring alternative
   ONNX-based deployment paths (e.g., onnxruntime-DPU, OpenVINO, other
   FPGA toolchains) has a clean starting point

---

## Known limitations

### Quantization noise on small models

As documented in "Capacity vs quantization" above, int8 PTQ on
yolov11n produces a large false-positive rate on fine-grained
single-class detection tasks (egg detection on industrial backgrounds).
The mitigation is yolov11s or larger; raising the confidence threshold
to 0.85+ also helps but at the cost of recall on hard true positives.

### Architecture-induced accuracy ceiling (vs stock YOLOv11)

Replacing softmax attention with element-wise HardSigmoid gating
reduces the model's selectivity for spatial patterns where attention
would normally focus on specific anchor positions. For most detection
tasks this isn't an issue (detection is largely a per-cell
classification problem). For tasks with heavy spatial relationships
(multi-object association, attribute detection conditioned on spatial
relationships) the modified architecture may underperform stock
YOLOv11. The eggs task doesn't exercise these patterns so the ceiling
isn't visible.

If you need full-fidelity attention, deploy via
`vitis_ai_library.GraphRunner` on a multi-subgraph compile of the stock
YOLOv11n — this preserves the attention but is significantly slower
(CPU↔DPU boundaries at every attention block). This pipeline doesn't
currently support that path.

### Quantization-Aware Training (QAT)

QAT was attempted in early v0.9 development. The intent was to narrow
the float-vs-int8 gap by simulating int8 quantization inside the
training loop. The path proved difficult to integrate with Ultralytics'
training loop (Vitis-AI's `pytorch_nndct.QatProcessor` expects a
PyTorch-native training loop, and Ultralytics' trainer adds its own
forward hooks and module wrapping). Abandoned in favor of the PTQ +
hard-negative + capacity approach documented above, which proved
effective without requiring training-pipeline integration.

For future work: a clean QAT integration would either patch
Ultralytics' `BaseTrainer` to expose the right hooks, or train via a
custom loop that bypasses the Ultralytics trainer entirely.

---

## File reference

| File | Role |
|---|---|
| `lpr_pipeline/c2psa_dpu.py` | DPU-friendly `C2PSA_DPU` class and constructors |
| `lpr_pipeline/detect_dpu.py` | `apply_dwconv_monkey_patch()` function |
| `lpr_pipeline/compile/yolov11.py` | Family-specific compile module (delegates to yolov5) |
| `lpr_pipeline/compile/registry.py` | Maps `family="yolov11"` to the compile module |
| `lpr_pipeline/shared/models.py` | `yolov11n` and `yolov11s` registry entries; both status "full" |
| `lpr_pipeline/deploy/decoders.py` | `decode_yolov11()` alias of `decode_yolov5u()` |
| `lpr_pipeline/deploy/preprocess.py` | `Preprocessor` accepts `family="yolov11"` |
| `lpr_pipeline/deploy/runner.py` | `ModelRunner` family→decoder dispatch |
| `scripts/host/_train_yolov11.py` | User-facing training entry point |
| `scripts/host/_export_onnx_yolov11.py` (+`.sh`) | PyTorch → ONNX export (v0.10) |
| `scripts/host/_quantize_onnx_yolov11.py` (+`.sh`) | ONNX PTQ via `vai_q_onnx` (v0.10, blocked) |
| `scripts/host/02_compile.sh` | Accepts `yolov11` as family argument; works for both variants |
| `scripts/host/03_sync_to_kria.sh` | Generic — works for any compiled model |
| `scripts/kria/run_live.sh` | Dispatches `yolov11n` and `yolov11s` to the eggs notebook |
| `notebooks/eggs/05_deploy_visual.ipynb` | Live eggs detection demo |

---

## Troubleshooting

### "Verification reports ✗ WARNING: monkey-patches may not have held"

The training rebuilt the model from YAML but the rebuild didn't see the
patches. Most common cause: running the script with a Python path that
doesn't include the repo root. Make sure you run from the repo root
(`cd ~/Documents/Girona_Masters/Thesis/KriaKv260_Model_Compiler` or
wherever your clone lives) and that `lpr_pipeline/` is a sibling
directory.

### Compile fails with "Can't get attribute 'C2PSA_DPU'"

The Vitis-AI container can't find the surgery module. Verify:

```bash
ls lpr_pipeline/c2psa_dpu.py        # must exist
docker run --rm -v "$(pwd):/workspace" xilinx/vitis-ai-pytorch-cpu:latest \
    bash -lc 'python3 -c "import lpr_pipeline.c2psa_dpu; print(\"OK\")"'
```

If the import fails, your `02_compile.sh` may have stripped `PYTHONPATH`
or the volume mount isn't right. Check `02_compile.sh` near the
`docker_args` array — it should set `-e PYTHONPATH=/workspace`.

### Compile produces multi-subgraph xmodel (>5 subgraphs)

Run the diagnostic to see what's fragmenting:

```bash
docker run --rm -v "$(pwd):/workspace" xilinx/vitis-ai-pytorch-cpu:latest \
    bash -lc '
        source /opt/vitis_ai/conda/etc/profile.d/conda.sh
        conda activate vitis-ai-pytorch
        python3 -c "
import xir
g = xir.Graph.deserialize(\"/workspace/out/yolov11n/yolov11n_kv260.xmodel\")
for sg in g.get_root_subgraph().toposort_child_subgraph():
    dev = sg.get_attr(\"device\") if sg.has_attr(\"device\") else \"?\"
    print(f\"  device={dev} ops={len(sg.get_ops())} name={sg.get_name()[:80]}\")
"
    '
```

Expected output:

```
device=USER ops=1
device=DPU ops=352
device=CPU ops=1    # fix2float
device=CPU ops=1    # fix2float
device=CPU ops=1    # fix2float
```

If you see additional CPU subgraphs with names containing `DWConv` or
`nndct_repeat`, the monkey-patches didn't take effect during training.
Re-train from scratch with `_train_yolov11.py` and re-verify the saved
model's architecture before recompiling.

### High int8 false-positive rate on industrial imagery

If you see thousands of false positives at conf=0.85 on background
machinery while the PyTorch float model is clean, the cause is
quantization noise on a model that's near the int8 capacity floor for
your task. Options in order of effectiveness:

1. **Switch to yolov11s** (or larger). See "Capacity vs quantization"
   above — typically 60-70% FP reduction with no detection-precision
   loss; ~33% throughput penalty.
2. **Verify calibration set composition.** Use in-domain images only
   (don't mix in hard-negatives at calibration time).
3. **Raise the deployment confidence threshold** (cheapest; works if
   your true positives have separable confidence from your false
   positives).
4. **Add hard-negative training images** (helps the float model; may
   or may not transfer to int8 — measure both).

### Detection quality on deployment differs from PyTorch reference

Confirm calibration data matches deployment domain. If you trained on
eggs and calibrated with license plates (or vice versa), this is the
most likely cause. Re-run `02_compile.sh` with calibration images that
match your deployment scenario.

If calibration matches and you still see major drift, the issue is
likely numerical precision loss from quantization. Mitigations:

- Raise inference confidence threshold (cheapest)
- Switch to a larger variant — yolov11s instead of yolov11n
- Add more calibration images (`N_CALIB=500 bash scripts/host/02_compile.sh ...`)
- Re-check calibration composition (in-domain only — see warning above)

### ONNX export script works but vai_q_onnx fails in align_concat

This is the documented v0.10 blocker on the ONNX deployment path; see
"ONNX deployment path — investigated, not deployable" above. The
NNDCT path (default `02_compile.sh`) is the working alternative for
this YOLOv11 architecture in `vai_q_onnx 1.14.0`.
