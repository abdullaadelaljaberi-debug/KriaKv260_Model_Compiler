# YOLOv11 on KV260 — User Guide

This document covers everything needed to take a stock Ultralytics YOLOv11n
model and deploy it to the Kria KV260 DPU through this pipeline.

It's longer than `docs/YOLOV5.md` because YOLOv11 needs architectural
modifications that YOLOv5n doesn't. Once you understand the additional
training step, the compile and deploy stages are identical to other
supported families.

---

## TL;DR (the three commands)

```bash
# 1. Train a DPU-friendly YOLOv11n on your dataset (once per dataset)
python3 scripts/host/_train_yolov11.py \
    --weights stock_yolo11n.pt \
    --data    my_dataset/data.yaml \
    --output  data/weights/my_model_dpu.pt \
    --epochs  50 --batch 16

# 2. Compile to xmodel (standard pipeline)
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

## The seven architectural modifications

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
   native XIR operation. For repeat factor 2 (the case in YOLOv11n):
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

For a YOLOv11n trained on the egg dataset (single class, 933 train / 32
val images) with default `_train_yolov11.py` settings:

| Metric | Value |
|---|---|
| Trained-model params | 3,600,083 (vs 2,590,035 stock; +1M from the surgery) |
| Trained-model layers (fused) | 117 |
| mAP@0.5 on val | 0.995 |
| mAP@0.5:0.95 on val | 0.97 |
| Quantized xmodel size | 4.7 MB |
| DPU fingerprint | 0x101000056010407 (KV260 B4096, VAI 3.5) |
| Subgraph count | 5 (1 USER input + 1 DPU with 352 ops + 3 CPU output fix2float) |
| Loadable via `DpuOverlay.load_model()` | yes |

The compile produces a single 352-op DPU subgraph carrying the entire
backbone, neck, modified attention, and detection conv layers. The three
output `fix2float` CPU operations are standard int8→float32 dequantization
boundaries that every quantized model has.

---

## Detailed workflow

### Step 1: produce a DPU-friendly trained model

```bash
python3 scripts/host/_train_yolov11.py \
    --weights yolo11n.pt \           # or yolo11n.yaml for from-scratch
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

GPU is strongly recommended. On an RTX A2000 8GB: 50 epochs at batch 16
on a ~900-image single-class dataset takes ~12 minutes. CPU-only training
on the same dataset takes 5+ hours.

### Step 2: compile to xmodel

```bash
NUM_CLASSES=1 bash scripts/host/02_compile.sh yolov11 yolov11n \
    data/weights/my_model_dpu.pt data/calib/
```

Note `yolov11` (the family) is now a valid first argument, as of v0.8.0.

The compile pipeline:

1. Activates the Vitis-AI 3.5 Docker container (the script auto-selects
   CPU or GPU image)
2. Inside the container, runs the standard pipeline:
   - Load `.pt` (pickle finds `lpr_pipeline.c2psa_dpu.C2PSA_DPU` via the
     repo's `/workspace` mount)
   - SiLU → LeakyReLU activation swap
   - Strip Detect head's inline post-processing
   - Wrap with NHWC permute
   - Calibrate with up to 200 images from `data/calib/`
   - Quantize via `pytorch_nndct`
   - Compile via `vai_c_xir` for KV260 B4096
3. Final xmodel written to `out/yolov11n/yolov11n_kv260.xmodel`

**Important: calibration data quality matters.** The quantizer learns
activation ranges from calibration images. If your training data is eggs
and your calibration set is license plates (or vice versa), the
quantization scales will be wrong for the deployment domain and you'll
see false positives or missed detections. Always calibrate with images
that match your deployment data distribution. The simplest approach is
to put a representative sample of your training images in `data/calib/`.

Compile time depends on whether you have the GPU VAI image:

- GPU: ~1 minute end-to-end
- CPU: ~5 minutes end-to-end (still acceptable; result is bit-identical to GPU)

### Step 3: sync to Kria

```bash
bash scripts/host/03_sync_to_kria.sh ubuntu@10.42.0.189 yolov11n
```

This copies the xmodel to `/home/ubuntu/xmodels_vai35/yolov11n/yolov11n_kv260.xmodel`
on the Kria. The Kria-side scripts (`run_benchmark.sh`, `run_live.sh`)
look there.

### Step 4: deploy

Multiple options on the Kria:

```bash
# Live demo
bash scripts/kria/run_live.sh yolov11n

# Benchmark
bash scripts/kria/run_benchmark.sh yolov11n

# Or use ModelRunner directly from Python
python3 -c "
from pynq_dpu import DpuOverlay
from lpr_pipeline.shared.models import get_spec
from lpr_pipeline.deploy.runner import ModelRunner

overlay = DpuOverlay('dpu.bit')
runner = ModelRunner(
    get_spec('yolov11n'),
    '/home/ubuntu/xmodels_vai35/yolov11n/yolov11n_kv260.xmodel',
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

---

## Known limitations

### Quantization accuracy gap

On the egg dataset, PyTorch float and DPU int8 produce very similar
detections on the same image (the same ~50 true positives plus the same
~12 false positives on background machinery and baskets). The
quantization gap is small.

That said, **the false positives in both models reflect dataset bias** —
the Roboflow eggs dataset is mostly close-up belt shots with limited
background diversity, so the model never learned "what's not an egg" for
machinery/basket backgrounds. The fix is more diverse training data, not
better quantization.

If your application sees novel backgrounds, expect false positives at low
confidence thresholds. Two mitigations:

1. **Raise the confidence threshold** at deploy time. In the egg test
   case, raising from 0.25 to 0.85 cleanly separates true eggs (typically
   0.90+) from background false positives.

2. **Add hard-negative images to training**. Sample frames of just the
   conveyor / machinery / background without the target object and label
   them as "no objects". 100-300 such frames typically suppress most
   background false positives.

### Architecture-induced accuracy ceiling

Replacing softmax attention with element-wise gating reduces the model's
selectivity for spatial patterns where attention would normally focus on
specific anchor positions. For most detection tasks this isn't an issue
(detection is largely a per-cell classification problem). For tasks with
heavy spatial relationships (multi-object association, attribute
detection conditioned on spatial relationships) the modified architecture
may underperform stock YOLOv11. The egg task doesn't exercise these
patterns so the ceiling isn't visible.

If you need full-fidelity attention, deploy via `vitis_ai_library.GraphRunner`
on a multi-subgraph compile of the stock YOLOv11n — this preserves the
attention but is significantly slower (CPU↔DPU boundaries at every
attention block). This pipeline doesn't currently support that path.

### Quantization-Aware Training (QAT) — future work

The current pipeline uses post-training quantization (PTQ). For sensitive
applications, QAT typically narrows the float-vs-int8 gap by simulating
int8 quantization inside the training loop. Vitis-AI supports QAT via
`pytorch_nndct.QatProcessor`. Adding a QAT mode to `_train_yolov11.py`
is straightforward but hasn't been implemented yet (not blocking for the
v0.8.0 release).

---

## File reference

| File | Role |
|---|---|
| `lpr_pipeline/c2psa_dpu.py` | DPU-friendly `C2PSA_DPU` class and constructors |
| `lpr_pipeline/detect_dpu.py` | `apply_dwconv_monkey_patch()` function |
| `lpr_pipeline/compile/yolov11.py` | Family-specific compile module (delegates to yolov5) |
| `lpr_pipeline/compile/registry.py` | Maps `family="yolov11"` to the compile module |
| `lpr_pipeline/shared/models.py` | `yolov11n` registry entry; promotes status to "full" |
| `lpr_pipeline/deploy/decoders.py` | `decode_yolov11()` alias of `decode_yolov5u()` |
| `lpr_pipeline/deploy/preprocess.py` | `Preprocessor` accepts `family="yolov11"` |
| `lpr_pipeline/deploy/runner.py` | `ModelRunner` family→decoder dispatch |
| `scripts/host/_train_yolov11.py` | User-facing training entry point |
| `scripts/host/02_compile.sh` | Accepts `yolov11` as family argument |
| `scripts/host/03_sync_to_kria.sh` | Generic — works for any compiled model |

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

### Detection quality on deployment differs from PyTorch reference

Confirm calibration data matches deployment domain. If you trained on
eggs and calibrated with license plates (or vice versa), this is the
most likely cause. Re-run `02_compile.sh` with calibration images that
match your deployment scenario.

If calibration matches and you still see major drift, the issue is
likely numerical precision loss from quantization. Mitigations:

- Raise inference confidence threshold (cheapest)
- Add more calibration images (`N_CALIB=500 bash scripts/host/02_compile.sh ...`)
- Implement QAT (most effective; see "Future work")
