# Supported model families

The pipeline supports six object-detection families targeting Vitis-AI 3.5
on the Kria KV260's B4096 DPU. Each family has its own compile path because
of differences in quantization conventions, head structure, and decoder logic.

| Family | Variants | Status | Decoder | Notes |
|---|---|---|---|---|
| **YOLOv5** (Ultralytics u-variant) | `yolov5n`, `yolov5s`, `yolov5s_eggs` | ✅ full | DFL anchor-free | Single DPU subgraph; uses `pynq_dpu.DpuOverlay.runner` |
| **YOLOv11** (Ultralytics) | `yolov11n`, `yolov11s` | ✅ full | DFL anchor-free | Requires architectural surgery at training time; see `docs/YOLOV11.md`. Single DPU subgraph after surgery. |
| **YOLOX** (Megvii) | `yolox_tiny`, `yolox_nano` | ✅ full | sigmoid + grid | Multi-DPU-subgraph; uses `vitis_ai_library.GraphRunner` |
| **YOLOv7** (WongKinYiu) | `yolov7-tiny` | 🚧 stub | anchor-based | Compile path scaffolded; family-specific code TBD |
| **YOLOv4-CSP** | `yolov4_csp` | 🚧 stub | anchor-based | Compile path scaffolded; family-specific code TBD |
| **SSD-MobileNetV2-TF** | `ssd_mobilenet_v2_coco` | 🚧 stub | TF-style box decode | Compile path scaffolded; family-specific code TBD |

✅ = end-to-end working with the included LPR demo
🚧 = directory structure + Python class skeleton present; raises `NotImplementedError`
       on `compile()`. Drop-in xmodels from AMD's model zoo will *deploy* on
       the Kria via the existing runner — the gap is only the host-side compile.

## Per-family details

### YOLOv5 (✅ full)

- **Source**: [ultralytics/yolov5](https://github.com/ultralytics/yolov5)
- **VAI 3.5 zoo entry**: `pt_yolov5_v6_640_640_3.5`
- **Variants in this pipeline**:
  - `yolov5n` — input 320×320, ~4 GMACs, fastest (~12 ms inference on KV260)
  - `yolov5s` — input 320×320, ~7 GMACs, ~19 ms inference
  - `yolov5s_eggs` — input 640×640, ~9.1M params, trained on eggs+hardneg for the
    capacity-vs-architecture comparison (see `docs/YOLOV11.md` § "Capacity vs
    architecture"). ~49 ms inference, 20.4 FPS. Demonstrates that within-family
    architectural choices matter more than capacity for int8 robustness.
- **Head**: anchor-free with Distribution Focal Loss (DFL); stripped for VAI
  (no inline decode/NMS — done on CPU after DPU)
- **Output channels per cell**: `4*reg_max + nc` where `reg_max=16`
- **Three output tensors** at strides 8, 16, 32 — shapes `[1, H, W, C]`
- **Preprocessing**: BGR → RGB, divide by 255 → float32 in `[0, 1]`
- **Calibration set**: any folder of representative images (UC3M-LP for the
  LPR demo)
- **Demo**: `yolov5n` for license plate detection, ~60 fps live (camera-bound)
- **Architectural modifications applied at compile time** (no retraining
  required): `SiLU → LeakyReLU(0.1015625)` via
  `lpr_pipeline.compile.yolov5._swap_silu_to_leakyrelu`. The Detect head's
  inline decode/NMS is stripped; decoding is done CPU-side after the DPU
  returns raw conv outputs. An NHWC permute wrapper matches the DPU's
  native input layout.

### YOLOv11 (✅ full)

- **Source**: [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics) (modern Ultralytics packaging covers YOLOv8, v9, v10, v11)
- **VAI 3.5 zoo entry**: none — YOLOv11 post-dates VAI 3.5's release; this pipeline
  ships its own compile path
- **Variants in this pipeline**:
  - `yolov11n` — input 640×640, ~3.6M params after DPU-friendly surgery, single
    352-op DPU subgraph on KV260 B4096, ~38 ms DPU latency
  - `yolov11s` — input 640×640, ~13.5M params after DPU-friendly surgery (3.7×
    yolov11n), single DPU subgraph, ~58 ms DPU latency. Added in v0.10 for
    the capacity-vs-quantization study; produces 67% fewer int8 false
    positives than yolov11n on industrial test imagery (see
    `docs/YOLOV11.md`)
- **Head**: anchor-free with Distribution Focal Loss (DFL), structurally
  identical to YOLOv5u; same `reg_max=16` and channel layout, same
  `_strip_detect_head_for_quant()` and same `decode_yolov5u()` decoder
- **Architectural modifications applied at training time**:
  - `C2PSA` → `C2PSA_DPU` (HardSigmoid + conv-based gating replaces
    softmax-based attention)
  - `DWConv` → plain `Conv` in the Detect head's `cv3` branch
  - These modifications are NOT mathematically equivalent to the originals;
    retraining is required. See `docs/YOLOV11.md` for the full operator-level
    breakdown and the rationale.
- **Output channels per cell**: `4*reg_max + nc` where `reg_max=16` — identical
  to YOLOv5u, so decoder is reused (`decode_yolov11()` is a thin alias)
- **Three output tensors** at strides 8, 16, 32 — shapes `[1, 80, 80, C]`,
  `[1, 40, 40, C]`, `[1, 20, 20, C]` at imgsz=640
- **Preprocessing**: identical to YOLOv5u (BGR → RGB, divide by 255 → float32
  in `[0, 1]`); `Preprocessor("yolov11", 640)` is supported
- **Training workflow**: use `scripts/host/_train_yolov11.py` — it applies the
  monkey-patches before any `YOLO()` construction, trains, and produces a
  DPU-ready `.pt`
- **Calibration data**: must match deployment domain. For the egg detection
  demo, calibration uses egg training images; using mismatched calibration
  (e.g., LPR images for an egg model) causes significant quantization drift.
  Calibrating with a *mixture* of in-domain and out-of-domain images can
  also hurt — see "calibration set composition" in `docs/YOLOV11.md`.
- **Hard-negative training**: for industrial deployments where background
  false positives matter, augment the training set with labeled empty
  frames (e.g., conveyor / machinery images with no target object). The
  v0.10 eggs demo uses 402 hard-negative images (Roboflow Production Line
  Package Tracking v8i) merged with 933 eggs images. See
  `docs/YOLOV11.md` for the workflow.
- **Validated on**: egg detection (Roboflow `egg.v4` dataset, 933 train /
  32 val images, single class) — mAP@0.5 = 0.995 in PyTorch, comparable
  detection quality after int8 quantization

### YOLOX (✅ full)

- **Source**: [Megvii-BaseDetection/YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)
- **VAI 3.5 zoo entry**: `pt_yolox_nano_TT100K_416_416_3.5`
- **Variants in this pipeline**:
  - `yolox_tiny` — input 416×416, ~92 ms inference (4 DPU subgraphs)
  - `yolox_nano` — input 416×416, ~680 ms inference (34 DPU subgraphs;
    structurally slow due to depthwise convolutions fragmenting the graph)
- **Head**: decoupled cls/obj/reg; we strip `decode_in_inference` so the
  graph emits a single `[1, 3549, 6]` output (3549 = 52² + 26² + 13²)
- **Output channels**: `(raw_cx, raw_cy, log_w, log_h, sigmoid(obj), sigmoid(cls))`
  — `cx/cy/w/h` are raw offsets; we finish decode on CPU with grid + stride
- **Input quantization**: `fix_point=-1`, equivalent to `int8 = uint8 / 2`
  (right-shift by 1)
- **Preprocessing**: BGR uint8 letterbox; no scaling beyond the int8 quantization
- **Multi-subgraph runner**: `vitis_ai_library.GraphRunner.create_graph_runner(graph)`
  — required because PYNQ-DPU's `overlay.load_model()` asserts single DPU subgraph

### YOLOv7 (🚧 stub)

- **Source**: [WongKinYiu/yolov7](https://github.com/WongKinYiu/yolov7)
- **VAI 3.5 zoo entry**: `pt_yolov7_640_640_3.5`
- **Planned variant**: `yolov7-tiny`
- **Head**: anchor-based (3 anchors × 3 scales); requires anchor box
  application + sigmoid + per-class softmax during decode
- **Stub status**: directory + class skeleton exist; `compile()` raises
  `NotImplementedError`; deployment runner has decoder stub

### YOLOv4-CSP (🚧 stub)

- **Source**: [AlexeyAB/darknet](https://github.com/AlexeyAB/darknet) +
  PyTorch port
- **VAI 3.5 zoo entry**: `pt_yolov4_csp_512_512_3.5`
- **Planned variant**: `yolov4_csp`
- **Head**: anchor-based, similar to YOLOv7 but slightly different anchor
  conventions and CIoU loss in training

### SSD-MobileNetV2-TF (🚧 stub)

- **Source**: TensorFlow Object Detection API
- **VAI 3.5 zoo entry**: `tf_ssdmobilenetv2_coco_300_300_3.5`
- **Planned variant**: `ssd_mobilenet_v2_coco`
- **Head**: prior-box based, classes-first output shape; needs `vai_q_tensorflow2`
  for quantization (different toolchain than the PyTorch families above)
- **Important difference**: this family uses TensorFlow rather than PyTorch
  for the source model, so the compile path will be substantially different

## Adding a new variant within a family

For YOLOv5 / YOLOX, adding e.g. `yolov5m` or `yolox_s` requires:

1. Train (or download) the variant weights to `data/weights/<name>.pt`
2. Add an entry to `lpr_pipeline/shared/models.py` with the right `imgsz`,
   `nc`, etc.
3. Run `bash scripts/host/02_compile.sh <family> <name> <weights> <calib>` —
   for example, `bash scripts/host/02_compile.sh yolov5 yolov5m data/weights/yolov5m.pt data/calib/`.
   The existing compile pipeline handles all variants of a supported family.

For YOLOv11, adding e.g. `yolov11m` requires the same three steps PLUS
re-training with `scripts/host/_train_yolov11.py` because the
DPU-friendly surgery (C2PSA_DPU, DWConv→Conv) is not weight-compatible
with stock Ultralytics checkpoints. The yolov11s entry added in v0.10
demonstrates the pattern.

## Adding a new family from scratch

1. Read the matching VAI 3.5 model zoo entry to understand head structure
2. Create `lpr_pipeline/compile/<family>.py` extending `BaseCompiler`
3. Implement the head-stripping, calibration, and quantization steps for
   that family's framework (PyTorch vs TensorFlow)
4. Implement `lpr_pipeline/deploy/decoders.py:decode_<family>`
5. Add to `lpr_pipeline/shared/models.py` registry
6. Update `MODELS.md` with the new entry

## DPU compatibility note

Every xmodel is compiled for the KV260's B4096 DPU at fingerprint
`0x101000056010407`. xmodels compiled for a different DPU configuration
(e.g., B1024, B2304, or VAI 2.5's older fingerprint `0x101000017010407`)
will load but fail at execution time with a fingerprint-mismatch error.
The compile pipeline always targets B4096 / VAI 3.5; reconfiguring is
out of scope.

## Activation function policy

The KV260's DPUCZDX8G has limited hardware support for activation functions:

- ✅ **ReLU**, **ReLU6**, **LeakyReLU(0.1015625)** — fully accelerated on the DPU
- ❌ **SiLU** (Swish), **GELU**, **Mish**, custom — must run on CPU

Most modern object detection models default to SiLU, which causes a problem
on the DPU: each SiLU op forces a CPU subgraph, and large models can fragment
into 100+ tiny DPU+CPU pieces. Two consequences:

1. Inference runs 2-4× slower (CPU↔DPU transfer overhead)
2. `pynq_dpu.DpuOverlay.load_model()` refuses to load multi-subgraph xmodels
   — its `assert len(subgraphs) == 1` fails. You'd need
   `vitis_ai_library.GraphRunner` (the multi-subgraph runner used for YOLOX)

The pipeline addresses this by automatically swapping SiLU → LeakyReLU(0.1015625)
before quantization. The slope `0.1015625` (= 13/128) is the only value the
DPU supports natively; using it directly avoids the train-vs-deploy numerical
drift that the quantizer's auto-correction would otherwise introduce.

### Four ways to handle SiLU in your trained model

In order of resulting accuracy:

**Option A — train with LeakyReLU from the start (best accuracy)**

Modify your training script to swap SiLU → LeakyReLU(0.1015625) before
training begins. The model adapts to LeakyReLU during training. Recommended
for new projects. See `02_train_yolov5.ipynb` in the upstream lpr_thesis
repo for an example that does exactly this.

**Option B — fine-tune after swap (good compromise)**

Take a SiLU-trained checkpoint, swap activations, fine-tune for 5-10 epochs
on the same dataset. Usually recovers >95% of the original accuracy.
Recommended when retraining from scratch is impractical but a short
fine-tune is feasible.

**Option C — swap at compile time (default; small accuracy hit)**

This is what `02_compile.sh` does by default. The model's weights were
tuned for SiLU; replacing the activation introduces error throughout the
network. For YOLOv5 the SiLU and LeakyReLU(0.1) curves are similar enough
that the accuracy hit is usually 1-3 mAP@0.5 points. Acceptable when
accuracy isn't tightly constrained.

**Option D — keep SiLU, deploy with multi-subgraph runner (exact accuracy, much slower)**

Disable the swap with the `SWAP_ACTIVATIONS=false` env var:

```bash
SWAP_ACTIVATIONS=false bash scripts/host/02_compile.sh yolov5 yolov5n \
    data/weights/best.pt data/calib/
```

The xmodel will fragment into many DPU+CPU subgraphs. Won't load via
`pynq_dpu.overlay.load_model()`. Must use `vitis_ai_library.GraphRunner`
(the deploy path used for YOLOX). Inference will be 2-4× slower than the
swapped version due to CPU↔DPU bouncing.

Rarely the right choice. Use only when exact accuracy preservation
matters more than inference speed, and you're prepared to use the
multi-subgraph deploy runner.
