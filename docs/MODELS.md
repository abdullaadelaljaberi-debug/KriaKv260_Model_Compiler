# Supported model families

The pipeline supports five object-detection families targeting Vitis-AI 3.5
on the Kria KV260's B4096 DPU. Each family has its own compile path because
of differences in quantization conventions, head structure, and decoder logic.

| Family | Variants | Status | Decoder | Notes |
|---|---|---|---|---|
| **YOLOv5** (Ultralytics u-variant) | `yolov5n`, `yolov5s` | ✅ full | DFL anchor-free | Single DPU subgraph; uses `pynq_dpu.DpuOverlay.runner` |
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
- **Head**: anchor-free with Distribution Focal Loss (DFL); stripped for VAI
  (no inline decode/NMS — done on CPU after DPU)
- **Output channels per cell**: `4*reg_max + nc` where `reg_max=16`
- **Three output tensors** at strides 8, 16, 32 — shapes `[1, H, W, C]`
- **Preprocessing**: BGR → RGB, divide by 255 → float32 in `[0, 1]`
- **Calibration set**: any folder of representative images (UC3M-LP for the
  LPR demo)
- **Demo**: `yolov5n` for license plate detection, ~60 fps live (camera-bound)

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
3. Run `bash scripts/host/02_compile.sh <family> <name>` — the existing
   compile pipeline handles all variants of a supported family

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
