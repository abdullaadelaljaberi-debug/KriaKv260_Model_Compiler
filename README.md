# Kria KV260 Model Compiler

Object detection pipeline targeting the AMD Kria KV260 SOM. Compiles
PyTorch YOLO models (YOLOv5, YOLOv11) on a laptop via the Vitis AI 3.5
NNDCT toolchain and deploys them to the KV260's B4096 DPU for live
inference.

> *This is a working thesis project. Two end-to-end demos are validated:
> license plate detection with YOLOv5n at **60 FPS live, ~12 ms/frame**,
> and egg detection with YOLOv11n at **~25 FPS** / YOLOv11s at **~17 FPS**.*

## What this does

```
Laptop                                                  Kria KV260
──────                                                  ──────────

YOLOv5n/v5s, YOLOv11n/v11s (.pt PyTorch)
      │
      ▼
DPU-friendly surgery (SiLU→LeakyReLU; for v11: C2PSA_DPU + DWConv→Conv)
      │
      ▼
NNDCT quantize (INT8)                              Live camera + DPU inference
      │                                                      ▲
      ▼                                                      │
vai_c_xir compile                                  DpuOverlay + ModelRunner
      │                                                      ▲
      ▼                                                      │
*_kv260.xmodel ────────── rsync over SSH ────────► /home/ubuntu/xmodels_vai35/

```

Tested with:
- **Hardware**: Kria KV260 Vision AI Starter Kit + Logitech Brio
- **Host**: Ubuntu 20.04 / 22.04, Vitis AI 3.5 Docker (PyTorch CPU + GPU images)
- **Board**: Ubuntu 22.04 LTS + Kria-PYNQ 3.0 + Vitis AI 3.5 runtime
- **Validated tasks**:
  - License plate detection (single class), YOLOv5n + YOLOv5s
  - Egg detection on industrial conveyor (single class), YOLOv11n + YOLOv11s

## Quick start

```bash
# Clone the repo (on both laptop and Kria)
git clone https://github.com/abdullaadelaljaberi-debug/KriaKv260_Model_Compiler.git
cd KriaKv260_Model_Compiler
```

If you already have a Kria with our scripts installed:

```bash
# Laptop — compile + sync (license plate / YOLOv5n)
bash scripts/host/02_compile.sh yolov5 yolov5n \
     data/weights/yolov5n_lpr.pt data/calib/
bash scripts/host/03_sync_to_kria.sh ubuntu@<kria-ip> yolov5n

# Kria — run live
sudo bash scripts/kria/run_live.sh yolov5n visual
```

For YOLOv11 (egg detection), training is a separate step because the
architecture needs DPU-friendly modifications. See
[**docs/YOLOV11.md**](docs/YOLOV11.md) for the workflow.

```bash
# Laptop — compile + sync (egg detection / YOLOv11s)
NUM_CLASSES=1 bash scripts/host/02_compile.sh yolov11 yolov11s \
     data/weights/yolo11s_eggs_dpu.pt data/calib_v2_hardneg/
bash scripts/host/03_sync_to_kria.sh ubuntu@<kria-ip> yolov11s

# Kria — run live
sudo bash scripts/kria/run_live.sh yolov11s
```

Then open the URL the script prints in your laptop's browser.

If you're starting from a fresh Kria SD card, see
[**docs/KRIA_SETUP.md**](docs/KRIA_SETUP.md).

## Documentation

| Doc | When to read |
|---|---|
| [**docs/KRIA_SETUP.md**](docs/KRIA_SETUP.md) | One-time install on a fresh SD card |
| [**docs/HOST_SETUP.md**](docs/HOST_SETUP.md) | One-time install on your laptop (Vitis AI Docker, NVIDIA, etc.) |
| [**docs/USAGE.md**](docs/USAGE.md) | Daily workflow + adding new variants/families |
| [**docs/MODELS.md**](docs/MODELS.md) | Supported model families + how to add new ones |
| [**docs/YOLOV11.md**](docs/YOLOV11.md) | YOLOv11-specific workflow + architecture rationale + capacity findings |
| [**docs/TROUBLESHOOTING.md**](docs/TROUBLESHOOTING.md) | Every issue we've hit, with forensic detail |
| [**docs/CHANGELOG.md**](docs/CHANGELOG.md) | Version history |
| [**docs/vai35_benchmark_report.md**](docs/vai35_benchmark_report.md) | VAI 3.5 model zoo benchmark results (33 models, current reference) |
| [**docs/vai25_vs_vai35_comparison.md**](docs/vai25_vs_vai35_comparison.md) | Cross-runtime comparison: VAI 2.5 → 3.5 reproducibility + methodology evolution |
| [**docs/vai25_benchmark_report.md**](docs/vai25_benchmark_report.md) | VAI 2.5 benchmark results (historical baseline, v0.4 era) |

## Performance (as of `v0.11.0`, 2026-05)

| Model | Task | Pure DPU (ms) | End-to-end (FPS) | Notes |
|---|---|---:|---:|---|
| YOLOv5n | License plate | 7.74 | 60 (camera-bound) | Original validation; primary throughput demo |
| YOLOv11n | Egg detection | ~38 | ~26 (synthetic input) | After DPU-friendly architecture surgery; int8 eggs F1 = 0.863 |
| YOLOv5s (eggs) | Egg detection | ~49 | ~20 (synthetic input) | Architecture comparison: 9.1M params but **F1 0.673**, demonstrates capacity ≠ int8 robustness |
| YOLOv11s | Egg detection | ~58 | ~17 (synthetic input) | 3.7× params vs YOLOv11n, 67% fewer int8 FPs, **best int8 eggs F1 0.842** |

See [KRIA_SETUP.md §11](docs/KRIA_SETUP.md#11-validated-performance) for
the YOLOv5n per-stage breakdown and [YOLOV11.md](docs/YOLOV11.md) for
the capacity-vs-quantization comparison.

## VAI 3.5 model zoo benchmark (separate workflow)

For benchmarking the AMD VAI 3.5 model zoo (~34 pre-compiled classification
+ detection models against COCO val2017, VOC2007 test, and ImageNetV2),
the workflow is host-driven to keep heavy downloads off the Kria's SD card:

```bash
# Laptop: download + stage data (~12 GB, ~60 min)
bash scripts/host/04_stage_benchmark.sh

# Laptop: push to Kria
bash scripts/host/05_sync_benchmark_to_kria.sh ubuntu@<kria-ip>

# Kria: run the benchmark
sudo bash scripts/kria/run_live.sh yolov5n
# Then open notebooks/04_vai35_benchmark.ipynb in the browser
```

See [USAGE.md §13](docs/USAGE.md#13-vai-35-model-zoo-benchmark) for details.

## Repo layout

```
scripts/
  host/                # laptop-side: compile + sync + benchmark staging + ONNX export
  kria/                # board-side: install + tune + run

lpr_pipeline/
  shared/models.py     # ModelSpec registry (yolov5n/s, yolov11n/s, yolox_*)
  c2psa_dpu.py         # DPU-friendly C2PSA replacement (YOLOv11)
  detect_dpu.py        # DWConv → Conv monkey-patch (YOLOv11)
  compile/             # host-only: PyTorch → ONNX → quantize → xmodel
  deploy/              # board-only: xmodel → live detections

notebooks/
  01_compile.ipynb              # optional walk-through of the compile pipeline
  02_deploy_text.ipynb          # max-throughput text-mode live demo (YOLOv5)
  03_deploy_visual.ipynb        # visual live demo with bounding boxes + sliders (YOLOv5)
  04_vai35_benchmark.ipynb      # VAI 3.5 model zoo benchmark (host-staged data)
  eggs/05_deploy_visual.ipynb   # eggs interactive demo (YOLOv11)

docs/                  # this directory
```

See [USAGE.md §12](docs/USAGE.md#12-whats-where) for a fuller tour.

## Supported models

| Variant | Family | Status | DPU latency (ms) | Notes |
|---|---|---|---:|---|
| yolov5n | YOLOv5 | ✓ validated end-to-end | 7.74 | License plate demo (60 FPS camera-bound) |
| yolov5s | YOLOv5 | ✓ compiles + runs | ~15-20 (est.) | Not benchmarked live |
| yolov11n | YOLOv11 | ✓ validated end-to-end | ~38 | Egg detection demo; DPU-friendly surgery applied |
| yolov11s | YOLOv11 | ✓ validated end-to-end | ~58 | Capacity-vs-quantization study (v0.10) |
| yolox_tiny | YOLOX | spec only | — | Decoder + runner branch not yet implemented |
| yolox_nano | YOLOX | spec only | — | Same |

Adding a new variant within an existing family: edit
[`lpr_pipeline/shared/models.py`](lpr_pipeline/shared/models.py), drop
the weights at the expected path, run `scripts/host/02_compile.sh`. See
[USAGE.md §10](docs/USAGE.md#10-deep-dive-adding-a-new-yolov5-variant)
for the YOLOv5 path, [YOLOV11.md](docs/YOLOV11.md) for the YOLOv11 path
(which requires DPU-friendly training).

## License

Apache License 2.0 — see [LICENSE](LICENSE) for the full text.

In short: you can use, modify, and distribute this code freely, including
in commercial work, as long as you preserve the copyright notice and the
license file. If you modify the code, mark the files you changed.

## Acknowledgments

Built on top of AMD/Xilinx's [Kria-PYNQ](https://github.com/Xilinx/Kria-PYNQ),
[DPU-PYNQ](https://github.com/Xilinx/DPU-PYNQ), and
[Kria-RoboticsAI](https://github.com/amd/Kria-RoboticsAI). The Vitis AI 3.5
upgrade procedure is adapted from AMD's reference scripts; documented in
detail in [KRIA_SETUP.md](docs/KRIA_SETUP.md).
