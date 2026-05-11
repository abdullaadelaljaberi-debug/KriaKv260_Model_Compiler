# Kria KV260 Model Compiler

License plate recognition pipeline targeting the AMD Kria KV260 SOM.
Compiles PyTorch YOLOv5 models on a laptop (Vitis AI 3.5 NNDCT toolchain)
and deploys them to the KV260's B4096 DPU for live camera inference.

> *This is a working thesis project, not a polished library — but the
> pipeline reproducibly hits **60 fps live with ~12 ms per frame** on
> yolov5n.*

## What this does

```
Laptop                                                  Kria KV260
──────                                                  ──────────

yolov5n.pt (PyTorch)
      │
      ▼
Auto-swap SiLU → LeakyReLU
      │
      ▼                                                      
NNDCT quantize (INT8)                              Live camera + DPU inference
      │                                                      ▲
      ▼                                                      │
vai_c_xir compile                                  DpuOverlay + ModelRunner
      │                                                      ▲
      ▼                                                      │
yolov5n_kv260.xmodel ───── rsync over SSH ─────►  /home/ubuntu/xmodels_vai35/


```

Tested with:
- **Hardware**: Kria KV260 Vision AI Starter Kit + Logitech Brio
- **Host**: Ubuntu 20.04, Vitis AI 3.5 Docker
- **Board**: Ubuntu 22.04 LTS + Kria-PYNQ 3.0 + Vitis AI 3.5 runtime
- **Target task**: single-class license plate detection (extensible to
  multi-class via the model spec registry)

## Quick start

```bash
# Clone the repo (on both laptop and Kria)
git clone https://github.com/abdullaadelaljaberi-debug/KriaKv260_Model_Compiler.git
cd KriaKv260_Model_Compiler
```

If you already have a Kria with our scripts installed:

```bash
# Laptop: compile + sync
bash scripts/host/02_compile.sh yolov5 yolov5n \
     data/weights/yolov5n_lpr.pt data/calib/
bash scripts/host/03_sync_to_kria.sh ubuntu@<kria-ip> yolov5n

# Kria: run live
sudo bash scripts/kria/run_live.sh yolov5n visual
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
| [**docs/TROUBLESHOOTING.md**](docs/TROUBLESHOOTING.md) | Every issue we've hit, with forensic detail |
| [**docs/CHANGELOG.md**](docs/CHANGELOG.md) | Version history |

## Performance (as of `v0.6-pass6-validated`, 2026-05)

| Metric | Value |
|---|---|
| Pure DPU inference (yolov5n, imgsz=320) | 7.74 ms / 129 fps |
| End-to-end pipeline (pre + DPU + decode) | 12.38 ms / 80 fps |
| Live camera throughput | 59.87 fps |
| Hit rate (60 s LPR run) | 18.98% across 3620 frames |

Camera-bound at 60 fps; DPU has ~30% spare capacity on yolov5n. See
[KRIA_SETUP.md §11](docs/KRIA_SETUP.md#11-validated-performance) for
full per-stage breakdown.

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
  host/                # laptop-side: compile + sync + benchmark staging
  kria/                # board-side: install + tune + run

lpr_pipeline/
  shared/models.py     # ModelSpec registry (yolov5n, yolov5s, yolox_*)
  compile/             # host-only: PyTorch → ONNX → quantize → xmodel
  deploy/              # board-only: xmodel → live detections

notebooks/
  01_compile.ipynb         # optional walk-through of the compile pipeline
  02_deploy_text.ipynb     # max-throughput text-mode live demo
  03_deploy_visual.ipynb   # visual live demo with bounding boxes + sliders
  04_vai35_benchmark.ipynb # VAI 3.5 model zoo benchmark (host-staged data)

docs/                  # this directory
```

See [USAGE.md §12](docs/USAGE.md#12-whats-where) for a fuller tour.

## Supported models

| Variant | Status | DPU latency (ms) | Notes |
|---|---|---|---|
| yolov5n | ✓ validated end-to-end | 7.74 | Recommended for camera-bound demos |
| yolov5s | ✓ compiles + runs | ~15-20 (est.) | Not yet benchmarked live |
| yolox_tiny | spec only | — | Decoder + runner branch not yet implemented |
| yolox_nano | spec only | — | Same |

Adding a new YOLOv5 variant: edit
[`lpr_pipeline/shared/models.py`](lpr_pipeline/shared/models.py), drop
the weights at the expected path, run `scripts/host/02_compile.sh`. See
[USAGE.md §10](docs/USAGE.md#10-deep-dive-adding-a-new-yolov5-variant).

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
