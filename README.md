# Kria KV260 Model Compiler

License plate recognition pipeline targeting the AMD Kria KV260 SOM.
Compiles PyTorch YOLOv5 models on a laptop (Vitis AI 3.5 NNDCT toolchain)
and deploys them to the KV260's B4096 DPU for live camera inference.

> *This is a working thesis project, not a polished library — but the
> pipeline reproducibly hits **60 fps live with ~12 ms per frame** on
> yolov5n.*

## What this does

```
Laptop                              Kria KV260
──────                              ──────────
yolov5n.pt (PyTorch)                Live camera + DPU inference
   │                                          ▲
   ▼                                          │
Auto-swap SiLU → LeakyReLU                  Compiled xmodel
   │                                          │
   ▼                                          │
NNDCT quantize (INT8)               ──────────┘
   │                                rsync over SSH
   ▼
vai_c_xir compile
   │
   ▼
yolov5n_kv260.xmodel ───────────────┐
```

Tested with:
- **Hardware**: Kria KV260 Vision AI Starter Kit + Logitech Brio
- **Host**: Ubuntu 20.04, Vitis AI 3.5 Docker
- **Board**: Ubuntu 22.04 LTS + Kria-PYNQ 3.0 + Vitis AI 3.5 runtime
- **Target task**: single-class license plate detection (extensible to
  multi-class via the model spec registry)

## Quick start

If you already have a Kria with our scripts installed:

```bash
# Laptop: compile + sync
bash scripts/host/01_compile.sh yolov5n
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
| [**docs/USAGE.md**](docs/USAGE.md) | Daily workflow + adding new variants/families |
| [**docs/TROUBLESHOOTING.md**](docs/TROUBLESHOOTING.md) | Every issue we've hit, with forensic detail |

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

## Repo layout

```
scripts/
  host/                # laptop-side: compile + sync
  kria/                # board-side: install + tune + run

lpr_pipeline/
  shared/models.py     # ModelSpec registry (yolov5n, yolov5s, yolox_*)
  compile/             # host-only: PyTorch → ONNX → quantize → xmodel
  deploy/              # board-only: xmodel → live detections

notebooks/
  01_compile.ipynb         # optional walk-through of the compile pipeline
  02_deploy_text.ipynb     # max-throughput text-mode live demo
  03_deploy_visual.ipynb   # visual live demo with bounding boxes + sliders

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
the weights at the expected path, run `scripts/host/01_compile.sh`. See
[USAGE.md §10](docs/USAGE.md#10-deep-dive-adding-a-new-yolov5-variant).

## License

(Whatever your existing README states.)

## Acknowledgments

Built on top of AMD/Xilinx's [Kria-PYNQ](https://github.com/Xilinx/Kria-PYNQ),
[DPU-PYNQ](https://github.com/Xilinx/DPU-PYNQ), and
[Kria-RoboticsAI](https://github.com/amd/Kria-RoboticsAI). The Vitis AI 3.5
upgrade procedure is adapted from AMD's reference scripts; documented in
detail in [KRIA_SETUP.md](docs/KRIA_SETUP.md).
