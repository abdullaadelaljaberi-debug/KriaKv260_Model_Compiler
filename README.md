# KriaKv260_Model_Compiler

End-to-end pipeline for compiling object-detection models into Vitis-AI 3.5
xmodels and deploying them to a Xilinx Kria KV260 board.

Built around the Logitech Brio USB camera, with a hook for AR1335 (J7) support
to be added later.

```
┌──────────────────────┐         ┌──────────────────────┐
│   Host PC (NVIDIA)   │  scp    │      Kria KV260      │
│                      │  ────>  │                      │
│  • PyTorch weights   │         │  • DPU bitstream     │
│  • Vitis-AI 3.5      │ xmodel  │  • PYNQ + VART 3.5   │
│    quantize+compile  │         │  • Camera + display  │
└──────────────────────┘         └──────────────────────┘
       (Pass 2-4)                       (Pass 5-6)
```

## Status

This is a thesis-companion pipeline. v1 ships:

- **Full pipeline support** for **YOLOv5** (Ultralytics u-variant) — proven
  on KV260 at 60 fps live with `yolov5n` license-plate detection
- **Full pipeline support** for **YOLOX** (Megvii) — multi-DPU-subgraph,
  uses `vitis_ai_library.GraphRunner`
- **Skeleton** support for **YOLOv7**, **YOLOv4-CSP**, **SSD-MobileNetV2-TF**
  — directory structure + class scaffolds in place; family-specific compile
  logic to be filled in later

## Quick start

### One-time host setup

```bash
git clone https://github.com/AbdullaAdel/KriaKv260_Model_Compiler.git
cd KriaKv260_Model_Compiler
bash scripts/host/00_check_prereqs.sh        # verify docker, nvidia, etc.
bash scripts/host/01_install_vai.sh           # pull Vitis-AI 3.5 docker
bash scripts/host/download_weights.sh         # pull pretrained LPR weights
```

See [`docs/HOST_SETUP.md`](docs/HOST_SETUP.md) for the full host install.

### One-time Kria setup

The Kria install is **wide scope**: it includes flashing the SD card from the
host PC. See [`docs/KRIA_SETUP.md`](docs/KRIA_SETUP.md).

### Compile a model

```bash
bash scripts/host/02_compile.sh yolov5 yolov5n \
     data/weights/yolov5n_lpr.pt \
     data/calib/ \
     out/yolov5n_kv260.xmodel
```

### Deploy to Kria

```bash
bash scripts/host/03_sync_to_kria.sh ubuntu@10.42.0.27 yolov5n
ssh ubuntu@10.42.0.27 'bash KriaKv260_Model_Compiler/scripts/kria/run_live.sh yolov5n'
```

See [`docs/USAGE.md`](docs/USAGE.md) for the full day-to-day workflow.

## Repository layout

```
KriaKv260_Model_Compiler/
├── docs/             Markdown documentation (host setup, Kria setup, usage, models)
├── notebooks/        Jupyter notebooks — visible workflow for thesis demo
├── scripts/host/     Bash helpers run on the development PC
├── scripts/kria/     Bash helpers run on the KV260 board
├── lpr_pipeline/     Python package — shared compile/deploy logic
├── data/             User data (weights, calibration images, eval images) — gitignored
└── tests/            Smoke tests (minimal in v1)
```

## Documentation

- [`docs/HOST_SETUP.md`](docs/HOST_SETUP.md) — one-time setup of the host PC (Docker, NVIDIA Container Toolkit, Vitis-AI image)
- [`docs/KRIA_SETUP.md`](docs/KRIA_SETUP.md) — one-time setup of the Kria board (SD flash, PYNQ, VAI 3.5 stack, camera)
- [`docs/USAGE.md`](docs/USAGE.md) — daily workflow (compile, deploy, demo, eval)
- [`docs/MODELS.md`](docs/MODELS.md) — supported model families, variants, conventions

## License

Apache 2.0 — see [`LICENSE`](LICENSE).

Compatible with AMD/Xilinx Vitis-AI (also Apache 2.0). Suitable for
academic, research, and commercial use; includes patent grant.

## Hardware tested

- AMD Xilinx Kria KV260 Vision AI Starter Kit
- Logitech Brio (USB 3.0)
- Host PC: x86_64 Linux (Ubuntu 22.04 / 24.04), NVIDIA GPU with CUDA 11.x

## Citation

If this pipeline supports your research, please cite the underlying frameworks:

```
@article{vitis_ai,
  title  = {Vitis AI},
  author = {Advanced Micro Devices, Inc.},
  url    = {https://github.com/Xilinx/Vitis-AI},
  year   = {2023}
}
```

(Replace with your thesis citation once published.)
