# Usage

Daily workflow for the KriaKv260 model compiler / deployment pipeline.

This document assumes you've already set up the Kria via
[KRIA_SETUP.md](./KRIA_SETUP.md). If not, do that first.

## Contents

1. [Hello-world smoke test](#1-hello-world-smoke-test)
2. [Daily-driver workflow](#2-daily-driver-workflow)
3. [Compiling a model (laptop side)](#3-compiling-a-model-laptop-side)
4. [Syncing to the Kria](#4-syncing-to-the-kria)
5. [Running the live demo](#5-running-the-live-demo)
6. [Text vs visual mode](#6-text-vs-visual-mode)
7. [Switching models](#7-switching-models)
8. [Reading the performance numbers](#8-reading-the-performance-numbers)
9. [Common failures + recovery](#9-common-failures--recovery)
10. [Deep dive: adding a new YOLOv5 variant](#10-deep-dive-adding-a-new-yolov5-variant)
11. [Deep dive: adding a new model family](#11-deep-dive-adding-a-new-model-family)
12. [What's where](#12-whats-where)

## 1. Hello-world smoke test

After a fresh install or a reboot, run this to verify everything works
end-to-end in ~2 minutes:

```bash
# On the Kria. Assumes you've already synced a yolov5n xmodel.
cd ~/KriaKv260_Model_Compiler
sudo bash scripts/kria/run_live.sh yolov5n text
```

Then in your laptop's browser, open the URL printed by the script. In
JupyterLab:

1. Open `notebooks/02_deploy_text.ipynb`
2. Run cells 1-10 (configuration through ModelRunner warmup)
3. Check that cell 10 prints something like:
   ```
   ModelRunner built:
     input  dims = [1, 320, 320, 3]
     output[0] = [1, 40, 40, 65]
     output[1] = [1, 20, 20, 65]
     output[2] = [1, 10, 10, 65]
   ```
   The `65` is `4 × reg_max(16) + nc(1)` — confirms the xmodel and spec
   match. Warmup times should stabilize around 12-13 ms by the third run.

If both checks pass, the install is healthy. You can stop here (Kernel
→ Interrupt) without running the live loop. Total time: ~2 minutes
including kernel startup.

If anything fails, see [TROUBLESHOOTING.md](./TROUBLESHOOTING.md).

## 2. Daily-driver workflow

Once everything is set up, the day-to-day workflow is:

```bash
# 1. On laptop: maybe edit training/compile config, retrain, etc.
# Then compile:
bash scripts/host/02_compile.sh yolov5 yolov5n data/weights/yolov5n_lpr.pt data/calib/

# 2. On laptop: sync the new xmodel to the Kria
bash scripts/host/03_sync_to_kria.sh ubuntu@10.42.0.27 yolov5n

# 3. On Kria: run the live demo
sudo bash scripts/kria/run_live.sh yolov5n visual

# 4. In your laptop's browser: open the URL printed by the script,
#    click Run All in the notebook
```

No SSH tunnel. No second terminal on the laptop. The Kria's IP is
auto-discovered and printed in the launch banner.

## 3. Compiling a model (laptop side)

```bash
bash scripts/host/02_compile.sh <family> <variant> <weights.pt> <calib_dir>
```

Where `<variant>` is one of the entries in
[`lpr_pipeline/shared/models.py`](../lpr_pipeline/shared/models.py)
(`yolov5n`, `yolov5s`, `yolox_tiny`, `yolox_nano`).

The script:

1. Loads the trained `.pt` weights
2. Auto-replaces SiLU activations with LeakyReLU(0.1015625) if needed
   (the DPU doesn't support SiLU natively; LeakyReLU is the closest
   supported activation and the slope `0.1015625` is the closest
   representable value to `0.1` in the DPU's fixed-point format)
3. Strips the Detect head for export and emits a raw multi-scale conv
   output (the head's CPU-side decode is what `decode_yolov5u` does at
   runtime)
4. Exports to ONNX
5. Quantizes via NNDCT to INT8 with calibration data from
   `data/calib_images/`
6. Compiles to xmodel via `vai_c_xir --arch /opt/vitis_ai/compiler/arch/DPUCZDX8G/KV260/arch.json`

Output: `xmodels_vai35/<variant>/<variant>_kv260.xmodel`

> The script is **idempotent** — if you re-run it with the same weights,
> it skips finished stages. Force a fresh compile with `REBUILD=1` env
> var. Force re-quantization but not re-compile with `REQUANT=1`.

> **SiLU swap opt-out**: if your trained model already uses LeakyReLU
> (e.g., the `leakyrelu.pt` weights in `models/yolov5n/`), set
> `SWAP_ACTIVATIONS=false` to skip the auto-swap. The pipeline detects
> SiLU/LeakyReLU counts automatically and only swaps when SiLU is
> dominant.

## 4. Syncing to the Kria

```bash
bash scripts/host/03_sync_to_kria.sh <user@host> <variant>
```

Example:

```bash
bash scripts/host/03_sync_to_kria.sh ubuntu@10.42.0.27 yolov5n
```

This rsyncs `xmodels_vai35/<variant>/` from your laptop to
`/home/ubuntu/xmodels_vai35/<variant>/` on the Kria. If you've set up
SSH key auth (recommended — see [KRIA_SETUP.md §4](./KRIA_SETUP.md#recommended-set-up-ssh-key-auth)),
the sync is silent. Otherwise you'll be prompted for the Kria's password.

Only the `.xmodel` file is needed at runtime. The quantized ONNX and
intermediate artifacts stay on your laptop.

## 5. Running the live demo

```bash
sudo bash scripts/kria/run_live.sh <variant> [text|visual]
```

The second argument picks the notebook. Default is `text`.

The script prepares the FPGA + Jupyter environment:

- Verifies running as root (PYNQ-DPU needs root for FPGA mmap)
- Verifies the xmodel and notebook exist
- Re-applies CPU governor if needed
- Unloads `k26-starter-kits` (default boot firmware) so PYNQ can program
  the DPU bitstream
- Sources `/etc/profile.d/pynq_venv.sh` so the Jupyter kernel inherits
  `XILINX_XRT=/usr` and `LD_LIBRARY_PATH=/usr/lib`
- Launches `jupyter lab` bound to `0.0.0.0:8888` with `--allow-root`

The terminal stays attached to Jupyter. To stop: Ctrl-C twice in the
Kria terminal.

> **Cross-LAN access**: by default Jupyter binds to all interfaces. The
> printed URL contains a single-use token (~48 chars); anyone on your
> LAN with that token can access the notebook. For thesis/private
> network use this is fine. For semi-public networks set
> `JUPYTER_HOST=127.0.0.1` (via `sudo -E`) to revert to localhost-only,
> then SSH-tunnel from your laptop.

## 6. Text vs visual mode

Two notebooks ship, designed for different use cases:

### `notebooks/02_deploy_text.ipynb` (text mode)

- **What it shows**: HTML status block that refreshes once per second.
  Detection events stream below as text (throttled to once per 0.4 s).
- **No per-frame rendering** — no JPEG encode, no video widget update.
- **Use for**: thesis benchmark numbers, headless validation runs, when
  you need "the model runs at X fps" measurements that aren't degraded
  by the display path.
- **Typical end-to-end FPS**: 60 (camera-bound at the Brio's rate).

### `notebooks/03_deploy_visual.ipynb` (visual mode)

- **What it shows**: live MJPG video feed with bounding boxes overlaid
  on each frame. Plus the same text status block above the video.
- **Interactive controls** below the video — sliders for `conf`, `iou`,
  `max_detections`; toggles for `show_labels`, `show_confidence`, and
  `show_stats_overlay` (FPS/latency on the video itself); a **■ Stop**
  button.
- **Per-frame rendering cost**: 5-8 ms for JPEG encode + widget update.
- **Use for**: demos, presentations, parameter exploration, visually
  debugging a specific detection.
- **Typical end-to-end FPS**: 40-50 (display path is the bottleneck;
  DPU compute time is unchanged from text mode).

You can run both during one Kria session — just stop one (Ctrl-C
JupyterLab) and start the other.

## 7. Switching models

In the notebook, edit cell 2 (Configuration), change `VARIANT` from
e.g. `yolov5n` to `yolov5s`, **restart the kernel** (Kernel → Restart),
and re-run cells from the top.

You must restart because:

1. The DPU runtime caches metadata for the previously-loaded xmodel
2. `overlay.runner` is bound to whatever xmodel was last `load_model`'d
3. The preprocessor's pre-allocated buffers are sized for the previous
   imgsz

Restart is fast (~3 seconds).

When launching via `run_live.sh`, the second argument sets `LPR_VARIANT`
which the notebook's cell 2 reads:

```bash
sudo bash scripts/kria/run_live.sh yolov5s visual
```

So you don't even need to edit the notebook — just launch with the
right variant name.

## 8. Reading the performance numbers

### In the text notebook's status block

```
yolov5n  elapsed=  30.0s  frames=  1800
inf_fps= 60.0  cam_fps= 60.0  theoretical_max= 80.5 fps
pre=0.41  dpu=15.62  dec=0.59  (total=16.62 ms)
detections=42  hit_rate= 14.5%
```

| Field | Meaning | Bound by |
|---|---|---|
| `inf_fps` | End-to-end inferences per second | camera or DPU, whichever is slower |
| `cam_fps` | Unique frames delivered by the camera | hardware (Brio @ 60 fps) |
| `theoretical_max` | `1000 / (pre + dpu + dec)` — pipeline ceiling | DPU + CPU work, no camera |
| `pre` | Preprocess time (letterbox + RGB + /255 to float32) | CPU |
| `dpu` | DPU compute time | FPGA |
| `dec` | Decode time (DFL softmax + NMS) | CPU |
| `total` | Sum of the three stages — the actual per-frame time | — |
| `detections` | Cumulative detections (one frame can have multiple) | model recall |
| `hit_rate` | % of inferred frames that found at least one detection | model precision under thresholds + content |

### In the visual notebook's status block

Same fields, plus:

- `display_ms` — JPEG encode + widget update time per frame

If `display_ms` is significant (>5 ms) and `inf_fps < cam_fps`, the
display path is your bottleneck. That's normal for visual mode.

### Reportable numbers for the thesis

| Claim | Source |
|---|---|
| "Pure inference latency (DPU only)" | Cell 6 in text notebook, `dpu` field of pure benchmark |
| "End-to-end inference latency (DPU + CPU pre/post)" | Cell 6, `total` field |
| "End-to-end throughput, live camera" | Cell 14 final stats, `inference fps` |
| "Theoretical max throughput (pipeline)" | Cell 6, `1000 / total mean` |

Don't mix them. The text notebook's measurements are clean; the visual
notebook's are display-degraded.

## 9. Common failures + recovery

Quick reference. For full forensic detail see [TROUBLESHOOTING.md](./TROUBLESHOOTING.md).

| Symptom | Recovery |
|---|---|
| `No Devices Found` from DpuOverlay | `sudo xmutil unloadapp` then retry. `run_live.sh` does this automatically since `v0.6` |
| `Root permissions required` | Launch via `sudo bash run_live.sh ...`, not `bash run_live.sh ...` |
| `cannot open camera 0` | `lsusb \| grep -i logitech` to confirm it's enumerated; if not, replug |
| `no frames received in 3s` | `sudo bash scripts/kria/02_apply_tuning.sh` to re-tune v4l2 settings |
| Live demo at 15 fps instead of 60 | USB autosuspend kicked back in; re-run the tuning script |
| Jupyter URL says `Connection refused` | Kernel restarted Jupyter or you closed the terminal; relaunch via `run_live.sh` |
| Tuning script reports `Device or resource busy` | Jupyter still holds the camera; Kernel → Restart in Jupyter, then re-run tuning |
| Kria can't find the xmodel | Did you `bash scripts/host/03_sync_to_kria.sh ...` from the laptop? |

## 10. Deep dive: adding a new YOLOv5 variant

Suppose you train a `yolov5m` for LPR and want it in the pipeline.

### 10.1 Add the spec

Edit [`lpr_pipeline/shared/models.py`](../lpr_pipeline/shared/models.py):

```python
"yolov5m": ModelSpec(
    name="yolov5m",
    family="yolov5",
    weights="models/yolov5m/training/weights/best.pt",
    imgsz=320,
    nc=1,
    reg_max=16,
),
```

### 10.2 Make sure the weights exist

Put the trained `.pt` at `models/yolov5m/training/weights/best.pt`.

### 10.3 Compile

```bash
bash scripts/host/02_compile.sh yolov5 yolov5m data/weights/yolov5m_lpr.pt data/calib/
```

The pipeline will:

1. Detect that the model has SiLU activations (Ultralytics default)
2. Auto-swap to LeakyReLU(0.1015625)
3. Strip the Detect head, export to ONNX
4. Quantize via NNDCT (uses `data/calib_images/`)
5. Compile to xmodel

Output: `xmodels_vai35/yolov5m/yolov5m_kv260.xmodel`

### 10.4 Sync + run

```bash
# Laptop
bash scripts/host/03_sync_to_kria.sh ubuntu@10.42.0.27 yolov5m

# Kria
sudo bash scripts/kria/run_live.sh yolov5m visual
```

### 10.5 Performance expectations

yolov5m is much larger than yolov5n. Expect DPU latency to roughly
triple (from ~7.7 ms to ~20-30 ms) and end-to-end FPS to drop from
camera-bound (60) to DPU-bound (~30-40). For a thesis comparison, that's
exactly the interesting tradeoff to measure.

## 11. Deep dive: adding a new model family

YOLOX is partially scaffolded in `lpr_pipeline.shared.models` but the
deploy side needs work:

### Required additions

| Component | Status |
|---|---|
| Compile pipeline for YOLOX | Not implemented in this repo (the YOLOX `.pt` would need its own export path, since the head structure differs from YOLOv5u) |
| `lpr_pipeline.deploy.preprocess.Preprocessor` for YOLOX | Currently raises `NotImplementedError` — needs the int8 right-shifted path (see the docstring; YOLOX's DPU input is `uint8 >> 1` viewed as `int8`) |
| `lpr_pipeline.deploy.decoders.decode_yolox` | Not yet present. YOLOX emits a permuted `[1, 3549, 6]` output (cx, cy, log_w, log_h, sigmoid(obj), sigmoid(cls)). Decoder needs grid-cell projection. |
| `lpr_pipeline.deploy.runner.ModelRunner` YOLOX branch | YOLOX is multi-subgraph after VAI compilation. PYNQ-DPU's `overlay.runner` only handles single-subgraph; YOLOX needs `vitis_ai_library.GraphRunner.create_graph_runner(graph)` instead. |

The previous-generation notebook (`09_yolov5n_final_v2.ipynb`, not
shipped in this repo but preserved in earlier branches) has working
YOLOX code that can be ported. See its `Preprocessor.process` for the
int8 path and `decode_yolox` for the grid-cell decode.

### Why we deferred YOLOX

Three reasons:

1. yolov5n already meets the thesis throughput targets (60 fps live)
2. YOLOX's multi-subgraph compilation pushed install complexity higher
   than the marginal FPS gain justified
3. Adding YOLOX would mean diverging the notebook structure (different
   runner type), and we wanted Pass 6 to ship clean

If you want YOLOX support, the route is:
1. Reinstate `decode_yolox` from the previous notebook into `decoders.py`
2. Extend `Preprocessor` with the int8 right-shift path
3. Refactor `ModelRunner` to dispatch between `overlay.runner` and
   `GraphRunner` based on `spec.family`

Each is ~1-2 hours of work but they're separable, so you can do them
one at a time without breaking the YOLOv5 path.

## 12. What's where

```
scripts/
  host/                     ← laptop-side workflow
    02_compile.sh           ← .pt → xmodel
    03_sync_to_kria.sh      ← rsync xmodel to board
  kria/                     ← board-side workflow
    01_install_vai35.sh     ← one-time install (Pass 5)
    02_apply_tuning.sh      ← runtime tuning (called by systemd unit)
    03_install_systemd.sh   ← install the systemd unit
    run_live.sh             ← launch Jupyter for the live demo
    lib/common.sh           ← shared logging, summary table, helpers

lpr_pipeline/
  shared/                   ← used by both host and Kria
    models.py               ← ModelSpec registry: get_spec(name)
  compile/                  ← host-only: PyTorch → xmodel
    yolov5.py               ← Detect head stripping, SiLU swap
    quantize.py             ← NNDCT quantize wrapper
    ...
  deploy/                   ← Kria-only: xmodel → live detections
    __init__.py             ← public exports
    preprocess.py           ← letterbox + RGB + /255 → float32
    decoders.py             ← decode_yolov5u (DFL softmax + NMS)
    camera.py               ← ThreadedCamera (BUFFERSIZE=4, MJPG, 60fps)
    runner.py               ← ModelRunner — ties it all together
    draw.py                 ← bounding boxes + stats overlay

notebooks/
  01_compile.ipynb          ← host: walk through a compile (optional)
  02_deploy_text.ipynb      ← Kria: max-throughput text-mode live demo
  03_deploy_visual.ipynb    ← Kria: visual live demo with sliders

docs/
  KRIA_SETUP.md             ← one-time install on a fresh SD card
  USAGE.md                  ← this file
  TROUBLESHOOTING.md        ← every issue we've hit, with detail
  img/                      ← screenshots (TODO: live demo screenshot)

models/                     ← trained weights (.pt, not git-tracked)
xmodels_vai35/              ← compiled xmodels (host side, not git-tracked)
data/                       ← calibration + eval images (not git-tracked)
```

## Next: troubleshooting

When something specific breaks, [TROUBLESHOOTING.md](./TROUBLESHOOTING.md)
has the deep forensic detail on every issue we've encountered during
development.

## 13. VAI 3.5 model zoo benchmark

A separate workflow from the LPR pipeline above: benchmarks AMD's
pre-compiled VAI 3.0 KV260 model binaries (34 classification + detection
models) against COCO val2017, VOC2007 test, and ImageNetV2. Useful for
thesis comparison numbers ("how does our LPR-tuned yolov5n compare to
the stock model zoo on standard benchmarks?").

### Why this workflow is host-driven

The benchmark needs ~12 GB of downloads (models + datasets). An earlier
version did the downloads on the Kria itself, which corrupted a 256 GB
SD card under sustained writes. Consumer SD card controllers handle
sustained heavy I/O badly. The current workflow does all downloads on
the laptop's SSD instead and pushes the result to the Kria via rsync.

### One-time setup

```bash
# On laptop — ~60 min of mostly download time
cd ~/Documents/Girona_Masters/Thesis/KriaKv260_Model_Compiler

bash scripts/host/04_stage_benchmark.sh
```

This downloads:
- ~9 GB of VAI 3.0 pre-compiled KV260 xmodels (34 models)
- ~1.3 GB ImageNetV2 (10,000 labeled images, MIT licensed)
- ~1.0 GB COCO val2017 (images + annotations)
- ~430 MB VOC2007 test set

Output: `build/benchmark_stage/`. The `build/` directory is gitignored.

The script:
- Refuses to start unless ≥15 GB free disk
- Uses fsync after every 16 MB to bound data loss on crashes
- Resumes interrupted downloads via HTTP Range
- Verifies download sizes
- Falls back to unverified SSL if the system CA bundle is incomplete

Interrupt with Ctrl-C and re-run to resume — completed items skip.

### Push to the Kria

```bash
# Still on laptop
bash scripts/host/05_sync_benchmark_to_kria.sh ubuntu@10.42.0.27
```

This rsyncs the staged data to
`/home/ubuntu/KriaKv260_Model_Compiler/notebooks/`. The script:
- Sets up SSH key auth on first run (one password prompt)
- Pre-flight checks remote disk space
- Uses `rsync --inplace --partial` (resumable)
- Spot-checks key files on the remote after completion
- Excludes intermediate cache dirs (`_downloads/`, the extracted source
  tree for ImageNetV2 — only the symlinked staging dir is synced)

Add `--dry-run` to preview without transferring:

```bash
bash scripts/host/05_sync_benchmark_to_kria.sh ubuntu@10.42.0.27 --dry-run
```

### Run the benchmark on the Kria

```bash
# On Kria
sudo bash scripts/kria/run_live.sh yolov5n
# (variant doesn't matter; we just need Jupyter + DPU access)
```

Then in your laptop's browser, open `notebooks/04_vai35_benchmark.ipynb`
and run cells top to bottom.

The notebook:
1. Verifies prerequisites (the data you just rsync'd is in place)
2. Loads the model catalogue (34 entries)
3. Smoke-tests each model (1 inference) — excludes broken ones from full run
4. Runs the 5-criteria benchmark (DPU FPS, latency, power, accuracy, camera FPS)
5. Computes COCO mAP for detection models on val2017
6. Computes VOC mAP for VOC-trained models on VOC2007 test
7. Generates a combined markdown report

CSV outputs (gitignored):
- `vai35_smoke_test.csv` — which models pass the smoke test
- `vai35_benchmark_results.csv` — main 5-criteria results
- `vai35_coco_map_results.csv` — COCO mAP per model
- `vai35_voc_map_results.csv` — VOC mAP per model
- `vai35_benchmark_report.md` — combined markdown report

Full benchmark run takes ~3-5 hours depending on how many models pass
the smoke test.

### Re-running with updated data

If you need to refresh datasets or models on the Kria:

```bash
# Laptop: re-download (will skip what's already there)
bash scripts/host/04_stage_benchmark.sh

# Laptop: re-sync (rsync only transfers changed files)
bash scripts/host/05_sync_benchmark_to_kria.sh ubuntu@10.42.0.27
```

Both scripts are fully idempotent — running them on a clean system or
on a fully-staged system both behave correctly.

### Re-running with subset

To download only a specific model (e.g. when iterating on the catalogue):

```bash
bash scripts/host/04_stage_benchmark.sh --only resnet50
```

To skip the models or datasets:

```bash
bash scripts/host/04_stage_benchmark.sh --skip-datasets
bash scripts/host/04_stage_benchmark.sh --skip-models
```

### Where things end up

| Path on laptop | Path on Kria | Contents |
|---|---|---|
| `build/benchmark_stage/Models_VAI35/` | `notebooks/Models_VAI35/` | 34 model directories |
| `build/benchmark_stage/Datasets/imagenet_sample/` | `notebooks/Datasets/imagenet_sample/` | 10K symlinked images + labels.txt |
| `build/benchmark_stage/Datasets/coco_val2017/` | `notebooks/Datasets/coco_val2017/` | 5K COCO images + annotations |
| `build/benchmark_stage/Datasets/voc2007_test/` | `notebooks/Datasets/voc2007_test/` | 5K VOC images + annotations |
| `build/benchmark_stage/Datasets/imagenet_class_index.json` | (same path on Kria) | ImageNet 1000-class taxonomy |

The `notebooks/Models_VAI35/` and `notebooks/Datasets/` directories on
the Kria are gitignored — they exist only after the sync.
