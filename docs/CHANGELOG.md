# Changelog

## v0.8.0 — YOLOv11 family support (2026-05-18)

### Added

- **YOLOv11 as a first-class supported family** (`family="yolov11"`).
  Promoted `yolov11n` from `status="stub"` to `status="full"`. See
  `docs/YOLOV11.md` for the full user guide.

- **`lpr_pipeline/c2psa_dpu.py`** — DPU-friendly replacement for Ultralytics'
  C2PSA attention module. Element-wise gating + HardSigmoid replaces
  matmul + softmax; conv-based channel selectors replace `chunk`/`split`;
  `torch.cat` replaces `tensor.repeat` (the latter emitted `nndct_repeat`
  which isn't in the XIR op set). Includes `from_original()` and
  `repair_from_legacy()` constructors for in-memory and pickle-recovery
  swaps.

- **`lpr_pipeline/detect_dpu.py`** — `apply_dwconv_monkey_patch()` function
  that rebinds `ultralytics.nn.modules.head.DWConv` to plain `Conv`. The
  Detect head's `cv3` branch then builds with plain conv blocks at YAML
  parse time, eliminating the CPU-subgraph fragmentation that YOLOv11n's
  depthwise convolutions produce on DPUCZDX8G_ISA1.

- **`lpr_pipeline/compile/yolov11.py`** — family-specific compile module.
  Thin subclass of `YOLOv5Compiler` that adds defensive imports of
  `lpr_pipeline.c2psa_dpu` and `lpr_pipeline.detect_dpu` (so pickle can
  resolve the surgery classes during `torch.load` inside the Vitis-AI
  container) and delegates to the parent's compile flow.

- **`scripts/host/_train_yolov11.py`** — user-facing training entry point.
  Applies both monkey-patches before YOLO construction, auto-detects GPU,
  resolves Roboflow-style relative dataset paths, locates the trained
  `best.pt` even when Ultralytics writes it to a non-default location,
  and verifies the saved model has zero `DWConv` modules.

- **`decode_yolov11()`** in `lpr_pipeline/deploy/decoders.py` — thin alias
  for `decode_yolov5u()`. YOLOv5u and YOLOv11 share the Detect head's
  output format exactly (anchor-free DFL, `reg_max=16`, same channel
  layout, same anchor convention), so the underlying math is identical.

- **`docs/YOLOV11.md`** — full user guide covering: when to use this
  family, why training is a separate step, the seven architectural
  modifications with rationale, the train/compile/sync/deploy workflow,
  result expectations, known limitations (PTQ accuracy gap, dataset
  bias, QAT as future work), and a troubleshooting section.

### Changed

- **`lpr_pipeline/compile/registry.py`** — added `"yolov11"` family
  dispatch entry.

- **`lpr_pipeline/shared/models.py`** — `Family` Literal extended to
  include `"yolov11"`; the `yolov11n` variant moved into its own family
  section with `status="full"` and updated notes describing the post-
  surgery architecture (3.6M params, single 352-op DPU subgraph,
  mAP@0.5 ≈ 0.99 on the egg validation set).

- **`lpr_pipeline/deploy/preprocess.py`** — `Preprocessor` now accepts
  `family="yolov11"`. YOLOv5u and YOLOv11 use identical input
  preprocessing (letterbox + BGR→RGB + [0,1] float32 + NHWC), so the
  same code path serves both.

- **`lpr_pipeline/deploy/runner.py`** — `ModelRunner._SUPPORTED_FAMILIES`
  extended to `("yolov5", "yolov11")`. Added `_DECODERS` dispatch dict
  mapping family → decoder function. Updated docstrings throughout.

- **`docs/MODELS.md`** — added YOLOv11 row to the families table and a
  per-family section between YOLOv5 and YOLOX.

### Why

YOLOv11 brings two improvements over YOLOv5: the C3k2 block in the
backbone/neck (more parameter-efficient than C3) and the C2PSA self-
attention module (which improves localization on cluttered scenes).
Supporting it on the KV260 required replacing both the attention block
(which uses matmul + softmax + chunk operations that the DPU can't
accelerate) and the Detect head's depthwise convolutions (which hit a
shape constraint of DPUCZDX8G_ISA1 and force CPU subgraphs). With the
surgery in place, YOLOv11n compiles to a single DPU subgraph and runs
at full hardware speed, like the existing YOLOv5 family.

### Validation

On the Roboflow `egg.v4` dataset (single class, 933 train / 32 val images):

- Trained model (post-surgery): mAP@0.5 = 0.995, mAP@0.5:0.95 = 0.97,
  3,600,083 parameters, 117 fused layers
- Compiled xmodel: 4.7 MB, single 352-op DPU subgraph + 3 output
  fix2float CPU ops, loads cleanly via `pynq_dpu.DpuOverlay.load_model()`
- DPU int8 detections match PyTorch float reference on the same input
  (both detect the same ~50 true positives; both share the same ~12
  background false positives that reflect dataset bias rather than
  quantization)

### Migration notes

For users with existing v0.7.x workflows: nothing changes. YOLOv5 and
YOLOX paths are untouched. The yolov11 family is purely additive.

If you have a previously-trained YOLOv11n `.pt` from a non-DPU path (e.g.,
stock Ultralytics training): you cannot use it directly with this
pipeline. The DPU-friendly architecture has different operations
(C2PSA_DPU instead of C2PSA, plain Conv instead of DWConv) that need to
be retrained from scratch with the substitutions in place. Use
`scripts/host/_train_yolov11.py` to produce a compatible model.

---

## v0.7 — VAI 3.5 benchmark + SD-card hardening (2026-05-11)

### Added

- **`scripts/host/04_stage_benchmark.sh`** — host-side staging wrapper
  for the VAI 3.5 model zoo benchmark. Downloads ~12 GB (34 models +
  COCO val2017 + VOC2007 test + ImageNetV2) into `build/benchmark_stage/`.
- **`scripts/host/_stage_benchmark.py`** — hardened Python downloader:
  fsync every 16 MB, atomic `.part` → final rename with directory fsync,
  HTTP Range resume support, size verification, atomic state log,
  certifi-first SSL with unverified fallback.
- **`scripts/host/05_sync_benchmark_to_kria.sh`** — rsync wrapper that
  pushes staged data to the Kria. Mirrors the SSH-key-auth pattern from
  `03_sync_to_kria.sh`. Includes pre-flight remote disk check and
  post-sync verification (spot-checks for xmodels and dataset annotations).
- **`notebooks/04_vai35_benchmark.ipynb`** — VAI 3.5 model zoo benchmark
  notebook. Reads pre-staged data; does NO downloads. 33 cells covering:
  catalogue load, prerequisite check, smoke test, main 5-criteria
  benchmark, COCO mAP loop, VOC mAP loop, combined markdown report.
- **`docs/USAGE.md` §13** — documents the VAI 3.5 benchmark workflow.
- **`docs/TROUBLESHOOTING.md`** — new section on benchmark-workflow issues
  (SSL, permission, disk space, SD-card corruption recovery).
- **`docs/CHANGELOG.md`** — this file (was `CHANGELOG_pass6_final.md`
  in the repo root).

### Changed

- **All docs**: `scripts/host/01_compile.sh` references corrected to
  `scripts/host/02_compile.sh` (the actual script name).
- **`README.md`**: explicit `git clone https://github.com/abdullaadelaljaberi-debug/KriaKv260_Model_Compiler.git`
  URL added to the Quick start.
- **`docs/HOST_SETUP.md`**: GitHub URL placeholder corrected; disk space
  notes updated to include the benchmark workflow's 15 GB requirement.
- **`docs/KRIA_SETUP.md`**: GitHub URL placeholder `<your-username>`
  replaced with the actual repo URL.

### Removed

- **`APPLY_INSTRUCTIONS.md`** — one-time tarball-apply instructions from
  Pass 5 + 6 delivery; no longer relevant.
- **`tests/`** and **`docs/img/`** empty directories.

### Why

The previous in-notebook auto-download for the VAI 3.5 benchmark
corrupted a 256 GB SD card on the Kria under sustained writes. Consumer
SD card controllers handle sustained heavy I/O badly, and the corruption
window from the kernel write-back cache was too wide. Moved all
downloads to the laptop SSD, which is properly built for this load,
and reduced the Kria's role to a single rsync receive + read-only access
during the benchmark run.

The hardening primitives in `_stage_benchmark.py` (fsync, atomic rename,
resume, size verification) are now redundant on the host PC's SSD but
were preserved because they're cheap and harmless — if anyone ever runs
the staging on a flakier filesystem, they're covered.

---

## v0.6 — Pass 5 + Pass 6 final (Apr 2026)

# Pass 5 + Pass 6 final consolidation

This drop consolidates everything that was iteratively patched across Pass 5
(9 patches) and adds the final Pass 6 deliverables.

## File-by-file changes

### `scripts/kria/lib/common.sh`
- `vai_installed_version()` now reads `libvart` (not `libvart-runtime` — AMD's
  VAI 3.5 debs use the shorter name), with `libxir` and `.so` filename
  fallbacks.
- Summary-table infrastructure: `summary_init`, `summary_stage_start`,
  `summary_stage_done`, `summary_stage_skipped`, `summary_stage_failed`,
  `summary_print`, `summary_set_action`.
- `die()` records the failure in the summary table before exiting.

### `scripts/kria/01_install_vai35.sh`
- Stage 4 + Stage 5 rewritten to match AMD's reference script at
  `github.com/amd/Kria-RoboticsAI/files/scripts/install_update_kr260_to_vitisai35.sh`,
  adapted for KV260.
  - URL: `xilinx.com/bin/public/openDownload?filename=vai3.5_kr260.zip`
  - Bundled `setup.sh` for deb install (correct order; AMD-tested)
  - `lack_lib.tar.gz` extraction and copy to `/usr/lib/`
  - `xbutil2` copy to `/usr/bin/unwrapped/`
  - DPU-PYNQ clone + pip install into pynq-venv
  - LD_LIBRARY_PATH + xdputil + xrt env patches
- Per-stage `summary_stage_start` / `summary_stage_done` calls.
- `pynqutils` `download_overlays.py` patch (idempotent sed) to fix a real
  bug: `Device.devices[0]` raises `IndexError` on empty list during build.
- `pip install` runs inside a single `sudo bash -c "..."` shell so the
  `pynq_venv.sh` source survives into the python invocation.
- Stage 3 no longer tries `pynq-get-notebooks pynq_composable` (always
  fails with "No device found" pre-overlay-load).
- Explicit `exit 0` at end of script (so systemd doesn't see non-zero $?
  from `|| log_warn` clauses).

### `scripts/kria/02_apply_tuning.sh`
- Same summary-table instrumentation as 01.
- Explicit `exit 0` at end.

### `scripts/kria/03_install_systemd.sh`
- Explicit `exit 0` at end.

### `scripts/kria/run_live.sh` (most-changed)
- Accepts an optional 2nd arg: `text` (default) or `visual` — picks which
  notebook to open.
- Requires root upfront with a clear "use sudo" error message (PYNQ-DPU
  needs root to mmap FPGA configuration registers).
- Passes `--allow-root` to Jupyter when running as root.
- Auto-unloads `k26-starter-kits` via `xmutil unloadapp` so PYNQ can program
  the DPU bitstream into the empty PL. Without this, `DpuOverlay()` raises
  "No Devices Found".
- Sources `/etc/profile.d/pynq_venv.sh` and `/opt/xilinx/xrt/setup.sh`
  (if present) before exec'ing Jupyter — gets `XILINX_XRT=/usr` +
  `LD_LIBRARY_PATH=/usr/lib` into the kernel env. Sudo strips env by
  default; this replicates what AMD's `sudo su` would do.
- Exports `REPO_ROOT` so the notebook's path-detection picks it up.
- Removed the 09v2 fallback (now obsolete with our own notebooks shipping).

## New in Pass 6

### `lpr_pipeline/deploy/__init__.py`
Public API exports: `ModelRunner`, `Preprocessor`, `ThreadedCamera`,
`decode_yolov5u`, `unletterbox`, `draw_detections`, `draw_stats_overlay`.

### `lpr_pipeline/deploy/preprocess.py`
`Preprocessor` and `unletterbox`. Optimized hot path: replaced
`canvas.astype(float32) / 255.0` (allocates a temp every call) with
`np.multiply(..., out=out_f32[0], dtype=float32)` for a 2× speedup
empirically. Expected preprocess time on Kria: ~1-1.5 ms (down from 3.84 ms).

### `lpr_pipeline/deploy/decoders.py`
`decode_yolov5u`: anchor-free DFL decoder with softmax + projection +
NMS via OpenCV. Supports yolov5n and yolov5s (same head; different size).

### `lpr_pipeline/deploy/camera.py`
`ThreadedCamera`: BUFFERSIZE=4 + MJPG + 60 fps for the Brio. Cleanup-on-
init-failure (try/except around `__init__` body) so a failed construction
doesn't leak `/dev/video0`. First-frame timeout bumped from 1 s to 3 s to
accommodate USB camera mode-switching after re-enumeration.

### `lpr_pipeline/deploy/runner.py`
`ModelRunner`: wraps overlay + preprocessor + decoder. `infer()` returns
both detections (in camera-frame coords) and a per-stage timing dict
(`preprocess`/`dpu`/`decode` in ms).

### `lpr_pipeline/deploy/draw.py` (NEW)
`draw_detections(frame, dets, ...)`: in-place bounding-box rendering with
optional labels and confidence. Color-cycled palette indexed by class_idx.
`draw_stats_overlay(frame, fps=, inf_ms=, ...)`: optional FPS/latency
panel in the top-left corner of the frame.

### `notebooks/02_deploy_text.ipynb`
Live demo, max-throughput text-only HTML status. Same UX as the original
`02_deploy_live.ipynb` (which this replaces). 60 fps end-to-end on
yolov5n. Use for thesis benchmark numbers.

### `notebooks/03_deploy_visual.ipynb` (NEW)
Live demo with video preview + bounding boxes + interactive sliders for
conf/IOU/max_detections + toggles for labels/confidence/stats overlay.
40-50 fps end-to-end (display path is the bottleneck). Use for demos and
parameter exploration.

## Validation (May 2026)

On a fresh Ubuntu 22.04 LTS Kria KV260 SD card:
- 5/5 Pass 5 stages green in 2m 19s total install time
- Pass 6 text notebook: **60 fps live, 12.5 ms total per frame, 7.75 ms DPU**
- 694 detections across 3620 frames over a 60 s run (hit rate 19%)
