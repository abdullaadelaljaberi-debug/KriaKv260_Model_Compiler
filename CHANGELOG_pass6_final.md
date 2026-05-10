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
