# Kria KV260 setup

End-to-end procedure to set up a Kria KV260 board for running compiled
xmodels from this repo. From a fresh SD card to a validated live demo
at 60 fps.

Audience: someone comfortable with Linux command-line, familiar with the
general idea of FPGA-accelerated inference, with no prior Kria experience.

If something fails partway through, see [TROUBLESHOOTING.md](./TROUBLESHOOTING.md).

## Contents

1. [What this gets you](#1-what-this-gets-you)
2. [Hardware prerequisites](#2-hardware-prerequisites)
3. [Flash the SD card](#3-flash-the-sd-card)
4. [First boot + SSH access](#4-first-boot--ssh-access)
5. [Clone the repo](#5-clone-the-repo)
6. [Run the install script](#6-run-the-install-script)
7. [Apply runtime tuning + persist it](#7-apply-runtime-tuning--persist-it)
8. [Sync a compiled xmodel from your laptop](#8-sync-a-compiled-xmodel-from-your-laptop)
9. [Run the live demo](#9-run-the-live-demo)
10. [Architecture](#10-architecture)
11. [Validated performance](#11-validated-performance)
12. [What can go wrong (and where to look)](#12-what-can-go-wrong-and-where-to-look)

## 1. What this gets you

After completing this guide, the Kria will:

- Boot Ubuntu 22.04 LTS with Kria-PYNQ and Vitis AI 3.5 runtime installed
- Have a tuned USB camera at 60 fps MJPG
- Run any xmodel compiled by this repo's `host/01_compile.sh` pipeline
- Expose JupyterLab on the LAN at port 8888 for browser access from your
  laptop
- Persist all runtime tuning across reboots via a systemd unit

## 2. Hardware prerequisites

You need:

- A Kria KV260 Vision AI Starter Kit (KV260 SOM + the carrier card it
  ships on)
- An SD card, ≥16 GB recommended. The install plus VAI 3.5 fills ~10 GB.
- A USB camera (this guide assumes a Logitech Brio for the tuning specs;
  any USB-3 UVC camera that supports MJPG at 640×480 60 fps will work
  with minor adjustments to `scripts/kria/02_apply_tuning.sh`)
- Ethernet cable + a way to share network from your laptop, **or** the
  Kria on a regular LAN with DHCP. NetworkManager's "Shared to other
  computers" mode works well for thesis development (gives the Kria a
  10.42.0.x address)
- Power supply for the Kria (12V, included with the Starter Kit)

## 3. Flash the SD card

Get the Ubuntu image:

```
https://ubuntu.com/download/amd
```

Pick **Ubuntu 22.04 LTS for AMD Kria KV260** (NOT KR260, KD240, or any
"Server" variant). Download the `.img.xz` file.

Flash with **Raspberry Pi Imager** (cross-platform, handles the .xz
decompression automatically):

```
sudo apt install rpi-imager       # or download the .deb/.dmg
rpi-imager
```

In the GUI: *Choose OS → Use custom → pick the .img.xz file*. Choose your
SD card under *Choose Storage*. Click *Write*. Takes 5-10 minutes.

> **Note**: you can also use `dd`, `balenaEtcher`, or AMD's bootgen flow.
> rpi-imager is the simplest path because it decompresses the `.img.xz`
> directly without you needing a separate decompression step.

When flashing finishes, eject the SD card from the laptop and insert it
into the Kria's SD slot (on the side of the carrier card).

## 4. First boot + SSH access

Plug in:
- SD card (already inserted)
- Ethernet cable to your laptop (or to a LAN switch)
- USB camera (optional at first boot — can plug in later)
- Power

The Kria will boot in 30-60 seconds. The LED next to the SOM should
turn from red → green when boot completes.

### Get the Kria's IP

On your laptop:

```bash
arp -a | grep '00:0a:35'
```

Xilinx's OUI prefix is `00:0a:35:*`. The line in the output that starts
with that prefix tells you the Kria's IP. Example:

```
? (10.42.0.27) at 00:0a:35:0f:d8:c6 [ether] on enp0s31f6
```

So the Kria is at `10.42.0.27`. If `arp -a` shows nothing matching, wait
30 more seconds and try again (DHCP takes a moment).

### SSH in

```bash
ssh ubuntu@10.42.0.27          # replace with your Kria's IP
# password: ubuntu (default — change it after first login)
```

You're now on the Kria as user `ubuntu`. Verify:

```bash
uname -a
# Linux kria 5.15.0-1069-xilinx-zynqmp ... aarch64 GNU/Linux

dpkg -l | head -3
# Should show some packages — kernel, etc.
```

### Recommended: set up SSH key auth

You'll be SSH'ing here a lot. Avoid typing the password every time:

```bash
# On your laptop:
ssh-keygen -t ed25519 -C "kria-thesis"           # if you don't already have a key
ssh-copy-id ubuntu@10.42.0.27                     # pushes your key to the Kria
ssh ubuntu@10.42.0.27                             # should now skip the password prompt
```

### Recommended: change the default password

```bash
# On the Kria:
passwd
```

The default `ubuntu:ubuntu` is a known credential set. Change it before
the Kria sees any network beyond your laptop.

## 5. Clone the repo

On the Kria:

```bash
cd ~
git clone https://github.com/<your-username>/KriaKv260_Model_Compiler.git
cd KriaKv260_Model_Compiler
```

(Use your fork's URL.)

## 6. Run the install script

This is the big one. Single command, but takes ~25 minutes the first
time. Subsequent runs (re-installs, partial recovery) are fast because
of stamp files.

```bash
bash scripts/kria/01_install_vai35.sh
```

This script handles 5 stages. Here's a walk-through:

### Stage 1: First-boot config

Sets hostname to `kria-lpr` (so multiple Krias on the same network are
distinguishable) and nudges you to change the default password if you
haven't already.

Interactive — accepts the defaults.

### Stage 2: xlnx-config.sysinit

Installs the `xlnx-config` snap and runs `xlnx-config.sysinit`, which
configures system-level Kria knobs that AMD's standard image expects.

Interactive on first run; the script remembers it ran via
`/var/local/kriakv260_sysinit.done` and skips on subsequent invocations.

Takes 5-10 minutes if the snap needs downloading.

### Stage 3: Kria-PYNQ install

Clones `Xilinx/Kria-PYNQ` (branch v3.0) and runs `install.sh -b KV260`.
This installs the PYNQ runtime, JupyterLab, numpy, OpenCV, and a virtual
environment at `/usr/local/share/pynq-venv`.

Also applies a post-install `numpy<2` pin — Kria-PYNQ's bundled cv2 was
built against numpy 1.x and breaks if numpy 2.x is installed.

Takes ~15 minutes (large download).

### Stage 4: Vitis AI 3.5 runtime upgrade

The default install gives you VAI 2.5. This stage upgrades to 3.5 by
following AMD's procedure from
[`amd/Kria-RoboticsAI`](https://github.com/amd/Kria-RoboticsAI),
adapted for KV260.

The script:

1. Downloads `vai3.5_kr260.zip` from `xilinx.com/bin/public/openDownload`
   (~50 MB)
2. Extracts and runs AMD's bundled `setup.sh` to install the .deb
   packages (`libvart`, `libxir`, `libunilog`, `libvitis-ai-library`,
   `libtarget-factory`) in the correct order
3. Extracts `lack_lib.tar.gz` and copies its contents to `/usr/lib/`
   (these are extra .so files AMD ships separately from the .debs)
4. Copies `xbutil2` to `/usr/bin/unwrapped/`

> **Note**: despite the zip's filename containing "kr260", the contents
> work for KV260 too. The PL bitstream differs between the boards, but
> the VAI runtime libraries are identical.

Takes ~1 minute.

### Stage 5: DPU-PYNQ + post-install patches

Clones `Xilinx/DPU-PYNQ` (branch `design_contest_3.5`) and pip-installs
it into the pynq-venv. Then applies the runtime patches:

- Adds `export LD_LIBRARY_PATH=/usr/lib` to `/etc/profile.d/pynq_venv.sh`
  so VART can find `libunilog.so` etc.
- Strips the `/usr/bin/` prefix from `/usr/bin/xdputil` so it uses the
  pynq-venv's Python.
- Refreshes the pynq-dpu notebooks at `/home/root/jupyter_notebooks/`

This stage also includes a sed patch to `pynqutils/setup_utils/download_overlays.py`
to fix a known IndexError during install. See [TROUBLESHOOTING.md](./TROUBLESHOOTING.md#pynqutils-indexerror)
for the forensic detail.

Takes ~1.5 minutes.

### What you'll see at the end

```
━━ Install summary ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  [1/5] First-boot config                       [✓] DONE     0s   hostname=kria-lpr
  [2/5] xlnx-config.sysinit                     [✓] DONE  6m12s   ran sysinit, accepted defaults
  [3/5] Kria-PYNQ stack                         [✓] DONE 15m04s   fresh install; numpy=1.26.4, healthy
  [4/5] VAI 3.5 runtime upgrade                 [✓] DONE  0m41s   installed via AMD setup.sh
  [5/5] DPU-PYNQ for VAI 3.5                    [✓] DONE  1m28s   DPU-PYNQ installed; patches applied

━━ Time ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Total elapsed: 23m 25s
  Logs: /home/ubuntu/kriakv260_install.log

[OK] Install complete.
[!]   ⚠ AMD recommends rebooting the board after this install.
[!]     Run: sudo reboot
```

Reboot when prompted:

```bash
sudo reboot
```

Wait 60 seconds, SSH back in, and continue.

## 7. Apply runtime tuning + persist it

Three system-level knobs the live pipeline depends on:

- **USB autosuspend off**: stops the kernel from putting the Brio's USB
  hub to sleep mid-stream, which would otherwise halve framerate
- **CPU governor = performance**: pins all four Cortex-A53 cores at 1.5
  GHz instead of letting the `ondemand` governor scale them down,
  reducing inference jitter
- **Camera tuning**: sets MJPG @ 640×480 @ 60 fps, manual exposure
  (1/100 s), gain 200 — Brio-specific values that work for indoor
  scenes

Apply once (effective only for the current boot):

```bash
sudo bash scripts/kria/02_apply_tuning.sh
```

Then make it persistent across reboots:

```bash
sudo bash scripts/kria/03_install_systemd.sh
```

This installs `/etc/systemd/system/kriakv260-tuning.service` which runs
the tuning script at every boot. Verify:

```bash
sudo systemctl status kriakv260-tuning.service
# Should show: Active: active (exited)
```

> **Camera hot-plug note**: the systemd unit runs at boot. If you
> hot-plug the camera after boot (or it gets enumerated late by the
> kernel), the v4l2 settings are at firmware defaults, not the tuned
> values. Re-run `sudo bash scripts/kria/02_apply_tuning.sh` manually
> after any camera hot-plug. There's a TODO to make this automatic via
> a udev rule.

## 8. Sync a compiled xmodel from your laptop

On the laptop side, you compile an xmodel using `scripts/host/01_compile.sh`
(see [USAGE.md](./USAGE.md#compiling-a-model) for that side of the
workflow). Then push it to the Kria:

```bash
# On your laptop, from the repo root:
bash scripts/host/03_sync_to_kria.sh ubuntu@10.42.0.27 yolov5n
```

The xmodel lands at `/home/ubuntu/xmodels_vai35/yolov5n/yolov5n_kv260.xmodel`
on the Kria.

## 9. Run the live demo

On the Kria:

```bash
sudo bash scripts/kria/run_live.sh yolov5n visual
```

(Use `text` instead of `visual` for the max-throughput notebook with no
video preview — see [USAGE.md](./USAGE.md#text-vs-visual-mode) for the
difference.)

The script will:

1. Verify you're running as root (PYNQ needs it for FPGA mmap)
2. Verify the xmodel exists and the notebook exists
3. Unload the `k26-starter-kits` firmware app via `xmutil unloadapp` so
   PYNQ can program the DPU bitstream
4. Source `/etc/profile.d/pynq_venv.sh` (env vars for VAI runtime)
5. Launch JupyterLab on `0.0.0.0:8888`

The terminal will print a banner with a copy-pasteable URL like:

```
  ┌─ Open this in your laptop's browser ─────────────────────
  │
  │   Once Jupyter prints its URL below (with the token),
  │   replace 'localhost' or '127.0.0.1' with the Kria's IP:
  │
  │      http://10.42.0.27:8888/lab?token=<copy-from-below>
  │
  └──────────────────────────────────────────────────────────
```

Followed by Jupyter's own output containing the token. Paste the URL
(with token) into your laptop's browser.

In JupyterLab, open the notebook and run all cells. See
[USAGE.md](./USAGE.md#running-the-live-demo) for what to expect.

To stop, click the **■ Stop** button in the visual notebook's controls
panel, or press Ctrl-C twice in the Kria terminal.

## 10. Architecture

ASCII overview of which scripts touch which parts of the system:

```
┌──────────────────────────────────────────────────────────────────────────┐
│ LAPTOP                                                                   │
│                                                                          │
│  scripts/host/01_compile.sh                                              │
│   ├─ PyTorch (.pt)                                                       │
│   ├─ ONNX export                                                         │
│   ├─ NNDCT quantize                                                      │
│   ├─ vai_c_xir compile                                                   │
│   └─→ xmodels_vai35/<variant>/<variant>_kv260.xmodel                     │
│                                                                          │
│  scripts/host/03_sync_to_kria.sh ubuntu@<ip> <variant>                   │
│   └─ rsync xmodel to Kria over SSH                                       │
│                                                                          │
└────────────────────────────────┬─────────────────────────────────────────┘
                                 │ SSH (key-based)
                                 │ rsync xmodel
                                 ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ KRIA KV260                                                               │
│                                                                          │
│  ┌─ Pass 5: Install scripts (one-time) ───────────────────────────┐     │
│  │  scripts/kria/01_install_vai35.sh    VAI 3.5 + PYNQ + DPU-PYNQ │     │
│  │  scripts/kria/02_apply_tuning.sh     USB + CPU + camera        │     │
│  │  scripts/kria/03_install_systemd.sh  persist tuning            │     │
│  └────────────────────────────────────────────────────────────────┘     │
│                                                                          │
│  ┌─ Daily workflow: run_live.sh ────────────────────────────────────┐   │
│  │   xmutil unloadapp ← clear PL                                    │   │
│  │   pynq_venv.sh source ← XILINX_XRT, LD_LIBRARY_PATH              │   │
│  │   jupyter lab --ip 0.0.0.0 --allow-root                          │   │
│  │   └─→ http://<kria-ip>:8888/lab?token=...                        │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                          │
│  ┌─ Notebook (02_deploy_text.ipynb or 03_deploy_visual.ipynb) ─────┐    │
│  │   DpuOverlay("dpu.bit") ← program FPGA fabric                   │    │
│  │   ModelRunner(spec, xmodel, overlay) ← load xmodel              │    │
│  │   ThreadedCamera() ← drain /dev/video0 in bg thread             │    │
│  │   while True:                                                   │    │
│  │       frame = cam.read_new()                                    │    │
│  │       dets, timings = runner.infer(frame, conf, iou, ...)       │    │
│  │       draw_detections(frame, dets) ← visual mode only           │    │
│  │       widget.value = cv2.imencode(...).tobytes()                │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                          │
│        FPGA fabric (PL)                                                  │
│        │   ├─ DPU IP core (B4096)                                        │
│        │   └─ DMA / AXI fabric                                           │
│        │                                                                 │
│       PS (4× Cortex-A53)                                                 │
│        │   ├─ Python / VART runtime                                      │
│        │   ├─ V4L2 driver / uvcvideo                                     │
│        │   └─ Linux 5.15 + zocl module                                   │
│        │                                                                 │
│       I/O                                                                │
│            ├─ USB 3.0 ─── Logitech Brio                                  │
│            └─ Ethernet ─── laptop                                        │
└──────────────────────────────────────────────────────────────────────────┘
```

## 11. Validated performance

The numbers below were measured on a freshly-installed system following
this guide on **2026-05-10** at git tag **`v0.6-pass6-validated`**.

### Pure inference benchmark (`02_deploy_text.ipynb` cell 6)

yolov5n at imgsz=320 on KV260 B4096 DPU, VAI 3.5:

| Stage | Mean (ms) | p50 | p95 | p99 |
|---|---|---|---|---|
| Total | 12.38 | 12.35 | 12.71 | 13.08 |
| Preprocess | 3.80 | 3.77 | 4.09 | 4.47 |
| DPU | 7.74 | 7.74 | 7.82 | 7.88 |
| Decode | 0.80 | 0.79 | 0.82 | 0.87 |

**Throughput: 80.8 fps** (pure inference; no camera, no display).

### Live demo (`02_deploy_text.ipynb` cell 14, 60-second run)

| Metric | Value |
|---|---|
| Frames inferred | 3620 |
| Unique camera frames | 3621 |
| Inference fps | 59.87 |
| Camera fps | 59.88 |
| Avg preprocess (ms) | 3.84 |
| Avg DPU (ms) | 7.75 |
| Avg decode (ms) | 0.92 |
| Total (ms) | 12.51 |
| Theoretical max | 79.95 fps |
| Total detections | 694 |
| Frames with detection | 687 |
| Hit rate | 18.98% |

**End-to-end: 60 fps live, camera-bound.** The DPU is idle ~30% of the
time waiting for camera frames.

> The preprocess at 3.84 ms ms includes a `canvas.astype(float32) / 255.0`
> allocation that was later optimized via `np.multiply(..., out=...)`
> (commit after the validation tag). With the optimization, preprocess
> drops to ~1-1.5 ms and the theoretical max rises to ~110 fps.
> End-to-end stays at 60 fps because the camera is the bottleneck.

## 12. What can go wrong (and where to look)

| Symptom | Likely cause | Where to look |
|---|---|---|
| Install script stops at Stage 3 with `apt` errors | Disk full or network drop mid-install | `df -h`, then re-run the script (it resumes via stamps) |
| Install script stops at Stage 4 with `setup.sh failed` | VAI 3.5 deb dependency missing | [TROUBLESHOOTING.md → VAI 3.5 install](./TROUBLESHOOTING.md#vai-35-install-failures) |
| Install script stops at Stage 5 with `IndexError` building pynq-dpu | pynqutils bug on empty device list | [TROUBLESHOOTING.md → pynqutils IndexError](./TROUBLESHOOTING.md#pynqutils-indexerror) |
| `DpuOverlay("dpu.bit")` raises `No Devices Found` | `k26-starter-kits` is loaded, blocking PYNQ device enumeration | [TROUBLESHOOTING.md → No Devices Found](./TROUBLESHOOTING.md#no-devices-found) |
| `DpuOverlay("dpu.bit")` raises `Root permissions required` | Notebook running as non-root user | Make sure you launched via `sudo bash run_live.sh`; see [TROUBLESHOOTING.md → root permissions](./TROUBLESHOOTING.md#root-permissions-required) |
| `ThreadedCamera()` raises `cannot open camera 0` | `/dev/video0` doesn't exist | Camera not enumerated; replug, see [TROUBLESHOOTING.md → camera missing](./TROUBLESHOOTING.md#camera-missing) |
| `ThreadedCamera()` raises `no frames received in 3s` | Camera in wrong mode (YUYV instead of MJPG) | Re-run tuning, see [TROUBLESHOOTING.md → camera no frames](./TROUBLESHOOTING.md#camera-no-frames) |
| Live demo runs but `inf_fps` is ~15 fps not ~60 | USB autosuspend re-enabled (kernel quirk) | Re-run `sudo bash scripts/kria/02_apply_tuning.sh` |

For the full forensic treatment of every issue we've hit, see
[TROUBLESHOOTING.md](./TROUBLESHOOTING.md).

## Next: daily workflow

You have a working install. The day-to-day workflow (compile a model →
sync → run live, switch between text and visual modes) is documented in
[USAGE.md](./USAGE.md).
