# Kria board setup

One-time setup of the KV260 board to run compiled xmodels. After this is
done, day-to-day usage is just `bash scripts/kria/run_live.sh <variant>`.

## Hardware requirements

- **AMD Kria KV260 Vision AI Starter Kit** (KR260 / KD240 NOT supported — different DPU)
- **microSD card** ≥ 32 GB (~12 GB for OS + VAI stack, rest for headroom)
- **Power supply** that came with the kit
- **Ethernet** (Wi-Fi works but Ethernet is more reliable for the install)
- **Camera** for the live demo: Logitech Brio (USB 3.0) preferred — any
  MJPG-capable UVC camera works as fallback
- **Optional: HDMI display** — Pass 5 doesn't require it; we use SSH + Jupyter

## Software requirements on the board

- **Ubuntu 22.04 LTS for Kria SOMs** (Canonical's Kria image). 20.04 is NOT
  supported by this pipeline — re-flash if you have it.

## Step 0 — Flash the SD card (manual)

> **This step is NOT scripted** — it requires destructive disk operations
> and a different procedure for each user's hardware. Follow the official
> Canonical tutorial:
>
> https://canonical-kria.readthedocs-hosted.com/en/latest/
>
> Confirm the resulting image is **22.04 LTS** (not 20.04). The pipeline
> refuses to proceed on 20.04 because the VAI 3.5 deb packages are not
> built for it.

Once you have a 22.04 SD card in the Kria and can SSH into it as `ubuntu`,
the rest of this guide takes over.

## Step 1 — Clone the repo on the Kria

From your laptop, SSH in:

```bash
ssh ubuntu@<kria-ip>
```

On the Kria:

```bash
cd ~
git clone https://github.com/abdullaadelaljaberi-debug/KriaKv260_Model_Compiler.git
cd KriaKv260_Model_Compiler
```

## Step 2 — Verify the board is ready

```bash
bash scripts/kria/00_check_prereqs.sh
```

This runs 11 checks. Critical ones (script refuses to proceed if these fail):

1. **Hardware = Kria SOM** (device-tree check)
2. **Model = KV260** (KR260/KD240 are blocked because of different DPU)
3. **OS = Ubuntu 22.04 LTS** (20.04 blocked with reflash instructions)
4. **Architecture = aarch64**
5. **Repo layout sane**

Informational checks (warn but don't block):

- sudo access
- Disk space (≥5 GB)
- Internet connectivity
- xlnx-config presence
- Kria-PYNQ presence
- VAI version (so we know whether to upgrade)

If everything is green, proceed to step 3. Otherwise the script prints the
fix for each failure.

## Step 3 — Install Kria-PYNQ + VAI 3.5

```bash
bash scripts/kria/01_install_vai35.sh
```

This is the long step (20-30 min on first run). It does five things:

| Stage | What | Skipped if |
|---|---|---|
| 1 | First-boot config (hostname prompt) | hostname already set to something custom |
| 2 | `xlnx-config.sysinit` | a stamp file at `/var/local/kriakv260_sysinit.done` exists |
| 3 | Kria-PYNQ install (clone + `install.sh -b KV260`) | `/usr/local/share/pynq-venv` already exists |
| 4 | VAI 2.5 → 3.5 deb upgrade | `libvart-runtime` already reports version 3.5 |
| 5 | DPU bitstream (`DPU-PYNQ design_contest_3.5` branch) | `~/dpu_pynq_vai35/boards/kv260/` already populated |

All stages are idempotent. Re-running on an already-installed board completes
in seconds and reports "already done" for everything.

### What if VAI install fails

Logs go to `~/kriakv260_install.log`. Common failures:

- **Network drop**: re-run the script; it picks up where it left off
- **Disk full**: `df -h /` to confirm; `sudo apt clean` to free space
- **Deb dependency error**: the script falls back to `apt --fix-broken install` automatically; if that doesn't work, inspect the log
- **Wrong VAI zip URL**: AMD sometimes moves the download. Override with:
  ```bash
  VAI35_KR260_ZIP_URL=https://...new-location... bash scripts/kria/01_install_vai35.sh
  ```

## Step 4 — Apply runtime tuning

```bash
bash scripts/kria/02_apply_tuning.sh
```

Achieves 60 fps live demo performance by:

1. **Disabling USB autosuspend** on every USB device (this is the biggest
   single fix — silent 15 fps cap without it)
2. **Setting CPU governor to `performance`** (prevents downclocking)
3. **Tuning the camera via `v4l2-ctl`**:
   - MJPG @ 60 fps
   - Manual exposure (`auto_exposure=1, exposure_time_absolute=100, gain=200`)
   - `exposure_dynamic_framerate=0`
4. **Sanity-testing** with a 5-second capture, saving the first frame to
   `/tmp/kriakv260_test_frame.jpg` so you can eyeball it

The camera tuning auto-detects a Logitech Brio first; falls back to any
MJPG-capable USB camera. Controls not supported by the active camera are
silently skipped.

**Note**: this script is one-shot. Settings don't persist across reboots
unless you do step 5.

## Step 5 — Persist tuning across reboots

```bash
bash scripts/kria/03_install_systemd.sh
```

Installs `kriakv260-tuning.service` — a systemd unit that re-runs
`02_apply_tuning.sh` on every boot.

Verify it works:

```bash
sudo systemctl status kriakv260-tuning
sudo journalctl -u kriakv260-tuning -n 30
```

Or reboot the board and confirm `cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor`
still says `performance`.

To uninstall: `sudo systemctl disable kriakv260-tuning && sudo rm /etc/systemd/system/kriakv260-tuning.service`.

## Step 6 — Sync your first xmodel from the host

From your laptop (NOT the Kria):

```bash
cd ~/path/to/KriaKv260_Model_Compiler  # your laptop's copy of the repo
bash scripts/host/03_sync_to_kria.sh ubuntu@<kria-ip> yolov5n
```

The first run prompts once for the Kria password to set up SSH key
authentication. Subsequent runs are silent. The xmodel lands at
`/home/ubuntu/xmodels_vai35/yolov5n/yolov5n_kv260.xmodel` on the Kria,
which is where `run_live.sh` looks for it.

## Step 7 — Run the live demo

Back on the Kria:

```bash
cd ~/KriaKv260_Model_Compiler
bash scripts/kria/run_live.sh yolov5n
```

Jupyter starts on `localhost:8888` by default (not network-exposed). To
access from your laptop's browser, in a separate terminal on your laptop:

```bash
ssh -L 8888:localhost:8888 ubuntu@<kria-ip>
# leave this terminal open; then open http://localhost:8888 in your browser
```

Jupyter's first output line is `http://localhost:8888/lab?token=...` —
copy-paste that whole URL into the browser. Inside Jupyter, open
`notebooks/02_deploy_live.ipynb` (delivered in Pass 6) and Run All.

### Why SSH tunnel instead of binding to the network

Two reasons:

1. **Security**: a token-based Jupyter on `0.0.0.0:8888` is one mistake
   away from running arbitrary Python on your Kria from anyone on the LAN
2. **Convenience**: no need to type the Kria's IP into the browser —
   `localhost:8888` always points to the right place

If you really want network-bound for some reason:

```bash
JUPYTER_HOST=0.0.0.0 bash scripts/kria/run_live.sh yolov5n
# then your-laptop$ open http://<kria-ip>:8888
```

## Restoring from backup

If something goes wrong with the install scripts and you've backed up the
SD card per the recommended workflow:

```bash
# On your laptop, with the SD card in a reader:
lsblk                              # identify the SD device, e.g., /dev/sdc

# Restore. Replace /dev/sdX with the actual device.
gunzip -c ~/kria-vai25-backup-*.img.gz | sudo dd of=/dev/sdX bs=4M status=progress
sync
```

Then re-insert the SD card and the board boots into your pre-install state.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `00_check_prereqs.sh` fails on Ubuntu 20.04 | Wrong OS version | Re-flash with 22.04 (intentional refusal; see step 0) |
| Kria-PYNQ install fails halfway | Network drop or transient apt failure | Re-run `01_install_vai35.sh` — it's idempotent |
| `dpkg` fails with "depends on libfoo" | Missing apt dep | The script auto-runs `apt --fix-broken install`; if that fails too, paste the error |
| Camera produces all-black frames | Manual exposure too low for ambient light | Increase `exposure_time_absolute` in `02_apply_tuning.sh` |
| Live demo runs at 15 fps not 60 | USB autosuspend re-enabled (boot before systemd persistence) | Run `02_apply_tuning.sh` again, then install the systemd unit |
| `xmodel not found` on Kria | Didn't sync from host | Run `bash scripts/host/03_sync_to_kria.sh ubuntu@<ip> <variant>` |
| Jupyter URL doesn't open | SSH tunnel not active | Open a second terminal with `ssh -L 8888:localhost:8888 ubuntu@<ip>` |
| Multi-subgraph DPU error at deploy | xmodel has SiLU activations | Re-compile with the default `SWAP_ACTIVATIONS=true` (see `docs/MODELS.md` → Activation function policy) |

## What this pipeline does NOT do on the Kria

By design:

- **No password change automation** — `passwd` is interactive and you
  should change the default `ubuntu` password yourself
- **No network reconfiguration** — too easy to brick a headless board
- **No SD flashing** — destructive operation, follow Canonical's tutorial
- **No 20.04 → 22.04 OS upgrade** — high risk on Xilinx kernels; reflash instead
