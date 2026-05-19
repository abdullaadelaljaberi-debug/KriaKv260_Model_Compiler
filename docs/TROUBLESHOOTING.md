# Troubleshooting

Forensic detail for every issue we've hit during Pass 5 and Pass 6
development. Each section explains:

1. **The symptom** — exactly what the user sees
2. **The cause** — what's actually going wrong (often not what the
   error message implies)
3. **The fix** — minimal command to recover
4. **The prevention** — what `run_live.sh` / install scripts do
   automatically to avoid the issue in the future

If something not on this list breaks, the patterns documented here are
usually the right starting point — Linux/FPGA/sudo interactions tend to
share root causes.

## Contents

### Install-time (Pass 5 stages)
- [`libvart-runtime` version check fails after VAI 3.5 install](#libvart-runtime-version-check-fails)
- [pynqutils IndexError when building pynq-dpu](#pynqutils-indexerror)
- [`setup.sh` failed installing VAI 3.5 debs](#vai-35-install-failures)
- [Stage 3 numpy 2.x breaks cv2](#stage-3-numpy-2x-breaks-cv2)
- [Systemd unit reports `failed` even though tuning applied](#systemd-unit-reports-failed)

### Runtime (using the notebooks)
- [`No Devices Found` from `DpuOverlay`](#no-devices-found)
- [`Root permissions required`](#root-permissions-required)
- [`Running as root is not recommended` — Jupyter refuses to start](#jupyter-refuses-to-start-as-root)
- [`Cannot find dpu.bit`](#cannot-find-dpubit)
- [`Camera opened but no frames received`](#camera-no-frames)
- [`cannot open camera 0` / `/dev/video*` doesn't exist](#camera-missing)
- [Live demo runs at 15 fps not 60](#live-demo-15-fps)
- [`Check failed: r == 0 cannot set read range!` glog message](#fingerprint-glog-noise)

### Connectivity
- [Can't find Kria's IP](#cant-find-krias-ip)
- [Browser shows ERR_CONNECTION_REFUSED](#err_connection_refused)

### v0.8-v0.10 sections

- [QAT abandoned for PTQ + hard-negative training](#qat-abandoned-for-ptq--hard-negative-training)
- [High int8 false-positive rate despite clean float model](#high-int8-false-positive-rate-despite-clean-float-model)
- [Mixed calibration set INCREASES int8 false positives](#mixed-calibration-set-increases-int8-false-positives)
- [`vai_q_onnx` crashes in `align_concat` on YOLOv11](#vai_q_onnx-crashes-in-align_concat-on-yolov11)
- [DPU resource lock — `xrt_device_handle_imp` Check failed](#dpu-resource-lock--xrt_device_handle_imp-check-failed)
- [Vitis-AI container OOM during PTQ calibration](#vitis-ai-container-oom-during-ptq-calibration)
- [Pip install fails as non-root inside the Vitis-AI conda env](#pip-install-fails-as-non-root-inside-the-vitis-ai-conda-env)
- [Numpy import error after partial pip install in the container](#numpy-import-error-after-partial-pip-install-in-the-container)
- [`measure_fps_kria.py` hardcoded to yolov11n](#measure_fps_kriapy-hardcoded-to-yolov11n)
- [`run_live.sh` rejects new variants with "unknown variant"](#run_livesh-rejects-new-variants-with-unknown-variant) — *RESOLVED in v0.11*
- [Eggs notebook hardcoded `get_spec("yolov11n")` — cosmetic warning](#eggs-notebook-hardcoded-get_specyolov11n--cosmetic-warning)
- [bash `!` history expansion mangles commit messages](#bash--history-expansion-mangles-commit-messages)


---

## `libvart-runtime` version check fails

### Symptom

After Stage 4 (VAI 3.5 install), the script fails with:

```
[FAIL] VAI install completed but version is '', expected '3.5'.
  Check /home/ubuntu/kriakv260_install.log for clues, or inspect:
    dpkg -s libvart-runtime
```

But `dpkg -l | grep vart` shows `libvart 3.5.0-1 arm64` — the runtime
**is** installed correctly.

### Cause

AMD's VAI 3.5 debs use the package name `libvart`, not `libvart-runtime`.
The detection helper `vai_installed_version()` was originally written to
match VAI 2.5's naming and got the package name wrong.

### Fix

Already fixed in the scripts:

```bash
# In scripts/kria/lib/common.sh, vai_installed_version() now checks
# libvart first, with libxir and .so filename fallbacks.
```

If you're running an older version of the script (before the fix),
update via `git pull`. Or work around it once by manually creating the
stamp file:

```bash
sudo touch /var/local/kriakv260_vai35.done
bash scripts/kria/01_install_vai35.sh   # will skip Stage 4
```

### Why the helper has three fallback layers

After the libvart fix, the helper looks like this:

```bash
vai_installed_version() {
    # 1. Primary: libvart package
    pkg_ver=$(dpkg -s libvart 2>/dev/null | awk -F': *' '/^Version:/ {print $2}')
    if [[ "$pkg_ver" == 3.5* ]]; then echo "3.5"; return; fi

    # 2. Fallback: libxir package (always co-installed with libvart)
    pkg_ver=$(dpkg -s libxir 2>/dev/null | ...)
    ...

    # 3. Last-ditch: parse from /usr/lib/libvart-runner.so.3.5.0 filename
    ls /usr/lib/libvart-runner.so.*.*.* | sed -E 's|.*\.so\.([0-9]+\.[0-9]+)\..*|\1|'
}
```

Defense in depth: if AMD renames packages again in a future VAI release,
the fallbacks should still detect a working install.

## pynqutils IndexError

### Symptom

During Stage 5, `pip install pynq-dpu` fails:

```
File ".../pynqutils/setup_utils/download_overlays.py", line 127, in _resolve_devices_overlay_res
    if len(devices) == 0 and type(Device.devices[0]) == EmbeddedDevice:
IndexError: list index out of range
```

### Cause

Two compounding issues:

1. **pynqutils has a real bug**. Line 127 checks `len(devices) == 0`
   first, then accesses `Device.devices[0]`. But if `Device.devices` is
   *also* empty (true during a fresh install, before any FPGA bitstream
   has been loaded), the `[0]` access raises IndexError.
2. **`sudo` strips env vars** by default. When the pip install runs via
   `sudo /usr/local/.../python3 -m pip install ...`, the venv setup from
   `pynq_venv.sh` is lost, so XRT can't enumerate any device even if
   one is present.

### Fix

The install script applies two fixes:

1. **Sed-patch the pynqutils bug** in `download_overlays.py`:

   ```bash
   sed -i 's|len(devices) == 0 and type(Device.devices\[0\]) == EmbeddedDevice|len(devices) == 0 and (not Device.devices or type(Device.devices[0]) == EmbeddedDevice)|' \
       /usr/local/share/pynq-venv/lib/python3.10/site-packages/pynqutils/setup_utils/download_overlays.py
   ```

   The new guard `not Device.devices or ...` short-circuits when
   `Device.devices` is empty (which is correct for an embedded board:
   if no devices are discovered yet, assume the default is embedded).

2. **Run pip via `sudo bash -c "..."`** so the env stays sourced:

   ```bash
   sudo bash -c "
       source /etc/profile.d/pynq_venv.sh
       cd /home/ubuntu/.cache/kriakv260_install/DPU-PYNQ
       /usr/local/share/pynq-venv/bin/python3 -m pip install . --no-build-isolation
   "
   ```

Both are idempotent — the install script checks before applying.

### Prevention

The patch is part of `scripts/kria/01_install_vai35.sh` Stage 5b-i. If
you're starting fresh, you get the fixed version automatically.

## VAI 3.5 install failures

### Symptom

Stage 4 fails inside AMD's bundled `setup.sh`:

```
[*] Running AMD's setup.sh (installs VAI 3.5 debs)
dpkg: error processing package libvart (--install):
 dependency problems - leaving unconfigured
[FAIL] setup.sh failed. Check /home/ubuntu/kriakv260_install.log
```

### Cause

Usually a missing apt dependency for one of the VAI 3.5 debs. Most
common: `libgoogle-glog0v5`, `libboost-program-options1.74.0`.

### Fix

```bash
# See what dpkg reports as missing:
sudo apt-get install -f -y

# That auto-resolves most missing dependencies. Then re-run:
bash scripts/kria/01_install_vai35.sh
```

Stage 1-3 will skip (stamps), Stage 4 will retry.

### Prevention

We could call `apt-get install -f` inside the script before running
`setup.sh`, but that's destructive in edge cases. Better to let the
user opt into the fix-broken-deps action explicitly.

## Stage 3 numpy 2.x breaks cv2

### Symptom

Stage 3 (Kria-PYNQ install) completes but you see:

```
ImportError: numpy.core.multiarray failed to import
AttributeError: _ARRAY_API not found
```

When trying to `import cv2` in the pynq-venv.

### Cause

Kria-PYNQ's `install.sh` pulls in pip-installed numpy which may resolve
to numpy 2.x. But the bundled OpenCV was compiled against numpy 1.x
ABI, and the ABI break between 1.x and 2.x means cv2 can't load.

### Fix

The install script pins numpy<2 in pynq-venv unconditionally as a
post-Stage-3 patch:

```bash
sudo /usr/local/share/pynq-venv/bin/pip install --quiet 'numpy<2'
```

This is a no-op if numpy is already in the 1.x range. It only takes
effect if pip already resolved numpy 2.x at install time.

### Prevention

The numpy<2 pin is part of `scripts/kria/01_install_vai35.sh` Stage 3
post-install fixes. Always applied; idempotent.

## Systemd unit reports failed

### Symptom

```
sudo systemctl status kriakv260-tuning.service
● kriakv260-tuning.service - KriaKv260_Model_Compiler runtime tuning
     Active: failed (Result: exit-code)
    Process: 2482 ExecStart=/bin/bash ... (code=exited, status=1/FAILURE)
```

But the summary table inside the unit's output shows all stages green:

```
[1/3] USB autosuspend     [✓] DONE  disabled on 5 devices
[2/3] CPU governor        [✓] DONE  performance on 4 cores
[3/3] Camera tuning       [✓] DONE  Logitech BRIO: MJPG@60fps
```

### Cause

The tuning script's last `$?` came from a `|| log_warn` clause (e.g.,
a v4l2 control that's not supported on the camera, which is fine to
skip but returns non-zero). Without an explicit `exit 0` at the end of
the script, bash inherits that non-zero exit code, and systemd interprets
it as failure.

### Fix

Already fixed: every Kria-side script now has an explicit `exit 0` at
the end:

```bash
# At the bottom of scripts/kria/02_apply_tuning.sh:
log_ok "Tuning applied (this session only)."
log_info "Next:  bash scripts/kria/03_install_systemd.sh"

# Explicit exit 0. Without this, $? may be non-zero from a `|| log_warn`
# clause earlier, even though semantically everything succeeded.
exit 0
```

If you have an older version, `git pull` to get the fix.

### Prevention

Pattern is now standard for any script that uses `|| log_warn`
constructs.

## No Devices Found

### Symptom

```python
overlay = DpuOverlay("dpu.bit")
```

Raises:

```
RuntimeError: No Devices Found
  at: cls._active_device = cls.devices[0]
```

Preceded by warning:

```
UserWarning: No devices found, is the XRT environment sourced?
```

### Cause

Two things conspire:

1. **`k26-starter-kits` firmware app is loaded at boot.** It uses
   `XRT_FLAT` mode, which doesn't expose an xclbin-compatible PL. PYNQ's
   device enumeration looks for an XRT-compatible device and finds
   nothing.
2. **XRT environment isn't sourced** in the Jupyter kernel. If you
   launched Jupyter directly with `jupyter lab` (no script), the env
   vars `XILINX_XRT`, `LD_LIBRARY_PATH` aren't set, and XRT's device
   probe can't find the FPGA even with the starter-kit unloaded.

### Fix

```bash
# Unload the starter-kit
sudo xmutil unloadapp

# Source the env vars
source /etc/profile.d/pynq_venv.sh

# Then start Jupyter (as root)
sudo bash scripts/kria/run_live.sh yolov5n
```

### Prevention

`run_live.sh` now does both automatically:

```bash
# Inside run_live.sh:

# Detect + unload starter-kit
if sudo xmutil listapps | grep -q "k26-starter-kits.*XRT_FLAT.*0,"; then
    sudo xmutil unloadapp
fi

# Source env before exec'ing jupyter
[[ -f /etc/profile.d/pynq_venv.sh ]] && source /etc/profile.d/pynq_venv.sh
[[ -f /opt/xilinx/xrt/setup.sh   ]] && source /opt/xilinx/xrt/setup.sh
```

If you launched via `run_live.sh` and still see this error, the
starter-kit might have been re-loaded by something else. Verify:

```bash
sudo xmutil listapps
# The Active_slot column should NOT show k26-starter-kits
# If it does, unload manually and retry.
```

### Why the previous run "just worked"

In our testing, the first run after install succeeded — but only
because of luck (the previous boot's xmutil state happened to be
clean). After a reboot, the starter-kit reloads by default. That's why
we now unload it programmatically every launch.

## Root permissions required

### Symptom

```python
overlay = DpuOverlay("dpu.bit")
```

Raises:

```
OSError: Root permissions required.
  at: pynq.pl_server.embedded_device.EmbeddedDevice.mmap
```

### Cause

PYNQ-DPU mmaps FPGA configuration registers (clock, reset, etc.) via
`/dev/mem`. That requires root. As regular user `ubuntu`, the mmap
fails with EACCES.

### Fix

Launch Jupyter as root:

```bash
# NOT:
bash scripts/kria/run_live.sh yolov5n

# YES:
sudo bash scripts/kria/run_live.sh yolov5n
```

### Prevention

`run_live.sh` checks `$EUID` at the start:

```bash
if [[ $EUID -ne 0 ]]; then
    log_err "This script must be run as root: the PYNQ-DPU stack mmaps the FPGA"
    log_err "configuration registers, which requires root permissions."
    log_err "Re-run as:"
    log_err "  sudo bash scripts/kria/run_live.sh <variant>"
    exit 1
fi
```

So you get a clear error before Jupyter even starts, not deep inside the
notebook.

## Jupyter refuses to start as root

### Symptom

After `sudo bash run_live.sh`:

```
[C ... ServerApp] Running as root is not recommended. Use --allow-root to bypass.
```

(Then Jupyter exits.)

### Cause

Jupyter's safety check. Running as root is intentional here (we need
it for FPGA mmap) but Jupyter wants explicit acknowledgment.

### Fix

`run_live.sh` passes `--allow-root` automatically when invoked as root:

```bash
if [[ $EUID -eq 0 ]]; then
    jupyter_args+=( --allow-root )
fi
exec jupyter lab "${jupyter_args[@]}"
```

If you somehow hit this, update via `git pull`.

## Cannot find dpu.bit

### Symptom

```
FileNotFoundError: Cannot find dpu.bit.
```

When constructing `DpuOverlay("dpu.bit")`.

### Cause

`DpuOverlay` looks for the bitstream in pynq-dpu's data directory. If
Stage 5 of the install was interrupted (or skipped), the bitstream
files weren't extracted there.

### Fix

```bash
# Verify pynq-dpu is installed:
sudo /usr/local/share/pynq-venv/bin/pip show pynq-dpu

# If installed but bitstream files are missing, force a notebook refresh:
sudo bash -c "
    source /etc/profile.d/pynq_venv.sh
    cd /home/root/jupyter_notebooks
    rm -rf pynq-dpu
    /usr/local/share/pynq-venv/bin/pynq get-notebooks pynq-dpu -p . --force
"

# If pynq-dpu isn't installed, re-run Stage 5:
sudo rm /var/local/kriakv260_vai35.done
bash scripts/kria/01_install_vai35.sh
```

### Prevention

The stamp file `kriakv260_vai35.done` only gets written after Stage 5
completes successfully, so an interrupted install will retry on the
next run.

## Camera no frames

### Symptom

In the notebook's live-loop cell:

```
RuntimeError: Camera opened but no frames received in 3.0s.
```

The camera is plugged in. `/dev/video0` exists. `v4l2-ctl -d /dev/video0 -V`
shows MJPG mode. But OpenCV can't read frames.

### Cause

OpenCV's `VideoCapture` might be silently downgrading the camera's
format. The camera was MJPG when v4l2-ctl checked, but `cv2.set(FOURCC,
MJPG)` re-negotiates the format and the renegotiation can stall waiting
for the camera to settle.

In practice this happens after:

1. A previous failed `ThreadedCamera()` attempt left the device in a
   weird state
2. The camera was hot-plugged after boot, so the systemd tuning script
   never applied its v4l2 settings to *this* device instance
3. Another process is holding `/dev/video0`

### Fix

```bash
# 1. In Jupyter: Kernel → Restart (releases any held device handle)

# 2. From the Kria shell:
sudo bash scripts/kria/02_apply_tuning.sh

# 3. Verify the camera is in MJPG mode:
v4l2-ctl -d /dev/video0 -V
# Should show: Pixel Format : 'MJPG'

# 4. Re-run the notebook from the top
```

If step 3 shows the wrong format and tuning reported success: check
`sudo fuser /dev/video0` for any process still holding it.

### Prevention

`ThreadedCamera.__init__` now cleans up on failure (try/except wraps
the body, releases the `cv2.VideoCapture` on any exception). The
first-frame timeout was bumped from 1s to 3s to give the camera time
to negotiate MJPG after a fresh enumeration.

Long-term TODO: add a udev rule that auto-runs the tuning script when
a new V4L2 device appears, so hot-plug after boot is handled.

## Camera missing

### Symptom

```
$ ls -la /dev/video*
ls: cannot access '/dev/video*': No such file or directory
```

The Kria can't see the camera. Notebook fails with `cannot open camera 0`.

### Cause

The camera isn't enumerated by USB at all. Most common causes:

1. **Loose USB cable** — the Brio's cable can feel "clicked in" while
   only partially seated
2. **Faulty USB port** on the KV260
3. **Faulty USB cable**
4. **Camera was unplugged at boot** and udev hasn't run for it yet

### Fix

```bash
# 1. lsusb to confirm the camera is on the bus
lsusb
# Should show: Bus 00X Device 00Y: ID 046d:085e Logitech, Inc. BRIO ...
# If not, the camera isn't electrically connected.

# 2. Replug the camera:
#    - Remove the USB cable from the Kria
#    - Wait 3 seconds
#    - Plug back in, preferably to a different USB port

# 3. Verify enumeration:
ls -la /dev/video*

# 4. Re-apply tuning since the v4l2 state is now at defaults:
sudo bash scripts/kria/02_apply_tuning.sh
```

The Brio has a small white LED on the front; it lights briefly when the
camera enumerates correctly. If you see no light at all when plugged in,
the camera isn't getting power — try a different port or cable.

### Prevention

None — this is a hardware-level issue. The notebook gives a clear error
("cannot open camera 0") that points at the right diagnostic command
in the message.

## Live demo 15 fps

### Symptom

Live demo runs but `inf_fps` and `cam_fps` are stuck around 15, not 60.
DPU is sitting idle most of the time.

### Cause

The Brio's USB hub got its `power/control` set back to `auto`, which
enables autosuspend. When that triggers mid-stream, the camera's
effective fps halves (or quarters, or worse).

This can happen if:

1. The systemd tuning service didn't run (e.g., disabled, or unit
   removed)
2. The hub was hot-swapped after the service ran
3. A kernel update changed the default USB autosuspend behavior

### Fix

```bash
# Re-apply tuning manually:
sudo bash scripts/kria/02_apply_tuning.sh

# Verify all USB devices have power/control = on:
for f in /sys/bus/usb/devices/*/power/control; do
    echo "$f: $(cat $f)"
done
```

If everything reads `on` and you still see 15 fps, the issue is
elsewhere — check whether the Brio negotiated YUYV instead of MJPG (see
[camera no frames](#camera-no-frames) section).

### Prevention

The systemd unit re-applies tuning at every boot. Verify it's enabled:

```bash
sudo systemctl status kriakv260-tuning.service
# Active: active (exited) — good
# Active: inactive (dead) or failed — bad; re-enable:
sudo systemctl enable --now kriakv260-tuning.service
```

## Fingerprint glog noise

### Symptom

During inference or model load:

```
WARNING: Logging before InitGoogleLogging() is written to STDERR
F20260510 23:15:25.252251 xrt_device_handle_imp.cpp:327]
    Check failed: r == 0 cannot set read range!
    cu_index 0 cu_base_addr 2147549184 fingerprint 0x101000056010407 :
    Invalid argument [22]
```

The `F` prefix is glog's FATAL level.

### Cause

This is an internal XRT sanity check that fires during DPU handle
setup. It logs a FATAL message but doesn't actually abort — the inference
continues normally. The fingerprint `0x101000056010407` is exactly the
KV260 B4096 fingerprint our pipeline targets, so it's not a real
mismatch.

The "Logging before InitGoogleLogging()" prefix means the message was
emitted before glog's env-var-driven log level was applied — that's why
`GLOG_minloglevel=3` doesn't suppress it.

### Fix

Nothing to fix. The message is cosmetic noise.

If you find it distracting, you can redirect Jupyter's stderr to
`/dev/null` at launch (in `run_live.sh`), but that also hides real
errors, so we don't.

### Prevention

The notebook sets `GLOG_minloglevel=3` etc. in an early cell, which
suppresses most glog output. The pre-InitGoogleLogging message slips
through.

This bug has been present since VAI 2.5 (we hit it in `09_yolov5n_final_v2.ipynb`)
and is documented in AMD's release notes as a known issue with no
ETA for a fix.

## Can't find Kria's IP

### Symptom

After flashing + booting, `arp -a` shows nothing matching the Xilinx
OUI prefix.

### Cause

Either:

1. DHCP hasn't completed yet (wait 30-60 seconds after power-on)
2. The Kria isn't on a network that gives it an IP — for example, it's
   plugged into a switch with no DHCP server
3. The Kria failed to boot (different problem; check the green LED on
   the SOM)

### Fix

```bash
# 1. Verify the Kria is powered + booted:
#    The green status LED on the SOM should be solid (not blinking).

# 2. If using NetworkManager "Shared to other computers":
#    Make sure the connection is enabled.
nmcli connection show
# Look for a connection in "shared" mode. Activate it if it's not already:
nmcli connection up "<name>"

# 3. Wait 30 seconds, then retry:
arp -a | grep '00:0a:35'

# 4. If still nothing, plug a monitor + USB keyboard into the Kria and
#    log in directly to check `ip a`. The Kria might have an IP your
#    laptop can't see.
```

### Prevention

If you use a static IP (recommended for thesis development), set it via
`netplan` on the Kria after first boot. Then you always know where it
is.

## ERR_CONNECTION_REFUSED

### Symptom

After `sudo bash run_live.sh ...`, Jupyter starts and prints a URL.
Pasting that URL in your laptop's browser gives `ERR_CONNECTION_REFUSED`.

### Cause

You're trying to connect to `localhost` but Jupyter is on the Kria.

If `JUPYTER_HOST=0.0.0.0` (the default since `v0.6`), the URL in your
browser needs the Kria's IP, not `localhost`. The script's banner
prints the correct URL — the issue is usually that you copy-pasted
Jupyter's own output (which prints `localhost:8888/...`) instead of
the banner's URL.

### Fix

Find the **banner** that `run_live.sh` printed (above Jupyter's output):

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

Take the token from Jupyter's output, paste into the banner's URL
template.

If you really want to use `localhost:8888` in the browser, you need an
SSH tunnel — see [USAGE.md → Running the live demo](./USAGE.md#5-running-the-live-demo)
for the `JUPYTER_HOST=127.0.0.1` workflow.

### Prevention

The banner is meant to be unambiguous. If users still get this wrong,
we could add a "click here" link directly in the banner (with the
auto-detected IP pre-filled), but that requires terminal escape codes
that don't always render correctly in SSH.

---

## Found something not on this list?

The patterns documented here are stable — `sudo` strips env vars, FPGA
state survives across kernel restarts but not reboots, `cv2.VideoCapture`
calls are hint-based — and they explain about 95% of new failures
empirically.

When debugging a new issue, the most useful one-liner is:

```bash
# What's the state of the env, the FPGA, and the camera?
echo "EUID: $EUID, USER: $USER, SUDO_USER: $SUDO_USER"
echo "XILINX_XRT: $XILINX_XRT"
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
echo "---"
sudo xmutil listapps 2>&1 | tail -5
echo "---"
lsusb | grep -iE "logitech|046d"
ls -la /dev/video* 2>&1 | head -3
echo "---"
sudo fuser /dev/video0 2>&1
```

That snapshot is almost always enough to identify which section above
applies, or to localize a new issue.

## VAI 3.5 benchmark workflow issues

### Stage script fails with `SSL: CERTIFICATE_VERIFY_FAILED`

```
[get]  resnet50: https://www.xilinx.com/bin/public/...
    FAILED: <urlopen error [SSL: CERTIFICATE_VERIFY_FAILED]
    certificate verify failed: unable to get local issuer certificate>
```

**Cause**: Your laptop's CA bundle doesn't trust xilinx.com's cert chain.
This was specifically a Kria issue (Ubuntu 22.04 LTS for Kria has an
older CA bundle) but can also happen on older laptop installs.

**Fix**: Install `certifi`. The stage script uses certifi's CA bundle
preferentially when available, falling back to unverified SSL only as a
last resort.

```bash
pip install certifi
# Then re-run
bash scripts/host/04_stage_benchmark.sh
```

If certifi was already installed and SSL still fails, the script
automatically falls back to unverified SSL — you'll see `[warn]
downloaded with unverified SSL` in the output. This is acceptable for
the trusted AMD URLs we hit.

### `rsync: mkdir failed: Permission denied`

```
rsync: [Receiver] mkdir "/home/ubuntu/KriaKv260_Model_Compiler/notebooks/..." failed: Permission denied (13)
```

**Cause**: An earlier Jupyter session running as root created
`notebooks/` subdirectories that are root-owned. Your SSH session as
`ubuntu` can't write into them.

**Fix**: Reset ownership on the Kria.

```bash
ssh -t ubuntu@<kria-ip> 'sudo chown -R ubuntu:ubuntu /home/ubuntu/KriaKv260_Model_Compiler/notebooks/'
```

Note `-t` to allocate a TTY for the sudo prompt. Then re-run rsync.

### Benchmark notebook fails: `RuntimeError: Benchmark data not staged`

The notebook's prerequisite-check cell raises this when models or
datasets are missing.

**Cause**: You launched the notebook before running the host-side staging
scripts, or the sync didn't complete cleanly.

**Fix**: From your laptop, run both stages of the workflow:

```bash
bash scripts/host/04_stage_benchmark.sh                          # downloads
bash scripts/host/05_sync_benchmark_to_kria.sh ubuntu@<kria-ip>  # push to Kria
```

Then re-run the prerequisite-check cell.

### Stage script aborts with `insufficient free disk space`

```
ABORT: insufficient free disk space.
  Need:  15.0 GB
  Have:  8.2 GB
```

**Cause**: The script's pre-flight check requires ≥15 GB free at the
stage root (catalogue size + ~3 GB safety margin).

**Fix options**:

1. Free space on your laptop:
   ```bash
   docker system prune -a       # Vitis-AI Docker images can be ~25 GB
   du -sh ~/Downloads/* | sort -h
   ```

2. Override the check if you know what you're doing:
   ```bash
   bash scripts/host/04_stage_benchmark.sh --min-free-gb 5
   ```
   The check is conservative — actual peak usage is ~12 GB during
   downloads, dropping to ~6 GB after tarball cleanup.

### Kria SD card corrupted under sustained writes

This is the issue that motivated the host-driven workflow in the first
place. Symptoms:

- `fsck exited with status 4` on next boot
- `unable to set superblock flags` when running fsck manually
- BusyBox emergency shell on boot

**Cause**: Consumer SD card controllers handle sustained heavy I/O badly.
The original in-notebook download wrote ~10 GB sequentially while the
controller was also serving system reads. Combined with kernel write-back
cache, the corruption window was large.

**Recovery** (if it happens):

1. Try fsck with backup superblocks:
   ```bash
   # From the emergency shell
   fsck.ext4 -y -b 32768 /dev/mmcblk1p2
   # If that fails:
   fsck.ext4 -y -b 98304 /dev/mmcblk1p2
   ```

2. If fsck can't recover, re-flash the SD card with a fresh image. The
   Pass 5 install scripts (`scripts/kria/01_install_vai35.sh` etc.) make
   recovery reliable.

**Prevention**: always use the host-driven workflow
(`scripts/host/04_stage_benchmark.sh` + `05_sync_benchmark_to_kria.sh`)
for benchmark data. Never run sustained downloads directly on the Kria.

### Sync to Kria fails with `Insufficient remote disk space`

```
Insufficient remote disk: need ~13 GB, have 4 GB.
```

**Cause**: Your Kria's SD card doesn't have room for the staged data.

**Fix options**:

1. Clean up old benchmark data on the Kria:
   ```bash
   ssh ubuntu@<kria-ip> '
       rm -rf ~/KriaKv260_Model_Compiler/notebooks/Models_VAI35
       rm -rf ~/KriaKv260_Model_Compiler/notebooks/Datasets
   '
   ```

2. If still tight, use a larger SD card. 32 GB minimum is recommended
   for this workflow; 64 GB gives comfortable headroom.

### ImageNetV2 label resolution shows `N labels not in name-to-index mapping`

```
warning: 2 labels not in name-to-index mapping
ImageNet sample: 4998 images
```

**Cause**: Some class names in your labels.txt don't match Keras's
standard `imagenet_class_index.json` names (case, spacing, etc.).

**Fix**: Usually harmless — you lose a few of ~10K images. If you need
exact resolution, check the staged `imagenet_class_index.json` matches
the labels.txt format. The staging script generates labels.txt from
ImageNetV2's directory structure (where directory name IS the class
index), so this shouldn't happen if you used the host script. If it does,
your `labels.txt` may have been manually edited — re-run staging.

## VAI 3.5 install reports "Current VAI version: 3.5. Upgrading to 3.5." then fails

**Symptom**: Re-running `scripts/kria/01_install_vai35.sh` on a Kria where
VAI 3.5 is *already correctly installed* (verified via `dpkg -l libvart`
showing `3.5.0-1`), stage 4 reports it's about to "upgrade" and then
fails downloading from xilinx.com:

```
━━ [4/5] VAI 3.5 runtime upgrade
[*] Current VAI version: 3.5. Upgrading to 3.5.
[*]   Downloading from https://www.xilinx.com/bin/public/openDownload?filename=vai3.5_kr260.zip
ERROR: cannot verify www.xilinx.com's certificate, issued by 'CN=R13,O=Let's Encrypt,C=US':
  Unable to locally verify the issuer's authority.
[FAIL] Download failed.
```

### Two stacked issues

**Issue 1**: The stage 4 idempotency check requires *both* the installed
version to be 3.5 *and* the stamp file `/var/local/kriakv260_vai35.done`
to exist. On a Kria where VAI 3.5 was installed via some prior path that
didn't write the stamp, the check fails despite the runtime being correct,
and the script proceeds to "upgrade VAI 3.5 → VAI 3.5".

**Issue 2**: Once it gets to the download, `wget` on some Kria images
doesn't pick up the system CA bundle automatically, causing SSL
verification to fail on xilinx.com's Let's Encrypt R-series intermediate
certificates — even though `ca-certificates` is properly installed.

### Detection

Confirm VAI 3.5 is actually correctly installed:

```bash
# Runtime packages
dpkg -l 2>/dev/null | grep -E '^ii\s+(libvart|libxir|libvitis-ai-library|libtarget-factory)' \
    | awk '{print $2, $3}'
# All four should print version 3.5.0-1
```

Confirm DPU programming works:

```bash
sudo XILINX_XRT=/usr LD_LIBRARY_PATH=/usr/lib \
    /usr/local/share/pynq-venv/bin/python3 -c "
import subprocess
subprocess.run(['xmutil', 'unloadapp'], capture_output=True)
from pynq_dpu import DpuOverlay
DpuOverlay('dpu.bit')
print('OK — VAI 3.5 functional')
"
```

If both succeed, your VAI 3.5 install is fine; the script was wrongly
trying to "upgrade" it.

### Fix (immediate — for a system that's already in this state)

```bash
sudo touch /var/local/kriakv260_vai35.done
bash scripts/kria/01_install_vai35.sh
```

The stamp file unblocks the idempotency check. The re-run should now
show stages 4 and 5 as `↷ SKIPPED`.

### Fix (permanent — patched in v0.7.1)

The install script was patched to:

1. **Detect when current VAI version equals the target (3.5)** and write
   the stamp if missing, instead of requiring both conditions before
   skipping. The version match is sufficient evidence of a correct
   install; the stamp can be regenerated.

2. **Pass `--ca-directory=/etc/ssl/certs` to wget** so it uses the system
   CA bundle reliably, fixing the Let's Encrypt R13 cert verification
   failure on Kria images where wget doesn't auto-pick the default CA dir.

3. **Print clearer diagnostics** distinguishing "stamp missing but
   runtime present" from "runtime genuinely needs install".

### Workaround if you can't update the script (older Kria with restricted network)

Download the zip on your laptop (where SSL works) and scp to the Kria:

```bash
# On laptop
wget --ca-directory=/etc/ssl/certs \
    "https://www.xilinx.com/bin/public/openDownload?filename=vai3.5_kr260.zip" \
    -O vai3.5_kr260.zip
scp vai3.5_kr260.zip ubuntu@<kria-ip>:/tmp/

# On Kria: pre-populate the script's staging directory so it doesn't try to download
sudo mkdir -p /home/ubuntu/.cache/kriakv260_install
sudo cp /tmp/vai3.5_kr260.zip /home/ubuntu/.cache/kriakv260_install/

# Re-run — the script detects the existing zip and skips the wget
bash scripts/kria/01_install_vai35.sh
```

## Benchmark sync to Kria: ImageNet images are broken symlinks

**Symptom**: After running `scripts/host/05_sync_benchmark_to_kria.sh`,
the Kria reports a much smaller `Datasets/` size than the laptop. For
example: laptop shows 3.3 GB, Kria shows 2.1 GB — and the
`imagenet_sample/images/` directory contains broken symbolic links
pointing back at the laptop's filesystem.

```
ubuntu@kria:~$ ls -la ~/KriaKv260_Model_Compiler/notebooks/Datasets/imagenet_sample/images/ | head -3
lrwxrwxrwx 1 ubuntu ubuntu 191 May 11 19:37 cls0000_xxx.jpeg ->
    /home/aaljaberi/.../build/benchmark_stage/Datasets/imagenetv2-matched-frequency-format-val/0/xxx.jpeg
```

The `file` command confirms it:

```
broken symbolic link to /home/aaljaberi/.../...
```

### Cause

The host-side staging script (`scripts/host/_stage_benchmark.py`) creates
symlinks under `imagenet_sample/images/` that point to the extracted
ImageNetV2 source tree at
`Datasets/imagenetv2-matched-frequency-format-val/`. This keeps the
local staging area compact (each image lives once).

In versions before v0.7.2, the sync script used `rsync --archive` which
preserves symlinks **as symlinks** — copying the link text but not the
target. Combined with an `--exclude` rule that intentionally skipped
the source tree (since it was thought to be redundant), the Kria
received broken pointers.

### Detection

```bash
# On the Kria
ssh ubuntu@<kria-ip> '
  FIRST=$(ls ~/KriaKv260_Model_Compiler/notebooks/Datasets/imagenet_sample/images/ | head -1)
  F=~/KriaKv260_Model_Compiler/notebooks/Datasets/imagenet_sample/images/$FIRST
  file "$F"
'
```

If `file` reports "broken symbolic link", you have this bug.

### Fix (immediate — re-sync just the ImageNet subtree)

```bash
# On laptop
cd ~/Documents/Girona_Masters/Thesis/KriaKv260_Model_Compiler

rsync -ah --info=progress2 --copy-links \
    build/benchmark_stage/Datasets/imagenet_sample/ \
    ubuntu@<kria-ip>:/home/ubuntu/KriaKv260_Model_Compiler/notebooks/Datasets/imagenet_sample/
```

The `--copy-links` flag dereferences symlinks during transfer, sending
the real JPEG content in place of each link. ~1.3 GB transfer over LAN.

### Verify the fix

```bash
ssh ubuntu@<kria-ip> '
  du -sh ~/KriaKv260_Model_Compiler/notebooks/Datasets/imagenet_sample
  # Should be ~1.3G

  FIRST=$(ls ~/KriaKv260_Model_Compiler/notebooks/Datasets/imagenet_sample/images/ | head -1)
  F=~/KriaKv260_Model_Compiler/notebooks/Datasets/imagenet_sample/images/$FIRST
  ls -la "$F"
  # Should show -rw-r--r-- (regular file), not lrwxrwxrwx (symlink)
  file "$F"
  # Should report JPEG image data, not "broken symbolic link"
'
```

### Permanent fix in v0.7.2

The sync script (`scripts/host/05_sync_benchmark_to_kria.sh`) was patched
to pass `--copy-links` to rsync by default, so future fresh syncs send
real images. The `--inplace` flag was also dropped (it's incompatible
with overwriting symlinks with regular files).
## QAT abandoned for PTQ + hard-negative training

### Symptom

Vitis-AI's `pytorch_nndct.QatProcessor` integration with Ultralytics'
training loop fails or produces unstable training, despite the QAT API
being documented in Vitis-AI 3.5.

### Cause

`QatProcessor` expects a PyTorch-native training loop where forward
hooks, backward hooks, and parameter management are directly
accessible. Ultralytics' `BaseTrainer` wraps modules with its own
forward hooks (for loss callbacks), uses its own DDP setup, and
re-instantiates the model from YAML during `setup_model()` in ways
that conflict with NNDCT's wrapping.

Specifically:
1. NNDCT wraps modules with `QuantStub` markers, but Ultralytics'
   re-instantiation strips the markers
2. NNDCT expects to control `optimizer.zero_grad()` / `loss.backward()`
   ordering, but Ultralytics' trainer owns the optimizer
3. NNDCT's gradient hooks interact badly with Ultralytics' AMP scaler

Workarounds (custom training loop bypassing the Ultralytics trainer)
were investigated but proved fragile and added significant
maintenance debt.

### Fix

Abandon QAT in favor of:

1. **PTQ via NNDCT** — the default path; no training-loop integration
   needed
2. **Hard-negative training** — augment your dataset with images that
   contain no target-class objects but visually match the deployment
   environment (see [YOLOV11.md "Hard-negative training workflow"](./YOLOV11.md#hard-negative-training-workflow))
3. **Larger model capacity** — switch to yolov11s (or larger) to get
   more weight redundancy against int8 noise (see
   [YOLOV11.md "Capacity vs quantization"](./YOLOV11.md#capacity-vs-architecture-what-int8-quantization-actually-depends-on))

Empirically, this combination reduces deployment int8 false positives
by 67% on the eggs benchmark without any training-pipeline changes.

### Prevention

For future work on a different DPU target or with a different training
framework, QAT may still be worth attempting. The infrastructure for
hardware-friendly training (the monkey-patches in
`scripts/host/_train_yolov11.py`) is independent of the
QAT-vs-PTQ choice.

---

## High int8 false-positive rate despite clean float model

### Symptom

A trained YOLOv11n model in PyTorch float (`.pt`) shows zero or very
few false positives on a held-out industrial test set at conf=0.85.
After compiling to xmodel and deploying on the Kria, the same model
produces thousands of false positives on the same images. Mean
confidence of the false positives is suspiciously saturated (mean ≈
0.96).

### Cause

This is the int8 quantization tax. The PyTorch float model has
~32-bit float precision throughout; the DPU runs ~8-bit fixed-point
with per-tensor activation scales. For fine-grained single-class
discrimination tasks (e.g., "is this egg-shaped object actually an
egg, vs a plastic basket or cardboard packaging?"), the int8 precision
loss erodes the decision boundary, producing high-confidence false
positives on visually similar non-target objects.

The DPU hardware constraint (per-tensor, not per-channel, activation
scales) means there's no quantizer-level fix — the constraint is
baked into the silicon.

### Fix

In order of effectiveness (and complexity):

1. **Raise the deployment confidence threshold.** Cheapest. Works if
   your true positives have separable confidence from false positives.
   For the eggs deployment: threshold 0.85 separates clean true
   positives from background false positives reasonably well.

2. **Switch to a larger model variant.** Most effective. yolov11s vs
   yolov11n produced **67% fewer FPs** at conf=0.85 with zero detection
   precision loss. Throughput cost: ~33% (25.8 → 17.2 FPS). See
   [YOLOV11.md "Capacity vs quantization"](./YOLOV11.md#capacity-vs-architecture-what-int8-quantization-actually-depends-on).

3. **Add hard-negative training data.** Helps the float model
   significantly (eggs deployment: from ~12 FPs at float to 0 FPs at
   float). May or may not transfer to int8 — measure both. For the
   eggs deployment, hard-neg training fixed float but did not by
   itself reduce int8 FPs.

4. **Verify calibration set composition.** Use in-domain images only.
   Mixing in-domain and out-of-domain calibration images *increases*
   int8 FPs by widening per-tensor activation scales. See "Mixed
   calibration set INCREASES int8 false positives" below.

### Prevention

Choose the smallest model variant that fits your **deployment int8
accuracy budget**, not your **training float accuracy budget**. They
are different. A model that's perfect at float can still produce
catastrophic int8 deployment performance on cluttered backgrounds.

Measure deployment int8 quality on representative test imagery early
(after the first compile + sync), before committing to a model size.

---

## Mixed calibration set INCREASES int8 false positives

### Symptom

You augment your training data with hard-negative images (industrial
background frames without target class). You decide to also include
those hard-neg images in the calibration set so the quantizer "knows
about them." Your int8 false-positive rate gets *worse*, not better.

### Cause

The DPU hardware uses **per-tensor** (not per-channel) activation
scales for all int8 quantization. The calibration step measures
activation min/max across the calibration images and chooses a single
scale per tensor that covers the observed range.

When you mix in-domain images (e.g., eggs on conveyor) with
out-of-domain images (e.g., empty conveyor / packaging machinery), you
*widen* the observed activation range across the calibration set. The
chosen per-tensor scales become coarser. Each int8 bucket now covers a
wider float range, so per-layer quantization noise increases.

That extra noise amplifies through the network depth and produces
**more** spurious high-confidence detections, not fewer.

Empirical observation (v0.10 yolov11n):

| Calibration set | int8 FPs @ 0.85 |
|---|---:|
| 600 in-domain (eggs) images | 4,220 |
| 300 in-domain + 300 hard-neg | 6,609 (+57%) |

### Fix

Calibrate with **in-domain images only**. Even when your training
set includes hard-negative images, your calibration set should not.

```bash
# Eggs example: hardneg dir is for training, calib dir is in-domain only
python3 scripts/host/_train_yolov11.py \
    --data data/datasets/eggs_hardneg/data.yaml \   # mixed training data
    ...

NUM_CLASSES=1 bash scripts/host/02_compile.sh yolov11 yolov11n \
    data/weights/yolo11n_eggs_dpu.pt \
    data/calib/                                      # in-domain only!
```

### Prevention

Treat calibration set composition as a hyperparameter. Measure int8
deployment quality with in-domain vs mixed calibration; pick whichever
performs better on real test imagery. The default should be
in-domain only.

For per-channel-quantization DPUs (newer Versal AI Edge devices,
possibly future Kria revisions), this constraint may not apply. The
per-tensor scale limitation is specific to DPUCZDX8G_ISA1 (KV260
B4096).

---

## `vai_q_onnx` crashes in `align_concat` on YOLOv11

### Symptom

You try the alternative ONNX-based PTQ path via `vai_q_onnx`:

```bash
bash scripts/host/_quantize_onnx_yolov11.sh \
    out/yolov11n/yolov11n.onnx \
    out/yolov11n/yolov11n_quant.onnx
```

It crashes during quantization with:

```
TypeError: '<' not supported between instances of 'NoneType' and 'int'
  at: vai_q_onnx/quantize/refine_model.py:..., in pass_align_concat
```

### Cause

`vai_q_onnx 1.14.0`'s `align_concat` refinement pass assumes all
Concat-node inputs have inferred shapes available. In our YOLOv11
graph (which combines a custom `C2PSA_DPU` attention block, stripped
detect head, and NHWC permute wrapper), some Concat inputs end up
with `None` in their shape inference. The refinement pass then tries
to compare `None < int` and crashes.

The failure is independent of:
- `per_channel` (True or False)
- `quant_format` (`VitisQuantFormat.FixNeuron` or `VitisQuantFormat.QDQ`)
- `N_CALIB` value (tested 50, 200, 600)
- `optimize_model` (True or False)

### Fix

There is no workaround in `vai_q_onnx 1.14.0` for our YOLOv11
architecture. Use the default NNDCT path instead:

```bash
# Default path; works correctly
NUM_CLASSES=1 bash scripts/host/02_compile.sh yolov11 yolov11n \
    data/weights/yolo11n_eggs_dpu.pt data/calib/
```

The ONNX export script (`_export_onnx_yolov11.sh`) still works and is
useful for other purposes (e.g., evaluating the model in
onnxruntime, validating the graph topology). Only the PTQ step is
blocked.

### Prevention

The DPU hardware uses per-tensor activation scales — the main
theoretical advantage of the ONNX path (per-channel weight
quantization) is negated by the hardware constraint anyway. For this
DPU, the NNDCT path is the appropriate choice.

A future `vai_q_onnx` release may fix the `align_concat` issue. The
ONNX export script is retained in-tree so this path can be revisited
without redoing the graph-preparation work.

---

## DPU resource lock — `xrt_device_handle_imp` Check failed

### Symptom

You try to launch a notebook or run an inference script on the Kria
and get:

```
F0517 23:14:07.123456  4567 xrt_device_handle_imp.cpp:101]
Check failed: r == 0 (1 vs. 0) cannot set read range!
*** Check failure stack trace: ***
```

The Kria's DPU appears locked; subsequent attempts produce the same
error.

### Cause

A previous Python process holding the DPU device handle exited
ungracefully (e.g., kernel killed by OOM, notebook kernel crashed
without proper cleanup, ssh disconnected during inference). The XRT
runtime considers the device still claimed.

### Fix

Kill any stale Python processes, then retry:

```bash
# On Kria
sudo pkill -f 'python3.*xmodel'
sudo pkill -f 'jupyter'

# If that's not enough, force-unload and reload the DPU overlay
sudo xmutil unloadapp
sleep 2

# Now retry your inference / notebook launch
sudo bash scripts/kria/run_live.sh yolov11n
```

If the issue persists, reboot the Kria:

```bash
sudo reboot
```

The systemd unit will reapply tuning after reboot.

### Prevention

Always exit Python sessions gracefully. In notebooks, use "Kernel →
Shutdown" rather than just closing the browser tab. For long-running
scripts, install a SIGTERM handler that releases the DPU runner:

```python
import signal
def cleanup(*args):
    runner.close()  # or del runner; del overlay
    sys.exit(0)
signal.signal(signal.SIGTERM, cleanup)
```

The PYNQ-DPU library should auto-release on Python interpreter exit,
but in practice it relies on Python's garbage collection running
before XRT's device-handle finalizer — which doesn't always happen on
abnormal termination.

---

## Vitis-AI container OOM during PTQ calibration

### Symptom

Inside the Vitis-AI container, the NNDCT quantize step (`_quantize.py`
or the ONNX `quantize_static`) is killed by the kernel with no error:

```
[*] Running NNDCT calibration with N_CALIB=200 images...
Killed
```

`dmesg` on the host shows:

```
Out of memory: Killed process 12345 (python3) ...
```

### Cause

NNDCT loads the entire model graph plus all calibration activations
into memory for analysis. For YOLOv11 at imgsz=640 with N_CALIB=200,
peak memory is ~14 GB. If the container has less, or if the host has
less free RAM than the container's limit, the kernel OOM-killer fires.

### Fix

Reduce `N_CALIB`:

```bash
# Default
N_CALIB=200 bash scripts/host/02_compile.sh yolov11 yolov11n ...

# Reduced for memory-constrained environments
N_CALIB=50 bash scripts/host/02_compile.sh yolov11 yolov11n ...
```

`N_CALIB=50` typically uses ~4-6 GB and works on 16 GB hosts.
Quantization quality is slightly worse than at `N_CALIB=200` but
usually within 1-2% of the same int8 FP count.

If the host has more RAM but the container is capped, increase the
container's memory limit in `02_compile.sh` (look for `--memory` or
`-m` in the `docker run` invocation).

### Prevention

Monitor RAM during compile:

```bash
# In a separate terminal while 02_compile.sh runs:
watch -n 1 'free -h; docker stats --no-stream'
```

If RAM is consistently >80% used during the NNDCT step, lower N_CALIB
preemptively rather than waiting for the OOM-killer.

For YOLOv5 at imgsz=320, memory usage is much lower (~2 GB even at
N_CALIB=500); the constraint is specific to larger YOLOv11 input
sizes.

---

## Pip install fails as non-root inside the Vitis-AI conda env

### Symptom

Inside the Vitis-AI container:

```
$ pip install onnx onnxruntime
ERROR: Could not install packages due to an OSError: [Errno 13]
Permission denied: '/opt/vitis_ai/conda/envs/vitis-ai-pytorch/lib/python3.7/site-packages/...'
```

### Cause

The Vitis-AI container's conda environment is owned by root. The
container's default user is `vitis-ai-user` (UID 1000) for security
reasons. Pip can't write to the system-wide site-packages without
sudo, but `pip install --user` fails for other reasons (see "Numpy
import error" below).

### Fix

Two options:

**Option A: install as root via sudo inside the container**

```bash
docker exec -u root <container-id> bash -lc '
    source /opt/vitis_ai/conda/etc/profile.d/conda.sh
    conda activate vitis-ai-pytorch
    pip install onnx onnxruntime
'
```

**Option B: build a derived image with the packages baked in (recommended for repeatable workflows)**

```dockerfile
# Dockerfile.eggs
FROM xilinx/vitis-ai-onnx-cpu:latest
USER root
RUN source /opt/vitis_ai/conda/etc/profile.d/conda.sh && \
    conda activate vitis-ai-pytorch && \
    pip install onnx onnxruntime onnx-simplifier
USER vitis-ai-user
```

Then build and use the derived image:

```bash
docker build -f Dockerfile.eggs -t vitis-ai-onnx-cpu:eggs .
# Use it in 02_compile.sh by setting VAI_IMAGE=vitis-ai-onnx-cpu:eggs
```

This is what the v0.10 ONNX investigation used (the derived image
`vitis-ai-onnx-cpu:eggs` exists on the host with `vai_q_onnx`'s
runtime dependencies pre-installed).

### Prevention

Treat the Vitis-AI base images as read-only and build derived images
for any persistent package needs. This avoids both the permission
issue and the partial-install corruption problem below.

---

## Numpy import error after partial pip install in the container

### Symptom

After a `pip install` that failed midway (e.g., interrupted by Ctrl-C,
or hit the permission error above and was retried with `--user`),
imports start failing:

```python
>>> import numpy
ImportError: numpy.core.multiarray failed to import
RuntimeError: module compiled against API version 0x10 but this version of numpy is 0xf
```

### Cause

Pip with `--user` writes to `~/.local/lib/python3.7/site-packages/`.
If this path appears earlier than the conda env in `sys.path`, a
partial numpy install in `--user` shadows the working numpy in the
conda env. The mismatch in C-extension API versions breaks the import.

### Fix

```bash
# Inside the container, as the user who ran the bad install:
rm -rf ~/.local/lib/python3.7/site-packages/numpy*
rm -rf ~/.local/lib/python3.7/site-packages/scipy*  # often co-corrupted
```

Then verify:

```bash
python3 -c "import numpy; print(numpy.__file__)"
# Should print a path inside /opt/vitis_ai/conda/...
```

If you need additional packages, use one of the methods in "Pip
install fails as non-root" above (don't use `--user` again).

### Prevention

Never use `pip install --user` inside the Vitis-AI container. Either
install as root via `docker exec -u root`, or build a derived image
with the packages pre-installed.

---

## `measure_fps_kria.py` hardcoded to yolov11n

### Symptom

You're measuring FPS for `yolov11s` and the output shows yolov11n
input dimensions / spec parameters, even though the xmodel path
clearly points to a yolov11s file.

### Cause

The original `measure_fps_kria.py` script had `spec_name = "yolov11n"`
hardcoded near the top, even though the xmodel path was a CLI
argument. The spec was used to size preprocessor buffers and to drive
the decoder.

For yolov11n and yolov11s, this is mostly harmless because the spec
parameters (family, imgsz, nc, reg_max) are identical for both. But
the output report is misleading and the spec is wrong-by-name.

### Fix

Patch the script to derive `spec_name` from the xmodel path:

```python
# At the top of measure_fps_kria.py
from pathlib import Path
spec_name = Path(XMODEL).parent.name   # e.g. "yolov11s" from /path/yolov11s/yolov11s_kv260.xmodel
spec = get_spec(spec_name)
```

This works because the sync layout always uses
`/home/ubuntu/xmodels_vai35/<variant>/<variant>_kv260.xmodel`.

### Prevention

When adding new variants, audit all standalone measurement scripts on
the Kria for hardcoded variant names. The repo's `run_live.sh` and
notebooks read `LPR_VARIANT` from the environment, but ad-hoc
benchmark scripts (`measure_fps_kria.py`, `benchmark_fps.py`,
`sanity_*.py`) tend to hardcode.

A more robust pattern: always derive variant from the xmodel path,
not from a hardcoded constant.

---

## `run_live.sh` rejects new variants with "unknown variant"

> **RESOLVED in v0.11.** `scripts/kria/run_live.sh` now resolves
> `variant → family` via the `lpr_pipeline.shared.models` registry and
> dispatches on `family`, not on a hardcoded variant whitelist. Any
> variant added to the registry works automatically — no script edit
> required. The historical entry below describes the v0.10-era issue
> and is retained for users on older versions.

### Symptom (v0.10 and earlier)

You add a new variant (e.g., `yolov11s`) to the registry, sync its
xmodel to the Kria, and try to launch:

```bash
$ sudo bash scripts/kria/run_live.sh yolov11s
[FAIL] Unknown variant: yolov11s
```

### Cause (v0.10 and earlier)

`run_live.sh` had a `case` statement that mapped variant names to
notebook paths. When a new variant is added to the registry, the
script's `case` block needs an explicit entry — it doesn't read the
registry.

### Fix (v0.10 and earlier)

Edit `scripts/kria/run_live.sh`, find the variant dispatch (search
for "Unknown variant"), and add the new variant to the case branch
that uses the right notebook:

```bash
case "${VARIANT}" in
    yolov5n|yolov5s)
        NOTEBOOK="${NOTEBOOK:-notebooks/02_deploy_text.ipynb}"
        ;;
    yolov11n|yolov11s)              # add new variant here
        NOTEBOOK="notebooks/eggs/05_deploy_visual.ipynb"
        ;;
    *)
        log_err "Unknown variant: ${VARIANT}"
        exit 1
        ;;
esac
```

Then re-sync the script to the Kria:

```bash
# From laptop
rsync -av scripts/kria/run_live.sh ubuntu@<kria-ip>:/home/ubuntu/KriaKv260_Model_Compiler/scripts/kria/
```

### Resolution (v0.11)

The variant whitelist was replaced with a family-based dispatch that
reads the registry as the single source of truth:

```bash
FAMILY=$(python3 -c "
import sys
sys.path.insert(0, '$REPO_ROOT')
from lpr_pipeline.shared.models import get_spec
print(get_spec('$VARIANT').family)
")

case "$FAMILY" in
    yolov5)  ... ;;
    yolov11) ... ;;
    yolox)   ... ;;
esac
```

Adding a new variant within an existing family (e.g., `yolov5m`,
`yolov11m`) requires only a registry entry and a weights file. Adding
a new family (e.g., a hypothetical `yolov12`) still requires a new
`case` arm because the family-to-notebook mapping is genuinely
family-specific. The error messages distinguish "variant not in
registry" from "family has no notebook" so the right fix is obvious.

---

## Eggs notebook hardcoded `get_spec("yolov11n")` — cosmetic warning

### Symptom

You launch `run_live.sh yolov11s` and the eggs notebook
(`notebooks/eggs/05_deploy_visual.ipynb`) loads. Inference works
correctly — detections appear, FPS is reported. But the notebook's
status panel says `spec=yolov11n` even though you're running yolov11s.

### Cause

The notebook has `spec = get_spec("yolov11n")` hardcoded in
configuration cell 2. The yolov11n and yolov11s specs happen to be
identical (same family, imgsz, nc, reg_max), so the runner works
correctly regardless — but the displayed spec name is wrong.

### Fix

For correctness, change the notebook's cell 2 to read the variant
from the environment:

```python
import os
VARIANT = os.environ.get("LPR_VARIANT", "yolov11n")
spec = get_spec(VARIANT)
```

This is the same pattern the YOLOv5 notebooks use.

### Prevention

When adding a notebook for a new variant or family, always read
`LPR_VARIANT` from the environment (set by `run_live.sh`) rather than
hardcoding a name. Hardcoded names are a regression risk every time a
new variant is added to that family.

---

## bash `!` history expansion mangles commit messages

### Symptom

You're crafting a git commit message in a multi-line `git commit -m
"..."` invocation. One of the lines contains a `!` character (e.g.,
"override the !data/weights/*.pt rule"). Bash interrupts with:

```
bash: !data/weights/: event not found
```

The commit still succeeds, but the message in the commit object may
be missing the line containing `!`.

### Cause

Interactive bash (with `set +H` not in effect) treats `!` as the
history-expansion sigil. In `"..."` double-quoted strings, `!` is
still expanded; you'd need single quotes or to escape the `!` to
disable expansion.

### Fix

Three workarounds, in order of convenience:

**Option A: use single quotes for the commit message**

```bash
git commit -m '
v0.10: gitignore changes
Override the !data/weights/*.pt rule to track trained checkpoints.
'
```

**Option B: disable history expansion for this shell**

```bash
set +H
git commit -m "..."  # now ! is literal
```

**Option C: escape the bang**

```bash
git commit -m "Override the \!data/weights/*.pt rule"
# Note: bash will keep the backslash in the commit message; you may not want this
```

**Option D: use `git commit` with an editor and a draft file**

```bash
$EDITOR /tmp/commit-msg.txt   # edit freely; no shell interpretation
git commit -F /tmp/commit-msg.txt
```

### Verifying the commit message

After committing despite the warning, verify the message looks right:

```bash
git log --format='%H%n%n%B%n---%n' -1 HEAD
```

If `!`-containing lines are missing, redo the commit with
`git commit --amend` using one of the workarounds above.

### Prevention

For multi-line commit messages with shell metacharacters, prefer
**Option D** (editor + `-F`). It avoids all shell-quoting issues
permanently.

For everyday two-line commit messages, just remember: `!` in
double-quoted bash strings is dangerous.
