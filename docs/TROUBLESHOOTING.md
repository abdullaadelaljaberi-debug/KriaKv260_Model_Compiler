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
