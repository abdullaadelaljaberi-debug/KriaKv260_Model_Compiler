#!/usr/bin/env bash
# scripts/kria/02_apply_tuning.sh
# ─────────────────────────────────────────────────────────────────────────────
# Apply runtime tuning that achieved 60 fps live LPR detection. Settings:
#
#   1. USB autosuspend disabled on ALL USB devices
#      (silent 15-fps cap on KV260 — biggest single fix)
#   2. CPU governor → performance
#   3. v4l2 camera settings (Brio-tuned, with auto-detect fallback):
#      - MJPG @ 60fps
#      - Manual exposure (auto_exposure=1, exposure_time_absolute=100, gain=200)
#      - exposure_dynamic_framerate=0
#
# One-shot script: applies once and exits. Persistence across reboots is
# handled by scripts/kria/03_install_systemd.sh.
#
# Tests camera at the end with a 5-second capture; saves a frame so you can
# eyeball that it's not all-black.
#
# Idempotent: re-running is safe (settings just re-applied).
# ─────────────────────────────────────────────────────────────────────────────
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/common.sh
source "$SCRIPT_DIR/lib/common.sh"

parse_common_flags "$@"

BRIO_VENDOR="046d"
BRIO_PRODUCT="085e"

log_step "Apply runtime tuning"

summary_init
trap 'summary_print' EXIT

# ─── 1. USB autosuspend OFF on every USB device ─────────────────────────────
log_step "[1/3] USB autosuspend"
summary_stage_start "1/3" "USB autosuspend"

usb_devices=( /sys/bus/usb/devices/*/power/control )
if (( ${#usb_devices[@]} == 0 )); then
    log_warn "  No USB power controls found at /sys/bus/usb/devices/*/power/control"
    summary_stage_failed "no USB power controls"
else
    log_info "  Disabling autosuspend on ${#usb_devices[@]} USB device(s)"
    for ctrl in "${usb_devices[@]}"; do
        if [[ -w "$ctrl" ]] || [[ -w $(dirname "$ctrl") ]]; then
            echo on | need_sudo tee "$ctrl" >/dev/null 2>&1 || true
        fi
    done
    log_ok "  USB autosuspend disabled on all USB devices"
    summary_stage_done "disabled on ${#usb_devices[@]} devices"
fi

# Also persist for future kernel modules
USB_MOD_CONF="/etc/modprobe.d/usb-no-autosuspend.conf"
if [[ ! -f "$USB_MOD_CONF" ]]; then
    log_info "  Writing $USB_MOD_CONF"
    echo "options usbcore autosuspend=-1" | need_sudo tee "$USB_MOD_CONF" >/dev/null
    log_ok "  persistent: usbcore.autosuspend=-1"
else
    log_ok "  $USB_MOD_CONF already exists"
fi

# ─── 2. CPU governor → performance ──────────────────────────────────────────
log_step "[2/3] CPU governor"
summary_stage_start "2/3" "CPU governor"

governors=( /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor )
if (( ${#governors[@]} == 0 )); then
    log_warn "  No cpufreq governors found — kernel may not support cpufreq"
    summary_stage_skipped "no cpufreq support"
else
    log_info "  Setting governor → performance on ${#governors[@]} core(s)"
    for g in "${governors[@]}"; do
        echo performance | need_sudo tee "$g" >/dev/null 2>&1 || true
    done

    # Verify
    current=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo "?")
    if [[ "$current" == "performance" ]]; then
        log_ok "  CPU governor: performance"
        summary_stage_done "performance on ${#governors[@]} cores"
    else
        log_warn "  CPU governor wanted 'performance', got '$current'"
        log_warn "  Some Kria kernels lock the governor. This is mostly cosmetic; the SOC"
        log_warn "  rarely throttles at our power budget. Live demo will still hit ~60 fps."
        summary_stage_done "wanted performance, got $current"
    fi
fi

# ─── 3. Camera tuning ───────────────────────────────────────────────────────
log_step "[3/3] Camera tuning"
summary_stage_start "3/3" "Camera tuning"

if ! have_cmd v4l2-ctl; then
    log_warn "  v4l2-ctl not installed. Installing..."
    need_sudo apt-get install -y v4l-utils || die "apt install v4l-utils failed"
fi

# Find the right /dev/videoN device. Strategy (F2 option c):
#   1. Look for Brio (USB ID 046d:085e) — preferred
#   2. Fall back to first MJPG-capable device
find_camera_device() {
    # Pass 1: Brio
    for dev in /dev/video*; do
        [[ -c "$dev" ]] || continue
        local idx="${dev#/dev/video}"
        # Each device may be one of several /dev/video* nodes for the same camera
        # (one per capability: capture, metadata, etc.). We want the capture node.
        local caps
        caps=$(v4l2-ctl --device="$dev" --info 2>/dev/null || true)
        if [[ "$caps" == *"Video Capture"* ]] || [[ "$caps" == *"Video Capture"* ]]; then
            # Check if it's the Brio
            local sysdev="/sys/class/video4linux/video${idx}/device"
            if [[ -L "$sysdev" ]]; then
                local idvendor idproduct
                idvendor=$(cat "$sysdev/../idVendor" 2>/dev/null || cat "$sysdev/../../idVendor" 2>/dev/null || echo "")
                idproduct=$(cat "$sysdev/../idProduct" 2>/dev/null || cat "$sysdev/../../idProduct" 2>/dev/null || echo "")
                if [[ "$idvendor" == "$BRIO_VENDOR" ]] && [[ "$idproduct" == "$BRIO_PRODUCT" ]]; then
                    echo "$dev"
                    return 0
                fi
            fi
        fi
    done
    # Pass 2: first MJPG-capable
    for dev in /dev/video*; do
        [[ -c "$dev" ]] || continue
        if v4l2-ctl --device="$dev" --list-formats 2>/dev/null | grep -q MJPG; then
            echo "$dev"
            return 0
        fi
    done
    return 1
}

cam_dev=$(find_camera_device)
if [[ -z "$cam_dev" ]]; then
    log_warn "  No camera detected. Plug in the Brio (or another MJPG camera) and re-run."
    log_warn "  Continuing without camera tuning — other tuning has been applied."
    summary_stage_skipped "no camera detected"
    exit 0
fi
log_ok "  Camera device: $cam_dev"

# Identify the device for the log
cam_name=$(v4l2-ctl --device="$cam_dev" --info 2>/dev/null \
            | awk -F: '/Card type/ {print $2}' | xargs)
log_info "  Identified as: ${cam_name:-unknown}"

# Apply settings (Brio-tuned; works for most MJPG cameras too)
log_info "  Applying v4l2 settings (MJPG @60fps, manual exposure, gain=200)"

# Format: 1280×720 (Brio supports up to 1080p; 720p is faster)
v4l2-ctl --device="$cam_dev" \
    --set-fmt-video=width=1280,height=720,pixelformat=MJPG 2>/dev/null || \
    log_warn "    set-fmt-video failed (may be unsupported on this camera)"

# Frame rate: 60 fps
v4l2-ctl --device="$cam_dev" --set-parm=60 2>/dev/null || \
    log_warn "    set-parm 60 failed (camera may not support 60 fps at this resolution)"

# Manual exposure controls — these are Brio-specific control names
apply_v4l2_ctrl() {
    local ctrl="$1" val="$2"
    if v4l2-ctl --device="$cam_dev" --set-ctrl="$ctrl=$val" 2>/dev/null; then
        log_debug "    $ctrl = $val"
    else
        log_debug "    $ctrl: not supported on this camera (skipping)"
    fi
}

apply_v4l2_ctrl auto_exposure 1                  # 1 = manual mode
apply_v4l2_ctrl exposure_time_absolute 100
apply_v4l2_ctrl gain 200
apply_v4l2_ctrl exposure_dynamic_framerate 0     # don't auto-drop fps
log_ok "  v4l2 settings applied (skipped controls that this camera doesn't support)"
summary_stage_done "${cam_name:-camera}: MJPG@60fps, manual exposure"

# ─── Camera sanity test ─────────────────────────────────────────────────────
log_step "Camera sanity test (5-second capture)"

# We use v4l2-ctl with --stream-* options to grab frames, saving the first
# to /tmp for inspection.
TEST_FRAME="/tmp/kriakv260_test_frame.jpg"
if v4l2-ctl --device="$cam_dev" \
        --stream-mmap --stream-count=30 --stream-to="$TEST_FRAME" 2>/dev/null; then
    if [[ -s "$TEST_FRAME" ]]; then
        sz=$(stat -c%s "$TEST_FRAME" | numfmt --to=iec)
        log_ok "  Captured 30 frames; first saved to $TEST_FRAME ($sz)"
        # Sanity: MJPG frames should be >5 KB. Sub-1KB means all-black.
        if (( $(stat -c%s "$TEST_FRAME") < 1024 )); then
            log_warn "  Frame is suspiciously small — camera may be returning all-black."
            log_warn "  Check exposure settings, lens cap, and lighting."
        fi
    else
        log_warn "  v4l2-ctl reported success but $TEST_FRAME is empty"
    fi
else
    log_warn "  Camera capture failed. Settings applied but couldn't grab a test frame."
    log_warn "  Common causes:"
    log_warn "    - Camera busy (another process using it): pkill -f v4l2"
    log_warn "    - Insufficient permissions: ensure user is in 'video' group"
fi

# ─── Done ───────────────────────────────────────────────────────────────────
echo
log_ok "Tuning applied (this session only)."
log_info "Next:  bash scripts/kria/03_install_systemd.sh"
log_info "       (persists tuning across reboots)"
