# Kria board setup

> **This document is filled in Pass 5** (Kria scripts). The full content
> walks a fresh user from "received the KV260 in the box" to "live demo
> running" with no manual intervention beyond physically inserting the SD
> card.
>
> Pass 5 deliverables (preview):
>
> - `scripts/host/04_flash_sd.sh` — host-side: download Ubuntu 22.04 LTS
>   for Kria, verify SHA256, list removable block devices, prompt with
>   model+size confirmation, dd image to chosen device, sync, eject
> - `scripts/kria/00_check_prereqs.sh` — verifies network, sudo, expected
>   hardware
> - `scripts/kria/01_install_vai35.sh` — automates the manual VAI 2.5 → 3.5
>   upgrade (downloads `vai3.5_kr260.zip`, dpkg-installs the 5 debs in order,
>   copies `lack_lib/*` to `/usr/lib/`, replaces XRT binaries)
> - `scripts/kria/02_apply_tuning.sh` — v4l2 + USB autosuspend + CPU governor
> - `scripts/kria/03_install_systemd.sh` — persists tuning across reboots
>
> A separate "preview" section below outlines the high-level steps so you
> have a sense of scope.

## High-level walkthrough (Pass 5 fills in details)

1. **Host PC**: download Ubuntu image, flash to SD card via `04_flash_sd.sh`
2. **Insert SD** into KV260, power on, connect via Ethernet
3. **First SSH**: `ssh ubuntu@<board-ip>` (default password from Canonical docs)
4. **Run base bring-up**: `xlnx-config.sysinit` (handled by `01_install_vai35.sh`)
5. **Install Kria-PYNQ**: cloned and run `install.sh -b KV260` (script handles)
6. **Upgrade VAI 2.5 → 3.5**: the manual deb-install dance, automated
7. **Tune camera + USB + CPU**: `02_apply_tuning.sh`
8. **Persist across reboots**: `03_install_systemd.sh`

After all of the above the board is ready to run any of the deploy notebooks
or scripts. Total automated runtime: ~1 hour (dominated by package downloads).

## SD card flashing safety (Pass 5 enforces)

The flashing script will:

- **Refuse to write to non-removable devices** (checks `/sys/block/<dev>/removable`)
- **Refuse to write to mounted root** (checks `/proc/mounts`)
- **Show device model + size** before any prompt
- **Require typing the device name twice** to confirm
- **Verify SHA256** of the downloaded image before flashing

This is the single most dangerous operation in the whole pipeline; the
script errs hard on the side of refusing to do anything risky.
