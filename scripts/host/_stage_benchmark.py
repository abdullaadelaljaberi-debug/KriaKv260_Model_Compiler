#!/usr/bin/env python3
"""scripts/host/_stage_benchmark.py — host-side staging for the VAI 3.5 benchmark.

Runs on the host PC (laptop). Downloads:
  - ~9 GB of VAI 3.0 KV260 pre-compiled xmodels
  - ~1.5 GB COCO val2017 (images + annotations)
  - ~430 MB VOC2007 test set
  - ~1.3 GB ImageNetV2 matched-frequency (with synthesized labels.txt)
  - imagenet_class_index.json

Outputs everything under: $REPO_ROOT/build/benchmark_stage/
  ├── Models_VAI35/
  │   ├── classification/
  │   └── detection/
  └── Datasets/
      ├── imagenet_sample/  (symlinked from imagenetv2-matched-frequency-format-val/)
      ├── coco_val2017/
      ├── voc2007_test/
      └── imagenet_class_index.json

Wrapped by scripts/host/04_stage_benchmark.sh — usually you call that, not
this module directly.

Hardening defenses (designed after the prior SD card corruption incident):
  - fsync every 16 MB during downloads → bounds data loss on crash
  - Atomic .part → final rename with directory fsync → durable file appearance
  - Resume via HTTP Range → interrupted downloads pick up where they left off
  - Size verification → catches truncated downloads
  - State log (.stage_state.json) → atomic writes for re-runnability
  - SSL: certifi first, unverified fallback (Kria's CA bundle was incomplete;
    laptop's usually fine but the fallback costs nothing)

These hardening measures matter much less on a laptop SSD than they did on
the Kria SD card, but they're cheap and harmless. The point of moving the
download to the host is precisely to keep this workload off the SD card.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import ssl
import sys
import tarfile
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Optional


# ─────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────

# Pre-flight: refuse to start unless this much disk is free at the staging root.
# Sized to: downloads (~12 GB) + safety margin (~3 GB) = 15 GB.
MIN_FREE_GB_REQUIRED = 15.0

# Sync after every N bytes downloaded. 16 MB bounds data loss on a crash to one
# 16 MB window. Smaller = safer, slower. Default download chunk is 64 KB.
SYNC_EVERY_BYTES = 16 * 1024 * 1024
DOWNLOAD_CHUNK   = 64 * 1024

# Per-chunk network timeout
NETWORK_TIMEOUT = 120

# Delete tarballs after extract — saves ~6 GB
DELETE_TARBALLS_AFTER_EXTRACT = True

MODEL_URL_TEMPLATE = (
    "https://www.xilinx.com/bin/public/openDownload"
    "?filename={name}-zcu102_zcu104_kv260-r3.0.0.tar.gz"
)

DATASET_URLS = {
    "imagenetv2":           "https://huggingface.co/datasets/vaishaal/ImageNetV2/resolve/main/imagenetv2-matched-frequency.tar.gz",
    "imagenet_class_index": "https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json",
    "coco_images":          "http://images.cocodataset.org/zips/val2017.zip",
    "coco_anns":            "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
    "voc_test":             "http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar",
}


# ─────────────────────────────────────────────────────────────────────
# MODEL CATALOGUE
# ─────────────────────────────────────────────────────────────────────

CATALOG = {
    # Classification (ImageNet)
    "resnet50":                       dict(category="classification", enabled=True),
    "resnet_v1_50_tf":                dict(category="classification", enabled=True),
    "resnet_v1_101_tf":               dict(category="classification", enabled=True),
    "resnet_v1_152_tf":               dict(category="classification", enabled=True),
    "inception_v1_tf":                dict(category="classification", enabled=True),
    "inception_v2_tf":                dict(category="classification", enabled=True),
    "inception_v3_tf":                dict(category="classification", enabled=True),
    "inception_v4_2016_09_09_tf":     dict(category="classification", enabled=True),
    "mobilenet_v1_0_25_128_tf":       dict(category="classification", enabled=True),
    "mobilenet_v1_1_0_224_tf":        dict(category="classification", enabled=True),
    "mobilenet_v2_1_0_224_tf":        dict(category="classification", enabled=True),
    "mobilenet_v2_1_4_224_tf":        dict(category="classification", enabled=True),
    "mobilenetv2_pt":                 dict(category="classification", enabled=True),
    "squeezenet_pt":                  dict(category="classification", enabled=True),
    "vgg_16_tf":                      dict(category="classification", enabled=True),
    "vgg_19_tf":                      dict(category="classification", enabled=True),
    "efficientnet-b0_tf2":            dict(category="classification", enabled=True),
    "efficientnet_edgetpu-S_tf":      dict(category="classification", enabled=True),
    "efficientnet_edgetpu-M_tf":      dict(category="classification", enabled=True),
    "efficientnet_edgetpu-L_tf":      dict(category="classification", enabled=True),
    "inception_resnet_v2_tf":         dict(category="classification", enabled=True),
    "resnet50_pt":                    dict(category="classification", enabled=True),
    "resnet50_pruned_0_4_pt":         dict(category="classification", enabled=True),
    "ofa_resnet50_0_9B_pt":           dict(category="classification", enabled=True),
    "mobilenet_edge_2_75_pt":         dict(category="classification", enabled=True),

    # Detection — COCO
    "ssd_mobilenet_v1_coco_tf":       dict(category="detection", enabled=True),
    "ssd_mobilenet_v2_coco_tf":       dict(category="detection", enabled=True),
    "ssdlite_mobilenetv2_coco_tf":    dict(category="detection", enabled=True),
    "ssd_inception_v2_coco_tf":       dict(category="detection", enabled=True),
    "ssd_resnet_50_fpn_coco_tf":      dict(category="detection", enabled=True),
    "yolov3_coco_416_tf2":            dict(category="detection", enabled=True),
    "yolov4_leaky_spp_m":             dict(category="detection", enabled=True),
    "yolov3":                         dict(category="detection", enabled=True),

    # Detection — VOC
    "yolov3_voc_tf":                  dict(category="detection", enabled=True),

    # Face / specialty
    "face_mask_detection_pt":         dict(category="detection", enabled=True),
    "densebox_320_320":               dict(category="detection", enabled=True),
    "densebox_640_360":               dict(category="detection", enabled=True),
}


# ─────────────────────────────────────────────────────────────────────
# HARDENING PRIMITIVES
# ─────────────────────────────────────────────────────────────────────

def get_free_gb(path: Path) -> float:
    return shutil.disk_usage(path).free / (1024 ** 3)


def fsync_file(f) -> None:
    """Flush + sync to physical storage. Without this, writes can stay in the
    kernel write-back cache and be lost on a crash."""
    f.flush()
    os.fsync(f.fileno())


def fsync_dir(d: Path) -> None:
    """Sync directory metadata. After a rename, the directory entry update lives
    in the parent's metadata — fsync_dir ensures the rename is durable."""
    if not d.exists() or not d.is_dir():
        return
    fd = os.open(str(d), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_rename(tmp: Path, dest: Path) -> None:
    """Atomic POSIX rename + fsync parent. Either dest names the new inode or
    the old — never an inconsistent half-state."""
    tmp.rename(dest)
    fsync_dir(dest.parent)


def human_size(n) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}PB"


def ssl_contexts():
    """certifi-first, unverified-fallback. The xilinx.com cert chain wasn't in
    the Kria's CA bundle; usually fine on laptops but the fallback is free."""
    out = []
    try:
        import certifi
        out.append(("certifi", ssl.create_default_context(cafile=certifi.where())))
    except ImportError:
        pass
    out.append(("unverified", ssl._create_unverified_context()))
    return out


# ─────────────────────────────────────────────────────────────────────
# CORE: HARDENED DOWNLOAD + EXTRACT
# ─────────────────────────────────────────────────────────────────────

def download(url: str, dest: Path, label: str = "") -> bool:
    """Hardened download with fsync, atomic rename, resume, size verify.

    Returns True on success. On failure, leaves a clean state — partial .part
    files preserved for next-attempt resume.
    """
    label = label or dest.name

    if dest.exists():
        print(f"  [skip] {label}: already at {dest} ({human_size(dest.stat().st_size)})")
        return True

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    resume_from = tmp.stat().st_size if tmp.exists() else 0

    if resume_from > 0:
        print(f"  [resume] {label}: continuing from {human_size(resume_from)}")
    else:
        print(f"  [get]  {label}: {url[:70]}...")

    last_err: Optional[Exception] = None
    for ctx_name, ctx in ssl_contexts():
        try:
            req = urllib.request.Request(url)
            if resume_from > 0:
                req.add_header("Range", f"bytes={resume_from}-")

            with urllib.request.urlopen(req, timeout=NETWORK_TIMEOUT, context=ctx) as r:
                total: Optional[int] = None
                if resume_from > 0 and r.status == 206:
                    cr = r.headers.get("Content-Range", "")
                    if "/" in cr:
                        try:
                            total = int(cr.rsplit("/", 1)[1])
                        except ValueError:
                            pass
                else:
                    cl = r.headers.get("Content-Length")
                    if cl is not None:
                        try:
                            total = int(cl) + resume_from
                        except ValueError:
                            pass

                # Per-download disk check
                if total is not None:
                    needed_gb = (total - resume_from) * 1.2 / (1024 ** 3)
                    free_gb = get_free_gb(dest.parent)
                    if free_gb < needed_gb:
                        print(f"      FAILED: need {needed_gb:.1f} GB free for "
                              f"{label}, have {free_gb:.1f} GB")
                        return False

                open_mode = "ab" if resume_from > 0 else "wb"
                with open(tmp, open_mode) as f:
                    got = resume_from
                    bytes_since_sync = 0
                    last_progress = time.time()

                    while True:
                        chunk = r.read(DOWNLOAD_CHUNK)
                        if not chunk:
                            break
                        f.write(chunk)
                        got += len(chunk)
                        bytes_since_sync += len(chunk)

                        if bytes_since_sync >= SYNC_EVERY_BYTES:
                            fsync_file(f)
                            bytes_since_sync = 0

                        if time.time() - last_progress > 2.0:
                            pct = (got / total * 100) if total else 0
                            print(f"      ... {human_size(got)} / "
                                  f"{human_size(total) if total else '?'} "
                                  f"({pct:.0f}%)  [{ctx_name}]")
                            last_progress = time.time()

                    fsync_file(f)

                actual = tmp.stat().st_size
                if total is not None and actual != total:
                    print(f"      FAILED: size mismatch — got {actual}, expected {total}")
                    tmp.unlink()
                    return False

                atomic_rename(tmp, dest)
                if ctx_name == "unverified":
                    print(f"      [warn] downloaded with unverified SSL")
                print(f"      [ok]  {label}: {human_size(actual)}")
                return True

        except (urllib.error.URLError, ssl.SSLError, ConnectionError, TimeoutError) as e:
            last_err = e
            print(f"      attempt with {ctx_name} failed: {e}")
            continue
        except KeyboardInterrupt:
            print(f"      interrupted; .part preserved for resume")
            raise
        except Exception as e:
            last_err = e
            print(f"      unexpected error with {ctx_name}: {e}")
            continue

    print(f"      FAILED after all SSL strategies: {last_err}")
    return False


def safe_extract(archive: Path, dest_dir: Path, label: str = "") -> bool:
    """Extract + fsync parent dir at end."""
    label = label or archive.name
    dest_dir.mkdir(parents=True, exist_ok=True)
    print(f"  [tar] {label}: extracting into {dest_dir}")
    try:
        if archive.suffix.lower() == ".zip":
            with zipfile.ZipFile(archive) as zf:
                zf.extractall(dest_dir)
        else:
            with tarfile.open(archive, "r:*") as tf:
                tf.extractall(dest_dir)
        fsync_dir(dest_dir)
        print(f"      [ok]  extracted to {dest_dir}")
        return True
    except (tarfile.TarError, zipfile.BadZipFile, OSError) as e:
        print(f"      FAILED to extract: {e}")
        return False


# ─────────────────────────────────────────────────────────────────────
# STATE LOG
# ─────────────────────────────────────────────────────────────────────

class StateLog:
    """Atomic JSON record of completed downloads for clean resume."""
    def __init__(self, path: Path):
        self.path = path
        self.data: dict = {}
        if path.exists():
            try:
                self.data = json.loads(path.read_text())
            except Exception:
                self.data = {}

    def mark(self, key: str, status: str) -> None:
        self.data[key] = {"status": status, "ts": time.time()}
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.data, indent=1))
        with open(tmp, 'rb') as f:
            fsync_file(f)
        atomic_rename(tmp, self.path)

    def is_done(self, key: str) -> bool:
        return self.data.get(key, {}).get("status") == "done"


# ─────────────────────────────────────────────────────────────────────
# DOWNLOAD ORCHESTRATION
# ─────────────────────────────────────────────────────────────────────

def download_models(stage_root: Path, only: Optional[str], state: StateLog) -> tuple:
    models_dir = stage_root / "Models_VAI35"
    models_dir.mkdir(exist_ok=True)
    cache = models_dir / "_downloads"
    cache.mkdir(exist_ok=True)

    enabled = [(n, c) for n, c in CATALOG.items() if c.get("enabled", True)]
    if only:
        enabled = [(n, c) for n, c in enabled if n == only]
        if not enabled:
            print(f"No model named {only!r} in catalogue")
            return 0, 0, 0

    print(f"\n[Models] {len(enabled)} to consider")
    n_ok = n_skip = n_fail = 0

    for i, (name, cfg) in enumerate(enabled, 1):
        key = f"model:{name}"
        cat_dir = models_dir / cfg["category"]
        cat_dir.mkdir(exist_ok=True)

        existing_xmodels = list(cat_dir.rglob("*.xmodel"))
        if state.is_done(key) or any(name.lower() in p.parent.name.lower()
                                       for p in existing_xmodels):
            n_skip += 1
            print(f"  [{i:>2}/{len(enabled)}] skip {name}")
            continue

        print(f"  [{i:>2}/{len(enabled)}] {name}")
        url = MODEL_URL_TEMPLATE.format(name=name)
        tarball = cache / f"{name}.tar.gz"

        if not download(url, tarball, label=name):
            n_fail += 1
            continue
        if not safe_extract(tarball, cat_dir, label=name):
            n_fail += 1
            continue

        if DELETE_TARBALLS_AFTER_EXTRACT:
            try:
                tarball.unlink()
                fsync_dir(tarball.parent)
            except OSError:
                pass

        state.mark(key, "done")
        n_ok += 1

    # Clean up empty cache after all done
    try:
        if cache.exists() and not any(cache.iterdir()):
            cache.rmdir()
    except OSError:
        pass

    print(f"  Models: {n_ok} new, {n_skip} skipped, {n_fail} failed")
    return n_ok, n_skip, n_fail


def download_datasets(stage_root: Path, state: StateLog) -> tuple:
    ds_dir = stage_root / "Datasets"
    ds_dir.mkdir(exist_ok=True)
    cache = ds_dir / "_downloads"
    cache.mkdir(exist_ok=True)

    print(f"\n[Datasets]")
    n_ok = n_skip = n_fail = 0

    # ─── 1. ImageNetV2 (with labels.txt synthesis) ───
    key = "dataset:imagenetv2"
    imagenet_imgs   = ds_dir / "imagenet_sample" / "images"
    imagenet_labels = ds_dir / "imagenet_sample" / "labels.txt"
    if state.is_done(key) or (imagenet_imgs.exists() and imagenet_labels.exists()
                                and any(imagenet_imgs.iterdir())):
        print("  [skip] ImageNetV2: already staged")
        n_skip += 1
    else:
        tarball = cache / "imagenetv2-matched-frequency.tar.gz"
        extracted = ds_dir / "imagenetv2-matched-frequency-format-val"
        ok = download(DATASET_URLS["imagenetv2"], tarball, label="ImageNetV2")
        if ok and not extracted.exists():
            ok = safe_extract(tarball, ds_dir, label="ImageNetV2")
        if ok and extracted.exists():
            print(f"  [stage] symlinking + building labels.txt in {imagenet_imgs.parent}")
            imagenet_imgs.mkdir(parents=True, exist_ok=True)
            label_lines = []
            n_staged = 0
            for class_dir in sorted(extracted.iterdir()):
                if not class_dir.is_dir():
                    continue
                try:
                    cls_idx = int(class_dir.name)
                except ValueError:
                    continue
                for img in class_dir.iterdir():
                    if img.suffix.lower() not in (".jpeg", ".jpg", ".png"):
                        continue
                    flat = f"cls{cls_idx:04d}_{img.name}"
                    target = imagenet_imgs / flat
                    if not target.exists():
                        try:
                            target.symlink_to(img.resolve())
                        except OSError:
                            shutil.copy2(img, target)
                    label_lines.append(f"{flat} {cls_idx}\n")
                    n_staged += 1
            tmp_labels = imagenet_labels.with_suffix(".txt.tmp")
            tmp_labels.write_text("".join(label_lines))
            with open(tmp_labels, "rb") as f:
                fsync_file(f)
            atomic_rename(tmp_labels, imagenet_labels)
            fsync_dir(imagenet_labels.parent)
            print(f"      [ok]  {n_staged} labeled images")
            if DELETE_TARBALLS_AFTER_EXTRACT:
                try:
                    tarball.unlink()
                    fsync_dir(tarball.parent)
                except OSError:
                    pass
            state.mark(key, "done")
            n_ok += 1
        else:
            n_fail += 1

    # ─── 2. ImageNet class index (small JSON) ───
    iv2_json = ds_dir / "imagenet_class_index.json"
    if not iv2_json.exists():
        if download(DATASET_URLS["imagenet_class_index"], iv2_json,
                     label="imagenet class index"):
            n_ok += 1
        else:
            n_fail += 1
    else:
        n_skip += 1
        print("  [skip] imagenet_class_index.json: already present")

    # ─── 3. COCO val2017 ───
    key = "dataset:coco"
    coco_root  = ds_dir / "coco_val2017"
    coco_imgs  = coco_root / "val2017"
    coco_ann   = coco_root / "annotations" / "instances_val2017.json"
    if state.is_done(key) or (coco_imgs.exists() and coco_ann.exists()):
        print("  [skip] COCO val2017: already staged")
        n_skip += 1
    else:
        img_zip = cache / "coco_val2017.zip"
        ann_zip = cache / "coco_annotations.zip"
        ok = True
        if not coco_imgs.exists():
            ok = ok and download(DATASET_URLS["coco_images"], img_zip,
                                  label="COCO images")
            ok = ok and safe_extract(img_zip, coco_root, label="COCO images")
        if not coco_ann.exists():
            ok = ok and download(DATASET_URLS["coco_anns"], ann_zip,
                                  label="COCO annotations")
            ok = ok and safe_extract(ann_zip, coco_root, label="COCO annotations")
        if ok:
            if DELETE_TARBALLS_AFTER_EXTRACT:
                for arch in (img_zip, ann_zip):
                    try:
                        if arch.exists():
                            arch.unlink()
                            fsync_dir(arch.parent)
                    except OSError:
                        pass
            state.mark(key, "done")
            n_ok += 1
        else:
            n_fail += 1

    # ─── 4. VOC2007 test ───
    key = "dataset:voc"
    voc_root = ds_dir / "voc2007_test"
    voc_imgs = voc_root / "VOCdevkit" / "VOC2007" / "JPEGImages"
    voc_ann  = voc_root / "VOCdevkit" / "VOC2007" / "Annotations"
    if state.is_done(key) or (voc_imgs.exists() and voc_ann.exists()):
        print("  [skip] VOC2007 test: already staged")
        n_skip += 1
    else:
        voc_tar = cache / "voc2007_test.tar"
        ok = download(DATASET_URLS["voc_test"], voc_tar, label="VOC2007 test")
        ok = ok and safe_extract(voc_tar, voc_root, label="VOC2007 test")
        if ok:
            if DELETE_TARBALLS_AFTER_EXTRACT:
                try:
                    voc_tar.unlink()
                    fsync_dir(voc_tar.parent)
                except OSError:
                    pass
            state.mark(key, "done")
            n_ok += 1
        else:
            n_fail += 1

    # Clean up empty cache after all done
    try:
        if cache.exists() and not any(cache.iterdir()):
            cache.rmdir()
    except OSError:
        pass

    print(f"  Datasets: {n_ok} new, {n_skip} skipped, {n_fail} failed")
    return n_ok, n_skip, n_fail


# ─────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Stage VAI 3.5 benchmark data on the host PC",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--stage-root", type=Path, required=True,
                        help="Where to stage everything")
    parser.add_argument("--skip-models", action="store_true",
                        help="Only download datasets")
    parser.add_argument("--skip-datasets", action="store_true",
                        help="Only download models")
    parser.add_argument("--only", type=str, default=None,
                        help="Download a single model by name (for catalogue iteration)")
    parser.add_argument("--min-free-gb", type=float,
                        default=MIN_FREE_GB_REQUIRED,
                        help=f"Pre-flight disk space requirement in GB (default: {MIN_FREE_GB_REQUIRED})")
    args = parser.parse_args()

    stage_root: Path = args.stage_root.resolve()
    stage_root.mkdir(parents=True, exist_ok=True)
    free = get_free_gb(stage_root)
    print(f"Stage root:   {stage_root}")
    print(f"Free disk:    {free:.1f} GB  (need ≥{args.min_free_gb:.1f} GB)")

    if free < args.min_free_gb:
        print()
        print(f"ABORT: insufficient free disk space.")
        print(f"  Need: {args.min_free_gb:.1f} GB, Have: {free:.1f} GB")
        sys.exit(2)

    state = StateLog(stage_root / ".stage_state.json")
    t_start = time.time()

    try:
        m_ok = m_skip = m_fail = 0
        d_ok = d_skip = d_fail = 0
        if not args.skip_models:
            m_ok, m_skip, m_fail = download_models(stage_root, args.only, state)
        if not args.skip_datasets and args.only is None:
            d_ok, d_skip, d_fail = download_datasets(stage_root, state)
    except KeyboardInterrupt:
        elapsed = time.time() - t_start
        print(f"\nInterrupted after {elapsed/60:.1f} min. State saved; re-run to resume.")
        sys.exit(130)

    elapsed = time.time() - t_start
    print()
    print(f"Models:    {m_ok} new, {m_skip} skipped, {m_fail} failed")
    print(f"Datasets:  {d_ok} new, {d_skip} skipped, {d_fail} failed")
    print(f"Time:      {elapsed/60:.1f} min")
    print(f"Free disk: {get_free_gb(stage_root):.1f} GB")

    if m_fail or d_fail:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
