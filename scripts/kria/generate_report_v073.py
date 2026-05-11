#!/usr/bin/env python3
"""
Generate the KV260 Vitis AI 3.5 benchmark report (markdown).

Reads CSVs produced by 04_vai35_benchmark.ipynb:
  - vai35_benchmark_results.csv  : FPS, power, top-1/top-5 accuracy per model
  - vai35_coco_map_results.csv   : COCO mAP per detection model
  - vai35_voc_map_results.csv    : VOC mAP per detection model

Writes:
  - vai35_benchmark_report.md    : comprehensive markdown report

Output format mirrors the VAI 2.5 reference report so the two are directly
comparable. Includes top-1 AND top-5 accuracy (v0.7.3+ CSV has both columns).

USAGE:
    On the Kria, after the benchmark notebook finishes:
        python3 /tmp/generate_report_v073.py

    Optionally override paths via environment variables:
        REPORT_INPUT_DIR=/path/to/csvs/  REPORT_OUTPUT=/path/to/out.md \\
            python3 /tmp/generate_report_v073.py
"""
import csv
import os
import time
from pathlib import Path

# ─── Paths (env-overridable for flexibility) ─────────────────────────────
NB_DIR       = Path(os.environ.get(
    'REPORT_INPUT_DIR',
    '/home/ubuntu/KriaKv260_Model_Compiler/notebooks'
))
RESULTS_CSV  = NB_DIR / 'vai35_benchmark_results.csv'
COCO_MAP_CSV = NB_DIR / 'vai35_coco_map_results.csv'
VOC_MAP_CSV  = NB_DIR / 'vai35_voc_map_results.csv'
REPORT_MD    = Path(os.environ.get(
    'REPORT_OUTPUT',
    str(NB_DIR / 'vai35_benchmark_report.md')
))


def to_float(v, default=None):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def fmt(v, fallback='—'):
    if v is None or v == '' or (isinstance(v, str) and not v.strip()):
        return fallback
    return str(v)


def load_csv(path):
    if not path.exists():
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def top_n(rows, key, n=10, reverse=True, require_no_error=True):
    """Return top N rows sorted by float(row[key]), filtering missing values."""
    def keep(r):
        if require_no_error and r.get('error'):
            return False
        return to_float(r.get(key)) is not None
    ranked = sorted([r for r in rows if keep(r)],
                    key=lambda r: to_float(r[key]),
                    reverse=reverse)
    return ranked[:n]


def main():
    if not RESULTS_CSV.exists():
        raise SystemExit(
            f"Main results CSV not found: {RESULTS_CSV}\n"
            f"Run the benchmark notebook first, then re-run this script.")

    rows      = load_csv(RESULTS_CSV)
    coco_rows = load_csv(COCO_MAP_CSV)
    voc_rows  = load_csv(VOC_MAP_CSV)

    # Join mAP data into main rows by model name
    coco_by_model = {r['model']: r for r in coco_rows}
    voc_by_model  = {r['model']: r for r in voc_rows}
    for r in rows:
        c = coco_by_model.get(r['model'], {})
        v = voc_by_model.get(r['model'], {})
        r['coco_map_50']    = c.get('map_50', '')
        r['coco_map_50_95'] = c.get('map_50_95', '')
        r['coco_skip']      = c.get('skip_reason', '')
        r['coco_n_imgs']    = c.get('n_images', '')
        r['voc_map_50']     = v.get('map_50', '')
        r['voc_skip']       = v.get('skip_reason', '')
        r['voc_n_imgs']     = v.get('n_images', '')

    cls = [r for r in rows if r.get('category') == 'classification']
    det = [r for r in rows if r.get('category') == 'detection']
    err = [r for r in rows if r.get('error')]
    det_with_coco_map = [r for r in det if to_float(r.get('coco_map_50')) is not None]
    det_with_voc_map  = [r for r in det if to_float(r.get('voc_map_50')) is not None]

    # Try to infer config from the first valid classification row
    n_acc = next((r['n_acc_images'] for r in cls if r.get('n_acc_images')), '—')

    out = []
    P = lambda *a: out.append(' '.join(str(x) for x in a))

    # ─── HEADER ──────────────────────────────────────────────────────────
    P("# KV260 Vitis AI 3.5 Benchmark Report")
    P("")
    P("Generated:", time.strftime('%Y-%m-%d %H:%M:%S'))
    P("")

    # ─── TEST ENVIRONMENT ────────────────────────────────────────────────
    P("## Test Environment")
    P("")
    P("| Field | Value |")
    P("|---|---|")
    P("| Board | Kria KV260 |")
    P("| Stack | Kria-PYNQ 3.0.1 / Vitis AI 3.5 / DPUCZDX8G ISA1 B4096 @ 300 MHz |")
    P("| DPU fingerprint | 0x101000056010407 |")
    P("| Camera | Logitech BRIO on /dev/video0, MJPG |")
    P(f"| Cls dataset | ImageNetV2 matched-frequency, N={n_acc}, shuffled (seed=42) |")
    if det_with_coco_map:
        P(f"| COCO mAP    | COCO val2017, N={det_with_coco_map[0].get('coco_n_imgs','—')} |")
    if det_with_voc_map:
        P(f"| VOC mAP     | VOC2007 test, N={det_with_voc_map[0].get('voc_n_imgs','—')} |")
    P("| Preprocessing | Per-model mean / scale from bundled .prototxt files |")
    P("")

    # ─── SUMMARY ─────────────────────────────────────────────────────────
    P("## Summary")
    P("")
    P("| Statistic | Value |")
    P("|---|---|")
    P(f"| Total models       | {len(rows)} |")
    P(f"| Classification     | {len(cls)} |")
    P(f"| Detection          | {len(det)} |")
    if det_with_coco_map:
        P(f"| Detection w/ COCO mAP | {len(det_with_coco_map)} |")
    if det_with_voc_map:
        P(f"| Detection w/ VOC mAP  | {len(det_with_voc_map)} |")
    P(f"| Errors             | {len(err)} |")
    P("")

    # ─── DETECTION ACCURACY (COCO mAP) ───────────────────────────────────
    if coco_rows:
        P("## Detection Accuracy (COCO mAP)")
        P("")
        P(f"COCO val2017, NMS IoU=0.45, score≥0.05, max 100 det/image.")
        P("")
        P("| Model | Status | mAP@0.5 | mAP@0.5:0.95 | n_imgs | Note |")
        P("|---|---|---:|---:|---:|---|")
        for r in sorted(coco_rows, key=lambda x: -to_float(x.get('map_50'), -1)):
            status = ('OK' if to_float(r.get('map_50')) is not None else 'SKIPPED')
            note = r.get('skip_reason', '') or r.get('error', '') or '—'
            P(f"| {r['model']} | {status} | {fmt(r.get('map_50'))} | "
              f"{fmt(r.get('map_50_95'))} | {fmt(r.get('n_images'))} | {note} |")
        P("")

    # ─── DETECTION ACCURACY (VOC mAP) ────────────────────────────────────
    if voc_rows:
        P("## Detection Accuracy (VOC mAP)")
        P("")
        P("VOC2007 test set, mAP@0.5 (standard PASCAL VOC metric).")
        P("")
        P("| Model | Status | mAP@0.5 | n_imgs | Note |")
        P("|---|---|---:|---:|---|")
        for r in sorted(voc_rows, key=lambda x: -to_float(x.get('map_50'), -1)):
            status = ('OK' if to_float(r.get('map_50')) is not None else 'SKIPPED')
            note = r.get('skip_reason', '') or r.get('error', '') or '—'
            P(f"| {r['model']} | {status} | {fmt(r.get('map_50'))} | "
              f"{fmt(r.get('n_images'))} | {note} |")
        P("")

    # ─── TOP RANKINGS ────────────────────────────────────────────────────
    P("## Top Rankings")
    P("")

    P("### Top 10 by Pure-DPU FPS (no camera bottleneck)")
    P("")
    P("| Rank | Model | Cat | Input | DPU FPS | Latency mean (ms) | Power (W) |")
    P("|---|---|---|---|---:|---:|---:|")
    for i, r in enumerate(top_n(rows, 'dpu_fps'), 1):
        P(f"| {i} | {r['model']} | {r['category'][:5]} | {r['input_shape']} | "
          f"{r['dpu_fps']} | {r['dpu_latency_mean_ms']} | {r['power_load_w']} |")
    P("")

    if any(to_float(r.get('cam_fps')) for r in rows):
        P("### Top 10 by Camera FPS (end-to-end with BRIO)")
        P("")
        P("| Rank | Model | Cat | Input | Cam res | Cam FPS | Cam-bound | Power (W) |")
        P("|---|---|---|---|---|---:|:---:|---:|")
        for i, r in enumerate(top_n(rows, 'cam_fps'), 1):
            cb = '✓' if str(r.get('cam_limited','')).lower() in ('true','1') else '—'
            P(f"| {i} | {r['model']} | {r['category'][:5]} | {r['input_shape']} | "
              f"{r['cam_resolution']} | {r['cam_fps']} | {cb} | {r['power_load_w']} |")
        P("")

        P("### Top 10 by FPS/W (camera-based efficiency)")
        P("")
        P("| Rank | Model | Cat | Cam FPS | Power (W) | FPS/W |")
        P("|---|---|---|---:|---:|---:|")
        for i, r in enumerate(top_n(rows, 'fps_per_w'), 1):
            P(f"| {i} | {r['model']} | {r['category'][:5]} | "
              f"{r['cam_fps']} | {r['power_load_w']} | {r['fps_per_w']} |")
        P("")

    if cls:
        P("### Top 10 Classification by Top-1 Accuracy")
        P("")
        P(f"(ImageNetV2 matched-frequency, N={n_acc}; ~±1.5% CI at this N)")
        P("")
        P("| Rank | Model | Top-1 | Top-5 | DPU FPS | Cam FPS | Power (W) |")
        P("|---|---|---:|---:|---:|---:|---:|")
        for i, r in enumerate(top_n(cls, 'accuracy_top1'), 1):
            P(f"| {i} | {r['model']} | {r['accuracy_top1']} | {fmt(r.get('accuracy_top5'))} | "
              f"{r['dpu_fps']} | {fmt(r.get('cam_fps'))} | {r['power_load_w']} |")
        P("")

        # Top-5 ranking is its own section since it's the more robust metric
        if any(to_float(r.get('accuracy_top5')) for r in cls):
            P("### Top 10 Classification by Top-5 Accuracy")
            P("")
            P("Top-5 is more robust to fine-grained class ambiguity and ImageNetV2's")
            P("known multi-subject images. AMD reports top-5 alongside top-1.")
            P("")
            P("| Rank | Model | Top-5 | Top-1 | DPU FPS | Power (W) |")
            P("|---|---|---:|---:|---:|---:|")
            for i, r in enumerate(top_n(cls, 'accuracy_top5'), 1):
                P(f"| {i} | {r['model']} | {r['accuracy_top5']} | {r['accuracy_top1']} | "
                  f"{r['dpu_fps']} | {r['power_load_w']} |")
            P("")

    if det_with_coco_map:
        P("### Top by COCO mAP@0.5")
        P("")
        P("| Rank | Model | mAP@0.5 | mAP@0.5:0.95 | DPU FPS | Cam FPS | Power (W) |")
        P("|---|---|---:|---:|---:|---:|---:|")
        for i, r in enumerate(top_n(det_with_coco_map, 'coco_map_50'), 1):
            P(f"| {i} | {r['model']} | {r['coco_map_50']} | {fmt(r.get('coco_map_50_95'))} | "
              f"{r['dpu_fps']} | {fmt(r.get('cam_fps'))} | {r['power_load_w']} |")
        P("")

    # ─── CAMERA-BOUND vs MODEL-BOUND ─────────────────────────────────────
    if any(to_float(r.get('cam_fps')) for r in rows):
        cb = sum(1 for r in rows
                 if str(r.get('cam_limited','')).lower() in ('true','1'))
        mb = sum(1 for r in rows
                 if str(r.get('cam_limited','')).lower() in ('false','0'))
        P("## Camera-Bound vs Model-Bound")
        P("")
        P(f"- **Camera-limited** (FPS ≥ 95% of target): {cb} models. DPU faster than")
        P("  camera frame delivery; use *Pure-DPU FPS* for comparison.")
        P(f"- **Model-limited**: {mb} models. DPU inference exceeds camera period;")
        P("  cam FPS reflects model's actual ceiling with this preprocessing pipeline.")
        P("")

    # ─── DETAILED RESULTS — CLASSIFICATION ───────────────────────────────
    def detail_section_cls(items):
        if not items: return
        P("## Detailed Results — Classification")
        P("")
        P("| Model | Input | DPU lat mean / p50 / p99 (ms) | DPU FPS | "
          "Pwr idle / load (W) | Top-1 | Top-5 | Cam res @ tgt | Cam FPS | "
          "cap / pre / inf (ms) | Bound | FPS/W |")
        P("|---|---|---|---:|---|---:|---:|---|---:|---|:---:|---:|")
        for r in sorted(items, key=lambda x: -to_float(x.get('dpu_fps'), 0)):
            if r.get('error'):
                P(f"| **{r['model']}** | {r.get('input_shape','—')} | "
                  f"ERROR: {r['error'][:40]} | | | | | | | | | |")
                continue
            lat = (f"{r.get('dpu_latency_mean_ms','—')} / "
                   f"{r.get('dpu_latency_p50_ms','—')} / "
                   f"{r.get('dpu_latency_p99_ms','—')}")
            pwr = f"{r.get('power_idle_w','—')} / {r.get('power_load_w','—')}"
            top1 = fmt(r.get('accuracy_top1'))
            top5 = fmt(r.get('accuracy_top5'))
            cres = f"{r.get('cam_resolution','—')} @ {r.get('cam_target_fps','—')}"
            stages = (f"{r.get('cam_capture_ms','—')} / "
                      f"{r.get('cam_preprocess_ms','—')} / "
                      f"{r.get('cam_inference_ms','—')}")
            cb_raw = str(r.get('cam_limited','')).lower()
            bound  = 'cam'   if cb_raw in ('true','1')  else (
                     'model' if cb_raw in ('false','0') else '—')
            P(f"| {r['model']} | {r.get('input_shape','—')} | {lat} | "
              f"{r.get('dpu_fps','—')} | {pwr} | {top1} | {top5} | {cres} | "
              f"{fmt(r.get('cam_fps'))} | {stages} | {bound} | "
              f"{fmt(r.get('fps_per_w'))} |")
        P("")

    # ─── DETAILED RESULTS — DETECTION ────────────────────────────────────
    def detail_section_det(items):
        if not items: return
        P("## Detailed Results — Detection")
        P("")
        P("| Model | Input | DPU lat mean / p50 / p99 (ms) | DPU FPS | "
          "Pwr idle / load (W) | COCO mAP@0.5 / @0.5:0.95 | VOC mAP@0.5 | "
          "Cam res @ tgt | Cam FPS | cap / pre / inf (ms) | Bound | FPS/W |")
        P("|---|---|---|---:|---|---:|---:|---|---:|---|:---:|---:|")
        for r in sorted(items, key=lambda x: -to_float(x.get('dpu_fps'), 0)):
            if r.get('error'):
                P(f"| **{r['model']}** | {r.get('input_shape','—')} | "
                  f"ERROR: {r['error'][:40]} | | | | | | | | | |")
                continue
            lat = (f"{r.get('dpu_latency_mean_ms','—')} / "
                   f"{r.get('dpu_latency_p50_ms','—')} / "
                   f"{r.get('dpu_latency_p99_ms','—')}")
            pwr = f"{r.get('power_idle_w','—')} / {r.get('power_load_w','—')}"
            coco = (f"{fmt(r.get('coco_map_50'))} / {fmt(r.get('coco_map_50_95'))}"
                    if to_float(r.get('coco_map_50')) is not None else '— / —')
            voc  = fmt(r.get('voc_map_50'))
            cres = f"{r.get('cam_resolution','—')} @ {r.get('cam_target_fps','—')}"
            stages = (f"{r.get('cam_capture_ms','—')} / "
                      f"{r.get('cam_preprocess_ms','—')} / "
                      f"{r.get('cam_inference_ms','—')}")
            cb_raw = str(r.get('cam_limited','')).lower()
            bound  = 'cam'   if cb_raw in ('true','1')  else (
                     'model' if cb_raw in ('false','0') else '—')
            P(f"| {r['model']} | {r.get('input_shape','—')} | {lat} | "
              f"{r.get('dpu_fps','—')} | {pwr} | {coco} | {voc} | {cres} | "
              f"{fmt(r.get('cam_fps'))} | {stages} | {bound} | "
              f"{fmt(r.get('fps_per_w'))} |")
        P("")

    detail_section_cls(cls)
    detail_section_det(det)

    # ─── ERRORS ──────────────────────────────────────────────────────────
    if err:
        P("## Errors")
        P("")
        for r in err:
            P(f"- **{r['model']}** ({r.get('category','?')}): `{r['error']}`")
        P("")

    # ─── METHODOLOGY ─────────────────────────────────────────────────────
    P("## Methodology")
    P("")
    P("Each model is measured on five criteria, plus mAP for detection models")
    P("where a decoder is implemented.")
    P("")
    P("1. **Power** — board-total power from `/sys/class/hwmon/`.")
    P("   *Idle* = avg over 0.3 s before DPU activity. *Load* = avg across DPU")
    P("   iterations.")
    P("2. **Latency (pure-DPU)** — wall-clock around `dpu.execute_async() + dpu.wait()`")
    P("   over N iterations on a static, in-memory image. Mean / p50 / p99 reported.")
    P("3. **Accuracy** — Top-1 AND Top-5 on ImageNetV2 matched-frequency for")
    P("   classification (sample shuffled with seed=42 to ensure class-balanced")
    P("   coverage). COCO and VOC mAP for detection.")
    P("4. **FPS** — two values:")
    P("   - **Pure-DPU FPS** = 1000 / mean DPU latency (no camera, no preprocessing)")
    P("   - **Camera FPS** = end-to-end with BRIO capture + cv2.resize + DPU")
    P("5. **FPS/W** — camera FPS divided by load power.")
    P("")
    P("**Preprocessing** uses the exact `mean` and `scale` from each model's")
    P("bundled `.prototxt` file. These are the values AMD used during quantization")
    P("calibration. Using catalogue defaults instead (as in earlier versions of")
    P("this notebook) drifts per-model and depresses top-1 accuracy by 10-22%.")
    P("Output is always converted to BGR (AMD VAI convention).")
    P("")
    P("**Top-1 vs Top-5 on ImageNetV2** — V2 is intrinsically harder than ImageNet")
    P("val (≈10% absolute drop, per Recht et al. 2019) and contains a non-trivial")
    P("fraction of multi-subject / unusually-cropped images. Top-5 is the more")
    P("interpretable metric for cross-model comparison; top-1 is reported for")
    P("completeness and matches AMD's published numbers within ~5%.")
    P("")
    P("**Camera resolution** is auto-selected per model: the smallest BRIO MJPG")
    P("preset at least as large as the model's input (640x480 @ 60 for most models;")
    P("1280x720 @ 30 for 640x640 inputs).")
    P("")
    P("**Camera-bound vs model-bound** — a model is *camera-bound* if measured FPS")
    P("is within 5% of the camera's target FPS (camera couldn't deliver frames")
    P("faster). Otherwise *model-bound*: FPS reflects the model's actual ceiling.")
    P("")
    P("**mAP details**")
    P("")
    P("- `mAP@0.5`: standard COCO/VOC IoU threshold of 0.5")
    P("- `mAP@0.5:0.95`: COCO primary metric, averaged over IoU thresholds 0.5..0.95")
    P("- pycocotools used for COCO mAP; standard 11-point AP for VOC")
    P("- Detection scores: NMS IoU=0.45, conf threshold ≥ 0.05, max 100 det/image")
    P("")

    # ─── COLUMN GLOSSARY ─────────────────────────────────────────────────
    P("## Column Glossary")
    P("")
    P("| Column | Meaning |")
    P("|---|---|")
    P("| `Input` | DPU tensor shape (NHWC) |")
    P("| `DPU lat mean / p50 / p99` | Pure-DPU inference latency in ms; image is in-memory |")
    P("| `DPU FPS` | 1000 / mean DPU latency — the model's raw ceiling |")
    P("| `Pwr idle / load` | Board total power: idle before warm-up vs avg under load |")
    P("| `Top-1` / `Top-5` | ImageNetV2 top-1 / top-5 accuracy (classification only) |")
    P("| `COCO mAP@0.5` / `@0.5:0.95` | COCO detection mAP at IoU thresholds |")
    P("| `VOC mAP@0.5` | Pascal VOC 2007 detection mAP at IoU 0.5 |")
    P("| `Cam res @ tgt` | Camera capture resolution and target FPS |")
    P("| `Cam FPS` | Measured end-to-end frames/s with BRIO + resize + DPU |")
    P("| `cap / pre / inf` | Per-frame timing: capture, preprocess, inference (ms) |")
    P("| `Bound` | `cam` if camera-limited, `model` if DPU-limited |")
    P("| `FPS/W` | Cam FPS / load power |")
    P("")

    # ─── WRITE OUT ───────────────────────────────────────────────────────
    REPORT_MD.write_text('\n'.join(out))
    print(f"[OK] Report written: {REPORT_MD}")
    print(f"     {len(out):,} lines, {REPORT_MD.stat().st_size:,} bytes")
    print(f"     {len(rows)} models ({len(cls)} cls, {len(det)} det, {len(err)} err)")
    if det_with_coco_map:
        print(f"     COCO mAP: {len(det_with_coco_map)} models")
    if det_with_voc_map:
        print(f"     VOC  mAP: {len(det_with_voc_map)} models")


if __name__ == '__main__':
    main()
