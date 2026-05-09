# Daily usage workflow

> **This document is filled in Pass 7** (final pass — once all components
> are in place). It describes the day-to-day workflow for someone who has
> already run the host setup (Pass 2) and Kria setup (Pass 5).
>
> Preview of the workflow:
>
> 1. Place trained weights in `data/weights/<name>.pt`
> 2. Place calibration images in `data/calib/`
> 3. Place held-out evaluation images in `data/eval/`
> 4. Run `bash scripts/host/02_compile.sh <family> <variant>`
> 5. Run `bash scripts/host/03_sync_to_kria.sh <user@host> <variant>`
> 6. Run `bash scripts/kria/run_live.sh <variant>` for live demo, or
>    `run_eval.sh <variant>` for batch mAP evaluation

## Typical compile time (rough)

| Stage | Time |
|---|---|
| Calibration (200 images) | 2-5 min on RTX 3060 |
| Quantization | 1-3 min |
| Compilation to xmodel | 30 sec |
| Total compile | ~5-8 min per model |
| Sync to Kria | <1 min for the xmodel |

## Typical demo time

| Stage | Time |
|---|---|
| Camera + system tuning (one-time per boot) | 5 sec |
| Model load on Kria | 2-3 sec |
| Live demo to first detection | <5 sec total |

(Filled in detail in Pass 7.)
