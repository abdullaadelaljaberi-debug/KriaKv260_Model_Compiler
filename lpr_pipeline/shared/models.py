"""Model registry — single source of truth for supported variants.

Every variant has a ``ModelSpec`` describing its input size, family,
quantization conventions, and decoder. Both the compile path and the
deploy path read from this registry, so changes in one place propagate
correctly.

A "family" is a model architecture (yolov5, yolox, yolov7, yolov4_csp,
ssd_mobilenetv2). A "variant" is a specific model within a family
(yolov5n, yolov5s, yolox_tiny, etc.).

Status per family:
    yolov5         — full pipeline support
    yolox          — full pipeline support
    yolov7         — stub (deploy-side decoder TBD; compile raises)
    yolov4_csp     — stub
    ssd_mobilenetv2 — stub
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


Family = Literal["yolov5", "yolox", "yolov7", "yolov4_csp", "ssd_mobilenetv2"]
Status = Literal["full", "stub"]


@dataclass(frozen=True)
class ModelSpec:
    """Describes a single supported variant.

    Attributes
    ----------
    name :
        Variant identifier (e.g. ``"yolov5n"``). Also used as the directory
        name for the compiled xmodel: ``out/<name>/<name>_kv260.xmodel``.
    family :
        Architecture family. Determines which compile path is used and which
        decoder runs at deploy time.
    imgsz :
        Square input size in pixels.
    nc :
        Number of object classes the model was trained on. License-plate
        detection is ``nc=1``; arbitrary use cases override this.
    reg_max :
        DFL distribution width (YOLOv5u only). Number of bins per box side
        in the Distribution Focal Loss head. ``-1`` for non-DFL families.
    status :
        ``"full"`` if both compile and deploy work end-to-end, ``"stub"`` if
        only the directory scaffolding exists.
    notes :
        Free-form per-variant notes. Surfaced in error messages and docs.
    """
    name:    str
    family:  Family
    imgsz:   int
    nc:      int = 1                       # default for LPR demo
    reg_max: int = -1                      # only YOLOv5u uses DFL
    status:  Status = "stub"
    notes:   str = ""
    # Files the deploy side needs alongside the xmodel (relative to
    # the xmodel directory). Currently empty for all families; reserved
    # for future use (e.g. anchor lists, prior-box files).
    extra_files: tuple[str, ...] = field(default_factory=tuple)


# ─────────────────────────────────────────────────────────────────────────────
# Registry
#
# To add a new variant within a supported family: append a ModelSpec.
# To add a new family: also implement the compile + decode logic in
#   lpr_pipeline/compile/<family>.py and lpr_pipeline/deploy/decoders.py.
# ─────────────────────────────────────────────────────────────────────────────
MODEL_REGISTRY: dict[str, ModelSpec] = {
    # ── YOLOv5 (Ultralytics u-variant; anchor-free DFL) ─────────────────────
    "yolov5n": ModelSpec(
        name="yolov5n", family="yolov5",
        imgsz=320, reg_max=16, status="full",
        notes="Smallest Ultralytics u-variant. ~12 ms inference on KV260.",
    ),
    "yolov5s": ModelSpec(
        name="yolov5s", family="yolov5",
        imgsz=320, reg_max=16, status="full",
        notes="~19 ms inference on KV260. Better mAP than n.",
    ),

    # ── YOLOX (Megvii; anchor-free, decoupled head) ─────────────────────────
    "yolox_tiny": ModelSpec(
        name="yolox_tiny", family="yolox",
        imgsz=416, status="full",
        notes="4 DPU subgraphs; ~92 ms inference. Uses GraphRunner.",
    ),
    "yolox_nano": ModelSpec(
        name="yolox_nano", family="yolox",
        imgsz=416, status="full",
        notes="34 DPU subgraphs (depthwise convs fragment the graph); "
              "structurally slow at ~680 ms. Documented limitation.",
    ),

    # ── YOLOv7 (WongKinYiu; anchor-based) ───────────────────────────────────
    "yolov7-tiny": ModelSpec(
        name="yolov7-tiny", family="yolov7",
        imgsz=640, status="stub",
        notes="VAI 3.5 zoo entry: pt_yolov7_640_640_3.5. Compile path stub.",
    ),

    # ── YOLOv4-CSP (anchor-based) ───────────────────────────────────────────
    "yolov4_csp": ModelSpec(
        name="yolov4_csp", family="yolov4_csp",
        imgsz=512, status="stub",
        notes="VAI 3.5 zoo entry: pt_yolov4_csp_512_512_3.5. Compile path stub.",
    ),

    # ── SSD MobileNetV2 (TF) ────────────────────────────────────────────────
    "ssd_mobilenet_v2_coco": ModelSpec(
        name="ssd_mobilenet_v2_coco", family="ssd_mobilenetv2",
        imgsz=300, nc=80, status="stub",
        notes="VAI 3.5 zoo entry: tf_ssdmobilenetv2_coco_300_300_3.5. "
              "Different toolchain (vai_q_tensorflow2). Stub.",
    ),
}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def get_spec(name: str) -> ModelSpec:
    """Look up a variant by name. Raises ``KeyError`` with a helpful message."""
    if name not in MODEL_REGISTRY:
        supported = ", ".join(sorted(MODEL_REGISTRY))
        raise KeyError(
            f"Unknown model variant: {name!r}\n"
            f"Supported variants: {supported}\n"
            f"To add one, edit lpr_pipeline/shared/models.py."
        )
    return MODEL_REGISTRY[name]


def list_supported_families() -> dict[Family, list[str]]:
    """Return a dict of family → list of variant names."""
    out: dict[Family, list[str]] = {}
    for spec in MODEL_REGISTRY.values():
        out.setdefault(spec.family, []).append(spec.name)
    return out


def list_full_support() -> list[str]:
    """Return only variants whose status is ``"full"`` (i.e. usable today)."""
    return [name for name, spec in MODEL_REGISTRY.items() if spec.status == "full"]
