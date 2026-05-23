"""Model registry — single source of truth for supported variants.

Every variant has a ``ModelSpec`` describing its input size, family,
quantization conventions, and decoder. Both the compile path and the
deploy path read from this registry, so changes in one place propagate
correctly.

A "family" is a model architecture (yolov5, yolox, yolov7, yolov4_csp,
ssd_mobilenetv2, ssdlite, retinanet, classification). A "variant" is a
specific model within a family (yolov5n, yolov5s, yolox_tiny, etc.).

Status per family:
    yolov5         — full pipeline support
    yolov11        — full pipeline support (DPU-friendly arch via training-time
                     monkey-patches; see scripts/host/_train_yolov11.py and
                     docs/YOLOV11.md)
    yolox          — full pipeline support
    yolov7         — stub (deploy-side decoder TBD; compile raises)
    yolov4_csp     — stub
    ssd_mobilenetv2 — stub
    ssdlite        — v0.12 (full compile path; variants stub until trained)
    retinanet      — v0.12 (full compile path; variants stub until trained)
    classification — v0.12 (full compile path; variants stub until trained;
                     sub-architecture (resnet50/mobilenetv2/inceptionv3) is
                     decoded from the variant name prefix — see
                     classification_subarch() below)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


Family = Literal[
    "yolov5", "yolov11", "yolox", "yolov7", "yolov4_csp", "ssd_mobilenetv2",
    "ssdlite", "retinanet", "classification",
]
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
    "yolov5s_eggs": ModelSpec(
        name="yolov5s_eggs", family="yolov5",
        imgsz=640, reg_max=16, status="full",
        notes="YOLOv5s (Ultralytics u-variant; anchor-free DFL, ~9.1M params). "
              "Trained on eggs+hardneg at imgsz=640 for the architecture-"
              "generation comparison vs yolov11s. v5 needs no training-time "
              "DPU surgery — only the standard SiLU → LeakyReLU swap at "
              "compile time. Sits between yolov11n (3.6M) and yolov11s (13.5M) "
              "in parameter count; isolates 'architecture generation' from "
              "'capacity' in the capacity-vs-quantization study.",
    ),

    # ── YOLOv11 (Ultralytics; anchor-free DFL with attention backbone) ──────
    # Requires architectural modifications for DPU compatibility (C2PSA →
    # C2PSA_DPU and DWConv → Conv) applied at training time. The compile
    # flow itself is identical to yolov5 once a DPU-friendly .pt exists.
    "yolov11n": ModelSpec(
        name="yolov11n", family="yolov11",
        imgsz=640, reg_max=16, status="full",
        notes="YOLOv11n (Ultralytics; anchor-free DFL, ~3.6M params after "
              "DPU-friendly architecture surgery). Requires the C2PSA → "
              "C2PSA_DPU and DWConv → Conv replacements applied at "
              "training time — see scripts/host/_train_yolov11.py. "
              "Compiles to a single ~352-op DPU subgraph on KV260 B4096. "
              "Validated on the egg detection task (mAP@0.5 ≈ 0.99).",
    ),

    "yolov11s": ModelSpec(
        name="yolov11s", family="yolov11",
        imgsz=640, reg_max=16, status="full",
        notes="YOLOv11s (Ultralytics; anchor-free DFL, ~9.4M params after "
              "DPU-friendly architecture surgery). Same DPU-friendly "
              "modifications as yolov11n (C2PSA → C2PSA_DPU, DWConv → "
              "Conv). 2.6x more parameters than yolov11n; investigates "
              "whether higher model capacity is more robust to int8 "
              "quantization noise for fine-grained discrimination tasks.",
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

    # ─────────────────────────────────────────────────────────────────────────
    # v0.12 multi-model multi-dataset pipeline
    #
    # Status note: all v0.12 variants are marked "stub" until training has
    # actually been run. Flip to "full" per-variant after `train_all.sh`
    # completes and the xmodel has been verified on the Kria.
    # ─────────────────────────────────────────────────────────────────────────

    # ── YOLOv5n on v0.12 datasets ───────────────────────────────────────────
    "yolov5n_bstld": ModelSpec(
        name="yolov5n_bstld", family="yolov5",
        imgsz=640, nc=4, reg_max=16, status="stub",
        notes="v0.12 — YOLOv5n on Bosch Small Traffic Lights (4 classes: "
              "red, yellow, green, off). Baseline against Amin & Hasan 2024 "
              "IEEE Access (YOLOv3-Tiny BSTLD on KV260: 15 FPS, 3.5 W, 99%).",
    ),
    "yolov5n_license_plates": ModelSpec(
        name="yolov5n_license_plates", family="yolov5",
        imgsz=640, nc=1, reg_max=16, status="stub",
        notes="v0.12 — YOLOv5n on Roboflow LPR (license-plate region "
              "detection, no OCR). Successor to yolov5n_lpr; same model, "
              "new larger Roboflow dataset.",
    ),
    "yolov5n_vineset": ModelSpec(
        name="yolov5n_vineset", family="yolov5",
        imgsz=640, nc=2, reg_max=16, status="stub",
        notes="v0.12 — YOLOv5n on VineSet (Magalhães Zenodo 5717293; grape "
              "bunches + trunks). Baseline against Magalhães 2022 EAAI "
              "RetinaNet-ResNet50 (KV260 14-25 FPS, ~5 W).",
    ),

    # ── YOLOv5s on v0.12 datasets ───────────────────────────────────────────
    "yolov5s_bstld": ModelSpec(
        name="yolov5s_bstld", family="yolov5",
        imgsz=640, nc=4, reg_max=16, status="stub",
        notes="v0.12 — YOLOv5s on BSTLD. Larger capacity than yolov5n; tests "
              "whether bigger non-attention model closes gap with YOLOv11n.",
    ),
    "yolov5s_license_plates": ModelSpec(
        name="yolov5s_license_plates", family="yolov5",
        imgsz=640, nc=1, reg_max=16, status="stub",
        notes="v0.12 — YOLOv5s on Roboflow LPR.",
    ),
    "yolov5s_vineset": ModelSpec(
        name="yolov5s_vineset", family="yolov5",
        imgsz=640, nc=2, reg_max=16, status="stub",
        notes="v0.12 — YOLOv5s on VineSet.",
    ),

    # ── YOLOv11n on v0.12 datasets ──────────────────────────────────────────
    # Same DPU-friendly architecture surgery as yolov11n: C2PSA → C2PSA_DPU
    # and DWConv → Conv applied at training time. See _train_yolov11.py.
    "yolov11n_bstld": ModelSpec(
        name="yolov11n_bstld", family="yolov11",
        imgsz=640, nc=4, reg_max=16, status="stub",
        notes="v0.12 — YOLOv11n on BSTLD. Attention-based architecture "
              "(C2PSA_DPU gated conv) on 4-class traffic-light detection.",
    ),
    "yolov11n_license_plates": ModelSpec(
        name="yolov11n_license_plates", family="yolov11",
        imgsz=640, nc=1, reg_max=16, status="stub",
        notes="v0.12 — YOLOv11n on Roboflow LPR.",
    ),
    "yolov11n_vineset": ModelSpec(
        name="yolov11n_vineset", family="yolov11",
        imgsz=640, nc=2, reg_max=16, status="stub",
        notes="v0.12 — YOLOv11n on VineSet. Direct comparison against "
              "Magalhães RetinaNet baseline on same dataset.",
    ),

    # ── YOLOv11s on v0.12 datasets ──────────────────────────────────────────
    "yolov11s_bstld": ModelSpec(
        name="yolov11s_bstld", family="yolov11",
        imgsz=640, nc=4, reg_max=16, status="stub",
        notes="v0.12 — YOLOv11s on BSTLD. Higher-capacity attention model; "
              "tests whether bigger gated-attention recovers more int8 "
              "robustness than larger non-attention (yolov5s).",
    ),
    "yolov11s_license_plates": ModelSpec(
        name="yolov11s_license_plates", family="yolov11",
        imgsz=640, nc=1, reg_max=16, status="stub",
        notes="v0.12 — YOLOv11s on Roboflow LPR.",
    ),
    "yolov11s_vineset": ModelSpec(
        name="yolov11s_vineset", family="yolov11",
        imgsz=640, nc=2, reg_max=16, status="stub",
        notes="v0.12 — YOLOv11s on VineSet.",
    ),

    # ── SSDLite-MobileNetV3-Large (new family: ssdlite) ─────────────────────
    # Anchor-based single-stage detector via torchvision.models.detection.
    # HardSwish backbone activations are swapped to HardSigmoid*x at compile
    # time for DPU compatibility (see lpr_pipeline/compile/ssd_mobilenet.py).
    "ssdlite_bstld": ModelSpec(
        name="ssdlite_bstld", family="ssdlite",
        imgsz=320, nc=4, reg_max=-1, status="stub",
        notes="v0.12 — SSDLite-MobileNetV3-Large on BSTLD. Non-YOLO "
              "single-stage detector for architectural philosophy "
              "comparison in defence table.",
    ),
    "ssdlite_license_plates": ModelSpec(
        name="ssdlite_license_plates", family="ssdlite",
        imgsz=320, nc=1, reg_max=-1, status="stub",
        notes="v0.12 — SSDLite-MobileNetV3-Large on Roboflow LPR.",
    ),
    "ssdlite_vineset": ModelSpec(
        name="ssdlite_vineset", family="ssdlite",
        imgsz=320, nc=2, reg_max=-1, status="stub",
        notes="v0.12 — SSDLite-MobileNetV3-Large on VineSet.",
    ),

    # ── RetinaNet-ResNet50-FPN (new family: retinanet) ──────────────────────
    # Focal-loss single-stage detector via torchvision.models.detection.
    # FPN's bilinear upsampling may produce multi-subgraph xmodel; deploy
    # via vitis_ai_library.GraphRunner on Kria if so.
    "retinanet_bstld": ModelSpec(
        name="retinanet_bstld", family="retinanet",
        imgsz=640, nc=4, reg_max=-1, status="stub",
        notes="v0.12 — RetinaNet-ResNet50-FPN on BSTLD. Heaviest variant in "
              "v0.12 (~37M params). May produce multi-subgraph xmodel.",
    ),
    "retinanet_license_plates": ModelSpec(
        name="retinanet_license_plates", family="retinanet",
        imgsz=640, nc=1, reg_max=-1, status="stub",
        notes="v0.12 — RetinaNet-ResNet50-FPN on Roboflow LPR.",
    ),
    "retinanet_vineset": ModelSpec(
        name="retinanet_vineset", family="retinanet",
        imgsz=640, nc=2, reg_max=-1, status="stub",
        notes="v0.12 — RetinaNet-ResNet50-FPN on VineSet. Closest "
              "peer-reviewed reproduction (Magalhães 2022 EAAI used "
              "RetinaNet on this exact dataset).",
    ),

    # ── Classification (new family: classification) ─────────────────────────
    # Sub-architecture is decoded from the variant name prefix via
    # classification_subarch(). See lpr_pipeline/compile/classification.py.
    "resnet50_gtsrb": ModelSpec(
        name="resnet50_gtsrb", family="classification",
        imgsz=224, nc=43, reg_max=-1, status="stub",
        notes="v0.12 — ResNet50 on GTSRB (43-class traffic-sign "
              "classification). ImageNet-pretrained transfer learning.",
    ),
    "resnet50_oxford_pets": ModelSpec(
        name="resnet50_oxford_pets", family="classification",
        imgsz=224, nc=37, reg_max=-1, status="stub",
        notes="v0.12 — ResNet50 on Oxford-IIIT Pets (37 cat+dog breeds). "
              "Fine-grained breed classification.",
    ),
    "mobilenetv2_gtsrb": ModelSpec(
        name="mobilenetv2_gtsrb", family="classification",
        imgsz=224, nc=43, reg_max=-1, status="stub",
        notes="v0.12 — MobileNetV2 on GTSRB. Efficiency baseline.",
    ),
    "mobilenetv2_oxford_pets": ModelSpec(
        name="mobilenetv2_oxford_pets", family="classification",
        imgsz=224, nc=37, reg_max=-1, status="stub",
        notes="v0.12 — MobileNetV2 on Oxford-IIIT Pets.",
    ),
    "inceptionv3_gtsrb": ModelSpec(
        name="inceptionv3_gtsrb", family="classification",
        imgsz=299, nc=43, reg_max=-1, status="stub",
        notes="v0.12 — InceptionV3 on GTSRB. Note: 299x299 input (not 224).",
    ),
    "inceptionv3_oxford_pets": ModelSpec(
        name="inceptionv3_oxford_pets", family="classification",
        imgsz=299, nc=37, reg_max=-1, status="stub",
        notes="v0.12 — InceptionV3 on Oxford-IIIT Pets.",
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


# ─────────────────────────────────────────────────────────────────────────────
# Classification sub-architecture dispatch
#
# The "classification" family covers three different torchvision models
# (ResNet50, MobileNetV2, InceptionV3). The compile path needs to know which
# factory to call; the sub-arch is encoded in the variant name prefix.
# ─────────────────────────────────────────────────────────────────────────────
CLASSIFICATION_SUBARCHS = ("resnet50", "mobilenetv2", "inceptionv3")


def classification_subarch(variant_name: str) -> str:
    """Return the torchvision sub-architecture name from a classification variant.

    Example::

        classification_subarch("resnet50_oxford_pets")     # → "resnet50"
        classification_subarch("mobilenetv2_gtsrb")        # → "mobilenetv2"
        classification_subarch("inceptionv3_oxford_pets")  # → "inceptionv3"

    Raises
    ------
    ValueError
        If ``variant_name`` does not begin with one of the known sub-arch
        prefixes followed by an underscore.
    """
    for arch in CLASSIFICATION_SUBARCHS:
        if variant_name.startswith(arch + "_"):
            return arch
    raise ValueError(
        f"Variant {variant_name!r} does not begin with a known classification "
        f"sub-architecture prefix (one of: {CLASSIFICATION_SUBARCHS})"
    )
