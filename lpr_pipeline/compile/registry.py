"""Family → Compiler dispatch.

Imports are deferred (``importlib``) so that a stubbed family can declare
``import torch`` without breaking the whole package on environments where
torch isn't installed. Only the family being compiled gets imported.
"""
from __future__ import annotations

import importlib

from .base import BaseCompiler, NotImplementedFamilyError


_FAMILY_TO_MODULE = {
    "yolov5":           "lpr_pipeline.compile.yolov5",
    "yolov11":          "lpr_pipeline.compile.yolov11",
    "yolox":            "lpr_pipeline.compile.yolox",
    "yolov7":           "lpr_pipeline.compile.yolov7",
    "yolov4_csp":       "lpr_pipeline.compile.yolov4_csp",
    "ssd_mobilenetv2":  "lpr_pipeline.compile.ssd_mobilenetv2",
    # v0.12 — multi-arch, multi-dataset benchmark:
    "ssdlite":          "lpr_pipeline.compile.ssd_mobilenet",
    "retinanet":        "lpr_pipeline.compile.retinanet",
    "classification":   "lpr_pipeline.compile.classification",
}


def get_compiler(family: str) -> BaseCompiler:
    """Return an instance of the Compiler for the given family.

    Raises:
        ValueError: family is not a known name
        NotImplementedFamilyError: family is known but its compile path is a stub
    """
    if family not in _FAMILY_TO_MODULE:
        known = ", ".join(_FAMILY_TO_MODULE)
        raise ValueError(f"Unknown family: {family!r}. Known: {known}")

    mod = importlib.import_module(_FAMILY_TO_MODULE[family])
    if not hasattr(mod, "Compiler"):
        raise RuntimeError(
            f"{_FAMILY_TO_MODULE[family]} does not export a Compiler class. "
            f"This is a bug in the stub."
        )
    return mod.Compiler()
