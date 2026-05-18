"""YOLOv11 compile path.

YOLOv11 shares the compile flow with YOLOv5u (modern Ultralytics anchor-free
DFL): same activation swap (SiLU → LeakyReLU(13/128)), same Detect-head
stripping (``cat(cv2[i], cv3[i])``), same calibration, same NNDCT
quantization, same ``vai_c_xir`` compile.

The only YOLOv11-specific concern is that the trained ``.pt`` references
``lpr_pipeline.c2psa_dpu.C2PSA_DPU`` (our DPU-friendly replacement for the
stock attention block). When ``torch.load`` deserializes the checkpoint
inside the Vitis-AI container, pickle's ``find_class`` needs to be able
to import that module. The repo is mounted at ``/workspace`` and
``PYTHONPATH=/workspace`` is set by ``02_compile.sh``, so the import
resolves automatically — but we explicitly import here too as a defensive
measure (makes any failure surface a clear ImportError rather than a
confusing pickle KeyError).

Most of the logic lives in ``yolov5.py`` and is inherited verbatim.
"""
from __future__ import annotations

from .base import CompileError, CompileInputs
from .yolov5 import Compiler as YOLOv5Compiler


class Compiler(YOLOv5Compiler):
    """YOLOv11 compiler.

    Inherits the full pipeline from ``YOLOv5Compiler``. The compile flow is
    architecture-agnostic for any modern Ultralytics u-variant (YOLOv5u,
    YOLOv8, YOLOv11) once the model has been trained with our DPU-friendly
    monkey-patches. The Detect head structure is identical — anchor-free,
    ``reg_max=16`` DFL, ``cv2``/``cv3`` split — so ``_strip_detect_head_for_quant``
    handles all of them the same way.

    What we override
    ----------------

    - ``family`` — for registry-vs-spec consistency checks
    - ``_compile_family`` — only to validate ``c2psa_dpu``/``detect_dpu``
      imports BEFORE the parent's torch.load runs, so failures surface
      with a clear cause instead of confusing pickle errors
    """

    family = "yolov11"

    def _compile_family(self, inputs: CompileInputs):
        # Defensive: surface a clear error if the surgery modules aren't
        # importable. Without these, torch.load will raise an opaque
        # pickle error when unmarshaling C2PSA_DPU instances.
        try:
            import lpr_pipeline.c2psa_dpu     # noqa: F401
            import lpr_pipeline.detect_dpu    # noqa: F401
        except ImportError as e:
            raise CompileError(
                f"Could not import lpr_pipeline.{{c2psa_dpu,detect_dpu}}: {e}\n"
                f"\n"
                f"These modules are required to deserialize a YOLOv11 .pt\n"
                f"that was trained with the DPU-friendly architecture.\n"
                f"\n"
                f"If you're inside the Vitis-AI container, ensure the repo\n"
                f"is mounted at /workspace and PYTHONPATH=/workspace.\n"
                f"02_compile.sh handles this automatically — if you're\n"
                f"running this compiler some other way, set PYTHONPATH\n"
                f"to include the repo root."
            ) from e

        # Delegate the rest of the compile flow to the YOLOv5 parent.
        return super()._compile_family(inputs)
