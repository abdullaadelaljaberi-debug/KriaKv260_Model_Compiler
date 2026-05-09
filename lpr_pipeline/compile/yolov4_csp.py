"""YOLOv4-CSP compile path — STUB.

Status: directory and class skeleton are in place, but the family-specific
quantize-and-compile logic is not implemented yet.

To implement:
  1. Study the VAI 3.5 zoo entry: pt_yolov4_csp_512_512_3.5
  2. YOLOv4-CSP head is anchor-based with CIoU loss; the export path
     differs slightly from YOLOv5 (different stride conventions)
  3. Calibration normalization: RGB float[0, 1]
  4. Add the corresponding decoder in lpr_pipeline/deploy/decoders.py
"""
from .base import BaseCompiler, NotImplementedFamilyError, CompileInputs


class Compiler(BaseCompiler):
    family = "yolov4_csp"

    def _compile_family(self, inputs: CompileInputs):
        raise NotImplementedFamilyError(
            "YOLOv4-CSP compile path is a stub.\n"
            "Implementation lives in: lpr_pipeline/compile/yolov4_csp.py\n"
            "See docs/MODELS.md → YOLOv4-CSP section for what's required.\n\n"
            "Workaround for now: drop a pre-compiled xmodel from AMD's "
            "model zoo (pt_yolov4_csp_512_512_3.5) into out/ manually."
        )
