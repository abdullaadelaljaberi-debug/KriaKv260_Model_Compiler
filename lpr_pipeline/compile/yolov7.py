"""YOLOv7 (WongKinYiu) compile path — STUB.

Status: directory and class skeleton are in place, but the family-specific
quantize-and-compile logic is not implemented yet.

To implement:
  1. Study the VAI 3.5 zoo entry: pt_yolov7_640_640_3.5
  2. Mirror the structure of compile/yolov5.py
  3. YOLOv7's detect head is anchor-based (3 anchors × 3 scales);
     strip the inline anchor-grid decode similarly to YOLOv5
  4. Calibration normalization: same as YOLOv5 (RGB float[0, 1])
  5. Add the corresponding decoder in lpr_pipeline/deploy/decoders.py
"""
from .base import BaseCompiler, NotImplementedFamilyError, CompileInputs


class Compiler(BaseCompiler):
    family = "yolov7"

    def _compile_family(self, inputs: CompileInputs):
        raise NotImplementedFamilyError(
            "YOLOv7 compile path is a stub.\n"
            "Implementation lives in: lpr_pipeline/compile/yolov7.py\n"
            "See docs/MODELS.md → YOLOv7 section for what's required.\n\n"
            "Workaround for now: drop a pre-compiled xmodel from AMD's "
            "model zoo (pt_yolov7_640_640_3.5) into out/ manually, and the "
            "deploy side will run it once the runner has a YOLOv7 decoder."
        )
