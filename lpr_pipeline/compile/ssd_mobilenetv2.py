"""SSD-MobileNetV2-TF compile path — STUB.

Status: directory and class skeleton are in place. This family uses
TensorFlow rather than PyTorch — quantization toolchain is
``vai_q_tensorflow2``, not ``vai_q_pytorch``.

To implement:
  1. Study the VAI 3.5 zoo entry: tf_ssdmobilenetv2_coco_300_300_3.5
  2. The compile path is substantially different: SavedModel → frozen graph →
     vai_q_tensorflow2 quant → vai_c_tensorflow → xmodel
  3. Output format is also different: SSD's prior boxes need their own
     decoder (different from YOLO families)
  4. Note: the Vitis-AI 3.5 Docker image used for PyTorch families is the
     PyTorch image; SSD-MobileNet may need the TF image
     (xilinx/vitis-ai-tensorflow2-gpu:3.5.0.001)
"""
from .base import BaseCompiler, NotImplementedFamilyError, CompileInputs


class Compiler(BaseCompiler):
    family = "ssd_mobilenetv2"

    def _compile_family(self, inputs: CompileInputs):
        raise NotImplementedFamilyError(
            "SSD-MobileNetV2 compile path is a stub.\n"
            "Implementation lives in: lpr_pipeline/compile/ssd_mobilenetv2.py\n"
            "See docs/MODELS.md → SSD-MobileNetV2 section for what's required.\n\n"
            "Note: this family requires the Vitis-AI TensorFlow Docker image, "
            "not the PyTorch one used by other families. The host script\n"
            "01_install_vai.sh would also need updating to pull that image."
        )
