"""
Compile path for SSDLite-MobileNetV3-Large detection on the Kria KV260.

SSDLite is a depthwise-separable variant of SSD using a MobileNetV3-Large
backbone. The architecture maps cleanly to DPU primitives:
- Backbone: Conv + BN + HardSwish/ReLU + DWConv (ALU engine)
- SSDLite head: DWConv + 1x1 Conv per feature scale
- Anchor decoding: CPU-side post-processing (no in-graph anchor math)

Note on HardSwish: SSDLite-MobileNetV3 uses HardSwish in the backbone, which
is NOT in the DPU's native ISA. We replace it with HardSigmoid * x at trace
time. This produces a similar curve and is DPU-native. The replacement adds
no parameters; the accuracy hit recovers in fine-tune.

Compile flow:
    1. Load the trained .pth (saved by train_detection.py with model_name='ssdlite')
    2. Rebuild SSDLite-MobileNetV3 architecture, replace classification head
       for our num_classes
    3. Swap HardSwish -> HardSigmoid*x in the backbone
    4. NNDCT calibrate on GPU with in-domain images
    5. vai_c_xir compile to KV260 xmodel
    6. Verify single-subgraph output

Usage:
    python3 -m lpr_pipeline.compile.ssd_mobilenet \\
        --variant ssdlite_bstld \\
        --weights data/weights/detection/ssdlite_bstld.pth \\
        --calib data/calib/detection/bstld/ \\
        --output out/ssdlite_bstld/
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path
from typing import Tuple

from .base import BaseCompiler, CompileError, CompileInputs



def log_step(msg: str) -> None:
    print(f"\n{'='*72}\n>>> {msg}\n{'='*72}")

def log_info(msg: str) -> None:
    print(f"    {msg}")

def log_err(msg: str) -> None:
    print(f"    ERROR: {msg}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Model factory and DPU-friendly substitutions
# ---------------------------------------------------------------------------

def swap_hardswish_to_dpu_friendly(model) -> None:
    """Replace nn.Hardswish in the model with HardSigmoid*x in-place.

    The DPU has HardSigmoid as a native operator but not HardSwish.
    Mathematically, HardSwish(x) = x * HardSigmoid(x), which IS DPU-compatible.
    This function walks the module tree and substitutes.
    """
    import torch.nn as nn  # type: ignore

    class HardSwishDpu(nn.Module):
        """HardSwish reformulated as x * HardSigmoid(x) — DPU-compatible."""
        def __init__(self):
            super().__init__()
            self.hardsigmoid = nn.Hardsigmoid()
        def forward(self, x):
            return x * self.hardsigmoid(x)

    for name, child in model.named_children():
        if isinstance(child, nn.Hardswish):
            setattr(model, name, HardSwishDpu())
        else:
            swap_hardswish_to_dpu_friendly(child)


def build_ssdlite_model(num_classes: int):
    """Construct SSDLite-MobileNetV3-Large with replaced classification head."""
    from torchvision.models.detection import ssdlite320_mobilenet_v3_large  # type: ignore
    from torchvision.models.detection.ssdlite import SSDLiteClassificationHead  # type: ignore
    from functools import partial
    import torch.nn as nn  # type: ignore

    # No pretrained weights needed — we load the trained checkpoint below
    model = ssdlite320_mobilenet_v3_large(weights=None, weights_backbone=None)

    # Replace classification head for the target num_classes
    in_channels = [m.in_channels for m in model.head.classification_head.module_list]
    num_anchors = model.anchor_generator.num_anchors_per_location()
    norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.03)
    # +1 because torchvision detection models count background as class 0
    model.head.classification_head = SSDLiteClassificationHead(
        in_channels, num_anchors, num_classes + 1, norm_layer)

    return model


def get_input_shape() -> Tuple[int, int, int, int]:
    """SSDLite-MobileNetV3-Large standard input shape."""
    return (1, 3, 320, 320)


# ---------------------------------------------------------------------------
# Compile flow
# ---------------------------------------------------------------------------

def compile_ssdlite(
    variant: str,
    weights_path: Path,
    calib_dir: Path,
    output_dir: Path,
    num_calib_images: int = 200,
) -> int:
    """Run the full compile pipeline for one SSDLite variant."""
    log_step(f"Compiling {variant} (SSDLite-MobileNetV3-Large)")

    try:
        import torch  # type: ignore
        from torchvision import transforms  # type: ignore
        from PIL import Image  # type: ignore
    except ImportError as e:
        log_err(f"torch / torchvision / PIL not available: {e}")
        return 1

    try:
        from pytorch_nndct.apis import torch_quantizer  # type: ignore
    except ImportError:
        log_err("pytorch_nndct not available. Run this inside the Vitis AI Docker:")
        log_err("  docker run -it xilinx/vitis-ai-pytorch-gpu:3.5.0 ...")
        return 1

    # ---- Load checkpoint ----
    log_info(f"loading checkpoint: {weights_path}")
    ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict) or "model_state_dict" not in ckpt:
        log_err("checkpoint format unexpected: missing 'model_state_dict' key")
        return 1

    state_dict = ckpt["model_state_dict"]
    num_classes = ckpt.get("num_classes")
    if num_classes is None:
        log_err("checkpoint missing num_classes")
        return 1
    # train_detection.py stores num_classes already +1 for background, so subtract back
    user_num_classes = num_classes - 1
    log_info(f"  num_classes (excluding background): {user_num_classes}")
    log_info(f"  trained dataset: {ckpt.get('dataset', '?')}")
    log_info(f"  trained for {ckpt.get('epoch', '?')} epochs, "
             f"loss={ckpt.get('loss', '?')}")

    # ---- Build model with float weights ----
    model = build_ssdlite_model(user_num_classes)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    log_info("float model built and weights loaded")

    # ---- Swap HardSwish -> DPU-friendly form ----
    log_info("swapping HardSwish -> HardSigmoid*x for DPU compatibility")
    swap_hardswish_to_dpu_friendly(model)

    # ---- Determine calibration device ----
    calib_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_info(f"calibration device: {calib_device}")

    # ---- Calibration ----
    log_info(f"running calibration pass with images from {calib_dir}")
    input_shape = get_input_shape()
    dummy_input = torch.zeros(input_shape, dtype=torch.float32)

    output_dir.mkdir(parents=True, exist_ok=True)
    quant_dir = output_dir / "quantized"
    quant_dir.mkdir(parents=True, exist_ok=True)

    # NNDCT calibration pass.
    # Note: torchvision detection models accept a list of tensors in training mode
    # and a list in eval mode. For NNDCT trace we need a single-tensor entry point,
    # which is why we wrap the model:
    class SSDForwardWrapper(torch.nn.Module):
        def __init__(self, ssd_model):
            super().__init__()
            self.ssd = ssd_model
            self.ssd.eval()

        def forward(self, x):
            # The SSDLite model in eval mode normally returns post-processed
            # detection results. For quantization we want the raw conv outputs.
            # Walk through the backbone -> head manually:
            features = self.ssd.backbone(x)
            head_outputs = self.ssd.head(list(features.values()))
            # head returns {'bbox_regression': ..., 'cls_logits': ...}
            return head_outputs['bbox_regression'], head_outputs['cls_logits']

    model_wrapped = SSDForwardWrapper(model)

    quantizer = torch_quantizer(
        quant_mode="calib",
        module=model_wrapped,
        input_args=(dummy_input.to(calib_device),),
        output_dir=str(quant_dir),
        bitwidth=8,
        device=calib_device,
    )
    quant_model = quantizer.quant_model
    quant_model.to(calib_device)

    # Build calibration data loader
    imgsz = input_shape[2]
    tf = transforms.Compose([
        transforms.Resize((imgsz, imgsz)),
        transforms.ToTensor(),
    ])
    calib_images = sorted(
        list(calib_dir.glob("*.jpg")) +
        list(calib_dir.glob("*.png")) +
        list(calib_dir.glob("*.jpeg"))
    )[:num_calib_images]
    if not calib_images:
        log_err(f"no calibration images found under {calib_dir}")
        return 1
    log_info(f"calibrating with {len(calib_images)} images")

    quant_model.eval()
    with torch.no_grad():
        for i, img_path in enumerate(calib_images):
            img = Image.open(img_path).convert("RGB")
            x = tf(img).unsqueeze(0).to(calib_device)
            _ = quant_model(x)
            if (i + 1) % 50 == 0:
                log_info(f"  calibrated {i+1}/{len(calib_images)}")

    quantizer.export_quant_config()
    log_info("calibration complete")

    # ---- Test/Deploy pass ----
    log_info("running test pass to export quantized xmodel...")
    quantizer = torch_quantizer(
        quant_mode="test",
        module=model_wrapped,
        input_args=(dummy_input.to(calib_device),),
        output_dir=str(quant_dir),
        bitwidth=8,
        device=calib_device,
    )
    quant_model = quantizer.quant_model
    quant_model.to(calib_device)
    quant_model.eval()
    with torch.no_grad():
        _ = quant_model(dummy_input.to(calib_device))
    quantizer.export_xmodel(output_dir=str(quant_dir), deploy_check=False)
    log_info(f"xmodel exported to {quant_dir}")

    # ---- vai_c_xir compile to KV260 target ----
    import subprocess
    xmodel_candidates = list(quant_dir.glob("*_int.xmodel"))
    if not xmodel_candidates:
        log_err(f"no _int.xmodel produced in {quant_dir}")
        return 1
    xmodel_in = xmodel_candidates[0]
    log_info(f"input xmodel: {xmodel_in}")

    arch_json = Path("/opt/vitis_ai/compiler/arch/DPUCZDX8G/KV260/arch.json")
    if not arch_json.exists():
        log_err(f"KV260 arch.json not found at {arch_json}")
        log_err("not running inside Vitis AI Docker?")
        return 1

    final_xmodel = output_dir / f"{variant}_kv260.xmodel"
    cmd = [
        "vai_c_xir",
        "--xmodel", str(xmodel_in),
        "--arch", str(arch_json),
        "--output_dir", str(output_dir),
        "--net_name", variant,
    ]
    log_info(f"running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        log_err("vai_c_xir failed:")
        log_err(result.stdout)
        log_err(result.stderr)
        return 1
    log_info("vai_c_xir compile successful")

    # Rename for clarity
    raw_out = output_dir / f"{variant}.xmodel"
    if raw_out.exists() and not final_xmodel.exists():
        raw_out.rename(final_xmodel)
    log_info(f"final xmodel: {final_xmodel}")

    # ---- Verify subgraph count ----
    log_info("verifying subgraph count...")
    try:
        import xir  # type: ignore
        graph = xir.Graph.deserialize(str(final_xmodel))
        root = graph.get_root_subgraph()
        dpu_subs = [s for s in root.toposort_child_subgraph()
                    if s.has_attr("device") and s.get_attr("device") == "DPU"]
        log_info(f"  DPU subgraphs: {len(dpu_subs)}")
        if len(dpu_subs) != 1:
            log_err(f"WARNING: expected 1 DPU subgraph, found {len(dpu_subs)}")
            log_err("this xmodel will NOT load via pynq_dpu.DpuOverlay.load_model")
            log_err("use vitis_ai_library.GraphRunner instead on the Kria side")
            return 2
    except ImportError:
        log_info("  (xir not available for verification; skipping)")

    log_info(f"DONE: {final_xmodel}")
    return 0


# ---------------------------------------------------------------------------
# BaseCompiler adapter — plugs into lpr_pipeline.compile.registry
# ---------------------------------------------------------------------------
class Compiler(BaseCompiler):
    """Thin adapter so the registry can dispatch ``ssdlite`` cleanly.

    The standalone ``compile_ssdlite()`` function above and its ``main()``
    CLI remain usable for direct invocation. This class is what
    ``02_compile.sh`` drives via ``get_compiler("ssdlite")``.
    """
    family = "ssdlite"

    def _compile_family(self, inputs: CompileInputs) -> Path:
        output_dir = inputs.out_xmodel.parent
        rc = compile_ssdlite(
            variant=inputs.spec.name,
            weights_path=inputs.weights,
            calib_dir=inputs.calib_dir,
            output_dir=output_dir,
            num_calib_images=inputs.n_calib,
        )
        if rc != 0:
            raise CompileError(
                f"ssdlite compile failed for {inputs.spec.name} "
                f"(compile_ssdlite returned {rc})"
            )
        if not inputs.out_xmodel.is_file():
            raise CompileError(
                f"compile reported success but {inputs.out_xmodel} is missing"
            )
        return inputs.out_xmodel


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", required=True,
                    help="name for this compiled variant, e.g. ssdlite_bstld")
    ap.add_argument("--weights", type=Path, required=True,
                    help="path to trained .pth file")
    ap.add_argument("--calib", type=Path, required=True,
                    help="directory of calibration images")
    ap.add_argument("--output", type=Path, required=True,
                    help="output directory for compiled xmodel")
    ap.add_argument("--num-calib-images", type=int, default=200)
    args = ap.parse_args()

    return compile_ssdlite(
        args.variant, args.weights, args.calib, args.output,
        args.num_calib_images,
    )


if __name__ == "__main__":
    sys.exit(main())
