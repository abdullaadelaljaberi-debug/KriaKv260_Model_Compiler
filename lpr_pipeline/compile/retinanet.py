"""
Compile path for RetinaNet-ResNet50-FPN detection on the Kria KV260.

RetinaNet is a single-stage detector with:
- ResNet50 backbone (Conv + BN + ReLU; fully DPU-native)
- Feature Pyramid Network (FPN) with lateral connections and top-down upsampling
- Classification head + bbox regression head per FPN level
- Focal loss (training-time only; inference is standard sigmoid)

The architecture is mostly DPU-friendly. The one concern is FPN's upsampling
(bilinear by default). The DPU supports nearest-neighbor upsampling natively;
bilinear may need substitution. Torchvision's FPN uses
F.interpolate(mode='nearest') by default, which IS DPU-compatible.

Compile flow:
    1. Load the trained .pth (saved by train_detection.py with model_name='retinanet')
    2. Rebuild RetinaNet-ResNet50-FPN architecture, replace classification head
       for our num_classes
    3. NNDCT calibrate on GPU with in-domain images
    4. vai_c_xir compile to KV260 xmodel
    5. Verify subgraph count (may not be 1 due to FPN; documented if so)

Usage:
    python3 -m lpr_pipeline.compile.retinanet \\
        --variant retinanet_vineset \\
        --weights data/weights/detection/retinanet_vineset.pth \\
        --calib data/calib/detection/vineset/ \\
        --output out/retinanet_vineset/
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
# Model factory
# ---------------------------------------------------------------------------

def build_retinanet_model(num_classes: int):
    """Construct RetinaNet-ResNet50-FPN with replaced classification head."""
    from torchvision.models.detection import retinanet_resnet50_fpn  # type: ignore
    from torchvision.models.detection.retinanet import RetinaNetClassificationHead  # type: ignore
    from functools import partial
    import torch.nn as nn  # type: ignore

    model = retinanet_resnet50_fpn(weights=None, weights_backbone=None)

    # Replace classification head — same recipe as in train_detection.py
    in_features = model.head.classification_head.cls_logits.in_channels
    num_anchors = model.head.classification_head.num_anchors
    # +1 because torchvision uses class 0 as background
    model.head.classification_head = RetinaNetClassificationHead(
        in_features, num_anchors, num_classes + 1,
        norm_layer=partial(nn.GroupNorm, 32))

    return model


def get_input_shape() -> Tuple[int, int, int, int]:
    """RetinaNet input shape — matches the 640x640 we trained with."""
    return (1, 3, 640, 640)


# ---------------------------------------------------------------------------
# Compile flow
# ---------------------------------------------------------------------------

def compile_retinanet(
    variant: str,
    weights_path: Path,
    calib_dir: Path,
    output_dir: Path,
    num_calib_images: int = 200,
) -> int:
    """Run the full compile pipeline for one RetinaNet variant."""
    log_step(f"Compiling {variant} (RetinaNet-ResNet50-FPN)")

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
    user_num_classes = num_classes - 1
    log_info(f"  num_classes (excluding background): {user_num_classes}")
    log_info(f"  trained dataset: {ckpt.get('dataset', '?')}")
    log_info(f"  trained for {ckpt.get('epoch', '?')} epochs, "
             f"loss={ckpt.get('loss', '?')}")

    # ---- Build model with float weights ----
    model = build_retinanet_model(user_num_classes)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    log_info("float model built and weights loaded")

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

    # Wrap RetinaNet so NNDCT sees a tensor-in, tensor-out signature.
    # The model in eval mode does heavy post-processing; we want raw outputs.
    class RetinaNetForwardWrapper(torch.nn.Module):
        def __init__(self, retinanet_model):
            super().__init__()
            self.retinanet = retinanet_model
            self.retinanet.eval()

        def forward(self, x):
            # Walk backbone -> FPN -> head, skipping post-processing
            features = self.retinanet.backbone(x)
            if isinstance(features, dict):
                feature_list = list(features.values())
            else:
                feature_list = features
            head_outputs = self.retinanet.head(feature_list)
            return head_outputs['bbox_regression'], head_outputs['cls_logits']

    model_wrapped = RetinaNetForwardWrapper(model)

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

    # Calibration data
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

    # ---- vai_c_xir compile ----
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

    raw_out = output_dir / f"{variant}.xmodel"
    if raw_out.exists() and not final_xmodel.exists():
        raw_out.rename(final_xmodel)
    log_info(f"final xmodel: {final_xmodel}")

    # ---- Verify subgraph count ----
    # NOTE: FPN may produce multi-subgraph xmodel due to lateral connections.
    # That's OK — we just need to know so the Kria-side runner uses GraphRunner.
    log_info("verifying subgraph count...")
    try:
        import xir  # type: ignore
        graph = xir.Graph.deserialize(str(final_xmodel))
        root = graph.get_root_subgraph()
        dpu_subs = [s for s in root.toposort_child_subgraph()
                    if s.has_attr("device") and s.get_attr("device") == "DPU"]
        log_info(f"  DPU subgraphs: {len(dpu_subs)}")
        if len(dpu_subs) > 1:
            log_info(f"  NOTE: multi-subgraph xmodel — deploy via "
                     f"vitis_ai_library.GraphRunner on Kria, not DpuOverlay.load_model")
    except ImportError:
        log_info("  (xir not available for verification; skipping)")

    log_info(f"DONE: {final_xmodel}")
    return 0


# ---------------------------------------------------------------------------
# BaseCompiler adapter — plugs into lpr_pipeline.compile.registry
# ---------------------------------------------------------------------------
class Compiler(BaseCompiler):
    """Thin adapter so the registry can dispatch ``retinanet`` cleanly.

    The standalone ``compile_retinanet()`` function above and its ``main()``
    CLI remain usable for direct invocation. This class is what
    ``02_compile.sh`` drives via ``get_compiler("retinanet")``.
    """
    family = "retinanet"

    def _compile_family(self, inputs: CompileInputs) -> Path:
        output_dir = inputs.out_xmodel.parent
        rc = compile_retinanet(
            variant=inputs.spec.name,
            weights_path=inputs.weights,
            calib_dir=inputs.calib_dir,
            output_dir=output_dir,
            num_calib_images=inputs.n_calib,
        )
        if rc != 0:
            raise CompileError(
                f"retinanet compile failed for {inputs.spec.name} "
                f"(compile_retinanet returned {rc})"
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
                    help="name for this compiled variant, e.g. retinanet_bstld")
    ap.add_argument("--weights", type=Path, required=True,
                    help="path to trained .pth file")
    ap.add_argument("--calib", type=Path, required=True,
                    help="directory of calibration images")
    ap.add_argument("--output", type=Path, required=True,
                    help="output directory for compiled xmodel")
    ap.add_argument("--num-calib-images", type=int, default=200)
    args = ap.parse_args()

    return compile_retinanet(
        args.variant, args.weights, args.calib, args.output,
        args.num_calib_images,
    )


if __name__ == "__main__":
    sys.exit(main())
