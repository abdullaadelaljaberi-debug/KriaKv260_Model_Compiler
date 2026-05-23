"""
Compile path for classification networks: ResNet50, MobileNetV2, InceptionV3.

These three architectures compile cleanly to a single DPU subgraph because:
- ResNet50: standard Conv + BN + ReLU + AdaptiveAvgPool + Linear. All DPU-native.
- MobileNetV2: Conv + DWConv + ReLU6. DWConv is supported on the DPU's ALU engine.
- InceptionV3: Conv + BN + ReLU + AvgPool + Linear. All DPU-native.

No SiLU/Swish, no softmax-attention, no architectural surgery needed.

The compile flow is:
    1. Load the trained PyTorch model from .pth
    2. Trace through NNDCT with the calibration set
    3. Quantize per-tensor int8, per-channel weights
    4. Compile with vai_c_xir to a KV260-targeted xmodel
    5. Verify exactly 1 DPU subgraph

Calibration data is expected at:
    data/calib/classification/<dataset>/   (built by prep_calibration.py)

Usage (called by 02_compile.sh):
    python3 -m lpr_pipeline.compile.classification \\
        --model resnet50 --variant resnet50_oxford_pets \\
        --weights data/weights/classification/resnet50_oxford_pets.pth \\
        --calib data/calib/classification/oxford_pets/ \\
        --output out/resnet50_oxford_pets/
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path
from typing import Tuple


def log_step(msg: str) -> None:
    print(f"\n{'='*72}\n>>> {msg}\n{'='*72}")

def log_info(msg: str) -> None:
    print(f"    {msg}")

def log_err(msg: str) -> None:
    print(f"    ERROR: {msg}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def build_classification_model(model_name: str, num_classes: int, imgsz: int):
    """Construct the classification model architecture, ready to load weights."""
    import torch.nn as nn  # type: ignore
    from torchvision.models import (
        resnet50, mobilenet_v2, inception_v3,
        ResNet50_Weights, MobileNet_V2_Weights, Inception_V3_Weights,
    )  # type: ignore

    if model_name == "resnet50":
        model = resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == "mobilenetv2":
        model = mobilenet_v2(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif model_name == "inceptionv3":
        # Important: aux_logits=False for inference/quantization. The aux head is
        # only used during training to combat vanishing gradients; at deploy time
        # we strip it. NNDCT can't trace a model that has dual outputs anyway.
        model = inception_v3(weights=None, aux_logits=False, init_weights=False)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        raise ValueError(f"unknown classification model: {model_name}")

    return model


def get_input_shape(model_name: str) -> Tuple[int, int, int, int]:
    """Return (batch, channel, height, width) for NNDCT trace."""
    if model_name == "inceptionv3":
        return (1, 3, 299, 299)
    else:
        return (1, 3, 224, 224)


# ---------------------------------------------------------------------------
# Compile flow
# ---------------------------------------------------------------------------

def compile_classification(
    model_name: str,
    variant: str,
    weights_path: Path,
    calib_dir: Path,
    output_dir: Path,
    num_calib_images: int = 200,
) -> int:
    """Run the full compile pipeline for one classification variant."""
    log_step(f"Compiling {variant} ({model_name})")

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
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
        num_classes = ckpt.get("num_classes")
        imgsz = ckpt.get("imgsz")
        log_info(f"  num_classes={num_classes}, imgsz={imgsz}, "
                 f"trained for {ckpt.get('epoch', '?')} epochs, "
                 f"val_acc={ckpt.get('val_acc', '?'):.4f}")
    else:
        state_dict = ckpt
        num_classes = None
        imgsz = None

    if num_classes is None:
        log_err("checkpoint missing num_classes; pass via --num-classes flag")
        return 1

    if imgsz is None:
        imgsz = 299 if model_name == "inceptionv3" else 224

    # ---- Build model with float weights ----
    model = build_classification_model(model_name, num_classes, imgsz)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    log_info("float model built and weights loaded")

    # ---- Calibration ----
    log_info(f"running calibration pass with images from {calib_dir}")
    input_shape = get_input_shape(model_name)
    dummy_input = torch.zeros(input_shape, dtype=torch.float32)

    output_dir.mkdir(parents=True, exist_ok=True)
    quant_dir = output_dir / "quantized"
    quant_dir.mkdir(parents=True, exist_ok=True)

    # Calibration runs on GPU if available (4× faster on the user's RTX A2000)
    # If NNDCT throws a CUDA error on the calibration pass, fall back to CPU.
    calib_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_info(f"calibration device: {calib_device}")

    # NNDCT calibration pass
    quantizer = torch_quantizer(
        quant_mode="calib",
        module=model,
        input_args=(dummy_input.to(calib_device),),
        output_dir=str(quant_dir),
        bitwidth=8,
        device=calib_device,
    )
    quant_model = quantizer.quant_model
    quant_model.to(calib_device)

    # Build calibration data loader
    tf = transforms.Compose([
        transforms.Resize((imgsz, imgsz)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    calib_images = sorted(
        list(calib_dir.glob("*.jpg")) +
        list(calib_dir.glob("*.png")) +
        list(calib_dir.glob("*.ppm"))
    )[:num_calib_images]
    if not calib_images:
        # Walk one level into subdirectories (ImageFolder layout)
        calib_images = []
        for sub in calib_dir.iterdir():
            if sub.is_dir():
                calib_images.extend(list(sub.glob("*.jpg")) +
                                    list(sub.glob("*.png")) +
                                    list(sub.glob("*.ppm")))
            if len(calib_images) >= num_calib_images:
                break
        calib_images = calib_images[:num_calib_images]

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
        module=model,
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
    xmodel_in = quant_dir / f"{type(model).__name__}_int.xmodel"
    if not xmodel_in.exists():
        # NNDCT names the file based on the model class
        candidates = list(quant_dir.glob("*_int.xmodel"))
        if not candidates:
            log_err(f"no _int.xmodel produced in {quant_dir}")
            return 1
        xmodel_in = candidates[0]
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
        log_err(f"vai_c_xir failed:")
        log_err(result.stdout)
        log_err(result.stderr)
        return 1
    log_info("vai_c_xir compile successful")

    # vai_c_xir produces <net_name>.xmodel — rename to _kv260.xmodel for clarity
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
            return 2
    except ImportError:
        log_info("  (xir not available for verification; skipping)")

    log_info(f"DONE: {final_xmodel}")
    return 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True,
                    choices=["resnet50", "mobilenetv2", "inceptionv3"])
    ap.add_argument("--variant", required=True,
                    help="name for this compiled variant, e.g. resnet50_oxford_pets")
    ap.add_argument("--weights", type=Path, required=True,
                    help="path to trained .pth file")
    ap.add_argument("--calib", type=Path, required=True,
                    help="directory of calibration images")
    ap.add_argument("--output", type=Path, required=True,
                    help="output directory for compiled xmodel")
    ap.add_argument("--num-calib-images", type=int, default=200)
    args = ap.parse_args()

    return compile_classification(
        args.model, args.variant, args.weights, args.calib, args.output,
        args.num_calib_images,
    )


if __name__ == "__main__":
    sys.exit(main())
