"""Export a trained YOLOv11 .pt to clean ONNX for the Vitis-AI ONNX-PTQ path.

Runs INSIDE the vitis-ai-onnx-cpu Docker container.

What this does:
  1. Apply DPU-friendly monkey-patches (C2PSA_DPU, DWConv → Conv)
  2. Load the trained .pt via Ultralytics
  3. Swap SiLU → LeakyReLU(0.1015625) (DPU-supported activation)
  4. Strip the Detect head's post-processing — output the three raw conv
     tensors (stride 8/16/32) instead of decoded boxes
  5. Wrap with NHWC permute (DPU accepts NHWC inputs)
  6. torch.onnx.export() with opset 17

Output: a clean ONNX file ready for vai_q_onnx.quantize_static().

Usage (inside container):
    python /workspace/scripts/host/_export_onnx_yolov11.py \
        --weights /workspace/data/weights/yolo11n_eggs_dpu.pt \
        --output  /workspace/build/yolov11n_onnx/yolov11n_eggs_dpu.onnx \
        --imgsz   640 \
        --nc      1
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", type=Path, required=True,
                   help="Trained .pt file (host path mounted at /workspace/...)")
    p.add_argument("--output",  type=Path, required=True,
                   help="Where to write the exported .onnx")
    p.add_argument("--imgsz",   type=int, default=640,
                   help="Input image size (must match training, default 640)")
    p.add_argument("--nc",      type=int, default=1,
                   help="Number of classes (default 1 for eggs)")
    p.add_argument("--opset",   type=int, default=17,
                   help="ONNX opset (default 17 — modern, well-supported)")
    return p.parse_args()


def swap_silu_to_leakyrelu(model):
    """Replace all SiLU activations with LeakyReLU(0.1015625).

    Same swap the NNDCT compile pipeline applies. DPU only supports
    LeakyReLU with negative_slope=13/128=0.1015625 natively.
    """
    import torch.nn as nn
    LEAKY_SLOPE = 13.0 / 128.0
    count = 0
    for name, child in model.named_children():
        if isinstance(child, nn.SiLU):
            inplace = getattr(child, "inplace", False)
            setattr(model, name, nn.LeakyReLU(negative_slope=LEAKY_SLOPE, inplace=inplace))
            count += 1
        else:
            count += swap_silu_to_leakyrelu(child)
    return count


def strip_detect_postprocessing(detect_module):
    """Replace Detect.forward with a version that returns raw conv outputs.

    The default Detect head decodes boxes, applies sigmoid, and runs NMS-like
    operations. We want the three raw [B, 4*reg_max+nc, H, W] tensors so
    quantization sees the model's full computation graph but stops before the
    int8-hostile box-decoding math. The host-side decoder
    (lpr_pipeline.deploy.decoders.decode_yolov11) then handles decoding on
    float DPU outputs.
    """
    import torch
    import torch.nn as nn

    def forward_raw(self, x):
        """Run cv2/cv3 branches, concatenate per-scale, return list of 3 tensors."""
        outputs = []
        for i in range(self.nl):
            cv2_out = self.cv2[i](x[i])  # box regression branch
            cv3_out = self.cv3[i](x[i])  # classification branch
            outputs.append(torch.cat((cv2_out, cv3_out), 1))
        return outputs

    # Bind the new forward to this specific instance
    import types
    detect_module.forward = types.MethodType(forward_raw, detect_module)


class NHWCWrapper:
    """Wraps a model so it accepts NHWC inputs (DPU layout).

    DPU expects NHWC; PyTorch/ONNX work in NCHW. This wrapper transposes the
    input once at the start. The transpose becomes part of the ONNX graph.
    """
    pass  # placeholder, real class defined inside main() to share imports


def main():
    args = parse_args()

    print()
    print("=" * 70)
    print("  ONNX export for YOLOv11 → Vitis-AI ONNX PTQ path")
    print("=" * 70)
    print(f"  Input:  {args.weights}")
    print(f"  Output: {args.output}")
    print(f"  imgsz:  {args.imgsz}")
    print(f"  nc:     {args.nc}")
    print(f"  opset:  {args.opset}")
    print()

    if not args.weights.is_file():
        print(f"ERROR: weights file not found: {args.weights}", file=sys.stderr)
        sys.exit(1)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # ── 1. Apply DPU-friendly monkey-patches ─────────────────────────────
    print("─── Applying DPU-friendly monkey-patches ─────────────────────────")
    sys.path.insert(0, '/workspace')
    from lpr_pipeline.c2psa_dpu  import C2PSA_DPU                # noqa: F401
    from lpr_pipeline.detect_dpu import apply_dwconv_monkey_patch
    apply_dwconv_monkey_patch()
    print("  ✓ Monkey-patches active")
    print()

    # ── 2. Load the trained model ─────────────────────────────────────────
    print("─── Loading model ────────────────────────────────────────────────")
    import torch
    import torch.nn as nn
    from ultralytics import YOLO

    yolo = YOLO(str(args.weights))
    model = yolo.model
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  ✓ Loaded: {n_params:,} params")

    # Sanity check architecture
    layer10_type = type(model.model[10]).__name__
    print(f"  layer 10: {layer10_type}  (expect C2PSA_DPU)")
    if layer10_type != "C2PSA_DPU":
        print(f"  WARN: layer 10 isn't C2PSA_DPU — monkey-patches may not have applied")

    print()

    # ── 3. Swap SiLU → LeakyReLU(0.1015625) ──────────────────────────────
    print("─── Swapping SiLU → LeakyReLU(0.1015625) ─────────────────────────")
    silu_count = swap_silu_to_leakyrelu(model)
    print(f"  ✓ Swapped {silu_count} SiLU → LeakyReLU")
    print()

    # ── 4. Strip Detect head's post-processing ────────────────────────────
    print("─── Stripping Detect head post-processing ───────────────────────")
    detect = model.model[-1]
    print(f"  detect head type: {type(detect).__name__}")
    print(f"  detect.nc        = {detect.nc}")
    print(f"  detect.nl        = {detect.nl}")
    print(f"  detect.reg_max   = {detect.reg_max}")
    strip_detect_postprocessing(detect)
    print(f"  ✓ Detect.forward replaced with raw-conv-output version")
    print()

    # ── 5. Put model in eval mode and run a forward to validate ──────────
    print("─── Validating forward pass with stripped detect ─────────────────")
    model = model.cpu().eval()
    dummy_nchw = torch.randn(1, 3, args.imgsz, args.imgsz)
    with torch.no_grad():
        out = model(dummy_nchw)
    if not isinstance(out, list):
        print(f"  ERROR: expected list output from stripped detect, got {type(out)}")
        sys.exit(2)
    for i, o in enumerate(out):
        print(f"  output[{i}]: shape={tuple(o.shape)} (expect [1, {4*detect.reg_max + detect.nc}, H_{i}, W_{i}])")
    print()

    # ── 6. Wrap with NHWC input permute ──────────────────────────────────
    # DPU consumes NHWC. Our existing NNDCT compile pipeline does this same
    # wrap. The transpose becomes an ONNX op at the start of the graph.
    print("─── Wrapping with NHWC permute ───────────────────────────────────")

    class NHWCModel(nn.Module):
        """Permutes NHWC → NCHW, runs the inner model, returns its outputs."""
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, x_nhwc):
            # x_nhwc: [B, H, W, C]; inner expects NCHW [B, C, H, W]
            x_nchw = x_nhwc.permute(0, 3, 1, 2).contiguous()
            return self.inner(x_nchw)

    nhwc_model = NHWCModel(model).cpu().eval()

    # Validate wrapper
    dummy_nhwc = torch.randn(1, args.imgsz, args.imgsz, 3)
    with torch.no_grad():
        out_w = nhwc_model(dummy_nhwc)
    print(f"  ✓ NHWC wrapper OK, produces {len(out_w)} output tensors")
    print()

    # ── 7. ONNX export ─────────────────────────────────────────────────────
    print("─── Exporting to ONNX ────────────────────────────────────────────")
    print(f"  input shape:  [1, {args.imgsz}, {args.imgsz}, 3] (NHWC)")
    print(f"  output:       {args.output}")
    print(f"  opset:        {args.opset}")

    output_names = [f"out_stride_{2**(3+i)}" for i in range(detect.nl)]  # out_stride_8, _16, _32
    input_names  = ["input_nhwc"]

    torch.onnx.export(
        nhwc_model,
        dummy_nhwc,
        str(args.output),
        opset_version    = args.opset,
        do_constant_folding = True,
        input_names      = input_names,
        output_names     = output_names,
        dynamic_axes     = None,  # static shapes for DPU
        verbose          = False,
    )

    sz_mb = args.output.stat().st_size / 1024 / 1024
    print(f"  ✓ ONNX written: {args.output} ({sz_mb:.1f} MB)")
    print()

    # ── 8. Verify the ONNX is loadable and ops are reasonable ────────────
    print("─── Verifying exported ONNX ──────────────────────────────────────")
    import onnx
    model_onnx = onnx.load(str(args.output))
    onnx.checker.check_model(model_onnx)
    print(f"  ✓ ONNX schema check passed")

    # Op type histogram
    from collections import Counter
    op_types = Counter(node.op_type for node in model_onnx.graph.node)
    print(f"  Op type histogram:")
    for op, count in op_types.most_common():
        print(f"    {op:25s}  {count}")
    print()

    # Input/output info
    print(f"  Inputs:")
    for inp in model_onnx.graph.input:
        shape = [d.dim_value for d in inp.type.tensor_type.shape.dim]
        print(f"    {inp.name}: shape={shape}")
    print(f"  Outputs:")
    for out_proto in model_onnx.graph.output:
        shape = [d.dim_value for d in out_proto.type.tensor_type.shape.dim]
        print(f"    {out_proto.name}: shape={shape}")
    print()

    # ── 9. Cross-validate ONNX vs PyTorch outputs ────────────────────────
    print("─── Cross-validating ONNX vs PyTorch numerics ────────────────────")
    import onnxruntime as ort
    import numpy as np

    sess = ort.InferenceSession(str(args.output), providers=['CPUExecutionProvider'])
    pt_in = torch.randn(1, args.imgsz, args.imgsz, 3)
    with torch.no_grad():
        pt_out = nhwc_model(pt_in)

    ort_out = sess.run(None, {"input_nhwc": pt_in.numpy()})

    for i, (pt_o, ort_o) in enumerate(zip(pt_out, ort_out)):
        diff = np.abs(pt_o.numpy() - ort_o).max()
        rel  = diff / (np.abs(pt_o.numpy()).max() + 1e-8)
        match = "✓" if diff < 1e-4 else ("⚠" if diff < 1e-2 else "✗")
        print(f"  output[{i}]: max abs diff = {diff:.6e}  rel = {rel:.2e}  {match}")
    print()

    print("=" * 70)
    print("  ONNX export complete")
    print("=" * 70)


if __name__ == "__main__":
    main()
