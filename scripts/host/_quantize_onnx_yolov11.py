"""ONNX PTQ via vai_q_onnx, then compile to xmodel via vai_c_xir.

Runs INSIDE the vitis-ai-onnx-cpu:eggs container.

Takes the exported ONNX from phase 2, runs PTQ with per-channel weight
quantization (the key improvement over NNDCT's default per-tensor),
then compiles the quantized model to a DPU xmodel.

Usage:
    python /workspace/scripts/host/_quantize_onnx_yolov11.py \\
        --input    /workspace/build/yolov11n_onnx/yolo11n_eggs_dpu.onnx \\
        --calib    /workspace/data/calib_v2_hardneg \\
        --output   /workspace/out/yolov11n_onnx/yolov11n_eggs_kv260.xmodel \\
        --imgsz    640 \\
        --n-calib  200 \\
        --arch     /opt/vitis_ai/compiler/arch/DPUCZDX8G/KV260/arch.json
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input",  type=Path, required=True,
                   help="Exported FP32 ONNX from phase 2")
    p.add_argument("--calib",  type=Path, required=True,
                   help="Directory of calibration JPGs")
    p.add_argument("--output", type=Path, required=True,
                   help="Final .xmodel path (e.g. .../yolov11n_eggs_kv260.xmodel)")
    p.add_argument("--imgsz",  type=int,  default=640)
    p.add_argument("--n-calib", type=int, default=200,
                   help="Number of calibration images to use (default 200)")
    p.add_argument("--arch", type=str,
                   default="/opt/vitis_ai/compiler/arch/DPUCZDX8G/KV260/arch.json",
                   help="DPU arch.json")
    p.add_argument("--net-name", default="yolov11n_onnx",
                   help="Compiled network name")
    p.add_argument("--per-channel", action="store_true", default=True,
                   help="Per-channel weight quantization (default True — the key win)")
    p.add_argument("--no-per-channel", action="store_false", dest="per_channel",
                   help="Disable per-channel (for ablation)")
    return p.parse_args()


class CalibDataReader:
    """Yields preprocessed calibration batches for vai_q_onnx.

    Each get_next() returns {input_name: NHWC_float32_array} or None when done.
    """
    def __init__(self, image_paths, input_name, imgsz=640, family="yolov11"):
        import sys
        sys.path.insert(0, '/workspace')
        from lpr_pipeline.deploy.preprocess import Preprocessor

        self.preprocessor = Preprocessor(family=family, imgsz=imgsz)
        self.image_paths = list(image_paths)
        self.input_name  = input_name
        self.imgsz       = imgsz
        self._iter       = iter(self.image_paths)
        self._count      = 0
        self._total      = len(self.image_paths)

    def get_next(self):
        try:
            img_path = next(self._iter)
        except StopIteration:
            return None

        self._count += 1
        if self._count % 25 == 0 or self._count == 1:
            print(f"    [calib] {self._count}/{self._total}: {img_path.name}")

        import cv2
        import numpy as np
        frame = cv2.imread(str(img_path))
        if frame is None:
            print(f"    [calib] WARNING: failed to read {img_path}, skipping")
            return self.get_next()

        # Run the same preprocessing the deployed pipeline uses
        # Preprocessor.process() returns NHWC float32 [1, H, W, 3] in [0, 1]
        result = self.preprocessor.process(frame); nhwc = (result[0] if isinstance(result, tuple) else result).copy()
        return {self.input_name: nhwc.astype(np.float32)}

    def rewind(self):
        self._iter = iter(self.image_paths)
        self._count = 0


def main():
    args = parse_args()

    print()
    print("=" * 70)
    print("  ONNX PTQ → xmodel via vai_q_onnx + vai_c_xir")
    print("=" * 70)
    print(f"  Input ONNX:   {args.input}")
    print(f"  Calib dir:    {args.calib}")
    print(f"  N calib:      {args.n_calib}")
    print(f"  Output:       {args.output}")
    print(f"  imgsz:        {args.imgsz}")
    print(f"  per_channel:  {args.per_channel}")
    print()

    if not args.input.is_file():
        print(f"ERROR: input ONNX not found: {args.input}", file=sys.stderr)
        sys.exit(1)
    if not args.calib.is_dir():
        print(f"ERROR: calib dir not found: {args.calib}", file=sys.stderr)
        sys.exit(1)
    if not Path(args.arch).is_file():
        print(f"ERROR: arch.json not found: {args.arch}", file=sys.stderr)
        sys.exit(1)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Working dir for intermediate files
    workdir = args.output.parent / f".{args.output.stem}_work"
    workdir.mkdir(parents=True, exist_ok=True)
    quantized_onnx = workdir / "model_int8.onnx"

    # ── 1. Identify input tensor name ─────────────────────────────────────
    print("─── Inspecting input ONNX ────────────────────────────────────────")
    import onnx
    onnx_model = onnx.load(str(args.input))
    input_name = onnx_model.graph.input[0].name
    print(f"  Input tensor name: {input_name}")
    print()

    # ── 2. Build calibration data reader ─────────────────────────────────
    print("─── Building calibration data reader ─────────────────────────────")
    all_calib = sorted(args.calib.glob("*.jpg"))
    print(f"  Found {len(all_calib)} .jpg images in calib dir")

    # Split calib pool into eggs and hardneg, then balance
    eggs    = [p for p in all_calib if not p.name.startswith("hardneg_")]
    hardneg = [p for p in all_calib if     p.name.startswith("hardneg_")]
    print(f"  Pool: {len(eggs)} eggs, {len(hardneg)} hardneg")

    n_each = args.n_calib // 2
    # Deterministic interleaved selection (first n_each of each, then interleave)
    import random
    rng = random.Random(42)
    eggs_pick    = rng.sample(eggs,    min(n_each, len(eggs)))
    hardneg_pick = rng.sample(hardneg, min(n_each, len(hardneg)))

    # Interleave so the calibration doesn\'t see all eggs then all hardneg
    calib_images = []
    for i in range(max(len(eggs_pick), len(hardneg_pick))):
        if i < len(eggs_pick):
            calib_images.append(eggs_pick[i])
        if i < len(hardneg_pick):
            calib_images.append(hardneg_pick[i])

    n_use = len(calib_images)
    print(f"  Using {n_use} balanced ({len(eggs_pick)} eggs + {len(hardneg_pick)} hardneg, interleaved)")
    print()

    reader = CalibDataReader(calib_images, input_name, imgsz=args.imgsz)

    # ── 3. Run vai_q_onnx PTQ ─────────────────────────────────────────────
    print("─── Running vai_q_onnx.quantize_static() ─────────────────────────")
    import vai_q_onnx
    from vai_q_onnx import (
        QuantType,
        VitisQuantFormat,
        PowerOfTwoMethod,
    )

    quant_args = dict(
        model_input             = str(args.input),
        model_output            = str(quantized_onnx),
        calibration_data_reader = reader,
        quant_format            = VitisQuantFormat.QDQ,    # DPU-compatible
        calibrate_method        = PowerOfTwoMethod.MinMSE,        # DPU-friendly
        activation_type         = QuantType.QInt8,
        weight_type             = QuantType.QInt8,
        per_channel             = args.per_channel,               # ← key setting
        use_dpu                 = True,                            # DPU target
        execution_providers     = ['CPUExecutionProvider'],
        optimize_model          = True,
    )
    print(f"  Config:")
    for k, v in quant_args.items():
        if k not in ('model_input', 'model_output', 'calibration_data_reader'):
            print(f"    {k:24s} = {v}")
    print()
    print(f"  Starting calibration (this will take 1-5 min depending on n_calib)...")
    print()

    vai_q_onnx.quantize_static(**quant_args)

    if not quantized_onnx.exists():
        print(f"ERROR: quantize_static did not produce {quantized_onnx}", file=sys.stderr)
        sys.exit(2)
    sz_mb = quantized_onnx.stat().st_size / 1024 / 1024
    print(f"  ✓ Quantized ONNX: {quantized_onnx} ({sz_mb:.1f} MB)")
    print()

    # ── 4. Compile quantized ONNX → xmodel via vai_c_xir ─────────────────
    print("─── Compiling to xmodel via vai_c_xir ────────────────────────────")
    compile_outdir = workdir / "compiled"
    compile_outdir.mkdir(exist_ok=True)

    cmd = [
        "vai_c_xir",
        "--xmodel",     str(quantized_onnx),
        "--arch",       args.arch,
        "--output_dir", str(compile_outdir),
        "--net_name",   args.net_name,
    ]
    print(f"  $ {' '.join(cmd)}")
    print()

    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print("STDERR:")
        print(result.stderr)
        print(f"ERROR: vai_c_xir failed (exit {result.returncode})", file=sys.stderr)
        sys.exit(3)

    # Find the produced xmodel
    produced = list(compile_outdir.glob("*.xmodel"))
    if not produced:
        print(f"ERROR: no xmodel produced in {compile_outdir}", file=sys.stderr)
        sys.exit(4)
    src_xmodel = produced[0]
    print(f"  ✓ Compiled: {src_xmodel}")

    # ── 5. Copy to final output path ─────────────────────────────────────
    import shutil
    shutil.copy2(src_xmodel, args.output)
    sz_mb = args.output.stat().st_size / 1024 / 1024
    print(f"  ✓ Final xmodel: {args.output} ({sz_mb:.1f} MB)")
    print()

    print("=" * 70)
    print("  ONNX PTQ + compile complete")
    print("=" * 70)
    print()
    print(f"  Next: sync to Kria and re-measure FPs")
    print(f"    scp {args.output} ubuntu@10.42.0.189:/home/ubuntu/xmodels_vai35/yolov11n/")


if __name__ == "__main__":
    main()
