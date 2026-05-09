"""YOLOv5 (Ultralytics u-variant) compile path.

Full pipeline:

    1. Load the trained .pt with ``torch.load(..., weights_only=False)``
       — the weight file contains the architecture as well, not just tensors
    2. Strip the inline detect-head NMS / decode so the DPU sees clean
       Conv outputs (3 tensors, one per stride: 8, 16, 32)
    3. Wrap the model in a thin ``InferenceModel`` that exposes the raw
       multi-scale outputs in NHWC layout (vai_q_pytorch's preferred form)
    4. Build a calibration DataLoader from ``inputs.calib_dir``
    5. Run vai_q_pytorch quant_mode='calib' → exports a fake-quant model
    6. Run vai_q_pytorch quant_mode='test' with deploy=True → exports
       the deployable .xmodel
    7. ``vai_c_xir`` compiles for B4096 fingerprint 0x101000056010407
       → final yolov5n_kv260.xmodel

Steps 5-7 require the Vitis-AI 3.5 PyTorch GPU docker image's environment
to be active (specifically: ``vai_q_pytorch`` and ``vai_c_xir`` on PATH).
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

from .base import BaseCompiler, CompileError, CompileInputs


# DPU fingerprint for KV260 B4096 in VAI 3.5. xmodels compiled with this
# fingerprint will fail to load on VAI 2.5 boards.
DPU_FINGERPRINT = "0x101000056010407"
# Architecture file that vai_c_xir uses to match the fingerprint. Bundled
# with the Vitis-AI image at this path.
ARCH_JSON       = "/opt/vitis_ai/compiler/arch/DPUCZDX8G/KV260/arch.json"


class Compiler(BaseCompiler):
    family = "yolov5"

    def _compile_family(self, inputs: CompileInputs) -> Path:
        # Defer heavy imports — only do them when actually compiling.
        # This lets the stub stub() error path import the registry without
        # pulling in torch.
        try:
            import torch
        except ImportError as e:
            raise CompileError(
                "PyTorch not available. The compile pipeline must run inside "
                "the Vitis-AI 3.5 Docker container. From the host, run:\n"
                "  bash scripts/host/02_compile.sh ..."
            ) from e

        try:
            import pytorch_nndct  # noqa: F401  — vai_q_pytorch
        except ImportError as e:
            raise CompileError(
                "vai_q_pytorch not available. You're not inside the Vitis-AI "
                "3.5 Docker container. From the host, run:\n"
                "  bash scripts/host/02_compile.sh ..."
            ) from e

        spec = inputs.spec
        print(f"\n══ YOLOv5 compile: {spec.name} (imgsz={spec.imgsz}, "
              f"nc={inputs.nc}, reg_max={spec.reg_max}) ══")

        # ── 1. Load checkpoint ──────────────────────────────────────────────
        ckpt_path = inputs.weights
        print(f"\n[1/5] Loading checkpoint: {ckpt_path}")
        # weights_only=False because Ultralytics ckpts ship the model object
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict) and "model" in ckpt:
            yolo_model = ckpt["model"]
        else:
            yolo_model = ckpt
        yolo_model = yolo_model.float().eval()

        # ── 2. Strip the detect head's inline post-processing ──────────────
        print("[2/5] Stripping detect head's inline NMS / decode")
        _strip_detect_head_for_quant(yolo_model)

        # ── 3. Wrap so DPU sees raw multi-scale conv outputs ───────────────
        wrapped = _InferenceModel(yolo_model, imgsz=spec.imgsz).eval()

        # ── 4. Calibrate ───────────────────────────────────────────────────
        print(f"[3/5] Calibrating with up to {inputs.n_calib} images "
              f"from {inputs.calib_dir}")
        calib_loader = _build_calib_loader(
            calib_dir=inputs.calib_dir,
            imgsz=spec.imgsz,
            n_calib=inputs.n_calib,
            seed=inputs.seed,
        )

        from pytorch_nndct.apis import torch_quantizer

        quant_dir = inputs.work_dir / "quant"
        quant_dir.mkdir(parents=True, exist_ok=True)

        dummy = torch.randn(1, 3, spec.imgsz, spec.imgsz)

        # 4a. CALIB pass — accumulates activation stats
        print("[4/5] Running quant_mode='calib' ...")
        quantizer = torch_quantizer(
            quant_mode="calib",
            module=wrapped,
            input_args=(dummy,),
            output_dir=str(quant_dir),
            device=torch.device("cpu"),
        )
        qmodel = quantizer.quant_model
        qmodel.eval()
        for batch in calib_loader:
            with torch.no_grad():
                qmodel(batch)
        quantizer.export_quant_config()

        # 4b. TEST pass with deploy=True — emits .xmodel
        print("[5/5] Running quant_mode='test' deploy=True ...")
        quantizer = torch_quantizer(
            quant_mode="test",
            module=wrapped,
            input_args=(dummy,),
            output_dir=str(quant_dir),
            device=torch.device("cpu"),
        )
        qmodel = quantizer.quant_model
        qmodel.eval()
        # deploy=True needs at least one forward to emit the xmodel
        with torch.no_grad():
            qmodel(dummy)
        quantizer.export_xmodel(deploy_check=False, output_dir=str(quant_dir))

        deployable = list(quant_dir.glob("*.xmodel"))
        if not deployable:
            raise CompileError(
                f"vai_q_pytorch did not produce an .xmodel in {quant_dir}. "
                "Inspect that directory for clues."
            )
        if len(deployable) > 1:
            print(f"  ⚠ Multiple .xmodel found, picking the most recent: "
                  f"{[p.name for p in deployable]}")
        deployable = max(deployable, key=lambda p: p.stat().st_mtime)
        print(f"  → quantized: {deployable}")

        # ── 5. Compile to KV260 B4096 ──────────────────────────────────────
        print("\nCompiling for KV260 B4096 ...")
        compiled = _vai_c_xir(
            xmodel=deployable,
            arch_json=Path(ARCH_JSON),
            output_dir=inputs.work_dir / "compiled",
            net_name=spec.name,
        )

        # ── 6. Move to the destination ─────────────────────────────────────
        inputs.out_xmodel.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(compiled, inputs.out_xmodel)
        print(f"\n✓ Final xmodel: {inputs.out_xmodel}")
        print(f"  fingerprint: {DPU_FINGERPRINT} (KV260 B4096, VAI 3.5)")
        return inputs.out_xmodel


# ─────────────────────────────────────────────────────────────────────────────
# Helpers — kept module-private so they don't leak into the public API
# ─────────────────────────────────────────────────────────────────────────────

def _strip_detect_head_for_quant(yolo_model) -> None:
    """In-place: zero out the head's inline post-processing.

    The Ultralytics detect head normally applies sigmoid + grid + anchor
    decoding before returning. The DPU can't run those ops efficiently;
    we strip them so the head returns raw conv outputs and we decode on CPU.
    """
    # Find the detect head — last module child by convention
    detect = None
    for m in yolo_model.modules():
        cls_name = type(m).__name__
        if cls_name in ("Detect", "DetectAux", "v8Detect", "v6Detect"):
            detect = m
            break
    if detect is None:
        raise CompileError(
            "Could not find a Detect head in the loaded checkpoint. "
            "Is this an Ultralytics YOLOv5 .pt? Architecture preview:\n"
            f"  {type(yolo_model).__name__} children: "
            f"{[type(c).__name__ for c in yolo_model.children()]}"
        )

    # Toggle flags Ultralytics checks at forward time
    detect.training = True             # bypasses the eval-time decode branch
    detect.export   = True             # tells head to return raw outputs
    if hasattr(detect, "inplace"):
        detect.inplace = False
    if hasattr(detect, "dynamic"):
        detect.dynamic = False
    print(f"  ✓ stripped {type(detect).__name__} head")


class _InferenceModel:
    """Thin wrapper that calls the YOLOv5 forward and reorders outputs.

    NCHW → NHWC because vai_q_pytorch's KV260 backend prefers NHWC.
    """
    def __init__(self, yolo_model, imgsz: int):
        import torch.nn as nn
        # We need to subclass nn.Module so vai_q_pytorch sees us as a module.
        # Building it dynamically here avoids requiring torch at module import.
        class _Wrap(nn.Module):
            def __init__(self, m, imgsz):
                super().__init__()
                self.m = m
                self.imgsz = imgsz

            def forward(self, x):
                outs = self.m(x)
                # Ultralytics returns a list of NCHW tensors when training=True
                if isinstance(outs, (list, tuple)):
                    return [o.permute(0, 2, 3, 1).contiguous() for o in outs]
                return outs.permute(0, 2, 3, 1).contiguous()

        self._impl = _Wrap(yolo_model, imgsz)
        self.imgsz = imgsz

    def __getattr__(self, name):
        return getattr(self._impl, name)

    def __call__(self, *a, **kw):
        return self._impl(*a, **kw)

    def eval(self):
        return self._impl.eval()


def _build_calib_loader(calib_dir: Path, imgsz: int, n_calib: int, seed: int):
    """A simple DataLoader yielding pre-letterboxed tensors for calibration."""
    import random

    import numpy as np
    import torch
    from torch.utils.data import DataLoader, Dataset
    from PIL import Image

    image_paths = sorted(
        p for p in calib_dir.iterdir()
        if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")
    )
    if len(image_paths) > n_calib:
        rng = random.Random(seed)
        image_paths = rng.sample(image_paths, n_calib)
    print(f"  → using {len(image_paths)} calibration images")

    class _CalibSet(Dataset):
        def __len__(self): return len(image_paths)
        def __getitem__(self, idx):
            img = Image.open(image_paths[idx]).convert("RGB")
            # Letterbox-resize to imgsz × imgsz, pad with 114 grey
            w, h = img.size
            r = min(imgsz / w, imgsz / h)
            new_w, new_h = int(round(w * r)), int(round(h * r))
            img = img.resize((new_w, new_h), Image.BILINEAR)
            arr = np.full((imgsz, imgsz, 3), 114, dtype=np.uint8)
            pad_x = (imgsz - new_w) // 2
            pad_y = (imgsz - new_h) // 2
            arr[pad_y:pad_y+new_h, pad_x:pad_x+new_w] = np.asarray(img)
            # NCHW float32 [0, 1]
            t = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
            return t

    return DataLoader(_CalibSet(), batch_size=1, shuffle=False, num_workers=0)


def _vai_c_xir(xmodel: Path, arch_json: Path, output_dir: Path,
                net_name: str) -> Path:
    """Run vai_c_xir to compile the quantized xmodel for KV260."""
    if not arch_json.is_file():
        raise CompileError(
            f"VAI 3.5 architecture file not found at {arch_json}. "
            f"Are you running inside the Vitis-AI 3.5 Docker container?"
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "vai_c_xir",
        "--xmodel",     str(xmodel),
        "--arch",       str(arch_json),
        "--output_dir", str(output_dir),
        "--net_name",   net_name,
    ]
    print(f"  $ {' '.join(cmd)}")
    r = subprocess.run(cmd, capture_output=True, text=True)
    print(r.stdout)
    if r.returncode != 0:
        raise CompileError(
            f"vai_c_xir failed (exit {r.returncode})\n"
            f"stdout: {r.stdout}\n"
            f"stderr: {r.stderr}"
        )

    out = output_dir / f"{net_name}.xmodel"
    if not out.is_file():
        # vai_c_xir sometimes uses a slightly different filename
        candidates = list(output_dir.glob("*.xmodel"))
        if not candidates:
            raise CompileError(
                f"vai_c_xir succeeded but produced no .xmodel in {output_dir}"
            )
        out = max(candidates, key=lambda p: p.stat().st_mtime)
        print(f"  ⚠ Expected {net_name}.xmodel; found {out.name}")
    return out
