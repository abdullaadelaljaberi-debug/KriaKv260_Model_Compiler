"""YOLOX (Megvii) compile path.

Differences vs YOLOv5:

  - Input quantization convention: BGR uint8 with fix_point=-1 (scale=0.5),
    not RGB float[0,1]
  - Detect head is decoupled (separate cls/obj/reg branches per scale)
  - We disable ``head.decode_in_inference`` so the graph emits raw
    pre-decode outputs (the GraphRunner-side decode in
    ``lpr_pipeline.deploy.decoders`` handles the rest)
  - Multi-DPU-subgraph at deploy time: yolox_tiny → 4 DPU subgraphs,
    yolox_nano → 34 DPU subgraphs (depthwise conv head fragments badly).
    The xmodel itself has no special handling; the runner side does.

The compile path here is structurally similar to YOLOv5 — the differences
are in the head-stripping step and the calibration normalization.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from .base import BaseCompiler, CompileError, CompileInputs


DPU_FINGERPRINT = "0x101000056010407"
ARCH_JSON       = "/opt/vitis_ai/compiler/arch/DPUCZDX8G/KV260/arch.json"


class Compiler(BaseCompiler):
    family = "yolox"

    def _compile_family(self, inputs: CompileInputs) -> Path:
        try:
            import torch
        except ImportError as e:
            raise CompileError(
                "PyTorch not available. Run inside the Vitis-AI 3.5 Docker "
                "container (use scripts/host/02_compile.sh)."
            ) from e

        try:
            import pytorch_nndct  # noqa: F401
        except ImportError as e:
            raise CompileError(
                "vai_q_pytorch not available. Run inside the Vitis-AI 3.5 "
                "Docker container."
            ) from e

        spec = inputs.spec
        print(f"\n══ YOLOX compile: {spec.name} (imgsz={spec.imgsz}, "
              f"nc={inputs.nc}) ══")

        # ── 1. Load checkpoint ──────────────────────────────────────────────
        print(f"\n[1/5] Loading checkpoint: {inputs.weights}")
        ckpt = torch.load(str(inputs.weights), map_location="cpu",
                           weights_only=False)
        # Megvii ckpts can be either {"model": ...} or directly the model
        if isinstance(ckpt, dict) and "model" in ckpt:
            yolox_model = ckpt["model"]
        else:
            yolox_model = ckpt
        yolox_model = yolox_model.float().eval()

        # ── 2. Disable inline decode in the head ────────────────────────────
        print("[2/5] Disabling head.decode_in_inference")
        head = getattr(yolox_model, "head", None)
        if head is None:
            raise CompileError(
                "Loaded checkpoint has no .head attribute. Is this a "
                f"YOLOX model? Got: {type(yolox_model).__name__}"
            )
        if hasattr(head, "decode_in_inference"):
            head.decode_in_inference = False
        else:
            print(f"  ⚠ head has no decode_in_inference attr "
                   f"(type={type(head).__name__}); proceeding anyway")

        wrapped = _InferenceModel(yolox_model, imgsz=spec.imgsz).eval()

        # ── 3. Calibrate ───────────────────────────────────────────────────
        print(f"[3/5] Calibrating with up to {inputs.n_calib} images "
              f"from {inputs.calib_dir}")
        calib_loader = _build_yolox_calib_loader(
            calib_dir=inputs.calib_dir,
            imgsz=spec.imgsz,
            n_calib=inputs.n_calib,
            seed=inputs.seed,
        )

        from pytorch_nndct.apis import torch_quantizer
        quant_dir = inputs.work_dir / "quant"
        quant_dir.mkdir(parents=True, exist_ok=True)
        dummy = torch.zeros(1, 3, spec.imgsz, spec.imgsz)

        # 3a. CALIB
        print("[4/5] Running quant_mode='calib' ...")
        quantizer = torch_quantizer(
            quant_mode="calib",
            module=wrapped,
            input_args=(dummy,),
            output_dir=str(quant_dir),
            device=torch.device("cpu"),
        )
        qmodel = quantizer.quant_model.eval()
        for batch in calib_loader:
            with torch.no_grad():
                qmodel(batch)
        quantizer.export_quant_config()

        # 3b. TEST + DEPLOY
        print("[5/5] Running quant_mode='test' deploy=True ...")
        quantizer = torch_quantizer(
            quant_mode="test",
            module=wrapped,
            input_args=(dummy,),
            output_dir=str(quant_dir),
            device=torch.device("cpu"),
        )
        qmodel = quantizer.quant_model.eval()
        with torch.no_grad():
            qmodel(dummy)
        quantizer.export_xmodel(deploy_check=False, output_dir=str(quant_dir))

        deployable = list(quant_dir.glob("*.xmodel"))
        if not deployable:
            raise CompileError(
                f"vai_q_pytorch did not produce an .xmodel in {quant_dir}"
            )
        deployable = max(deployable, key=lambda p: p.stat().st_mtime)

        # ── 4. Compile ─────────────────────────────────────────────────────
        print("\nCompiling for KV260 B4096 ...")
        compiled = _vai_c_xir(
            xmodel=deployable,
            arch_json=Path(ARCH_JSON),
            output_dir=inputs.work_dir / "compiled",
            net_name=spec.name,
        )

        inputs.out_xmodel.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(compiled, inputs.out_xmodel)
        print(f"\n✓ Final xmodel: {inputs.out_xmodel}")
        print(f"  fingerprint: {DPU_FINGERPRINT} (KV260 B4096, VAI 3.5)")
        return inputs.out_xmodel


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

class _InferenceModel:
    """Thin wrapper — produces NHWC outputs as the DPU expects."""
    def __init__(self, yolox_model, imgsz: int):
        import torch.nn as nn
        class _Wrap(nn.Module):
            def __init__(self, m, imgsz):
                super().__init__()
                self.m = m
                self.imgsz = imgsz
            def forward(self, x):
                # YOLOX expects BGR uint8-as-float (no /255 normalization)
                outs = self.m(x)
                if isinstance(outs, (list, tuple)):
                    return [o.permute(0, 2, 3, 1).contiguous() for o in outs]
                return outs.permute(0, 2, 3, 1).contiguous()
        self._impl = _Wrap(yolox_model, imgsz)

    def __getattr__(self, name):
        return getattr(self._impl, name)
    def __call__(self, *a, **kw):
        return self._impl(*a, **kw)
    def eval(self):
        return self._impl.eval()


def _build_yolox_calib_loader(calib_dir: Path, imgsz: int, n_calib: int, seed: int):
    """YOLOX calibration: BGR uint8 (cast to float, NO /255 normalization)."""
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
            # PIL gives RGB, YOLOX wants BGR
            arr = np.asarray(img)[:, :, ::-1].copy()
            h, w = arr.shape[:2]
            r = min(imgsz / w, imgsz / h)
            new_w, new_h = int(round(w * r)), int(round(h * r))
            resized = np.asarray(
                Image.fromarray(arr[:, :, ::-1]).resize(
                    (new_w, new_h), Image.BILINEAR
                )
            )[:, :, ::-1].copy()
            canvas = np.full((imgsz, imgsz, 3), 114, dtype=np.uint8)
            pad_x = (imgsz - new_w) // 2
            pad_y = (imgsz - new_h) // 2
            canvas[pad_y:pad_y+new_h, pad_x:pad_x+new_w] = resized
            # NCHW float (no /255 — YOLOX expects values in [0, 255])
            t = torch.from_numpy(canvas).permute(2, 0, 1).float()
            return t

    return DataLoader(_CalibSet(), batch_size=1, shuffle=False, num_workers=0)


def _vai_c_xir(xmodel: Path, arch_json: Path, output_dir: Path,
                net_name: str) -> Path:
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
            f"stdout: {r.stdout}\nstderr: {r.stderr}"
        )
    out = output_dir / f"{net_name}.xmodel"
    if not out.is_file():
        candidates = list(output_dir.glob("*.xmodel"))
        if not candidates:
            raise CompileError(f"vai_c_xir produced no xmodel in {output_dir}")
        out = max(candidates, key=lambda p: p.stat().st_mtime)
    return out
