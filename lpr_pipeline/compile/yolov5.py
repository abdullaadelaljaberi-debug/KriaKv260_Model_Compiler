"""YOLOv5 (Ultralytics u-variant) compile path.

Full pipeline:

    1. Load the trained .pt with ``torch.load(..., weights_only=False)``
    2. **Auto-swap SiLU → LeakyReLU(0.1015625)** for DPU compatibility.
       The KV260's DPUCZDX8G has no hardware SiLU; each SiLU op forces
       a CPU subgraph, fragmenting the compiled xmodel into many tiny
       pieces. The slope 0.1015625 (= 13/128) is the DPU's only supported
       value; using it directly avoids the quantizer auto-correction.
    3. Replace the Detect head's forward with a "raw outputs only" version.
       The DPU can't run sigmoid + grid + DFL decode efficiently, so we
       strip the post-processing and do it on CPU after DPU inference.
    4. Wrap the model in a thin ``_InferenceModel`` that exposes the raw
       multi-scale outputs in NHWC layout (vai_q_pytorch's preferred form)
    5. Build a calibration DataLoader from ``inputs.calib_dir``
    6. Run vai_q_pytorch quant_mode='calib' → exports a fake-quant model
    7. Run vai_q_pytorch quant_mode='test' with deploy=True → exports
       the deployable .xmodel
    8. ``vai_c_xir`` compiles for B4096 fingerprint 0x101000056010407

Steps 6-8 require the Vitis-AI 3.5 PyTorch Docker image's environment.

The activation swap is unconditional: it walks the model and swaps any
SiLU it finds. If the input is already LeakyReLU-only (e.g. ``leakyrelu.pt``
from a pre-swap pipeline), the swap is a no-op. If the input is the raw
trained ``best.pt`` (SiLU intact), the pipeline does the swap automatically.

This swap mirrors the function in the user's training notebook 02:
``swap_silu_to_leakyrelu``.
"""
from __future__ import annotations

import shutil
import subprocess
import types
from pathlib import Path

from .base import BaseCompiler, CompileError, CompileInputs


# DPU fingerprint for KV260 B4096 in VAI 3.5
DPU_FINGERPRINT = "0x101000056010407"
ARCH_JSON       = "/opt/vitis_ai/compiler/arch/DPUCZDX8G/KV260/arch.json"

# DPU's only supported negative_slope for LeakyReLU. = 13/128.
# The quantizer auto-corrects other values; using this directly avoids
# the train-vs-deploy numerical drift that the auto-correction causes.
DPU_LEAKY_SLOPE = 0.1015625


class Compiler(BaseCompiler):
    family = "yolov5"

    def _compile_family(self, inputs: CompileInputs) -> Path:
        try:
            import torch
        except ImportError as e:
            raise CompileError(
                "PyTorch not available. The compile pipeline must run inside "
                "the Vitis-AI 3.5 Docker container. From the host, run:\n"
                "  bash scripts/host/02_compile.sh ..."
            ) from e

        try:
            import pytorch_nndct  # noqa: F401
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
        print(f"\n[1/6] Loading checkpoint: {ckpt_path}")
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict) and "model" in ckpt:
            yolo_model = ckpt["model"]
        else:
            yolo_model = ckpt
        yolo_model = yolo_model.float().eval()

        # ── 2. Auto-swap SiLU → LeakyReLU for DPU compatibility ────────────
        if inputs.swap_activations:
            print(f"[2/6] Auto-swap SiLU → LeakyReLU({DPU_LEAKY_SLOPE}) for DPU compatibility")
            n_swapped = _swap_silu_to_leakyrelu(yolo_model, slope=DPU_LEAKY_SLOPE)
            if n_swapped > 0:
                print(f"  ✓ swapped {n_swapped} SiLU instances")
                print(f"    (without this, the model would compile but fragment into many DPU+CPU subgraphs,")
                print(f"     and pynq_dpu's overlay.load_model() would fail with assert len(subgraphs)==1)")
            else:
                print(f"  ✓ no SiLU found — model is already DPU-compatible")
        else:
            print(f"[2/6] SKIPPING activation swap (swap_activations=False)")
            print(f"  ⚠ If the model contains SiLU/GELU/Mish, the xmodel will fragment.")
            print(f"    pynq_dpu.overlay.load_model() will fail; use vitis_ai_library.GraphRunner.")
            print(f"    See docs/MODELS.md → 'Activation function policy' for trade-offs.")
            # Quick check + warning so the user knows what they got
            _warn_if_non_dpu_activations(yolo_model)

        # ── 3. Strip the detect head's inline post-processing ──────────────
        print("[3/6] Stripping detect head's inline NMS / decode")
        _strip_detect_head_for_quant(yolo_model)

        # ── 4. Wrap so DPU sees raw multi-scale conv outputs ───────────────
        wrapped = _make_inference_model(yolo_model)

        # ── 5. Calibrate ───────────────────────────────────────────────────
        print(f"[4/6] Calibrating with up to {inputs.n_calib} images "
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

        # 5a. CALIB pass — accumulates activation stats
        print("[5/6] Running quant_mode='calib' ...")
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

        # 5b. TEST pass with deploy=True — emits .xmodel
        print("[6/6] Running quant_mode='test' deploy=True ...")
        quantizer = torch_quantizer(
            quant_mode="test",
            module=wrapped,
            input_args=(dummy,),
            output_dir=str(quant_dir),
            device=torch.device("cpu"),
        )
        qmodel = quantizer.quant_model
        qmodel.eval()
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

        # ── 6. Compile to KV260 B4096 ──────────────────────────────────────
        print("\nCompiling for KV260 B4096 ...")
        compiled = _vai_c_xir(
            xmodel=deployable,
            arch_json=Path(ARCH_JSON),
            output_dir=inputs.work_dir / "compiled",
            net_name=spec.name,
        )

        # ── 7. Move to the destination ─────────────────────────────────────
        inputs.out_xmodel.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(compiled, inputs.out_xmodel)
        print(f"\n[OK] Final xmodel: {inputs.out_xmodel}")
        print(f"     fingerprint: {DPU_FINGERPRINT} (KV260 B4096, VAI 3.5)")
        return inputs.out_xmodel


# ─────────────────────────────────────────────────────────────────────────────
# Helpers — kept module-private
# ─────────────────────────────────────────────────────────────────────────────

def _warn_if_non_dpu_activations(yolo_model) -> None:
    """When --no-swap is used, audit the model and warn about non-DPU activations.

    The DPU only accelerates ReLU/ReLU6/LeakyReLU(0.1015625). Anything else
    falls back to CPU, fragmenting the graph.
    """
    import torch.nn as nn

    bad_types = (nn.SiLU, nn.GELU, nn.Mish)
    counts: dict[str, int] = {}

    for m in yolo_model.modules():
        for cls in bad_types:
            if isinstance(m, cls):
                counts[cls.__name__] = counts.get(cls.__name__, 0) + 1
        if hasattr(m, "act"):
            for cls in bad_types:
                if isinstance(m.act, cls):
                    name = f"{cls.__name__}(.act)"
                    counts[name] = counts.get(name, 0) + 1

    if counts:
        summary = ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
        print(f"  ⚠ Found non-DPU activations: {summary}")
        print(f"    Each will become a CPU subgraph at deploy time.")
    else:
        print(f"  ✓ No problematic activations detected — graph should stay clean.")


def _swap_silu_to_leakyrelu(yolo_model, slope: float = DPU_LEAKY_SLOPE) -> int:
    """Replace every SiLU with LeakyReLU(slope) in-place. Returns swap count.

    The KV260 DPU has no hardware SiLU op; each one becomes a CPU subgraph
    that fragments the compiled xmodel. LeakyReLU IS DPU-supported, but
    only at slope=0.1015625 (=13/128). Using that exact value here matches
    what the DPU runs, removing the train-vs-deploy mismatch the quantizer
    would otherwise introduce by auto-correcting from 0.1.

    Two-pass approach mirrors the user's training notebook 02:
      Pass 1: scan named children; replace any SiLU directly
      Pass 2: defensive — catch .act attributes that aren't in named_children
              (Ultralytics's Conv assigns .act in __init__; PyTorch normally
              registers Module assignments as children, but this pass handles
              edge cases like assignments via setattr_).
    """
    import torch.nn as nn

    n_swapped = 0

    def _recurse(module):
        nonlocal n_swapped
        # Pass 1: named children
        for name, child in list(module.named_children()):
            if isinstance(child, nn.SiLU):
                setattr(module, name, nn.LeakyReLU(slope, inplace=True))
                n_swapped += 1
            else:
                _recurse(child)
        # Pass 2: defensive — module.act is sometimes assigned outside named_children
        if hasattr(module, "act") and isinstance(module.act, nn.SiLU):
            module.act = nn.LeakyReLU(slope, inplace=True)
            n_swapped += 1

    _recurse(yolo_model)
    return n_swapped


def _strip_detect_head_for_quant(yolo_model) -> None:
    """In-place: replace the Detect head's forward with a raw-outputs version.

    See module docstring. Using ``types.MethodType`` to rebind ``forward``
    is more robust than toggling ``self.training``/``self.export`` flags
    because ``.eval()`` resets those flags.
    """
    import torch

    detect = None
    for m in yolo_model.modules():
        cls_name = type(m).__name__
        if cls_name in ("Detect", "DetectAux", "v8Detect", "v6Detect"):
            detect = m
            break
    if detect is None:
        raise CompileError(
            "Could not find a Detect head in the loaded checkpoint. "
            "Architecture preview:\n"
            f"  {type(yolo_model).__name__} children: "
            f"{[type(c).__name__ for c in yolo_model.children()]}"
        )

    cls_name = type(detect).__name__

    if hasattr(detect, "cv2") and hasattr(detect, "cv3"):
        def _stripped_forward(self, x):
            outputs = []
            for i in range(self.nl):
                outputs.append(torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1))
            return outputs
        detect.forward = types.MethodType(_stripped_forward, detect)
        print(f"  ✓ stripped {cls_name} head (modern Ultralytics: cv2+cv3)")

    elif hasattr(detect, "m"):
        def _stripped_forward(self, x):
            outputs = []
            for i in range(self.nl):
                outputs.append(self.m[i](x[i]))
            return outputs
        detect.forward = types.MethodType(_stripped_forward, detect)
        print(f"  ✓ stripped {cls_name} head (legacy YOLOv5: .m)")

    else:
        attrs = [a for a in dir(detect) if not a.startswith("_")][:30]
        raise CompileError(
            f"Don't know how to strip {cls_name} head — neither cv2/cv3 "
            f"(modern Ultralytics) nor .m (legacy YOLOv5) attributes found.\n"
            f"Public attrs: {attrs}"
        )


def _make_inference_model(yolo_model):
    """Build a thin nn.Module wrapper that produces NHWC outputs."""
    import torch
    import torch.nn as nn

    class _Wrap(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x):
            outs = self.m(x)
            if isinstance(outs, (list, tuple)):
                return [o.permute(0, 2, 3, 1).contiguous() for o in outs]
            return outs.permute(0, 2, 3, 1).contiguous()

    return _Wrap(yolo_model).eval()


def _build_calib_loader(calib_dir: Path, imgsz: int, n_calib: int, seed: int):
    """Simple DataLoader yielding pre-letterboxed tensors for calibration."""
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
            w, h = img.size
            r = min(imgsz / w, imgsz / h)
            new_w, new_h = int(round(w * r)), int(round(h * r))
            img = img.resize((new_w, new_h), Image.BILINEAR)
            arr = np.full((imgsz, imgsz, 3), 114, dtype=np.uint8)
            pad_x = (imgsz - new_w) // 2
            pad_y = (imgsz - new_h) // 2
            arr[pad_y:pad_y+new_h, pad_x:pad_x+new_w] = np.asarray(img)
            arr = arr.copy()
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
        candidates = list(output_dir.glob("*.xmodel"))
        if not candidates:
            raise CompileError(
                f"vai_c_xir succeeded but produced no .xmodel in {output_dir}"
            )
        out = max(candidates, key=lambda p: p.stat().st_mtime)
        print(f"  ⚠ Expected {net_name}.xmodel; found {out.name}")
    return out
