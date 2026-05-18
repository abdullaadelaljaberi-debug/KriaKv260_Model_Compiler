"""ModelRunner — load an xmodel + do inference + report per-stage timing.

Wraps the DPU overlay, preprocessor, and decoder behind a single `infer()`
call. Inference returns both detections and a per-stage timing dict so the
caller can build a meaningful performance breakdown.

The DPU itself is a singleton: `pynq_dpu.DpuOverlay("dpu.bit")` programs the
FPGA fabric with the DPU bitstream. After the bitstream is loaded, you can
swap xmodels onto it via `overlay.load_model(path)`. We don't recreate the
overlay across model swaps because reprogramming the fabric is ~3-5 seconds.

Currently supports two families via single-DPU-subgraph runner:

  * yolov5  (yolov5n, yolov5s for LPR)
  * yolov11 (yolov11n; DPU-friendly architecture via training-time monkey-
             patches in scripts/host/_train_yolov11.py)

Both families share the same DFL decoder; the family field in the spec
just controls which decode_* alias is dispatched.

YOLOX is not yet wired here — it needs `vitis_ai_library.GraphRunner` for
multi-subgraph models, a separate code path not yet implemented.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from lpr_pipeline.deploy.decoders   import (
    Detection,
    decode_yolov5u,
    decode_yolov11,
)
from lpr_pipeline.deploy.preprocess import Preprocessor, unletterbox
from lpr_pipeline.shared.models     import ModelSpec


# DPU fingerprint our pipeline targets. Set by `vai_c_xir --arch ...KV260/arch.json`
# at compile time. If a downloaded xmodel has a different fingerprint, the
# DPU runtime will reject it at load time with a clear error.
EXPECTED_FINGERPRINT = "0x101000056010407"


# Families supported by this runner. Both produce the same DFL output format
# at the Detect head and use the same single-DPU-subgraph runtime. They differ
# only in backbone architecture (which is internal to the xmodel and doesn't
# affect runner-level dispatch).
_SUPPORTED_FAMILIES = ("yolov5", "yolov11")


# Family → decoder dispatch. Both decoders implement the same math
# (decode_yolov11 is an alias) but the explicit dispatch makes the intent
# clear and leaves room for future per-family specialization (e.g. if
# YOLOv11 multi-class deployments need class-aware NMS).
_DECODERS = {
    "yolov5":  decode_yolov5u,
    "yolov11": decode_yolov11,
}


class ModelRunner:
    """One-shot wrapper around xmodel loading + inference.

    Parameters
    ----------
    spec : ModelSpec
        From `lpr_pipeline.shared.models.get_spec(variant)`. Provides imgsz,
        family, nc, reg_max.
    xmodel_path : str or Path
        Path to the compiled `.xmodel` on the Kria. Typically
        `/home/ubuntu/xmodels_vai35/<variant>/<variant>_kv260.xmodel`.
    overlay : pynq_dpu.DpuOverlay
        Pre-constructed overlay. The caller creates this once
        (`DpuOverlay("dpu.bit")`) and may reuse it across multiple ModelRunner
        instances if hot-swapping models.

    Notes
    -----
    Construction triggers `overlay.load_model(xmodel_path)`, which:

    - Deserializes the xmodel
    - Finds the single DPU subgraph (we built our xmodels with the SiLU swap
      to guarantee this — see `docs/MODELS.md` → Activation function policy)
    - Programs the runtime to send inputs to it

    The first inference is always slower than subsequent ones (JIT, buffer
    allocation, cache warming) — call `warmup(n=5)` after construction to
    avoid that overhead biasing your first measured timings.
    """

    def __init__(self, spec: ModelSpec, xmodel_path, overlay):
        self.spec        = spec
        self.xmodel_path = Path(xmodel_path)

        if not self.xmodel_path.is_file():
            raise FileNotFoundError(
                f"xmodel not found: {self.xmodel_path}\n"
                f"From your laptop, sync it:\n"
                f"  bash scripts/host/03_sync_to_kria.sh ubuntu@<kria-ip> {spec.name}"
            )

        if spec.family not in _SUPPORTED_FAMILIES:
            raise NotImplementedError(
                f"ModelRunner: family {spec.family!r} not yet supported. "
                f"Currently wired: {_SUPPORTED_FAMILIES}. "
                f"YOLOX requires GraphRunner and a different decoder; "
                f"YOLOv7/v4_csp/SSD are stubs."
            )

        # Build the preprocessor (allocates buffers).
        self.preprocess = Preprocessor(spec.family, spec.imgsz)

        # Resolve the decoder for this family.
        self._decode = _DECODERS[spec.family]

        # Load the xmodel into the DPU. The overlay is mutated in place; if
        # multiple ModelRunners share one overlay, only the last-loaded model
        # is callable via `overlay.runner`.
        overlay.load_model(str(self.xmodel_path))
        self.runner = overlay.runner

        # Pre-allocate output buffers in NHWC matching what the xmodel emits.
        # `get_output_tensors` returns one tensor per scale (3 scales for our
        # YOLO models: stride 8, 16, 32). Per-spec example shapes:
        #   yolov5n (imgsz=320): [1,40,40,65], [1,20,20,65], [1,10,10,65]
        #   yolov11n (imgsz=640): [1,80,80,65], [1,40,40,65], [1,20,20,65]
        self.out_buffers = [
            np.zeros(tuple(t.dims), dtype=np.float32, order="C")
            for t in self.runner.get_output_tensors()
        ]

        # Sanity-check expected output channel count. Helps catch
        # spec/xmodel mismatches early instead of producing garbage detections.
        expected_channels = 4 * spec.reg_max + spec.nc
        actual_channels   = self.out_buffers[0].shape[-1]
        if actual_channels != expected_channels:
            raise RuntimeError(
                f"Output channel mismatch: spec says {expected_channels} "
                f"(4*reg_max[{spec.reg_max}] + nc[{spec.nc}]), but xmodel "
                f"emits {actual_channels}. Likely cause: the xmodel was "
                f"compiled from a different model variant or with different "
                f"settings. Recompile with the spec we're trying to use."
            )

        # Record some shapes for debugging.
        self.input_dims  = list(self.runner.get_input_tensors()[0].dims)
        self.output_dims = [list(b.shape) for b in self.out_buffers]

    def warmup(self, n: int = 5, print_each: bool = False) -> List[float]:
        """Run `n` inferences on a random input to warm the JIT / caches.

        Returns the timings (ms) so the caller can verify they stabilize.
        Typical pattern: first warmup is 10-50ms; by the 3rd it's at the
        steady-state inference time.
        """
        dummy = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        times = []
        for i in range(n):
            t0 = time.perf_counter()
            self.infer(dummy)
            dt = (time.perf_counter() - t0) * 1000
            times.append(dt)
            if print_each:
                print(f"  warm-up #{i+1}: {dt:5.2f} ms")
        return times

    def infer(
        self,
        frame_bgr: np.ndarray,
        conf: float = 0.30,
        iou: float = 0.45,
        max_detections: int = 300,
    ) -> Tuple[List[Detection], Dict[str, float]]:
        """Run one inference. Return detections + per-stage timing.

        Parameters
        ----------
        frame_bgr : np.ndarray
            Raw camera frame (BGR, any size). Will be letterbox-resized to
            the model's input size internally.
        conf, iou : float
            Confidence and IoU thresholds. Per-call so you can sweep them
            interactively.
        max_detections : int
            Hard cap on detections after NMS.

        Returns
        -------
        detections : list of (x1, y1, x2, y2, score, class_idx) tuples in
            **original camera-frame coords** (already un-letterboxed).
        timings_ms : dict with keys 'preprocess', 'dpu', 'decode' — milliseconds
            spent in each stage of the pipeline. Useful for finding the
            bottleneck during optimization.
        """
        timings: Dict[str, float] = {}

        # Preprocess
        t0 = time.perf_counter()
        x, ratio, pad_x, pad_y = self.preprocess.process(frame_bgr)
        timings["preprocess"] = (time.perf_counter() - t0) * 1000

        # DPU inference
        t0 = time.perf_counter()
        job_id = self.runner.execute_async([x], self.out_buffers)
        self.runner.wait(job_id)
        timings["dpu"] = (time.perf_counter() - t0) * 1000

        # Decode (family-specific dispatch, but currently identical math).
        t0 = time.perf_counter()
        raw_dets = self._decode(
            self.out_buffers, self.spec.imgsz,
            self.spec.nc, self.spec.reg_max,
            conf, iou, max_detections,
        )
        timings["decode"] = (time.perf_counter() - t0) * 1000

        # Map boxes back to camera-frame coordinates.
        dets = [
            (*unletterbox(d[:4], ratio, pad_x, pad_y), d[4], d[5])
            for d in raw_dets
        ]

        return dets, timings
