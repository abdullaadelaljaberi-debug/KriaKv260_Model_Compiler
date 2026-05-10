"""Deploy-side modules for running compiled xmodels on the Kria KV260.

Public API:

    from lpr_pipeline.deploy import (
        ModelRunner, Preprocessor, ThreadedCamera, decode_yolov5u, unletterbox,
        draw_detections, draw_stats_overlay,
    )

Currently supports the YOLOv5u (anchor-free DFL) family — yolov5n and yolov5s.
YOLOX support exists in the model registry but needs the GraphRunner path
and a decoder; we'll add those in a later pass.

All modules in this package assume:
  - The Kria-side scripts have been run (Pass 5): VAI 3.5 runtime, pynq-venv,
    USB autosuspend off, governor=performance, camera tuned.
  - The notebook is running as root (PYNQ-DPU needs root to mmap the FPGA).
  - The xmodel was compiled by this pipeline (single DPU subgraph, NHWC outputs,
    DPU fingerprint 0x101000056010407 for KV260 B4096 / VAI 3.5).
"""
from lpr_pipeline.deploy.preprocess import Preprocessor, unletterbox
from lpr_pipeline.deploy.decoders   import decode_yolov5u
from lpr_pipeline.deploy.camera     import ThreadedCamera
from lpr_pipeline.deploy.runner     import ModelRunner
from lpr_pipeline.deploy.draw       import draw_detections, draw_stats_overlay

__all__ = [
    "ModelRunner",
    "Preprocessor",
    "ThreadedCamera",
    "decode_yolov5u",
    "unletterbox",
    "draw_detections",
    "draw_stats_overlay",
]
