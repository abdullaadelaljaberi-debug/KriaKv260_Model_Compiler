"""KriaKv260 Model Compiler pipeline.

This package provides shared compile + deploy logic for object-detection
models targeting the Xilinx Kria KV260 (Vitis-AI 3.5).

Subpackages:
    compile/   Family-specific quantize + compile logic (host-side, runs
               inside the Vitis-AI Docker image).
    deploy/    Runtime helpers for the Kria board (model runner, decoders,
               threaded camera, batch-eval).
    shared/    Code used by both sides — model registry, preprocessing,
               system tuning.
"""
__version__ = "0.1.0"
