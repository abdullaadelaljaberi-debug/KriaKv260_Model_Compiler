"""Compile pipeline — produces .xmodel files from PyTorch/TF checkpoints.

This subpackage runs INSIDE the Vitis-AI 3.5 Docker container on the host PC,
not on the Kria. It is not importable on the Kria (the Kria doesn't have
``vai_q_pytorch`` installed).

Family dispatch:

    >>> from lpr_pipeline.compile import get_compiler
    >>> compiler = get_compiler("yolov5")
    >>> compiler.compile(weights_path="...", calib_dir="...", ...)

Each family-specific module exports a ``Compiler`` class subclassing
``BaseCompiler``. Adding a new family means adding a new module here AND
registering its variants in ``lpr_pipeline.shared.models``.
"""
from .base import BaseCompiler, CompileError, NotImplementedFamilyError
from .registry import get_compiler

__all__ = ["BaseCompiler", "CompileError", "NotImplementedFamilyError", "get_compiler"]
