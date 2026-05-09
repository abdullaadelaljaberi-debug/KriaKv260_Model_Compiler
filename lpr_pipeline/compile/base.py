"""Base class + exceptions for family-specific compilers."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from lpr_pipeline.shared.models import ModelSpec


class CompileError(RuntimeError):
    """Raised when compilation fails for any reason internal to the pipeline."""


class NotImplementedFamilyError(NotImplementedError):
    """Raised by stub compilers (yolov7 / yolov4_csp / ssd_mobilenetv2).

    The compile path scaffolding exists but family-specific logic isn't
    implemented yet. The error message points the user at the stub file
    so they know where to add the implementation.
    """


@dataclass(frozen=True)
class CompileInputs:
    """Validated inputs to a compile run.

    All paths are absolute and have been verified to exist when this is
    constructed by ``BaseCompiler.run()`` — family-specific
    implementations don't need to re-validate.
    """
    spec:        ModelSpec       # ModelSpec for the variant being compiled
    weights:     Path            # input checkpoint (.pt or family-specific format)
    calib_dir:   Path            # directory of calibration images
    work_dir:    Path            # scratch directory for intermediates
    out_xmodel:  Path            # final xmodel destination
    nc:          int             # number of classes (overrides spec default for retrained models)
    n_calib:     int = 200       # number of calibration images to use
    seed:        int = 42        # for reproducible calibration sampling


class BaseCompiler(ABC):
    """Abstract compiler — every family subclasses this.

    Subclasses implement ``_compile_family`` with the family-specific dance:

      1. Load the trained checkpoint
      2. Strip / rewire the detect head for VAI compatibility
      3. Run calibration on ``inputs.calib_dir``
      4. Quantize (``vai_q_pytorch`` / ``vai_q_tensorflow2``)
      5. Compile the quantized graph to xmodel for B4096 / fingerprint
         0x101000056010407 (VAI 3.5)

    The base class' ``run()`` handles input validation, working-directory
    setup, and error wrapping.
    """

    family: str = ""             # set by subclass; matches ModelSpec.family

    def run(self, inputs: CompileInputs) -> Path:
        """Validate, compile, return the path to the produced xmodel."""
        if self.family != inputs.spec.family:
            raise CompileError(
                f"Compiler family mismatch: this is {self.family!r} but "
                f"spec.family is {inputs.spec.family!r}"
            )

        # Stub families raise NotImplementedFamilyError before we validate
        # input files — saves the user fixing inputs that won't matter.
        if inputs.spec.status == "stub":
            return self._compile_family(inputs)

        if not inputs.weights.is_file():
            raise CompileError(f"weights file not found: {inputs.weights}")

        if not inputs.calib_dir.is_dir():
            raise CompileError(f"calibration directory not found: {inputs.calib_dir}")

        n_calib_avail = sum(
            1 for p in inputs.calib_dir.iterdir()
            if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")
        )
        if n_calib_avail == 0:
            raise CompileError(
                f"no images found in {inputs.calib_dir}\n"
                "Calibration needs at least 100 representative images "
                "(JPG/PNG/BMP)."
            )
        if n_calib_avail < 50:
            print(
                f"⚠  Only {n_calib_avail} calibration images. Quantization "
                f"accuracy may suffer. Aim for ≥200 images covering the "
                f"full range of inputs the deployed model will see."
            )

        inputs.work_dir.mkdir(parents=True, exist_ok=True)
        inputs.out_xmodel.parent.mkdir(parents=True, exist_ok=True)

        try:
            return self._compile_family(inputs)
        except (CompileError, NotImplementedFamilyError):
            raise
        except Exception as e:
            raise CompileError(
                f"Unexpected error during {self.family} compile: "
                f"{type(e).__name__}: {e}"
            ) from e

    @abstractmethod
    def _compile_family(self, inputs: CompileInputs) -> Path:
        """Family-specific compile logic. Returns path to produced xmodel."""
