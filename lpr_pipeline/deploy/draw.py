"""Bounding-box drawing for the visual deploy notebook.

Renders detection boxes (and optional labels/confidence) onto a BGR frame
in-place. Designed to be fast — drawing at 60 fps means each call has a
budget of < 1 ms. We use cv2's native drawing primitives, which are SIMD-
accelerated on ARM.

The function takes the same detection tuple format the deploy pipeline
emits: `(x1, y1, x2, y2, score, class_idx)` in original camera-frame coords
(already un-letterboxed by `ModelRunner.infer`).
"""
from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np


# A small set of BGR colours that read well over varied backgrounds.
# Cycled by class index. For LPR (1 class) only the first is ever used.
_PALETTE: List[Tuple[int, int, int]] = [
    (  0, 220,   0),   # bright green
    (  0, 165, 255),   # orange
    (255, 100,   0),   # blue
    ( 40,  40, 255),   # red
    (255,   0, 255),   # magenta
    (255, 255,   0),   # cyan
]


def draw_detections(
    frame_bgr: np.ndarray,
    detections: Sequence[Tuple[float, float, float, float, float, int]],
    *,
    class_names: Optional[Sequence[str]] = None,
    show_labels: bool = True,
    show_confidence: bool = True,
    box_thickness: int = 2,
    font_scale: float = 0.5,
    font_thickness: int = 1,
) -> np.ndarray:
    """Draw detection boxes on `frame_bgr` in-place. Returns the same array.

    Parameters
    ----------
    frame_bgr : np.ndarray
        BGR image, modified in-place. If you need to preserve the original,
        pass `frame_bgr.copy()`.
    detections : sequence of (x1, y1, x2, y2, score, class_idx)
        Output of `ModelRunner.infer`. Coordinates are in the camera frame's
        coordinate system (already un-letterboxed).
    class_names : sequence of str, optional
        Names indexed by class_idx. If None, labels show the raw class_idx.
        For LPR with nc=1, pass `["plate"]`.
    show_labels : bool
        Show the class name above each box.
    show_confidence : bool
        Append the score (e.g., "plate 0.87") next to the label.
    box_thickness, font_scale, font_thickness : drawing params

    Returns
    -------
    The same `frame_bgr` array, modified in-place. Returned for chaining.
    """
    if len(detections) == 0:
        return frame_bgr

    h, w = frame_bgr.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX

    for x1, y1, x2, y2, score, cls_idx in detections:
        # Clamp to frame bounds; out-of-frame boxes can come from boxes
        # detected near the letterbox padding region after un-mapping.
        ix1 = max(0, min(int(round(x1)), w - 1))
        iy1 = max(0, min(int(round(y1)), h - 1))
        ix2 = max(0, min(int(round(x2)), w - 1))
        iy2 = max(0, min(int(round(y2)), h - 1))

        color = _PALETTE[int(cls_idx) % len(_PALETTE)]

        # Box
        cv2.rectangle(frame_bgr, (ix1, iy1), (ix2, iy2), color, box_thickness)

        # Label (with optional confidence)
        if show_labels:
            if class_names and 0 <= int(cls_idx) < len(class_names):
                name = class_names[int(cls_idx)]
            else:
                name = str(int(cls_idx))
            text = f"{name} {score:.2f}" if show_confidence else name

            # Measure text so we can draw a filled background rect for readability.
            (text_w, text_h), baseline = cv2.getTextSize(
                text, font, font_scale, font_thickness
            )
            # Position label just above the box; if the box is near the top
            # of the frame, draw inside instead.
            label_y_top = iy1 - text_h - baseline - 2
            if label_y_top < 0:
                label_y_top = iy1 + 2

            cv2.rectangle(
                frame_bgr,
                (ix1,             label_y_top),
                (ix1 + text_w + 4, label_y_top + text_h + baseline + 2),
                color, -1,                          # filled
            )
            # Use black text on the colored background — readable on any colour.
            cv2.putText(
                frame_bgr, text,
                (ix1 + 2, label_y_top + text_h),
                font, font_scale, (0, 0, 0), font_thickness, cv2.LINE_AA,
            )

    return frame_bgr


def draw_stats_overlay(
    frame_bgr: np.ndarray,
    *,
    fps: Optional[float] = None,
    inf_ms: Optional[float] = None,
    detections: Optional[int] = None,
    extra: Iterable[str] = (),
) -> np.ndarray:
    """Draw a small stats panel in the top-left corner of the frame.

    Optional. Useful when you want stats visible inside a recorded video file
    (the visual notebook also shows them as text in the cell, but for a
    demo where the user records the screen, having them in-frame is nicer).
    """
    lines: List[str] = []
    if fps is not None:        lines.append(f"FPS: {fps:5.1f}")
    if inf_ms is not None:     lines.append(f"inf: {inf_ms:5.2f} ms")
    if detections is not None: lines.append(f"det: {detections}")
    lines.extend(extra)
    if not lines:
        return frame_bgr

    font = cv2.FONT_HERSHEY_SIMPLEX
    fs, ft = 0.5, 1
    # Measure widest line
    widths = [cv2.getTextSize(s, font, fs, ft)[0][0] for s in lines]
    line_h = 18
    box_w  = max(widths) + 12
    box_h  = line_h * len(lines) + 8

    # Background rect (semi-transparent dark grey)
    overlay = frame_bgr[:box_h, :box_w].copy()
    cv2.rectangle(frame_bgr, (0, 0), (box_w, box_h), (30, 30, 30), -1)
    cv2.addWeighted(frame_bgr[:box_h, :box_w], 0.7, overlay, 0.3, 0,
                    dst=frame_bgr[:box_h, :box_w])

    for i, line in enumerate(lines):
        cv2.putText(frame_bgr, line, (6, 18 + i * line_h),
                    font, fs, (255, 255, 255), ft, cv2.LINE_AA)

    return frame_bgr
