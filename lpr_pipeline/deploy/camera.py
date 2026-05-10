"""Threaded camera capture for sustained 60 fps.

The naive `cv2.VideoCapture(0).read()` in a loop bottlenecks at ~15 fps on
the KV260 even with a 60 fps Brio — for non-obvious reasons documented below.
This module wraps the capture in a background thread with the right knobs
turned for high throughput.

Why a thread, not a polling loop:

  cv2.read() blocks on the V4L2 driver. If your inference is slower than the
  camera (e.g., yolox_tiny at ~10 fps vs camera at 60 fps), the camera's
  internal buffers fill up and `read()` starts returning stale frames (from
  several hundred ms ago — visible lag). A background thread that drains
  the camera as fast as it can deliver, while the main thread reads only the
  *latest* frame, keeps end-to-end latency low.

Why BUFFERSIZE=4 (not 1):

  This is the **non-obvious single biggest fix** on the KV260 + Brio + Ubuntu
  22.04 stack. With OpenCV's default BUFFERSIZE=1, the V4L2 backend somehow
  caps at 15 fps regardless of what the camera is actually delivering. Setting
  BUFFERSIZE=4 unblocks it to the camera ceiling. The exact mechanism isn't
  publicly documented, but empirically: 1 → 15 fps; 2-3 → still 15 fps;
  4+ → ~60 fps. We use 4 as the smallest known-working value.

Why MJPG (not YUYV):

  At 480p, the Brio's YUYV mode caps at 15 fps even with everything else
  tuned correctly. MJPG mode delivers a clean 60 fps. The decode cost of
  MJPG → BGR is negligible compared to the throughput gain.
"""
from __future__ import annotations

import threading
import time
from typing import Optional, Tuple

import cv2
import numpy as np


class ThreadedCamera:
    """Background-thread camera reader exposing always-latest frames.

    Construction blocks for up to `first_frame_timeout_s` (default 3 s) waiting
    for the first frame to arrive — so by the time `__init__` returns, calling
    `read_new()` is guaranteed to return a valid frame (no need to handle
    "warming up" in the consumer).

    Cleanup-on-failure: if `__init__` raises (no camera, no frame, etc.), the
    V4L2 handle is released and the capture thread is joined before the
    exception propagates. Without this, a failed construction would leak
    `/dev/video0` until the Python interpreter exited, blocking subsequent
    `ThreadedCamera()` attempts and any v4l2-ctl tuning from another shell.

    Usage::

        cam = ThreadedCamera()
        try:
            while True:
                frame, fid = cam.read_new()
                if frame is None:
                    time.sleep(0.001)   # no new frame since last read
                    continue
                # ... process `frame` ...
        finally:
            cam.close()

    Parameters
    ----------
    src : int or str
        Video device. Integer 0 → /dev/video0; string for an explicit path
        or URL. Default: 0.
    width, height : int
        Capture resolution. 640×480 is the Brio's clean 60 fps mode.
    fps : int
        Target framerate. The camera may not honour this exactly; check
        `cv2.CAP_PROP_FPS` after construction.
    buffersize : int
        V4L2 capture buffer count. **Must be ≥4 on this platform** or you'll
        get ~15 fps regardless of other settings.
    first_frame_timeout_s : float
        How long to wait for the first frame after the capture thread starts.
        The Brio takes ~1-2 s to negotiate MJPG mode after a fresh USB
        enumeration; 3 s is the smallest known-reliable value.
    """

    def __init__(self, src=0, width: int = 640, height: int = 480,
                 fps: int = 60, buffersize: int = 4,
                 first_frame_timeout_s: float = 3.0):
        # Initialize state to None/False first so cleanup-on-error knows what
        # to release. Without this, an early failure (e.g., VideoCapture open
        # succeeds but a property setter hangs) would leak the cv2 handle and
        # the thread, holding /dev/video0 forever (until kernel restart).
        self.cap:        Optional[cv2.VideoCapture] = None
        self._thread:    Optional[threading.Thread] = None
        self._running:   bool                       = False
        self._frame:     Optional[np.ndarray]       = None
        self._frame_id:  int                        = 0
        self._last_seen: int                        = -1
        self._lock = threading.Lock()

        try:
            self.cap = cv2.VideoCapture(src, cv2.CAP_V4L2)
            if not self.cap.isOpened():
                raise RuntimeError(
                    f"cannot open camera {src!r}. "
                    f"Check: ls -la /dev/video* ; v4l2-ctl --list-devices"
                )

            # Camera config. Order matters slightly — set codec first so subsequent
            # property setters know what format we want.
            self.cap.set(cv2.CAP_PROP_FOURCC,        cv2.VideoWriter_fourcc(*'MJPG'))
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,   width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  height)
            self.cap.set(cv2.CAP_PROP_FPS,           fps)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE,    buffersize)

            # Start the background capture thread.
            self._running = True
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()

            # Block waiting for the first frame.
            iters = max(1, int(first_frame_timeout_s / 0.02))
            for _ in range(iters):
                with self._lock:
                    if self._frame is not None:
                        break
                time.sleep(0.02)
            else:
                raise RuntimeError(
                    f"Camera opened but no frames received in "
                    f"{first_frame_timeout_s:.1f}s. Common causes:\n"
                    f"  - v4l2 settings conflict — re-run "
                    f"scripts/kria/02_apply_tuning.sh from a shell with the "
                    f"camera released (e.g., after Kernel → Restart in Jupyter)\n"
                    f"  - Permission issue — ensure user is in 'video' group\n"
                    f"  - Camera busy — `sudo fuser /dev/video0` shows who's "
                    f"holding it; `pkill -f v4l2` to clear stale processes."
                )

            # Record actual values (the camera may have negotiated different
            # numbers than what we requested).
            self.actual_width  = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.actual_fps    = self.cap.get(cv2.CAP_PROP_FPS)

        except BaseException:
            # Cleanup on failure: stop the thread, release the V4L2 handle.
            # Without this, /dev/video0 stays held by this process until the
            # interpreter exits, blocking subsequent ThreadedCamera() attempts
            # and any v4l2-ctl tuning from a separate shell.
            self._running = False
            if self._thread is not None and self._thread.is_alive():
                self._thread.join(timeout=1.0)
            if self.cap is not None:
                self.cap.release()
            raise

    def _loop(self) -> None:
        """Background loop: drain camera as fast as it delivers."""
        while self._running:
            ok, f = self.cap.read()
            if ok:
                with self._lock:
                    self._frame = f
                    self._frame_id += 1
            else:
                # Brief sleep so a failing camera doesn't burn 100% CPU.
                time.sleep(0.005)

    def read_new(self) -> Tuple[Optional[np.ndarray], int]:
        """Return the latest frame if it's newer than the last call.

        Returns
        -------
        frame : np.ndarray or None
            BGR image, or None if no new frame has arrived since the last
            call to `read_new()`.
        frame_id : int
            Monotonically-incrementing frame ID. Useful for tracking how
            many unique frames have arrived (camera FPS) vs how many you
            actually consumed (consumer FPS).

        Returns a reference to the internal buffer — **do not modify** the
        returned frame; copy first if you need to. The background thread will
        overwrite this buffer when the next frame arrives.
        """
        with self._lock:
            if self._frame is None or self._frame_id == self._last_seen:
                return None, self._frame_id
            self._last_seen = self._frame_id
            return self._frame, self._frame_id

    def close(self) -> None:
        """Stop the background thread and release the camera handle.

        Idempotent — safe to call multiple times. Safe to call on a
        partially-initialized instance (e.g., after `__init__` raised).
        """
        self._running = False
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def __enter__(self): return self
    def __exit__(self, exc_type, exc_val, exc_tb): self.close()
