#!/usr/bin/env python3
"""Reusable RTSP camera capture, shared across detection modules.

`GstNvdecCamera` opens an RTSP stream through one of three backends and hands back
plain BGR numpy frames (exactly what OpenCV/YOLO expect), so any module can swap
its capture layer for GPU decode without changing its detection code:

  * backend="nvdec"  (recommended for GPU decode): spawns `gst-launch-1.0.exe` as a
    child process running  rtspsrc ! ... ! nv{h264,h265}dec ! videoconvert !
    video/x-raw,format=BGR ! fdsink fd=1, and reads raw BGR bytes from the
    subprocess's stdout pipe. The H.264/H.265 stream is decoded on the NVIDIA GPU
    (NVDEC). **Needs only the GStreamer runtime + nvcodec plugin on disk — the
    stock pip `opencv-python` works as-is, no OpenCV rebuild.** Because raw BGR has
    no framing, the exact decoded width/height must be known (auto-probed via
    gst-discoverer if not supplied).

  * backend="ffmpeg" (default, most portable): OpenCV's stock FFmpeg backend with
    hardware acceleration requested (D3D11VA/NVDEC where available). Needs nothing
    beyond opencv-python.

  * backend="gstreamer": builds the same pipeline but ending in `appsink` and opens
    it with cv2.CAP_GSTREAMER. Lower overhead than the subprocess pipe, BUT requires
    an OpenCV built WITH_GSTREAMER=ON (the pip wheel is NOT — see has_gstreamer()).
    Kept for environments that have such a build; on a stock wheel use "nvdec".

All backends copy the decoded frame to a CPU numpy array for the consumer, so the
win of the GPU backends is offloading *decode* from CPU to GPU, not zero-copy.
"""
from __future__ import annotations

import logging
import re
import shutil
import subprocess
import time
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger("gstreamer_opt")

_VALID_CODECS = ("h264", "h265")
_VALID_BACKENDS = ("nvdec", "ffmpeg", "gstreamer")

# Default install location of the GStreamer Windows MSVC 64-bit package, used as a
# fallback when the gst tools aren't on PATH in the current shell/IDE (a recurring
# issue across terminals). Matches the layout confirmed on this machine.
_GST_BIN_FALLBACK = r"C:\Program Files\gstreamer\1.0\msvc_x86_64\bin"


# ---------------------------------------------------------------- gst tool discovery


def resolve_gst_exe(name: str, override: str | None = None) -> str:
    """Locate a GStreamer CLI tool: explicit override → PATH → default install dir."""
    if override:
        return override
    found = shutil.which(name) or shutil.which(f"{name}.exe")
    if found:
        return found
    fallback = Path(_GST_BIN_FALLBACK) / f"{name}.exe"
    if fallback.is_file():
        return str(fallback)
    raise FileNotFoundError(
        f"{name} not found on PATH or at the default GStreamer install location "
        f"({fallback}). Install the GStreamer runtime for Windows, or pass an explicit path."
    )


def probe_stream_info(rtsp_url: str, protocols: str = "tcp", discoverer_exe: str | None = None,
                      timeout_s: int = 20) -> dict:
    """Run gst-discoverer once to learn the stream's codec and decoded size.

    Returns {"codec": "h264"|"h265", "width": int, "height": int}. Raises if the
    stream can't be analyzed (e.g. camera busy) or the codec isn't H.264/H.265.
    """
    exe = resolve_gst_exe("gst-discoverer-1.0", discoverer_exe)
    proc = subprocess.run(
        [exe, "-v", rtsp_url],
        capture_output=True, text=True, timeout=timeout_s,
    )
    out = proc.stdout + proc.stderr
    if "video/x-h265" in out:
        codec = "h265"
    elif "video/x-h264" in out:
        codec = "h264"
    else:
        raise RuntimeError(f"Could not determine H.264/H.265 codec from gst-discoverer output for {rtsp_url}")
    w = re.search(r"width=\(int\)(\d+)", out)
    h = re.search(r"height=\(int\)(\d+)", out)
    if not (w and h):
        raise RuntimeError(f"Could not determine width/height from gst-discoverer output for {rtsp_url}")
    return {"codec": codec, "width": int(w.group(1)), "height": int(h.group(1))}


def has_gstreamer() -> bool:
    """True if this OpenCV build can use cv2.CAP_GSTREAMER (the pip wheel cannot)."""
    for line in cv2.getBuildInformation().splitlines():
        stripped = line.strip()
        if stripped.startswith("GStreamer:"):
            return stripped.split(":", 1)[1].strip().upper().startswith("YES")
    return False


class GstNvdecCamera:
    """RTSP camera with a swappable, optionally GPU-accelerated decode backend.

    Drop-in for a cv2.VideoCapture-style loop: use `read()` for a single frame,
    or iterate `frames()` for an endless, self-reconnecting stream of frames.
    """

    def __init__(
        self,
        rtsp_url: str,
        *,
        backend: str = "ffmpeg",
        codec: str | None = None,
        width: int | None = None,
        height: int | None = None,
        latency_ms: int = 200,
        protocols: str = "tcp",
        hw_accel: bool = True,
        connect_timeout_ms: int = 5000,
        gst_launch_exe: str | None = None,
    ) -> None:
        if backend not in _VALID_BACKENDS:
            raise ValueError(f"backend must be one of {_VALID_BACKENDS}, got {backend!r}")
        if codec is not None and codec not in _VALID_CODECS:
            raise ValueError(f"codec must be one of {_VALID_CODECS} or None, got {codec!r}")

        self.rtsp_url = rtsp_url
        self.backend = backend
        self.codec = codec
        self.width = width
        self.height = height
        self.latency_ms = latency_ms
        self.protocols = protocols
        self.hw_accel = hw_accel
        self.connect_timeout_ms = connect_timeout_ms
        self.gst_launch_exe = gst_launch_exe

        if backend == "gstreamer" and not has_gstreamer():
            raise RuntimeError(
                "backend='gstreamer' needs an OpenCV built WITH_GSTREAMER=ON, but this cv2 reports "
                "'GStreamer: NO'. Use backend='nvdec' (subprocess pipe, no rebuild) for GPU decode "
                "on a stock opencv-python wheel, or backend='ffmpeg'."
            )

        # For the GPU backends we need the codec; the nvdec pipe also needs exact
        # decoded dimensions. Auto-probe anything missing with gst-discoverer.
        if backend in ("nvdec", "gstreamer"):
            need_dims = backend == "nvdec" and (self.width is None or self.height is None)
            if self.codec is None or need_dims:
                logger.info("Probing stream (codec/size) with gst-discoverer...")
                info = probe_stream_info(rtsp_url, protocols, discoverer_exe=None)
                self.codec = self.codec or info["codec"]
                self.width = self.width if self.width is not None else info["width"]
                self.height = self.height if self.height is not None else info["height"]
                logger.info("Stream is %s %dx%d", self.codec, self.width, self.height)
        elif self.codec is None:
            self.codec = "h264"  # unused by the ffmpeg backend, but keep the attribute sane

        # Exactly one of these is active at a time, depending on backend.
        self._cap: cv2.VideoCapture | None = None
        self._proc: subprocess.Popen | None = None

    # ---------------------------------------------------------------- pipeline strings

    def build_pipeline(self) -> str:
        """cv2.CAP_GSTREAMER launch string (gstreamer backend; also handy for smoke tests)."""
        return (
            f"rtspsrc location={self.rtsp_url} latency={self.latency_ms} protocols={self.protocols} ! "
            f"rtp{self.codec}depay ! {self.codec}parse ! nv{self.codec}dec ! "
            f"videoconvert ! video/x-raw,format=BGR ! appsink drop=true max-buffers=1 sync=false"
        )

    def _nvdec_args(self) -> list[str]:
        """Argument list for the gst-launch subprocess (nvdec backend)."""
        exe = resolve_gst_exe("gst-launch-1.0", self.gst_launch_exe)
        return [
            exe, "-q",
            "rtspsrc", f"location={self.rtsp_url}", f"protocols={self.protocols}", f"latency={self.latency_ms}",
            "!", f"rtp{self.codec}depay",
            "!", f"{self.codec}parse",
            "!", f"nv{self.codec}dec",
            "!", "videoconvert",
            "!", f"video/x-raw,format=BGR,width={self.width},height={self.height}",
            # Leaky queue: if the Python-side reader falls behind (e.g. slow YOLO
            # inference), drop stale decoded frames instead of queuing them up.
            # Without this, a slow consumer causes ever-growing latency that looks
            # like the camera feed is running in slow motion.
            "!", "queue", "leaky=downstream", "max-size-buffers=1",
            "!", "fdsink", "fd=1",
        ]

    # ---------------------------------------------------------------- lifecycle

    def _open_cv2(self) -> cv2.VideoCapture:
        if self.backend == "gstreamer":
            return cv2.VideoCapture(self.build_pipeline(), cv2.CAP_GSTREAMER)
        # ffmpeg backend: explicit timeouts (so a stuck socket fails fast and retries
        # instead of hanging forever) plus a request for hardware decode.
        params = [
            cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, self.connect_timeout_ms,
            cv2.CAP_PROP_READ_TIMEOUT_MSEC, self.connect_timeout_ms,
        ]
        if self.hw_accel:
            params += [cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY]
        return cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG, params)

    def open(self) -> None:
        if self.is_opened():
            return
        logger.info("Opening camera (backend=%s, codec=%s)...", self.backend, self.codec)
        if self.backend == "nvdec":
            # stderr is silenced so gst-launch progress noise doesn't flood the log;
            # a failed pipeline surfaces as an EOF on stdout, which triggers reconnect.
            self._proc = subprocess.Popen(self._nvdec_args(), stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        else:
            self._cap = self._open_cv2()

    def is_opened(self) -> bool:
        if self.backend == "nvdec":
            return self._proc is not None and self._proc.poll() is None
        return self._cap is not None and self._cap.isOpened()

    def _read_nvdec(self) -> tuple[bool, np.ndarray | None]:
        assert self._proc is not None and self._proc.stdout is not None
        frame_bytes = self.width * self.height * 3
        buf = bytearray()
        while len(buf) < frame_bytes:
            chunk = self._proc.stdout.read(frame_bytes - len(buf))
            if not chunk:
                return False, None  # pipe closed / stream ended
            buf += chunk
        frame = np.frombuffer(bytes(buf), dtype=np.uint8).reshape((self.height, self.width, 3))
        return True, frame

    def read(self) -> tuple[bool, np.ndarray | None]:
        """Single frame, cv2.VideoCapture.read() semantics. Opens lazily on first call."""
        if not self.is_opened():
            self.open()
        if self.backend == "nvdec":
            return self._read_nvdec()
        assert self._cap is not None
        return self._cap.read()

    def release(self) -> None:
        if self._proc is not None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self._proc.kill()
            self._proc = None
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self) -> "GstNvdecCamera":
        self.open()
        return self

    def __exit__(self, *exc) -> None:
        self.release()

    # ---------------------------------------------------------------- streaming

    def frames(self, max_backoff_seconds: float = 30.0) -> Iterator[np.ndarray]:
        """Yield frames forever, reconnecting with exponential backoff on failure.

        A reconnect gap is transparent to the caller: it just pauses between yields
        rather than raising, so downstream per-track state and cooldowns are never
        reset merely because the stream blipped.
        """
        backoff = 1.0
        self.release()
        self.open()
        try:
            while True:
                if not self.is_opened():
                    logger.warning("Stream not open, retrying in %.1fs", backoff)
                    time.sleep(backoff)
                    backoff = min(backoff * 2, max_backoff_seconds)
                    self.release()
                    self.open()
                    continue

                ok, frame = self.read()
                if not ok or frame is None:
                    logger.warning("Read failed, reconnecting in %.1fs", backoff)
                    time.sleep(backoff)
                    backoff = min(backoff * 2, max_backoff_seconds)
                    self.release()
                    self.open()
                    continue

                backoff = 1.0
                yield frame
        finally:
            self.release()
