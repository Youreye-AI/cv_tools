"""RTSP frame source using OpenCV's GStreamer backend."""
from __future__ import annotations

from collections.abc import Iterator

import cv2
import numpy as np


def build_gstreamer_pipeline(rtsp_url: str) -> str:
    return (
        f"rtspsrc location={rtsp_url} latency=200 ! "
        "decodebin ! videoconvert ! video/x-raw,format=BGR ! appsink"
    )


class RtspFrameSource:
    """Pulls BGR numpy frames from an RTSP stream via OpenCV's GStreamer backend."""

    def __init__(self, rtsp_url: str, capture=None) -> None:
        if capture is not None:
            self._capture = capture
        else:
            pipeline = build_gstreamer_pipeline(rtsp_url)
            self._capture = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        if not self._capture.isOpened():
            raise RuntimeError(f"RTSP akisina baglanilamadi: {rtsp_url}")

    def frames(self) -> Iterator[np.ndarray]:
        try:
            while True:
                ret, frame = self._capture.read()
                if not ret:
                    raise RuntimeError(
                        "RTSP akisindan frame okunamadi (baglanti kopmus olabilir)"
                    )
                yield frame
        finally:
            self.close()

    def close(self) -> None:
        self._capture.release()
