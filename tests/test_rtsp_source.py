import numpy as np
import pytest

from rtsp_source import RtspFrameSource, build_gstreamer_pipeline


class _FakeCapture:
    def __init__(self, frames, opened=True):
        self._frames = list(frames)
        self._opened = opened
        self.released = False

    def isOpened(self):
        return self._opened

    def read(self):
        if self._frames:
            return True, self._frames.pop(0)
        return False, None

    def release(self):
        self.released = True


def test_build_gstreamer_pipeline_contains_url_and_appsink():
    pipeline = build_gstreamer_pipeline("rtsp://kamera/1")

    assert "rtsp://kamera/1" in pipeline
    assert "appsink" in pipeline


def test_frames_yields_frames_then_raises_on_read_failure():
    frame1 = np.zeros((2, 2, 3), dtype=np.uint8)
    frame2 = np.ones((2, 2, 3), dtype=np.uint8)
    fake_capture = _FakeCapture([frame1, frame2])
    source = RtspFrameSource("rtsp://kamera/1", capture=fake_capture)

    gen = source.frames()
    assert (next(gen) == frame1).all()
    assert (next(gen) == frame2).all()

    with pytest.raises(RuntimeError):
        next(gen)

    assert fake_capture.released is True


def test_init_raises_if_capture_not_opened():
    fake_capture = _FakeCapture([], opened=False)

    with pytest.raises(RuntimeError):
        RtspFrameSource("rtsp://kamera/1", capture=fake_capture)


def test_close_releases_capture():
    fake_capture = _FakeCapture([])
    source = RtspFrameSource("rtsp://kamera/1", capture=fake_capture)

    source.close()

    assert fake_capture.released is True
