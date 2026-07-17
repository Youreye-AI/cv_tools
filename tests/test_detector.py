import numpy as np

from detector import PersonDetector, crop_detections


def test_crop_detections_returns_correct_region():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    frame[10:20, 10:20] = 255

    crops = crop_detections(frame, [(10, 10, 20, 20)])

    assert len(crops) == 1
    assert crops[0].shape == (10, 10, 3)
    assert (crops[0] == 255).all()


def test_crop_detections_clamps_to_frame_bounds():
    frame = np.zeros((50, 50, 3), dtype=np.uint8)

    crops = crop_detections(frame, [(-5, -5, 60, 60)])

    assert len(crops) == 1
    assert crops[0].shape == (50, 50, 3)


def test_crop_detections_skips_invalid_box():
    frame = np.zeros((50, 50, 3), dtype=np.uint8)

    crops = crop_detections(frame, [(30, 30, 10, 10)])

    assert crops == []


class _FakeBox:
    def __init__(self, xyxy):
        self.xyxy = [np.array(xyxy, dtype=float)]


class _FakeResult:
    def __init__(self, boxes):
        self.boxes = boxes


class _FakeModel:
    def __init__(self, boxes_per_call):
        self._boxes_per_call = boxes_per_call

    def predict(self, frame, conf, classes, verbose):
        return [_FakeResult([_FakeBox(b) for b in self._boxes_per_call])]


def test_person_detector_detect_and_crop_uses_model_boxes():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    frame[0:10, 0:10] = 200
    fake_model = _FakeModel(boxes_per_call=[(0, 0, 10, 10)])
    detector = PersonDetector(confidence=0.5, model=fake_model)

    crops = detector.detect_and_crop(frame)

    assert len(crops) == 1
    assert crops[0].shape == (10, 10, 3)
    assert (crops[0] == 200).all()


def test_person_detector_no_detections_returns_empty_list():
    frame = np.zeros((50, 50, 3), dtype=np.uint8)
    fake_model = _FakeModel(boxes_per_call=[])
    detector = PersonDetector(confidence=0.5, model=fake_model)

    crops = detector.detect_and_crop(frame)

    assert crops == []
