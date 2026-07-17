"""Person detection and cropping using YOLO26."""
from __future__ import annotations

import numpy as np


def crop_detections(
    frame: np.ndarray, boxes: list[tuple[int, int, int, int]]
) -> list[np.ndarray]:
    crops: list[np.ndarray] = []
    height, width = frame.shape[:2]
    for x1, y1, x2, y2 in boxes:
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width, x2), min(height, y2)
        if x2 <= x1 or y2 <= y1:
            continue
        crops.append(frame[y1:y2, x1:x2].copy())
    return crops


def is_center_in_roi(
    box: tuple[int, int, int, int], roi: tuple[int, int, int, int]
) -> bool:
    bx1, by1, bx2, by2 = box
    rx1, ry1, rx2, ry2 = roi
    cx = (bx1 + bx2) / 2
    cy = (by1 + by2) / 2
    return rx1 <= cx < rx2 and ry1 <= cy < ry2


class PersonDetector:
    """Wraps an Ultralytics YOLO26 model to detect and crop persons (class 0)."""

    def __init__(
        self,
        model_path: str = "yolo26n.pt",
        confidence: float = 0.5,
        model=None,
    ) -> None:
        self._confidence = confidence
        if model is not None:
            self._model = model
        else:
            from ultralytics import YOLO

            self._model = YOLO(model_path)

    def detect_and_crop(
        self, frame: np.ndarray, roi: tuple[int, int, int, int] | None = None
    ) -> list[np.ndarray]:
        results = self._model.predict(
            frame, conf=self._confidence, classes=[0], verbose=False
        )
        boxes = [
            tuple(int(v) for v in box.xyxy[0].tolist())
            for result in results
            for box in result.boxes
        ]
        if roi is not None:
            boxes = [box for box in boxes if is_center_in_roi(box, roi)]
        return crop_detections(frame, boxes)
