"""Entry point: interactively collects human crops from an RTSP stream using YOLO26."""
from __future__ import annotations

import signal
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from detector import PersonDetector
from rtsp_source import RtspFrameSource
from session import SessionConfig, create_session_dir, generate_filename, write_metadata

_ROI_WINDOW_NAME = "ROI sec (Enter/Space: onayla, ESC: iptal)"


def ask(prompt: str, default: str | None = None) -> str:
    suffix = f" [{default}]" if default is not None else ""
    answer = input(f"{prompt}{suffix}: ").strip()
    return answer if answer else (default or "")


def to_xyxy(
    roi_xywh: tuple[int, int, int, int]
) -> tuple[int, int, int, int] | None:
    x, y, w, h = roi_xywh
    if w <= 0 or h <= 0:
        return None
    return (x, y, x + w, y + h)


def compute_resize_scale(
    height: int, width: int, max_width: int = 1280, max_height: int = 720
) -> float:
    return min(max_width / width, max_height / height, 1.0)


def select_roi(frame: np.ndarray) -> tuple[int, int, int, int] | None:
    height, width = frame.shape[:2]
    scale = compute_resize_scale(height, width)
    display_frame = (
        cv2.resize(frame, (int(width * scale), int(height * scale)))
        if scale != 1.0
        else frame
    )
    roi_xywh = cv2.selectROI(_ROI_WINDOW_NAME, display_frame, showCrosshair=True)
    cv2.destroyWindow(_ROI_WINDOW_NAME)
    scaled_roi_xywh = tuple(int(v / scale) for v in roi_xywh)
    return to_xyxy(scaled_roi_xywh)


def main() -> None:
    purpose = ask("Bu veriyi ne icin topluyorsunuz?")
    name = ask("Kayitlarin gidecegi klasor ismi")
    rtsp_url = ask("RTSP URL")

    interval_str = ask("Kac frame'de bir tespit yapilsin?", "30")
    try:
        interval = int(interval_str)
    except ValueError:
        print(f"Gecersiz frame araligi: {interval_str}")
        sys.exit(1)
    if interval <= 0:
        print(f"Frame araligi pozitif bir sayi olmali: {interval}")
        sys.exit(1)

    confidence_str = ask("Confidence esigi", "0.5")
    try:
        confidence = float(confidence_str)
    except ValueError:
        print(f"Gecersiz confidence esigi: {confidence_str}")
        sys.exit(1)

    image_format = ask("Gorsel formati (jpg/png)", "jpg").lower()
    if image_format not in ("jpg", "png"):
        print(f"Desteklenmeyen format: {image_format}")
        sys.exit(1)

    try:
        session_dir = create_session_dir(name, Path.cwd())

        detector = PersonDetector(confidence=confidence)
        source = RtspFrameSource(rtsp_url)
        frame_iter = source.frames()
        first_frame = next(frame_iter)

        roi: tuple[int, int, int, int] | None = None
        want_roi = ask("ROI secmek ister misiniz? (e/H)", "h").lower()
        if want_roi in ("e", "evet"):
            roi = select_roi(first_frame)

        config = SessionConfig(
            purpose=purpose,
            rtsp_url=rtsp_url,
            interval=interval,
            confidence=confidence,
            image_format=image_format,
            roi=roi,
        )
        write_metadata(session_dir, config)
        print(f"Klasor olusturuldu: {session_dir}")

        state = {"saved_count": 0}

        def handle_sigint(signum, frame):
            print(f"\nDurduruldu. Toplam kaydedilen gorsel: {state['saved_count']}")
            source.close()
            sys.exit(0)

        signal.signal(signal.SIGINT, handle_sigint)

        def process_frame(frame_index: int, frame: np.ndarray) -> None:
            if frame_index % interval != 0:
                return
            crops = detector.detect_and_crop(frame, roi=roi)
            for crop in crops:
                filename = generate_filename(
                    state["saved_count"], image_format, datetime.now()
                )
                cv2.imwrite(str(session_dir / filename), crop)
                state["saved_count"] += 1
            print(
                f"Frame {frame_index}: {len(crops)} kisi kaydedildi "
                f"(toplam {state['saved_count']})"
            )

        frame_index = 1
        process_frame(frame_index, first_frame)
        for frame in frame_iter:
            frame_index += 1
            process_frame(frame_index, frame)
    except (FileExistsError, RuntimeError) as exc:
        print(f"Hata: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
