"""Entry point: interactively collects human crops from an RTSP stream using YOLO26."""
from __future__ import annotations

import signal
import sys
from datetime import datetime
from pathlib import Path

import cv2

from detector import PersonDetector
from rtsp_source import RtspFrameSource
from session import SessionConfig, create_session_dir, generate_filename, write_metadata


def ask(prompt: str, default: str | None = None) -> str:
    suffix = f" [{default}]" if default is not None else ""
    answer = input(f"{prompt}{suffix}: ").strip()
    return answer if answer else (default or "")


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
        config = SessionConfig(
            purpose=purpose,
            rtsp_url=rtsp_url,
            interval=interval,
            confidence=confidence,
            image_format=image_format,
        )
        write_metadata(session_dir, config)
        print(f"Klasor olusturuldu: {session_dir}")

        detector = PersonDetector(confidence=confidence)
        source = RtspFrameSource(rtsp_url)

        state = {"saved_count": 0}

        def handle_sigint(signum, frame):
            print(f"\nDurduruldu. Toplam kaydedilen gorsel: {state['saved_count']}")
            source.close()
            sys.exit(0)

        signal.signal(signal.SIGINT, handle_sigint)

        frame_index = 0
        for frame in source.frames():
            frame_index += 1
            if frame_index % interval != 0:
                continue
            crops = detector.detect_and_crop(frame)
            for crop in crops:
                filename = generate_filename(state["saved_count"], image_format, datetime.now())
                cv2.imwrite(str(session_dir / filename), crop)
                state["saved_count"] += 1
            print(f"Frame {frame_index}: {len(crops)} kisi kaydedildi (toplam {state['saved_count']})")
    except (FileExistsError, RuntimeError) as exc:
        print(f"Hata: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
