"""Session folder and metadata management for the human image collector."""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass
class SessionConfig:
    purpose: str
    rtsp_url: str
    interval: int
    confidence: float
    image_format: str
    roi: tuple[int, int, int, int] | None = None


def create_session_dir(name: str, base_dir: Path) -> Path:
    session_dir = base_dir / name
    session_dir.mkdir(parents=True, exist_ok=False)
    return session_dir


def write_metadata(session_dir: Path, config: SessionConfig) -> Path:
    metadata = {
        "purpose": config.purpose,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "rtsp_url": config.rtsp_url,
        "interval": config.interval,
        "confidence": config.confidence,
        "image_format": config.image_format,
        "roi": (
            {
                "x1": config.roi[0],
                "y1": config.roi[1],
                "x2": config.roi[2],
                "y2": config.roi[3],
            }
            if config.roi is not None
            else None
        ),
    }
    metadata_path = session_dir / "metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return metadata_path


def generate_filename(index: int, image_format: str, timestamp: datetime) -> str:
    ts = timestamp.strftime("%Y%m%d_%H%M%S")
    return f"{ts}_{index:03d}.{image_format}"
