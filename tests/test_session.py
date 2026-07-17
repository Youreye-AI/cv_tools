from datetime import datetime
from pathlib import Path

import pytest

from session import SessionConfig, create_session_dir, generate_filename, write_metadata


def test_create_session_dir_creates_folder(tmp_path):
    session_dir = create_session_dir("test_seti", tmp_path)

    assert session_dir == tmp_path / "test_seti"
    assert session_dir.is_dir()


def test_create_session_dir_raises_if_exists(tmp_path):
    (tmp_path / "test_seti").mkdir()

    with pytest.raises(FileExistsError):
        create_session_dir("test_seti", tmp_path)


def test_write_metadata_writes_expected_fields(tmp_path):
    session_dir = tmp_path / "test_seti"
    session_dir.mkdir()
    config = SessionConfig(
        purpose="model egitimi",
        rtsp_url="rtsp://kamera/1",
        interval=30,
        confidence=0.5,
        image_format="jpg",
    )

    metadata_path = write_metadata(session_dir, config)

    assert metadata_path == session_dir / "metadata.json"
    import json

    data = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert data["purpose"] == "model egitimi"
    assert data["rtsp_url"] == "rtsp://kamera/1"
    assert data["interval"] == 30
    assert data["confidence"] == 0.5
    assert data["image_format"] == "jpg"
    assert "created_at" in data


def test_generate_filename_format():
    timestamp = datetime(2026, 7, 16, 15, 32, 45)

    filename = generate_filename(1, "jpg", timestamp)

    assert filename == "20260716_153245_001.jpg"


def test_generate_filename_zero_padded_index():
    timestamp = datetime(2026, 7, 16, 15, 32, 45)

    filename = generate_filename(42, "png", timestamp)

    assert filename == "20260716_153245_042.png"
