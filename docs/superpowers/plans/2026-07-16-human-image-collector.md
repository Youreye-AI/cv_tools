# RTSP + YOLO26 İnsan Görseli Toplama Script'i Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** RTSP kamera akışından GStreamer ile frame okuyup YOLO26 (Ultralytics) ile insan tespiti yapan, tespit edilen kişileri kırpılmış görseller olarak diske kaydeden, terminalden interaktif çalışan bir Python script'i oluşturmak.

**Architecture:** Üç bağımsız modül (`session.py` — klasör/metadata/dosya adı üretimi, `detector.py` — YOLO26 tespiti + crop, `rtsp_source.py` — `cv2.VideoCapture` + GStreamer backend ile RTSP frame kaynağı) `main.py` içinde birleştirilir. Her üç modülün saf mantığı, alt seviye bağımlılıklar (gerçek YOLO modeli / gerçek RTSP akışı) enjekte edilebilir tasarlanarak sahte (fake) nesnelerle birim testlerle doğrulanır; yalnızca gerçek bir RTSP kamerayla uçtan uca doğrulama kullanıcı tarafından manuel yapılır.

**Tech Stack:** Python 3.13 (conda ortamı: `cvstack313`), `ultralytics` (YOLO26), `opencv-python` (`cv2` — GStreamer 1.28.1 desteğiyle build edilmiş, RTSP okuma ve görsel kaydı için), `numpy`, `pytest`. Tüm bağımlılıklar `cvstack313` ortamında zaten kurulu — ek kurulum gerekmiyor.

## Global Constraints

- Hedef conda ortamı: **cvstack313** (Python 3.13.12). Script bu ortamda çalıştırılacak şekilde yazılır.
- RTSP okuma `cv2.VideoCapture(pipeline_str, cv2.CAP_GSTREAMER)` ile yapılır — `cv2` bu ortamda GStreamer 1.28.1 desteğiyle build edilmiş olduğundan doğrulandı, `gi`/PyGObject'e gerek yok, ek conda-forge kurulumu gerekmiyor.
- Frame interval (N) varsayılanı: **30**.
- Confidence eşiği varsayılanı: **0.5**.
- Görsel formatı varsayılanı: **jpg** (yalnızca `jpg`/`png` desteklenir).
- Kayıt klasörü zaten varsa script hata verip çıkar (üzerine yazmaz).
- Dosya adı deseni: `<YYYYMMDD_HHMMSS>_<3 haneli index>.<format>` (ör. `20260716_153245_001.jpg`).
- `metadata.json` alanları: `purpose`, `created_at` (ISO 8601, saniye hassasiyeti), `rtsp_url`, `interval`, `confidence`, `image_format`.
- YOLO tespiti yalnızca `person` sınıfı (class 0) için yapılır.
- Tüm kullanıcıya dönük terminal metinleri Türkçe olacak.
- Kapsam dışı: oturum devam ettirme (resume), RTSP otomatik yeniden bağlanma, GStreamer/gi kurulumu.

---

## Dosya Yapısı

```
cv_tools/
  session.py            # Klasör oluşturma, metadata.json yazımı, dosya adı üretimi
  detector.py            # crop_detections (saf fonksiyon) + PersonDetector (YOLO26 sarmalayıcı)
  rtsp_source.py          # cv2.VideoCapture + GStreamer backend RTSP frame kaynağı (RtspFrameSource)
  main.py                 # Terminal soruları + ana döngü + Ctrl+C handling
  .gitignore
  tests/
    test_session.py
    test_detector.py
    test_rtsp_source.py
```

---

### Task 1: Git deposu kurulumu + `session.py` (klasör/metadata/dosya adı mantığı)

**Files:**
- Create: `.gitignore`
- Create: `session.py`
- Test: `tests/test_session.py`

**Interfaces:**
- Produces:
  - `SessionConfig` (dataclass): alanlar `purpose: str`, `rtsp_url: str`, `interval: int`, `confidence: float`, `image_format: str`
  - `create_session_dir(name: str, base_dir: Path) -> Path` — `base_dir / name` oluşturur, zaten varsa `FileExistsError` fırlatır
  - `write_metadata(session_dir: Path, config: SessionConfig) -> Path` — `session_dir/metadata.json` yazar, dosya yolunu döner
  - `generate_filename(index: int, image_format: str, timestamp: datetime) -> str` — `"<YYYYMMDD_HHMMSS>_<3 haneli index>.<format>"` döner

- [ ] **Step 1: Git deposunu başlat**

```bash
git init
```

- [ ] **Step 2: `.gitignore` oluştur**

```
__pycache__/
*.pyc
.pytest_cache/
*.egg-info/
```

- [ ] **Step 3: İlk commit**

```bash
git add .gitignore
git commit -m "chore: initialize repository"
```

- [ ] **Step 4: Başarısız testleri yaz**

`tests/test_session.py`:

```python
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
```

- [ ] **Step 5: Testlerin başarısız olduğunu doğrula**

Run: `conda run -n cvstack313 python -m pytest tests/test_session.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'session'`

- [ ] **Step 6: `session.py`'yi yaz**

```python
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
    }
    metadata_path = session_dir / "metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return metadata_path


def generate_filename(index: int, image_format: str, timestamp: datetime) -> str:
    ts = timestamp.strftime("%Y%m%d_%H%M%S")
    return f"{ts}_{index:03d}.{image_format}"
```

- [ ] **Step 7: Testlerin geçtiğini doğrula**

Run: `conda run -n cvstack313 python -m pytest tests/test_session.py -v`
Expected: 5 passed

- [ ] **Step 8: Commit**

```bash
git add session.py tests/test_session.py
git commit -m "feat: add session folder, metadata, and filename generation"
```

---

### Task 2: `detector.py` — crop mantığı ve YOLO26 sarmalayıcısı

**Files:**
- Create: `detector.py`
- Test: `tests/test_detector.py`

**Interfaces:**
- Consumes: yok (bağımsız modül)
- Produces:
  - `crop_detections(frame: np.ndarray, boxes: list[tuple[int, int, int, int]]) -> list[np.ndarray]` — her `(x1, y1, x2, y2)` kutusunu frame sınırlarına clamp'leyip kırpar, geçersiz (sıfır/negatif alan) kutuları atlar
  - `PersonDetector` sınıfı: `__init__(self, model_path: str = "yolo26n.pt", confidence: float = 0.5, model=None)`, `detect_and_crop(self, frame: np.ndarray) -> list[np.ndarray]`. `model` parametresi test amaçlı enjekte edilebilir; `None` ise `ultralytics.YOLO(model_path)` lazy-import edilir.

- [ ] **Step 1: Başarısız testleri yaz**

`tests/test_detector.py`:

```python
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
```

- [ ] **Step 2: Testlerin başarısız olduğunu doğrula**

Run: `conda run -n cvstack313 python -m pytest tests/test_detector.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'detector'`

- [ ] **Step 3: `detector.py`'yi yaz**

```python
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

    def detect_and_crop(self, frame: np.ndarray) -> list[np.ndarray]:
        results = self._model.predict(
            frame, conf=self._confidence, classes=[0], verbose=False
        )
        boxes = [
            tuple(int(v) for v in box.xyxy[0].tolist())
            for result in results
            for box in result.boxes
        ]
        return crop_detections(frame, boxes)
```

- [ ] **Step 4: Testlerin geçtiğini doğrula**

Run: `conda run -n cvstack313 python -m pytest tests/test_detector.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add detector.py tests/test_detector.py
git commit -m "feat: add person detection and crop logic"
```

---

### Task 3: `rtsp_source.py` — `cv2.VideoCapture` + GStreamer backend frame kaynağı

**Files:**
- Create: `rtsp_source.py`
- Test: `tests/test_rtsp_source.py`

**Interfaces:**
- Consumes: yok
- Produces:
  - `build_gstreamer_pipeline(rtsp_url: str) -> str` — RTSP URL'sini içeren GStreamer pipeline string'i üretir
  - `RtspFrameSource` sınıfı: `__init__(self, rtsp_url: str, capture=None)` (`capture` test amaçlı enjekte edilebilir; `None` ise `cv2.VideoCapture(build_gstreamer_pipeline(rtsp_url), cv2.CAP_GSTREAMER)` kullanılır, açılamazsa `RuntimeError`), `frames(self) -> Iterator[np.ndarray]` (BGR numpy frame üretir, `read()` başarısız olursa `RuntimeError` fırlatır), `close(self) -> None`

`cvstack313` ortamındaki `cv2`, GStreamer 1.28.1 desteğiyle build edildiği için (`cv2.getBuildInformation()` ile doğrulandı) bu modül `cv2.VideoCapture`'ı doğrudan `cv2.CAP_GSTREAMER` backend'iyle kullanır — `gi`/PyGObject'e gerek yoktur. `capture` enjeksiyonu sayesinde gerçek RTSP bağlantısı olmadan tam birim testi mümkündür.

- [ ] **Step 1: Başarısız testleri yaz**

`tests/test_rtsp_source.py`:

```python
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
```

- [ ] **Step 2: Testlerin başarısız olduğunu doğrula**

Run: `conda run -n cvstack313 python -m pytest tests/test_rtsp_source.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'rtsp_source'`

- [ ] **Step 3: `rtsp_source.py`'yi yaz**

```python
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
```

- [ ] **Step 4: Testlerin geçtiğini doğrula**

Run: `conda run -n cvstack313 python -m pytest tests/test_rtsp_source.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add rtsp_source.py tests/test_rtsp_source.py
git commit -m "feat: add cv2/GStreamer RTSP frame source"
```

---

### Task 4: `main.py` — terminal akışı + ana döngü

**Files:**
- Create: `main.py`

**Interfaces:**
- Consumes:
  - `session.SessionConfig`, `session.create_session_dir`, `session.write_metadata`, `session.generate_filename`
  - `detector.PersonDetector` (`detect_and_crop`)
  - `rtsp_source.RtspFrameSource` (`frames`, `close`)
- Produces: `main()` — script giriş noktası

- [ ] **Step 1: `main.py`'yi yaz**

```python
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
    interval = int(ask("Kac frame'de bir tespit yapilsin?", "30"))
    confidence = float(ask("Confidence esigi", "0.5"))
    image_format = ask("Gorsel formati (jpg/png)", "jpg").lower()
    if image_format not in ("jpg", "png"):
        print(f"Desteklenmeyen format: {image_format}")
        sys.exit(1)

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


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Modülün sözdizimsel olarak doğru olduğunu kontrol et**

Run: `conda run -n cvstack313 python -c "import ast; ast.parse(open('main.py', encoding='utf-8').read())"`
Expected: hata vermeden çıkar

- [ ] **Step 3: Mevcut tüm testlerin hâlâ geçtiğini doğrula**

Run: `conda run -n cvstack313 python -m pytest tests/ -v`
Expected: 15 passed

- [ ] **Step 4: Commit**

```bash
git add main.py
git commit -m "feat: wire up interactive entry point"
```

- [ ] **Step 5: Manuel doğrulama (kullanıcı tarafından, gerçek RTSP kamerayla)**

```bash
conda run -n cvstack313 python main.py
```

çalıştırılıp gerçek bir RTSP URL ile: klasörün oluştuğu, `metadata.json`'ın doğru yazıldığı, N'inci frame'lerde crop'ların kaydedildiği ve Ctrl+C ile temiz durduğu gözlemlenerek doğrulanır.

---

## Self-Review Notu

- **Spec kapsamı:** Terminal soruları (amaç/isim/RTSP/interval/confidence/format), klasör+metadata oluşturma, cv2/GStreamer pipeline, N'inci frame'de YOLO26 person tespiti, crop+kaydetme, dosya adı deseni, Ctrl+C ile temiz durma, klasör-zaten-var hatası, cvstack313 hedef ortamı — tümü Task 1-4'te karşılanıyor.
- **Placeholder taraması:** Yok — tüm kod adımları tam içerik içeriyor.
- **Tip/isim tutarlılığı:** `SessionConfig`, `create_session_dir`, `write_metadata`, `generate_filename`, `PersonDetector.detect_and_crop`, `RtspFrameSource.frames`/`close`/`build_gstreamer_pipeline` tüm görevlerde aynı imzalarla kullanılıyor.
- **Değişiklik:** `gi`/PyGObject'e olan bağımlılık, `cvstack313`'teki `cv2`'nin GStreamer 1.28.1 desteğiyle build edilmiş olduğunun doğrulanmasıyla kaldırıldı; `rtsp_source.py` artık tamamen birim test edilebilir (Task 3), ek conda-forge kurulumu gerekmiyor.
