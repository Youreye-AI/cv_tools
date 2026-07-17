# ROI (İlgi Alanı) Seçimi Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** RTSP + YOLO26 insan görseli toplama script'ine, kullanıcının ilk frame üzerinde fare ile bir dikdörtgen ROI (ilgi alanı) seçebildiği ve sadece bu ROI ile kesişen kişi tespitlerinin kaydedildiği opsiyonel bir özellik eklemek.

**Architecture:** ROI'nin merkez-nokta filtreleme mantığı (`is_center_in_roi`) `detector.py`'ye eklenir çünkü `PersonDetector.detect_and_crop`'un bir parçasıdır. ROI seçim/GUI kodu (`to_xyxy`, `select_roi`) ayrı bir modül açılmadan doğrudan `main.py`'ye eklenir — kullanıcı tüm ROI akışını tek dosyadan takip edebilsin diye. `session.py`'ye ROI'yi `metadata.json`'a yazmak için bir alan eklenir.

**Tech Stack:** Python 3.13 (conda ortamı: `cvstack313`), `cv2.selectROI` (Win32 UI backend ile doğrulandı — `cvstack313`'teki `cv2` `GUI: WIN32UI` desteğiyle build edilmiş), `numpy`, `pytest`.

## Global Constraints

- Hedef conda ortamı: **cvstack313** (Python 3.13.12); test komutları `conda run -n cvstack313 python -m pytest ...` ile çalıştırılır.
- ROI filtreleme kuralı: bir tespitin bounding box'ının **merkez noktası** ROI dikdörtgeni içindeyse (`[x1, x2) x [y1, y2)` — sol/üst kenar dahil, sağ/alt kenar hariç, yarı-açık aralık) kaydedilir; değilse atlanır.
- ROI seçim kodu (`to_xyxy`, `select_roi`) **doğrudan `main.py` içinde** yaşar — ayrı bir `roi.py` modülü açılmaz.
- ROI filtreleme mantığı (`is_center_in_roi`) **`detector.py`'de** yaşar.
- ROI opsiyoneldir: terminalde "ROI secmek ister misiniz? (e/H)" sorulur, varsayılan (boş cevap) "hayır" — ROI yok, tüm frame kullanılır. Sadece "e"/"evet" (büyük/küçük harf duyarsız) ROI seçimini tetikler.
- `select_roi` içinde ESC'ye basılır veya sıfır/negatif boyutlu bir seçim yapılırsa (`cv2.selectROI`'nin döndürdüğü `w<=0` veya `h<=0`): ROI `None` sayılır, hata fırlatılmaz, script tüm frame ile devam eder.
- `metadata.json`'daki `roi` alanı: ROI seçildiyse `{"x1": .., "y1": .., "x2": .., "y2": ..}`, seçilmediyse `null`.
- `select_roi` gerçek bir OpenCV GUI penceresi açtığı için otomatik test kapsamı dışındadır (`rtsp_source.py`'nin gerçek RTSP bağlantısı gibi) — kullanıcı tarafından manuel doğrulanır. `to_xyxy` ve `is_center_in_roi` saf fonksiyonlardır ve otomatik test edilir.
- Tüm kullanıcıya dönük terminal metinleri Türkçe (ASCII-only, ör. "secmek" değil "seçmek" değil, mevcut dosyalardaki gibi "icin" tarzı) olacak.
- Kapsam dışı: ROI'nin oturum sırasında yeniden tanımlanması, çoklu/poligon ROI şekilleri.

---

## Dosya Yapısı

```
cv_tools/
  session.py    # DEĞİŞİKLİK: SessionConfig'e roi alanı, write_metadata roi'yi yazar
  detector.py    # DEĞİŞİKLİK: is_center_in_roi (yeni saf fonksiyon), detect_and_crop roi parametresi
  main.py         # DEĞİŞİKLİK: to_xyxy, select_roi (yeni), main() akışı ROI sorusu için güncellenir
  tests/
    test_session.py    # DEĞİŞİKLİK: roi metadata testleri eklenir
    test_detector.py    # DEĞİŞİKLİK: is_center_in_roi ve roi-filtreli detect_and_crop testleri eklenir
```

---

### Task 1: `session.py` — `SessionConfig`'e ROI alanı ve `metadata.json`'a yazımı

**Files:**
- Modify: `session.py`
- Modify: `tests/test_session.py`

**Interfaces:**
- Consumes: mevcut `SessionConfig`, `write_metadata` (bu task'ta değiştiriliyor)
- Produces:
  - `SessionConfig` artık `roi: tuple[int, int, int, int] | None = None` alanına sahip (varsayılan `None`, mevcut çağıranlar bozulmaz)
  - `write_metadata`, `metadata.json`'a `roi` alanını `{"x1": .., "y1": .., "x2": .., "y2": ..}` (roi doluysa) veya `null` (roi `None` ise) olarak yazar

- [ ] **Step 1: Başarısız testleri yaz**

`tests/test_session.py` dosyasının sonuna ekle:

```python
def test_write_metadata_includes_roi_when_set(tmp_path):
    session_dir = tmp_path / "test_seti"
    session_dir.mkdir()
    config = SessionConfig(
        purpose="model egitimi",
        rtsp_url="rtsp://kamera/1",
        interval=30,
        confidence=0.5,
        image_format="jpg",
        roi=(10, 20, 110, 220),
    )

    metadata_path = write_metadata(session_dir, config)

    import json

    data = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert data["roi"] == {"x1": 10, "y1": 20, "x2": 110, "y2": 220}


def test_write_metadata_roi_is_null_when_not_set(tmp_path):
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

    import json

    data = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert data["roi"] is None
```

- [ ] **Step 2: Testlerin başarısız olduğunu doğrula**

Run: `conda run -n cvstack313 python -m pytest tests/test_session.py -v`
Expected: FAIL — `test_write_metadata_includes_roi_when_set` bir `TypeError` ile başarısız olur (`roi` `SessionConfig`'in bilinmeyen bir alanı), `test_write_metadata_roi_is_null_when_not_set` ise `KeyError: 'roi'` ile başarısız olur (henüz `metadata.json`'da `roi` anahtarı yok).

- [ ] **Step 3: `session.py`'yi güncelle**

`session.py`'deki `SessionConfig` dataclass'ını şu hale getir:

```python
@dataclass
class SessionConfig:
    purpose: str
    rtsp_url: str
    interval: int
    confidence: float
    image_format: str
    roi: tuple[int, int, int, int] | None = None
```

`write_metadata` fonksiyonunu şu hale getir:

```python
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
```

(Diğer her şey — `create_session_dir`, `generate_filename`, importlar — aynı kalır.)

- [ ] **Step 4: Testlerin geçtiğini doğrula**

Run: `conda run -n cvstack313 python -m pytest tests/test_session.py -v`
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add session.py tests/test_session.py
git commit -m "feat: add optional ROI field to session config and metadata"
```

---

### Task 2: `detector.py` — ROI merkez-nokta filtreleme

**Files:**
- Modify: `detector.py`
- Modify: `tests/test_detector.py`

**Interfaces:**
- Consumes: mevcut `crop_detections`, `PersonDetector` (bu task'ta değiştiriliyor)
- Produces:
  - `is_center_in_roi(box: tuple[int, int, int, int], roi: tuple[int, int, int, int]) -> bool` — box'ın merkez noktasının ROI'nin `[x1, x2) x [y1, y2)` yarı-açık aralığında olup olmadığını döner
  - `PersonDetector.detect_and_crop(self, frame: np.ndarray, roi: tuple[int, int, int, int] | None = None) -> list[np.ndarray]` — `roi` verilirse kutular crop'lanmadan önce `is_center_in_roi` ile filtrelenir; `roi=None` ise mevcut (filtresiz) davranış korunur

- [ ] **Step 1: Başarısız testleri yaz**

`tests/test_detector.py` dosyasının sonuna ekle:

```python
from detector import is_center_in_roi


def test_is_center_in_roi_true_when_center_inside():
    assert is_center_in_roi((10, 10, 20, 20), (0, 0, 100, 100)) is True


def test_is_center_in_roi_false_when_center_outside():
    assert is_center_in_roi((200, 200, 220, 220), (0, 0, 100, 100)) is False


def test_is_center_in_roi_left_edge_inclusive():
    assert is_center_in_roi((0, 40, 20, 60), (10, 0, 100, 100)) is True


def test_is_center_in_roi_right_edge_exclusive():
    assert is_center_in_roi((90, 40, 110, 60), (0, 0, 100, 100)) is False


def test_person_detector_detect_and_crop_filters_by_roi():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    frame[0:10, 0:10] = 200
    frame[90:100, 90:100] = 150
    fake_model = _FakeModel(boxes_per_call=[(0, 0, 10, 10), (90, 90, 100, 100)])
    detector = PersonDetector(confidence=0.5, model=fake_model)

    crops = detector.detect_and_crop(frame, roi=(0, 0, 50, 50))

    assert len(crops) == 1
    assert crops[0].shape == (10, 10, 3)
    assert (crops[0] == 200).all()


def test_person_detector_detect_and_crop_roi_none_keeps_all():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    frame[0:10, 0:10] = 200
    frame[90:100, 90:100] = 150
    fake_model = _FakeModel(boxes_per_call=[(0, 0, 10, 10), (90, 90, 100, 100)])
    detector = PersonDetector(confidence=0.5, model=fake_model)

    crops = detector.detect_and_crop(frame, roi=None)

    assert len(crops) == 2
```

Not: `_FakeModel`, `_FakeBox`, `_FakeResult` sınıfları dosyanın üst kısmında zaten mevcut — yeniden tanımlamana gerek yok, sadece yeni testleri dosyanın sonuna ekle. `import numpy as np` da dosyanın en üstünde zaten mevcut.

- [ ] **Step 2: Testlerin başarısız olduğunu doğrula**

Run: `conda run -n cvstack313 python -m pytest tests/test_detector.py -v`
Expected: FAIL — `is_center_in_roi` testleri `ImportError`/`ImportError: cannot import name 'is_center_in_roi'` ile, roi-filtreli `detect_and_crop` testleri ise `TypeError: detect_and_crop() got an unexpected keyword argument 'roi'` ile başarısız olur.

- [ ] **Step 3: `detector.py`'yi güncelle**

`detector.py`'ye `crop_detections`'dan sonra, `PersonDetector` sınıfından önce şu fonksiyonu ekle:

```python
def is_center_in_roi(
    box: tuple[int, int, int, int], roi: tuple[int, int, int, int]
) -> bool:
    bx1, by1, bx2, by2 = box
    rx1, ry1, rx2, ry2 = roi
    cx = (bx1 + bx2) / 2
    cy = (by1 + by2) / 2
    return rx1 <= cx < rx2 and ry1 <= cy < ry2
```

`PersonDetector.detect_and_crop` metodunu şu hale getir:

```python
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
```

(Dosyanın geri kalanı — `crop_detections`, `PersonDetector.__init__` — aynı kalır.)

- [ ] **Step 4: Testlerin geçtiğini doğrula**

Run: `conda run -n cvstack313 python -m pytest tests/test_detector.py -v`
Expected: 11 passed

- [ ] **Step 5: Commit**

```bash
git add detector.py tests/test_detector.py
git commit -m "feat: add ROI center-point filtering to person detection"
```

---

### Task 3: `main.py` — ROI seçim akışı

**Files:**
- Modify: `main.py`

**Interfaces:**
- Consumes:
  - `session.SessionConfig` (artık `roi` alanı var), `session.create_session_dir`, `session.write_metadata`, `session.generate_filename` — Task 1'den
  - `detector.PersonDetector.detect_and_crop(frame, roi=...)` — Task 2'den
  - `rtsp_source.RtspFrameSource` (`frames()`, `close()`) — değişmedi
- Produces:
  - `to_xyxy(roi_xywh: tuple[int, int, int, int]) -> tuple[int, int, int, int] | None` — `(x, y, w, h)`'yi `(x1, y1, x2, y2)`'ye çevirir, `w<=0` veya `h<=0` ise `None` döner
  - `select_roi(frame: np.ndarray) -> tuple[int, int, int, int] | None` — `cv2.selectROI` penceresi açar, `to_xyxy` ile dönüştürüp döner
  - `main()` — güncellenmiş akış

Bu task, `main.py`'nin tamamını aşağıdaki içerikle değiştirir (mevcut dosyanın yerine geçer):

- [ ] **Step 1: `main.py`'yi tamamen şu içerikle değiştir**

```python
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


def select_roi(frame: np.ndarray) -> tuple[int, int, int, int] | None:
    roi_xywh = cv2.selectROI(_ROI_WINDOW_NAME, frame, showCrosshair=True)
    cv2.destroyWindow(_ROI_WINDOW_NAME)
    return to_xyxy(tuple(int(v) for v in roi_xywh))


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
```

- [ ] **Step 2: Modülün sözdizimsel olarak doğru olduğunu kontrol et**

Run: `conda run -n cvstack313 python -c "import ast; ast.parse(open('main.py', encoding='utf-8').read())"`
Expected: hata vermeden çıkar

- [ ] **Step 3: Mevcut tüm testlerin hâlâ geçtiğini doğrula**

Run: `conda run -n cvstack313 python -m pytest tests/ -v`
Expected: 22 passed

- [ ] **Step 4: Commit**

```bash
git add main.py
git commit -m "feat: add interactive ROI selection to main entry point"
```

- [ ] **Step 5: Manuel doğrulama notu (kullanıcı tarafından, gerçek RTSP kamerayla)**

```bash
conda run -n cvstack313 python main.py
```

çalıştırılıp "ROI secmek ister misiniz?" sorusuna "e" cevabı verildiğinde: bir OpenCV penceresinin açıldığı, fare ile çizilen dikdörtgenin Enter/Space ile onaylanabildiği, `metadata.json`'da `roi` alanının doğru `{x1,y1,x2,y2}` değerleriyle yazıldığı ve sadece ROI içinde kalan kişilerin crop'landığı gözlemlenerek doğrulanır. "h"/boş cevap verildiğinde önceki (ROI'siz) davranışın aynen sürdüğü de ayrıca doğrulanır.

---

## Self-Review Notu

- **Spec kapsamı:** ROI sorusu/varsayılan hayır, fare ile seçim, ESC/sıfır-boyut iptali, merkez-nokta filtre kuralı (yarı-açık aralık), `metadata.json`'a `roi` yazımı, `to_xyxy`/`select_roi`'nin `main.py`'de yaşaması, `is_center_in_roi`'nin `detector.py`'de yaşaması — tümü Task 1-3'te karşılanıyor.
- **Placeholder taraması:** Yok — tüm kod adımları tam içerik içeriyor.
- **Tip/isim tutarlılığı:** `SessionConfig.roi`, `write_metadata`, `is_center_in_roi`, `PersonDetector.detect_and_crop(frame, roi=...)`, `to_xyxy`, `select_roi` tüm görevlerde aynı imzalarla kullanılıyor.
- **Test sayısı:** Task 1 sonrası `test_session.py` 7 test (5 mevcut + 2 yeni); Task 2 sonrası `test_detector.py` 11 test (5 mevcut + 6 yeni); `test_rtsp_source.py` bu planda değişmiyor, 4 test. Task 3 sonrası tam suite: 7 + 11 + 4 = **22 passed**.
