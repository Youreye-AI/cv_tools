# ROI Penceresi için Otomatik Küçültme Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** ROI seçim penceresinde gösterilen frame'i, monitöre sığacak şekilde en-boy oranı korunarak küçültmek; kullanıcının küçültülmüş görsel üzerinde seçtiği ROI'yi orijinal frame koordinatlarına geri ölçeklemek.

**Architecture:** `main.py`'ye yeni bir saf fonksiyon `compute_resize_scale` eklenir (birim test edilir); `select_roi` bu fonksiyonu kullanarak gösterim öncesi `cv2.resize` uygular ve seçilen koordinatları geri ölçekler.

**Tech Stack:** Python 3.13 (conda ortamı: `cvstack313`), `cv2.resize`, `pytest`.

## Global Constraints

- Hedef conda ortamı: **cvstack313** (Python 3.13.12); test komutları `conda run -n cvstack313 python -m pytest ...` ile çalıştırılır.
- Maksimum gösterim boyutu sabit: **1280x720** (genişlik x yükseklik).
- `compute_resize_scale(height, width, max_width=1280, max_height=720) -> float` — en-boy oranını koruyan ölçek faktörünü `min(max_width/width, max_height/height, 1.0)` formülüyle döner; frame zaten sığıyorsa `1.0` (büyütme yapılmaz).
- Kullanıcının küçültülmüş görsel üzerinde seçtiği `(x, y, w, h)` değerleri `1/scale` ile orijinal koordinatlara geri çevrilip mevcut `to_xyxy`'ye verilir (sıfır/negatif boyut → `None` davranışı değişmez).
- `select_roi`'nin gerçek `cv2.resize`/`cv2.selectROI` içeren kısmı otomatik test kapsamı dışındadır (gerçek GUI gerektirir); `compute_resize_scale` saf fonksiyondur ve otomatik test edilir.
- Kapsam dışı: maksimum boyutun konfigüre edilebilir olması, gerçek ekran çözünürlüğü algılama.

---

## Dosya Yapısı

```
cv_tools/
  main.py                # DEĞİŞİKLİK: compute_resize_scale (yeni), select_roi güncellenir
  tests/
    test_main.py          # YENİ: compute_resize_scale testleri (main.py'nin ilk test dosyası)
```

---

### Task 1: `compute_resize_scale` + `select_roi` güncellemesi

**Files:**
- Modify: `main.py`
- Create: `tests/test_main.py`

**Interfaces:**
- Consumes: mevcut `to_xyxy` (bu task'ta değişmiyor, sadece `select_roi` içinden çağrılma şekli değişiyor)
- Produces:
  - `compute_resize_scale(height: int, width: int, max_width: int = 1280, max_height: int = 720) -> float`
  - `select_roi(frame: np.ndarray) -> tuple[int, int, int, int] | None` — davranışı: gösterim öncesi ölçekleme, seçim sonrası geri ölçekleme (imza değişmiyor)

- [ ] **Step 1: Başarısız testleri yaz**

`tests/test_main.py` (yeni dosya):

```python
from main import compute_resize_scale


def test_compute_resize_scale_no_scaling_when_already_fits():
    scale = compute_resize_scale(height=480, width=640, max_width=1280, max_height=720)

    assert scale == 1.0


def test_compute_resize_scale_no_scaling_when_exactly_fits():
    scale = compute_resize_scale(height=720, width=1280, max_width=1280, max_height=720)

    assert scale == 1.0


def test_compute_resize_scale_scales_down_when_width_overflows():
    scale = compute_resize_scale(height=720, width=2560, max_width=1280, max_height=720)

    assert scale == 0.5


def test_compute_resize_scale_scales_down_when_height_overflows():
    scale = compute_resize_scale(height=1440, width=1280, max_width=1280, max_height=720)

    assert scale == 0.5


def test_compute_resize_scale_uses_smaller_ratio_when_both_overflow():
    scale = compute_resize_scale(height=2160, width=3840, max_width=1280, max_height=720)

    assert scale == 1280 / 3840
```

Not: `2560/1280=2.0` -> `1/2.0=0.5`; `2560` yüksekliği 720 sınırını aşmıyor (`720/720=1.0`), bu yüzden genişlik oranı (`0.5`) küçük olan olarak seçilir — test bunu doğrular. Benzer şekilde `height=1440, width=1280`: `1280/1280=1.0` (genişlik sığıyor), `720/1440=0.5` (yükseklik taşıyor) — küçük olan `0.5` seçilir. `height=2160, width=3840` (4K): `1280/3840≈0.333`, `720/2160≈0.333` — bu örnekte ikisi de eşit (4K'nın 16:9 oranı 1280x720 ile aynı), bu yüzden `1280/3840` ile karşılaştır.

- [ ] **Step 2: Testlerin başarısız olduğunu doğrula**

Run: `conda run -n cvstack313 python -m pytest tests/test_main.py -v`
Expected: FAIL with `ImportError: cannot import name 'compute_resize_scale'`

- [ ] **Step 3: `main.py`'de `compute_resize_scale`'i ekle ve `select_roi`'yi güncelle**

`main.py`'deki `to_xyxy` fonksiyonundan sonra, `select_roi`'den önce şu fonksiyonu ekle:

```python
def compute_resize_scale(
    height: int, width: int, max_width: int = 1280, max_height: int = 720
) -> float:
    return min(max_width / width, max_height / height, 1.0)
```

`select_roi` fonksiyonunu şu hale getir:

```python
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
```

(Dosyanın geri kalanı — `ask`, `to_xyxy`, `main()` — aynı kalır.)

- [ ] **Step 4: Testlerin geçtiğini doğrula**

Run: `conda run -n cvstack313 python -m pytest tests/test_main.py -v`
Expected: 5 passed

- [ ] **Step 5: Sözdizimi kontrolü ve tam suite'in hâlâ geçtiğini doğrula**

Run: `conda run -n cvstack313 python -c "import ast; ast.parse(open('main.py', encoding='utf-8').read())"`
Expected: hata vermeden çıkar

Run: `conda run -n cvstack313 python -m pytest tests/ -v`
Expected: 27 passed

- [ ] **Step 6: Commit**

```bash
git add main.py tests/test_main.py
git commit -m "feat: resize ROI selection window to fit the monitor"
```

- [ ] **Step 7: Manuel doğrulama notu (kullanıcı tarafından, gerçek RTSP kamerayla)**

```bash
conda run -n cvstack313 python main.py
```

çalıştırılıp "ROI secmek ister misiniz?" sorusuna "e" cevabı verildiğinde: yüksek çözünürlüklü
bir kamerada ROI penceresinin artık monitöre sığdığı, seçilen ROI'nin `metadata.json`'a doğru
(orijinal frame çözünürlüğüne göre) koordinatlarla yazıldığı ve tespit filtrelemesinin
(`is_center_in_roi`) hâlâ doğru çalıştığı gözlemlenerek doğrulanır.

---

## Self-Review Notu

- **Spec kapsamı:** Sabit 1280x720 maksimum boyut, en-boy oranı koruma, büyütmeme (`scale<=1.0`), seçilen ROI'nin orijinal koordinatlara geri ölçeklenmesi — tümü Task 1'de karşılanıyor.
- **Placeholder taraması:** Yok — tüm kod adımları tam içerik içeriyor.
- **Tip/isim tutarlılığı:** `compute_resize_scale(height, width, max_width=1280, max_height=720) -> float`, `select_roi(frame) -> tuple|None` imzaları spec ile birebir uyumlu.
- **Test sayısı:** Mevcut tam suite 22 test; Task 1 `tests/test_main.py`'ye 5 yeni test ekliyor → toplam 27.
