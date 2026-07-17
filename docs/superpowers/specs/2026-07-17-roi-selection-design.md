# ROI (İlgi Alanı) Seçimi — Tasarım

**Tarih:** 2026-07-17
**Durum:** Onaylandı
**İlgili:** [2026-07-16-human-image-collector-design.md](2026-07-16-human-image-collector-design.md)

## Amaç

RTSP + YOLO26 insan görseli toplama script'ine opsiyonel bir ROI (region of interest / ilgi
alanı) seçme özelliği eklemek. Kullanıcı frame içinde bir dikdörtgen bölge seçtiğinde, sadece bu
bölgeyle örtüşen kişi tespitleri kaydedilir — böylece kadrajın istenmeyen kısımlarındaki (ör.
arka plandan geçen kişiler) tespitler filtrelenebilir.

## Kullanıcı Akışı

1. RTSP bağlantısı kurulur ve ilk frame okunur (mevcut ana döngüden önce, ayrı olarak).
2. Terminalde sorulur: "ROI secmek ister misiniz? (e/H)" — varsayılan boş cevap "hayır" olarak
   yorumlanır (mevcut davranış: tüm frame kullanılır, ROI yok).
3. "e"/"evet" cevabı verilirse: ilk frame bir OpenCV penceresinde (`cv2.selectROI`) gösterilir.
   Kullanıcı fare ile bir dikdörtgen çizip Enter/Space'e basar; ESC'ye basılırsa veya sıfır
   boyutlu bir seçim yapılırsa ROI seçilmemiş sayılır (tüm frame kullanılır). Pencere
   `cv2.destroyWindow` ile kapatılır.
4. Seçilen ROI `(x1, y1, x2, y2)` formatında saklanır ve `metadata.json`'a `roi` alanı olarak
   yazılır: `{"x1": .., "y1": .., "x2": .., "y2": ..}` ya da ROI seçilmediyse `null`.
5. Ana döngüde (ilk frame dahil), her N'inci frame'de YOLO26 tüm frame üzerinde person tespiti
   yapmaya devam eder. ROI tanımlıysa, tespit edilen bir kişinin bounding box'ının **merkez
   noktası** ROI dikdörtgeni içindeyse crop'lanıp kaydedilir; ROI dışındaysa atlanır. ROI
   tanımlı değilse mevcut davranış (tüm tespitler kaydedilir) aynen sürer.

## Bileşenler

- **`roi.py`** (yeni modül):
  - `to_xyxy(roi_xywh: tuple[int, int, int, int]) -> tuple[int, int, int, int] | None` — saf
    fonksiyon. `cv2.selectROI`'nin döndürdüğü `(x, y, w, h)` formatını `(x1, y1, x2, y2)`'ye
    çevirir; `w <= 0` veya `h <= 0` ise (iptal edilmiş/sıfır boyutlu seçim) `None` döner.
  - `select_roi(frame: np.ndarray) -> tuple[int, int, int, int] | None` — gerçek bir
    `cv2.selectROI` penceresi açar, kullanıcı seçimini `to_xyxy` ile dönüştürüp döner. Gerçek
    GUI etkileşimi gerektirdiğinden otomatik test kapsamı dışındadır (kapsam dışı bölümüne bkz.).

- **`detector.py`** (değişiklik):
  - Yeni saf fonksiyon `is_center_in_roi(box: tuple[int, int, int, int], roi: tuple[int, int, int, int]) -> bool`
    — box'ın merkez noktasının `(cx, cy)` ROI dikdörtgeni `[x1, x2) x [y1, y2)` içinde olup
    olmadığını döner.
  - `PersonDetector.detect_and_crop(self, frame: np.ndarray, roi: tuple[int, int, int, int] | None = None) -> list[np.ndarray]`
    — imzaya opsiyonel `roi` parametresi eklenir. `roi is not None` ise, kutular
    `crop_detections`'a verilmeden önce `is_center_in_roi` ile filtrelenir. `roi is None` ise
    mevcut davranış (filtresiz) korunur.

- **`session.py`** (değişiklik):
  - `SessionConfig`'e `roi: tuple[int, int, int, int] | None = None` alanı eklenir (varsayılan
    `None`, mevcut çağıranlar bozulmaz).
  - `write_metadata`, `config.roi` doluysa `{"x1": .., "y1": .., "x2": .., "y2": ..}` sözlüğünü,
    boşsa `null` değerini `metadata.json`'daki `roi` alanına yazar.

- **`main.py`** (değişiklik):
  - `RtspFrameSource` oluşturulduktan sonra, ana döngüden önce `frames()` generator'ından ilk
    frame ayrı okunur.
  - ROI sorusu terminalden sorulur; "e"/"evet" ise `roi.select_roi(first_frame)` çağrılır,
    sonucu (olabilir `None`) `roi` değişkeninde tutulur.
  - `SessionConfig`'e `roi=roi` geçirilir (metadata'ya yazılması için); bu nedenle
    `write_metadata` çağrısı ROI seçiminden SONRA yapılacak şekilde main.py'nin akış sırası
    güncellenir.
  - Ana döngü, ilk frame'i de işleyecek şekilde (frame_index=1'den başlayarak) güncellenir;
    ardından `frames()` generator'ının kalanı üzerinden devam eder (aynı RTSP bağlantısı, ikinci
    bir capture açılmaz).
  - Her `detector.detect_and_crop(frame)` çağrısı `detector.detect_and_crop(frame, roi=roi)`
    olarak güncellenir.

## Kapsam Dışı

- ROI'nin oturum sırasında yeniden tanımlanması (script çalışırken ROI değiştirme) — sadece
  başlangıçta bir kez seçilir.
- Çoklu/dairesel/poligon ROI şekilleri — yalnızca tek bir dikdörtgen ROI desteklenir.
- `select_roi`'nin otomatik testi — gerçek bir OpenCV GUI penceresi ve fare etkileşimi
  gerektirdiğinden, `rtsp_source.py`'nin gerçek RTSP bağlantısı gibi kapsam dışıdır; kullanıcı
  tarafından manuel doğrulanır.

## Hata Yönetimi

- ROI sorusuna boş cevap (Enter) veya "h"/"hayır" dışında bir şey: sadece "e"/"evet" (büyük/küçük
  harf duyarsız) ROI seçimini tetikler, diğer her şey "hayır" olarak yorumlanır.
- `select_roi` içinde ESC'ye basılırsa veya sıfır boyutlu seçim yapılırsa: `to_xyxy` `None`
  döner, script tüm frame ile (ROI'siz) devam eder — hata fırlatılmaz.

## Test Planı

- `to_xyxy`: geçerli `(x, y, w, h)` girdisi için doğru `(x1, y1, x2, y2)` dönüşümü; `w=0`/`h=0`/
  negatif genişlik-yükseklik için `None` dönüşü birim testle doğrulanır.
- `is_center_in_roi`: merkez ROI içinde, merkez ROI dışında, merkez tam sınırda (dahil/hariç
  davranışı net tanımlanmış) durumları birim testle doğrulanır.
- `PersonDetector.detect_and_crop(frame, roi=...)`: mevcut sahte (fake) model altyapısı
  kullanılarak, ROI içindeki ve dışındaki kutuların doğru filtrelendiği birim testle doğrulanır;
  `roi=None` durumunda mevcut (filtresiz) davranışın bozulmadığı da test edilir.
- `select_roi` ve `main.py`'nin güncellenmiş akışı: gerçek RTSP + gerçek GUI gerektirdiğinden
  kullanıcı tarafından manuel doğrulanır.
