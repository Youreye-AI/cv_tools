# ROI Penceresi için Otomatik Küçültme — Tasarım

**Tarih:** 2026-07-17
**Durum:** Onaylandı
**İlgili:** [2026-07-17-roi-selection-design.md](2026-07-17-roi-selection-design.md)

## Amaç

ROI seçim penceresi (`cv2.selectROI`), RTSP kameranın çözünürlüğüne göre bazen monitöre
sığmayacak kadar büyük açılıyor. Frame'i, seçim penceresinde gösterilmeden önce en-boy oranı
korunarak monitöre sığacak bir maksimum boyuta küçültmek gerekiyor. Kullanıcının küçültülmüş
görsel üzerinde seçtiği ROI, orijinal frame koordinatlarına geri ölçeklenmeli — böylece tespit
filtreleme mantığı (`is_center_in_roi`) hep orijinal çözünürlükte doğru çalışmaya devam eder.

## Tasarım

- Maksimum gösterim boyutu sabit: **1280x720**.
- `main.py`'ye yeni saf fonksiyon eklenir:
  `compute_resize_scale(height: int, width: int, max_width: int = 1280, max_height: int = 720) -> float`
  — `min(max_width / width, max_height / height, 1.0)` formülüyle en-boy oranını koruyan ölçek
  faktörünü döner. Frame zaten maksimum boyuttan küçükse/eşitse `1.0` döner (büyütme yapılmaz).
- `select_roi(frame)` şu şekilde güncellenir:
  1. `height, width = frame.shape[:2]`
  2. `scale = compute_resize_scale(height, width)`
  3. `scale != 1.0` ise `cv2.resize` ile `(int(width*scale), int(height*scale))` boyutunda bir
     gösterim kopyası oluşturulur; `scale == 1.0` ise orijinal frame doğrudan kullanılır.
  4. `cv2.selectROI` bu gösterim kopyası üzerinde çalıştırılır.
  5. Dönen `(x, y, w, h)` değerlerinin her biri `1/scale` ile çarpılıp (`int()`'e yuvarlanarak)
     orijinal frame koordinatlarına geri ölçeklenir.
  6. Ölçeklenmiş `(x, y, w, h)` `to_xyxy`'ye verilir (mevcut davranış — sıfır/negatif
     boyut → `None`).

## Kapsam Dışı

- Maksimum boyutun terminalden/konfigürasyondan özelleştirilmesi — sabit 1280x720.
- Gerçek ekran çözünürlüğünün dinamik olarak algılanması.

## Test Planı

- `compute_resize_scale`: frame zaten maksimum boyuttan küçük/eşit (ölçek 1.0, büyütme yok),
  frame yatayda taşıyor (genişliğe göre ölçekleniyor), frame dikeyde taşıyor (yüksekliğe göre
  ölçekleniyor), frame her iki boyutta da taşıyor (küçük olan oran kullanılıyor) durumları birim
  testle doğrulanır.
- `select_roi`'nin gerçek `cv2.resize`/`cv2.selectROI` çağrıları içeren kısmı, önceki ROI
  seçimi gibi gerçek GUI gerektirdiğinden otomatik test kapsamı dışındadır; kullanıcı tarafından
  manuel doğrulanır.
