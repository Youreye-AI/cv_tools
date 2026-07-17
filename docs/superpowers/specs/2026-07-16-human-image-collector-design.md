# RTSP + YOLO26 İnsan Görseli Toplama Script'i — Tasarım

**Tarih:** 2026-07-16
**Durum:** Onaylandı

## Amaç

CV (computer vision) çalışmaları için RTSP kaynaklı bir kameradan, YOLO26 modeli ile insan
tespiti yapıp tespit edilen kişileri kırpılmış (crop) görseller halinde diske kaydeden bir
Python script'i. Kullanıcı script'i başlattığında hangi amaçla veri topladığını ve kayıtların
gideceği klasör ismini terminalden girer.

## Hedef Ortam

Script, mevcut conda ortamı **cvstack313** (Python 3.13.12) hedeflenerek yazılır. Bu ortamda
`ultralytics` (8.4.34), `torch`/`torchvision` (CUDA 12.8), `numpy` ve `opencv-python` (`cv2`)
zaten kurulu. Doğrulandı: bu ortamdaki `cv2`, **GStreamer 1.28.1 desteğiyle build edilmiş**
(`cv2.getBuildInformation()` çıktısında `GStreamer: YES (1.28.1)`). Bu sayede RTSP akışı
`gi`/PyGObject gerekmeden doğrudan `cv2.VideoCapture(pipeline_str, cv2.CAP_GSTREAMER)` ile
okunabilir — ek bir conda-forge kurulumuna gerek yok, proje tüm bağımlılıklarıyla hazır.

## Kapsam dışı

- Kesintiye uğrayan bir toplama oturumunu kaldığı yerden devam ettirme (resume/pagination).
  Kullanıcı bunu ayrıca kendisi ekleyecek.
- RTSP bağlantısı koptuğunda otomatik yeniden bağlanma.

## Kullanıcı Akışı

1. Script çalıştırılır (`python main.py`).
2. Terminalde sırasıyla sorulur:
   - Veri toplama amacı (serbest metin)
   - Kayıtların gideceği klasör ismi
   - RTSP URL
   - Frame interval N — kaç frame'de bir tespit/kayıt yapılacağı (varsayılan: 30)
   - Confidence eşiği (varsayılan: 0.5)
   - Görsel formatı: jpg/png (varsayılan: jpg)
3. `<isim>/` klasörü oluşturulur (zaten varsa hata verir, üzerine yazmaz).
4. `<isim>/metadata.json` dosyası yazılır: amaç, oluşturma tarihi/saati, RTSP URL, interval,
   confidence, format.
5. `cv2.VideoCapture`, bir GStreamer pipeline string'i (`rtspsrc ! decodebin ! videoconvert !
   video/x-raw,format=BGR ! appsink`) ile RTSP akışına bağlanır ve BGR numpy frame'leri okur.
6. Her N'inci frame'de YOLO26 ile person (class 0) tespiti çalıştırılır. Confidence eşiğini
   geçen her bbox crop'lanıp `<isim>/<YYYYMMDD_HHMMSS>_<3 haneli index>.<format>` adıyla
   kaydedilir.
7. Ctrl+C ile script durur: pipeline temiz şekilde NULL state'e alınır, toplam kaydedilen
   görsel sayısı terminale yazılır.

## Bileşenler

- **`rtsp_source.py`** — `cv2.VideoCapture(pipeline_str, cv2.CAP_GSTREAMER)` ile RTSP akışına
  bağlanan, BGR numpy frame üreten bir generator arayüzü. Bağlantı kurulamazsa veya akış
  sırasında koparsa (`read()` başarısız olursa) anlaşılır bir hata fırlatır.
- **`detector.py`** — Ultralytics YOLO26 modelini yükler (otomatik indirilir, ör. `yolo26n.pt`),
  verilen frame üzerinde person tespiti yapar, confidence eşiğini geçen bbox'ları crop'layıp
  numpy görsel listesi olarak döner.
- **`main.py`** — Terminal soruları, klasör ve `metadata.json` oluşturma, ana döngü (frame
  sayacı, N'inci frame'de detector çağırma, dosya kaydetme), Ctrl+C (SIGINT) yakalayıp temiz
  kapanış ve özet yazdırma.

## Veri Akışı

```
RTSP kamera → rtsp_source (cv2.VideoCapture + GStreamer backend) → frame (numpy, BGR)
   → [her N. frame] → detector (YOLO26 person tespiti) → crop'lar
   → main (dosya adı üretimi) → <isim>/<timestamp>_<index>.<format>
```

## Hata Yönetimi

- RTSP bağlantısı kurulamazsa (`VideoCapture.isOpened()` False) veya akış sırasında koparsa
  (`read()` başarısız): anlaşılır bir hata mesajıyla `RuntimeError` fırlatılır, terminale
  yazılır, script düzgün şekilde sonlanır (retry/resume yok — kapsam dışı).
- Klasör zaten varsa: script hata verip çıkar, üzerine yazmaz (veri kaybını önlemek için).
- YOLO modeli ilk çalıştırmada indirilemezse (ağ hatası vb.): hata mesajı gösterilip script
  sonlanır.

## Test Planı

- `session.py` (klasör/metadata/dosya adı) ve `detector.py` (crop mantığı) bağımlılıksız birim
  testlerle doğrulanır.
- `rtsp_source.py`, `cv2.VideoCapture` nesnesinin enjekte edilebilmesi sayesinde sahte (fake)
  bir capture nesnesiyle birim testle doğrulanır — gerçek RTSP bağlantısı gerekmez.
- Gerçek RTSP + kamera ile uçtan uca doğrulama kullanıcı tarafından yapılacaktır.
