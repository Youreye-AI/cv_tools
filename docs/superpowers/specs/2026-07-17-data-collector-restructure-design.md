# `data_collector/` Klasör Yapısına Geçiş — Tasarım

**Tarih:** 2026-07-17
**Durum:** Onaylandı

## Amaç

`cv_tools` reposu ileride birden fazla bağımsız CV aracı barındıracak. Mevcut RTSP + YOLO26
insan görseli toplama aracı, gelecekteki araçlarla aynı klasör seviyesinde kardeş bir klasör
(`data_collector/`) altına taşınacak şekilde yeniden düzenlenir.

## Yeni Klasör Yapısı

```
cv_tools/
  requirements.txt        # kökte kalır — tüm araçların bağımlılıkları burada birikir
  docs/                    # proje geneli spec/plan dokümanları, kökte kalmaya devam eder
  data_collector/
    main.py
    session.py
    detector.py
    rtsp_source.py
    yolo26n.pt              # model dosyası artık aracın yanında
  # gelecekte: baska_arac/ (data_collector/ ile aynı seviyede kardeş klasör)
```

**Not:** `tests/` klasörü şu an ne diskte ne de git'te mevcut (proje sahibi tarafından ayrı bir
commit'te bilerek kaldırıldı). Bu yüzden bu tasarımda taşınacak bir test dosyası yok; taşıma
sadece kaynak kodu ve model dosyasını kapsıyor.

## Taşınacaklar

`git mv` ile (dosya içerikleri değişmeden, git geçmişi korunarak):
- `main.py` → `data_collector/main.py`
- `session.py` → `data_collector/session.py`
- `detector.py` → `data_collector/detector.py`
- `rtsp_source.py` → `data_collector/rtsp_source.py`
- `yolo26n.pt` → `data_collector/yolo26n.pt`

## Dokunulmayacaklar

- `requirements.txt` — kökte kalır.
- `docs/` — kökte kalır.
- `alfu_black_apron_17_07/`, `test/` (kullanıcının deneme oturum klasörleri) — dokunulmaz.

## Etkiler

- Modüller arası importlar (`from detector import ...` vb.) değişmiyor çünkü tüm modüller
  birlikte aynı klasöre taşınıyor.
- Script çalıştırma: `conda run -n cvstack313 python data_collector/main.py` (ya da
  `data_collector/` içine `cd` edip `python main.py`).
- `.gitignore`'daki proje-geneli girdiler (`__pycache__/`, `*.pyc`, vb.) değişmeden kalır —
  hem kökte hem `data_collector/` altında oluşan `__pycache__` klasörlerini zaten kapsar.

## Kapsam Dışı

- Yeni bir araç eklenmesi (`baska_arac/` gibi) — bu tasarım sadece mevcut aracı taşımayı
  kapsıyor, yeni araç eklenmesi ayrı bir görev/tasarım olacak.
- `requirements.txt`'in araç bazlı bölümlere ayrılması — şimdilik tek, birleşik dosya olarak
  kalıyor.

## Test Planı

Bu bir taşıma (dosya konumu değişikliği) işlemi olduğundan yeni davranış eklenmiyor ve şu an
otomatik test bulunmuyor (bkz. yukarıdaki not). Doğrulama, taşıma sonrası script'in hâlâ
sözdizimsel olarak doğru olduğu ve importların çalıştığı kontrol edilerek yapılır:
`conda run -n cvstack313 python -c "import ast; ast.parse(open('data_collector/main.py', encoding='utf-8').read())"`.
