## Instructions to run frame extraction (recorded video -> dataset folder)

Every other collector here reads a live RTSP stream, so footage that was already
recorded — a site visit, an NVR export — had no way into a dataset. `extract_frames.py`
decodes a video file and saves whole frames into a flat folder at a rate you choose,
ready to be labelled. Run it from the repo root:

```powershell
# 2 frames per second of video
python extract_frames.py --video clip.mp4 --out .\datasets\apron_batch_2 --crops-per-second 2

# one frame every 5 seconds, only 1:00 -> 4:30, capped at 500 images, downscaled to 1280px
python extract_frames.py --video clip.mp4 --out .\datasets\lobby -r 0.2 --start 60 --end 270 ^
    --max-frames 500 --max-width 1280
```

Frames are saved **whole and unmodified** — no detection, no cropping. Each image is
named after its position in the video (`clip_t00012500.jpg` = 12.5 s in), so you can
scrub straight to that moment when a label looks wrong, and a `metadata.json` recording
the source video and settings is written next to the images.

The output folder is exactly what `split_images.py` expects, so a batch goes straight to
the annotators:

```powershell
python split_images.py --src .\datasets\apron_batch_2 --dest .\datasets\assigned --n 5
```

If the video reports a wrong or missing frame rate (common on NVR exports) the tool
refuses to guess — pass the real one with `--source-fps 25`. Run
`python extract_frames.py --help` for the png/quality and other options.

## Instructions to split an annotated dataset (Roboflow export -> train/test)

Roboflow can export a dataset as **100% train**, which is what you want when the split
should be decided here rather than in the browser — but that export cannot be trained on
as-is: Ultralytics needs a validation set, and `data.yaml` has nowhere to point.
`split_dataset.py` turns it into an 80/20 `train/` + `test/` dataset.

Do **not** reach for `split_images.py` here. It splits a flat image folder for annotator
assignment, and pointed at a YOLO dataset it copies the images while silently dropping
every `.txt` label — only noticed once a training run produces garbage.

```powershell
# look at the split and its warnings first; nothing is written
python split_dataset.py --src .\datasets\apron_roboflow --dest .\datasets\apron_split --dry-run

# 80/20
python split_dataset.py --src .\datasets\apron_roboflow --dest .\datasets\apron_split
```

Our images come out of `extract_frames.py` sampling video a few frames per second, so
consecutive frames are near-identical. A plain shuffle would put frame N in train and
frame N+1 in test, and the reported mAP would be measured on images the model had
effectively memorised. So images are **grouped before they are split**: Roboflow's
augmented copies of one source image, and frames from the same video within
`--group-seconds` (default 10) of each other, always stay on the same side. Whole groups
are then assigned rarest-class-first, so every class still lands in the test set — and if
one does not, the script says so loudly rather than handing you a class whose mAP is
undefined.

The generated `data.yaml` points **both** `val:` and `test:` at `test/images` on purpose:
there are only two folders on disk, but Ultralytics validates against `val:` during
training and fails the run without it. `path:` is written absolute — it is the one line to
edit if you copy the dataset to the GPU server. A `split_report.json` records the seed,
the grouping and the per-class instance counts per split, which is the table to show
alongside the metrics.

```powershell
yolo detect train data=.\datasets\apron_split\data.yaml model=yolo11n.pt epochs=100 imgsz=640
```

The source export is never modified, so re-splitting with a different `--seed`,
`--val-fraction` or `--group-seconds` costs nothing. If the tool reports an empty test
split, the groups are coarser than the holdout — lower `--group-seconds`. Run
`python split_dataset.py --help` for the other options.

## Instructions to run GPU optimization (TensorRT)

To relieve GPU load, convert the two `.pt` models to optimized FP16 TensorRT
`.engine` files. TensorRT runs the models ~2x faster and uses roughly half the VRAM,
which keeps the inference server from congesting.

**Build the engines on the GPU server** (not a dev laptop) — a TensorRT engine only
loads on the exact GPU model + TensorRT/CUDA version it was built on:

```powershell
# First time only: install the export-side deps (tensorrt/onnx/onnxslim)
python scripts\optimize_to_engine.py --install-deps

# Convert both models (best.pt -> best.engine, yolov8m.pt -> yolov8m.engine)
python scripts\optimize_to_engine.py
```

Then run the pipeline against the engines — no other change needed, Ultralytics loads
a `.engine` transparently:

```powershell
python scripts\seat_zone_detection.py --rtsp-url "rtsp://user:pass@host:554/chX/0" ^
    --classifier best.engine --detector yolov8m.engine --show
```

The classifier engine is built with a max dynamic batch of 32 (person crops per
frame). If a scene ever exceeds that, rebuild with a higher `--batch`. Run
`python scripts\optimize_to_engine.py --help` for FP32/INT8 and other options.