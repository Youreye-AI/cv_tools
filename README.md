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