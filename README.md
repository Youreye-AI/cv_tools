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