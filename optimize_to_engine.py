"""Convert Ultralytics YOLO `.pt` weights into optimized TensorRT `.engine` files.

Why
---
The live pipelines (`seat_zone_detection.py`, `rtsp_waiter_client_detection.py`) run
two heavyweight FP32 PyTorch models per frame — a person *detector* (`yolov8m.pt`)
and a client/waiter *classifier* (`best.pt`). Under load the GPU server congests and
occasionally goes down. TensorRT rebuilds each model into a fused, FP16 `.engine`
that runs ~2x faster and uses roughly half the VRAM, relieving that pressure.

Nothing downstream has to change: Ultralytics loads a `.engine` transparently, so the
existing scripts pick the engines up by pointing their weight args at them, e.g.

    python scripts/seat_zone_detection.py --rtsp-url "rtsp://.../ch11/0" \
        --classifier best.engine --detector yolov8m.engine --show

IMPORTANT — engines are NOT portable
-------------------------------------
A TensorRT engine is compiled for the EXACT GPU model + TensorRT/CUDA version it was
built on. An engine built on one machine will fail to load on a different GPU. **Build
on the GPU server that will serve inference**, not on a dev laptop.

Run it (ON the server)
----------------------
    ..\\.venv\\Scripts\\Activate.ps1

    # Convert both repo models (best.pt + yolov8m.pt) with FP16 defaults:
    python scripts/optimize_to_engine.py

    # A single model, or arbitrary weights:
    python scripts/optimize_to_engine.py --models path/to/weights.pt

    # First-time deps (tensorrt/onnx/onnxslim) if not already installed:
    python scripts/optimize_to_engine.py --install-deps
"""
from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
import time

logger = logging.getLogger("optimize_to_engine")

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPT_DIR)

# Same weights the detection scripts default to (see seat_zone_detection.py:76-77).
DEFAULT_CLASSIFIER_PATH = os.path.join(_REPO_ROOT, "best.pt")
DEFAULT_DETECTOR_PATH = os.path.join(_REPO_ROOT, "yolov8m.pt")

# best.pt was trained at 224; yolov8m.pt (detect) runs at the YOLO default 640.
CLASSIFY_IMGSZ = 224
DETECT_IMGSZ = 640

# Export packages Ultralytics needs for the .pt -> ONNX -> .engine path. Kept out of
# the base requirements because they're only needed on the machine that BUILDS engines.
_EXPORT_DEPS = ("tensorrt", "onnx", "onnxslim")


# ---------------------------------------------------------------- preflight


def pick_device_index(requested: int) -> int:
    """Return a usable CUDA device index, or exit if no GPU is available.

    TensorRT export and engines both require a GPU; there is no CPU fallback, so a
    missing GPU is a hard error (fail loud) rather than a silent degrade.
    """
    try:
        import torch
    except Exception as exc:  # torch itself missing -> environment is not set up
        logger.error("PyTorch is not importable (%s). Install requirements first.", exc)
        raise SystemExit(2)
    if not torch.cuda.is_available():
        logger.error(
            "No CUDA GPU available. TensorRT engines must be built and run on a GPU — "
            "run this on the GPU server, not a CPU-only machine."
        )
        raise SystemExit(2)
    count = torch.cuda.device_count()
    if requested < 0 or requested >= count:
        logger.error("Requested --device %d but only %d CUDA device(s) present.", requested, count)
        raise SystemExit(2)
    name = torch.cuda.get_device_name(requested)
    cc_major, cc_minor = torch.cuda.get_device_capability(requested)
    logger.info("GPU               : cuda:%d  %s (compute %d.%d)", requested, name, cc_major, cc_minor)
    logger.info("torch / CUDA      : %s / %s", torch.__version__, torch.version.cuda)
    return requested


def ensure_export_deps(install: bool) -> None:
    """Verify tensorrt/onnx/onnxslim are importable; optionally pip-install them.

    Fails fast with the exact remediation command instead of letting Ultralytics'
    export raise a deep, confusing traceback when TensorRT is absent.
    """
    missing = []
    for mod in _EXPORT_DEPS:
        try:
            __import__(mod)
        except Exception:
            missing.append(mod)

    if missing and install:
        logger.info("Installing missing export deps: %s", " ".join(missing))
        subprocess.run([sys.executable, "-m", "pip", "install", *missing], check=True)
        missing = [m for m in missing if _still_missing(m)]

    if missing:
        logger.error(
            "Missing packages required to build engines: %s\n"
            "Install them on this GPU server with:\n"
            "    pip install %s\n"
            "or re-run this script with --install-deps.",
            ", ".join(missing), " ".join(_EXPORT_DEPS),
        )
        raise SystemExit(2)

    try:
        import tensorrt as trt
        logger.info("TensorRT          : %s", trt.__version__)
    except Exception:
        pass


def _still_missing(mod: str) -> bool:
    try:
        __import__(mod)
        return False
    except Exception:
        return True


# ---------------------------------------------------------------- per-model config


def resolve_model_config(pt_path: str, args: argparse.Namespace) -> dict:
    """Build the Ultralytics export kwargs for one .pt, keyed off its task.

    classify -> small square input, dynamic batch (the classifier is fed a
                variable-length batch of person crops each frame).
    detect   -> 640 input, static batch=1 (the detector sees one frame per track()).
    CLI flags override the per-task defaults so the module stays generic.
    """
    from ultralytics import YOLO

    model = YOLO(pt_path)
    task = getattr(model, "task", None)

    if task == "classify":
        imgsz = args.imgsz if args.imgsz is not None else CLASSIFY_IMGSZ
        dynamic = True if args.dynamic is None else args.dynamic
        batch = args.batch  # max batch for the dynamic profile
    elif task == "detect":
        imgsz = args.imgsz if args.imgsz is not None else DETECT_IMGSZ
        dynamic = False if args.dynamic is None else args.dynamic
        batch = 1 if not dynamic else args.batch
    else:
        # Unknown/other task (segment, pose, ...): use detect-like defaults and let
        # the operator override with flags.
        logger.warning("Model %s has task=%r; using detect-style defaults.", os.path.basename(pt_path), task)
        imgsz = args.imgsz if args.imgsz is not None else DETECT_IMGSZ
        dynamic = False if args.dynamic is None else args.dynamic
        batch = 1 if not dynamic else args.batch

    return {"model": model, "task": task, "imgsz": imgsz, "dynamic": dynamic, "batch": batch}


# ---------------------------------------------------------------- export + validate


def export_one(pt_path: str, cfg: dict, args: argparse.Namespace, device: int) -> str:
    """Export a single model to .engine and return the output path."""
    half = not args.fp32 and not args.int8  # int8 supersedes fp16
    precision = "int8" if args.int8 else ("fp16" if half else "fp32")

    logger.info("")
    logger.info("=== %s ===", os.path.basename(pt_path))
    logger.info("task=%s  imgsz=%s  dynamic=%s  batch=%s  precision=%s",
                cfg["task"], cfg["imgsz"], cfg["dynamic"], cfg["batch"], precision)

    if args.int8 and not args.data:
        logger.error(
            "--int8 needs calibration data: pass --data <dataset.yaml or image dir>. "
            "Without it, INT8 calibration cannot run."
        )
        raise SystemExit(2)

    export_kwargs = dict(
        format="engine",
        imgsz=cfg["imgsz"],
        dynamic=cfg["dynamic"],
        batch=cfg["batch"],
        half=half,
        int8=args.int8,
        workspace=args.workspace,
        simplify=True,
        device=device,
        verbose=args.verbose,
    )
    if args.data:
        export_kwargs["data"] = args.data

    t0 = time.perf_counter()
    engine_path = cfg["model"].export(**export_kwargs)
    build_s = time.perf_counter() - t0

    engine_path = str(engine_path)
    pt_mb = os.path.getsize(pt_path) / 1e6
    eng_mb = os.path.getsize(engine_path) / 1e6
    logger.info("built in %.1fs -> %s", build_s, engine_path)
    logger.info("size: %.1f MB (.pt) -> %.1f MB (.engine)", pt_mb, eng_mb)
    return engine_path


def validate_and_benchmark(engine_path: str, cfg: dict, device: int, iters: int = 30) -> None:
    """Smoke-test the engine and compare its speed/output against the .pt baseline.

    Uses a synthetic input of the right shape — enough to prove the engine loads and
    runs on this GPU and to measure the latency win. FP16 introduces minor numeric
    drift, so output differences are a warning, not a failure.
    """
    import numpy as np
    from ultralytics import YOLO

    imgsz = cfg["imgsz"]
    # One synthetic BGR frame/crop (uint8 HWC), what the real pipelines pass in.
    sample = (np.random.rand(imgsz, imgsz, 3) * 255).astype("uint8")

    def bench(model) -> float:
        # Warmup (first call builds context / caches), then time steady-state.
        for _ in range(3):
            model.predict(sample, imgsz=imgsz, device=device, verbose=False)
        t0 = time.perf_counter()
        for _ in range(iters):
            model.predict(sample, imgsz=imgsz, device=device, verbose=False)
        return (time.perf_counter() - t0) / iters * 1000.0  # ms/inference

    try:
        eng_model = YOLO(engine_path)
    except Exception as exc:
        logger.error("Engine failed to load — it may not match this GPU/TensorRT: %s", exc)
        raise SystemExit(3)

    eng_ms = bench(eng_model)
    pt_ms = bench(cfg["model"])
    speedup = pt_ms / eng_ms if eng_ms > 0 else float("nan")
    logger.info("latency: %.2f ms (.pt) -> %.2f ms (.engine)  =>  %.2fx faster", pt_ms, eng_ms, speedup)

    # Output sanity on one sample (task-specific).
    try:
        pt_res = cfg["model"].predict(sample, imgsz=imgsz, device=device, verbose=False)[0]
        eng_res = eng_model.predict(sample, imgsz=imgsz, device=device, verbose=False)[0]
        if cfg["task"] == "classify":
            pt_top = int(pt_res.probs.top1)
            eng_top = int(eng_res.probs.top1)
            delta = float(abs(pt_res.probs.data.cpu().numpy() - eng_res.probs.data.cpu().numpy()).max())
            if pt_top != eng_top or delta > 0.15:
                logger.warning("classifier drift: top1 pt=%d engine=%d, max prob delta=%.3f", pt_top, eng_top, delta)
            else:
                logger.info("output check OK: top1 match, max prob delta=%.3f", delta)
        else:
            n_pt = 0 if pt_res.boxes is None else len(pt_res.boxes)
            n_eng = 0 if eng_res.boxes is None else len(eng_res.boxes)
            logger.info("output check: detections pt=%d engine=%d (synthetic noise; counts are indicative)", n_pt, n_eng)
    except Exception as exc:
        logger.warning("output comparison skipped (%s)", exc)


# ---------------------------------------------------------------- driver


def run(args: argparse.Namespace) -> int:
    device = pick_device_index(args.device)
    ensure_export_deps(args.install_deps)

    models = args.models or [DEFAULT_CLASSIFIER_PATH, DEFAULT_DETECTOR_PATH]
    missing = [m for m in models if not os.path.exists(m)]
    if missing:
        logger.error("Model file(s) not found: %s", ", ".join(missing))
        return 1

    logger.info("Reminder: this engine is only valid on THIS GPU + TensorRT version.")

    built = []
    for pt_path in models:
        cfg = resolve_model_config(pt_path, args)
        engine_path = export_one(pt_path, cfg, args, device)
        if args.validate:
            validate_and_benchmark(engine_path, cfg, device)
        built.append(engine_path)

    logger.info("")
    logger.info("Done. Built %d engine(s):", len(built))
    for p in built:
        logger.info("  %s", p)
    logger.info("Point the pipeline at them, e.g. --classifier best.engine --detector yolov8m.engine")
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Convert YOLO .pt weights to optimized TensorRT .engine files (build on the GPU server).",
    )
    p.add_argument("--models", nargs="+", default=None,
                   help="one or more .pt paths to convert; omit to convert both repo models "
                        "(best.pt classifier + yolov8m.pt detector)")
    p.add_argument("--fp32", action="store_true",
                   help="build a full-precision engine (default is FP16, the recommended win)")
    p.add_argument("--int8", action="store_true",
                   help="build an INT8 engine (largest win, needs --data calibration; accuracy risk)")
    p.add_argument("--data", default=None,
                   help="calibration dataset (yaml or image dir) required for --int8")
    p.add_argument("--batch", type=int, default=32,
                   help="max batch size for dynamic engines (classifier); raise if a frame ever "
                        "yields more person crops than this, then rebuild (default 32)")
    p.add_argument("--imgsz", type=int, default=None,
                   help="override input size; default is per-task (224 classify / 640 detect)")
    p.add_argument("--dynamic", type=lambda s: s.lower() in ("1", "true", "yes"), default=None,
                   help="override dynamic-shape mode (true/false); default per-task")
    p.add_argument("--workspace", type=float, default=4.0,
                   help="TensorRT build workspace in GB (default 4)")
    p.add_argument("--device", type=int, default=0, help="CUDA device index (default 0)")
    p.add_argument("--no-validate", dest="validate", action="store_false", default=True,
                   help="skip the post-build load/benchmark/output check")
    p.add_argument("--install-deps", action="store_true",
                   help="pip install tensorrt/onnx/onnxslim if missing, then continue")
    p.add_argument("--verbose", action="store_true", help="verbose TensorRT build log")
    return p


def main() -> int:
    args = build_arg_parser().parse_args()
    logging.basicConfig(level="INFO", format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
