# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

`cv_tools` is a collection of independent computer-vision CLI tools for a road-damage/pothole
detection project, built around Ultralytics YOLO. There is no single application entry point —
each top-level script is a standalone stage in a pipeline: collect footage → extract frames →
build a labelled dataset → split it → train (outside this repo) → run/evaluate a trained model on
video → optimize weights for deployment. `data_collector/` is a separate, self-contained tool (an
interactive RTSP person-crop collector) and is not wired into the road-damage pipeline.

Every script is run directly with `python <script>.py --help` for full CLI docs — `README.md`
documents the *why* behind each tool's design decisions (thresholds, leakage handling, coordinate
systems) in detail and should be treated as the source of truth for tuning knobs; do not
re-derive numbers that are already measured there.

## Two Python environments — do not merge them

This project deliberately splits work across two environments that must stay separate:

- **Training / notebooks**: a `.venv` **one level above this repo**
  (`...\YourEye\.venv`, i.e. `..\.venv\Scripts\python.exe` from the repo root). Has torch,
  ultralytics, pandas, matplotlib. Its `cv2` is stock pip `opencv-python` (no GStreamer).
  Use this interpreter for `build_road_damage_dataset.py`, `fetch_rdd2022.py`,
  `split_dataset.py`, `predict_video.py`, `extract_frames.py`, `split_images.py`, notebooks, and
  any YOLO training (`yolo detect train ...`).
- **RTSP / `data_collector/`**: conda env `cvstack313` (Python 3.13) — the one
  `requirements.txt` targets. Its `cv2` is a hand-built OpenCV **with GStreamer** support,
  installed manually. Run `data_collector/main.py` with `conda run -n cvstack313 python
  data_collector/main.py` (or `cd data_collector` first — its modules import each other with
  bare names, e.g. `from detector import PersonDetector`, so it must run from inside that
  folder or with it on `sys.path`).

**Never `pip install` into the `cvstack313` env** — anything pulling in `opencv-python`
(ultralytics does) silently clobbers the hand-built GStreamer `cv2` and breaks RTSP reads. The
bare `python` on PATH is a plain 3.12 with neither environment's packages installed — don't use
it for anything in this repo.

## Commands

```powershell
# Frame extraction from a recorded video -> flat image folder
python extract_frames.py --video clip.mp4 --out .\datasets\batch --crops-per-second 2

# Assign a flat image folder to annotators
python split_images.py --src .\datasets\batch --dest .\datasets\assigned --n 5

# Fetch RDD2022 country archives (via Figshare range requests; upstream S3 links are dead)
python fetch_rdd2022.py --countries Czech,India,Japan,United_States --dest .\datasets\rdd2022_raw

# Merge RDD2022 + our own frames into one YOLO dataset (single `road_damage` class)
python build_road_damage_dataset.py --rdd .\datasets\rdd2022_raw --gaziantep .\datasets\gaziantepPotholesSplited --dest .\datasets\road_damage
python build_road_damage_dataset.py --rdd .\datasets\rdd2022_raw --dest .\datasets\rd --dry-run   # preview only

# 80/20 train/test split of a Roboflow "100% train" export (never use split_images.py for this — it drops .txt labels)
python split_dataset.py --src .\datasets\apron_roboflow --dest .\datasets\apron_split --dry-run
python split_dataset.py --src .\datasets\apron_roboflow --dest .\datasets\apron_split

# Train (outside this repo's tooling, straight Ultralytics CLI)
yolo detect train data=.\datasets\apron_split\data.yaml model=yolo11n.pt epochs=100 imgsz=640

# Run a trained model over recorded video, with tracking + noise-reduction rules
python predict_video.py --video clip.mp4
python predict_video.py --video clip.mp4 --weights runs\detect\train\weights\best.pt --conf 0.45 --no-track --hold-frames 0

# TensorRT engine export — must run ON the GPU server, not a dev laptop (engines aren't portable)
python optimize_to_engine.py --install-deps   # first time only
python optimize_to_engine.py

# RTSP person-crop collector (interactive prompts; needs the cvstack313 GStreamer cv2)
conda run -n cvstack313 python data_collector/main.py
```

There is no test suite, linter, or build step configured in this repo (`pytest` is listed in
`requirements.txt` for the `cvstack313` env but no test files currently exist).

## Architecture notes

**Dataset lineage matters more than the code.** Our own footage is a single 193-second drive
(68 labelled frames). `build_road_damage_dataset.py` exists because that alone is unusable for
training — it merges in RDD2022 (47k vehicle-mounted images, same viewpoint) and is careful about
two things: (1) **leakage** — RDD2022 and our own frames are sequential video captures, so
whole contiguous blocks are assigned to one split, never split mid-sequence (`split_dataset.py`
applies the same grouping logic via `--group-seconds`); (2) our 68 frames are repeated
`--gaziantep-repeat` times in train so they aren't drowned out by ~21k RDD images. All damage
types (cracks D00/D10/D20, potholes D40) collapse into one `road_damage` class because that's
what our own annotations already mark.

**`predict_video.py`'s detection pipeline is three-stage, in this order:**
1. YOLO model inference + ByteTrack.
2. Track-persistence filter (`--min-track-hits`, default 3) — drops single-frame flashes. Does
   almost all the noise reduction.
3. Vehicle/person veto — runs a second, vendored COCO detector (`data_collector/yolo26n.pt`) on
   the **native, unstretched frame** and drops pothole boxes mostly contained in a car/truck/
   bus/person box, using **intersection-over-the-pothole-box's-own-area**, not IoU (IoU would
   almost never trigger since a small pothole box inside a large car box has tiny IoU but IoA≈1).

Area/aspect-ratio envelope filters were tried and deliberately rejected (measured to remove
false positives at the cost of real, unusually-shaped potholes) — don't reintroduce them without
new measurements. `--conf`, `--iou`, and `--preprocess` are properties of *which weights* are
loaded, not general-purpose defaults — see the "Which weights" and "Thresholds belong to the
model" sections of `README.md` before changing them, and re-derive thresholds (via
`notebooks/gaziantep_demo.ipynb`) whenever new weights are trained.

**`data_collector/` is architecturally independent** from the rest of the repo: it's an
interactive tool (prompts for RTSP URL, ROI, save interval) that saves cropped person detections
to a session folder with a `metadata.json` sidecar, for later annotation — unrelated to the
road-damage dataset/training pipeline.

**Model weights (`models/`, `notebooks/*.pt`, root `yolo*.pt`) are gitignored** — they're build
artifacts, not source. `data_collector/yolo26n.pt` is the one deliberate exception (tracked,
vendored, scoped out of the root `/yolo*.pt` gitignore rule).

`gstreamer_opt.py` and `optimize_to_engine.py` carry docstrings referencing a `scripts/` layout
and other pipeline scripts (`seat_zone_detection.py`, `rtsp_waiter_client_detection.py`) that
don't exist in this repo — those tools were carried over from a related project; treat the
usage examples in their docstrings as needing a path adjustment (drop the `scripts/` prefix) when
actually running them here.
