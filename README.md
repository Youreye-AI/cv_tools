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

## Instructions to build a road-damage training set (RDD2022 + our frames)

Our own footage is one 193-second drive: 68 labelled frames, 54 instances. That is the reason the
first model scored 0.007 mAP50 on anything but itself. **RDD2022** supplies 47,420 vehicle-mounted
images from six countries — the same viewpoint as ours — and merging it takes the training set to
~36,700 instances.

```powershell
# 1. fetch the country archives (~2.3 GB; Norway is skipped, it is 10.6 GB on its own)
python fetch_rdd2022.py --countries Czech,India,Japan,United_States --dest .\datasets\rdd2022_raw

# 2. merge with our own frames into a YOLO dataset
python build_road_damage_dataset.py --rdd .\datasets\rdd2022_raw ^
    --gaziantep .\datasets\gaziantepPotholesSplited --dest .\datasets\road_damage

# see the counts and warnings without writing anything
python build_road_damage_dataset.py --rdd .\datasets\rdd2022_raw --dest .\datasets\rd --dry-run
```

**Getting the data is not as simple as the upstream README suggests.** The per-country S3 links in
`sekilab/RoadDamageDetector` are dead — every one returns `403 AccessDenied`. The surviving mirror
is Figshare, which hosts a single 13.26 GB bundle. `fetch_rdd2022.py` reads that archive's index
over HTTP range requests and pulls only the countries you ask for, so the full bundle never
transfers. (Licence note: the GitHub README says CC BY-SA 4.0, the Figshare record says CC BY 4.0.
Attribute the RDD2022 authors either way.) RDD2022's own `test/` folders contain images with **no
annotations**, so the held-out splits are carved out of `train/`.

**All four damage classes are merged into one.** D00/D10/D20 (crack types) and D40 (pothole) become
a single `road_damage` class, because that is what our own labels already mark — see the "Which
weights" section below. Filtering to D40 alone would have given 5,762 instances instead of 36,059,
*and* fought our own annotations.

Two things the builder is careful about, both learned the hard way:

- **Leakage.** RDD2022 images are sequential drive captures, like ours, so neighbouring frames are
  near-identical. Whole contiguous blocks of each country's sequence go to one split, never split
  across — the same trap `split_dataset.py` exists to avoid.
- **Our 68 frames not drowning.** Against ~21,000 RDD images they would contribute nothing, so they
  are repeated `--gaziantep-repeat` (default 15) times in train. The leakage check compares
  repeat-stripped stems, or a deliberate duplicate and a real leak look identical.

## Instructions to run a trained model over a recorded video

Once a model is trained, `predict_video.py` answers "what does it actually do on the
footage?" — it decodes a video, runs the detector with ByteTrack, and writes an
annotated copy. The source video is never modified.

```powershell
# annotate the whole video with the gaziantep pothole weights (the default --weights)
python predict_video.py --video C:\Users\RONSITO\Videos\gaziantep.mp4

# preview ten seconds before committing to a full pass
python predict_video.py --video clip.mp4 --start 10 --end 20 --out slice.mp4

# other weights, no tracking, raw un-smoothed boxes
python predict_video.py --video clip.mp4 --weights runs\detect\train\weights\best.pt ^
    --conf 0.45 --no-track --hold-frames 0
```

### Rule-based noise reduction

A detector trained on 40 pothole instances fires on plenty of things that are not potholes. Three
rules run over its output before anything is drawn or counted. On the full gaziantep video they
take **1368 raw detections down to 529**, and the unique-pothole counter from 201 to 87:

| rule | flag | dropped |
|---|---|---|
| track seen in fewer than 3 frames | `--min-track-hits 3` | **772** |
| sits mostly inside a car/truck/bus/person | `--suppress-classes` | 58 |
| centre above the road | `--roi-top 0.45` | 2 |

**Persistence does nearly all the work** — most false positives are one-frame flashes, and a
detection the tracker never confirmed can never reach the threshold. But persistence *cannot* catch
boxes on cars: 58 of the 90 vehicle-overlapping detections survive it, because a parked car is
stationary and ByteTrack follows it happily. That is why the vehicle veto exists as a separate rule,
running the already-vendored `data_collector/yolo26n.pt` alongside. The two are complementary.

Two implementation details worth not breaking:

- The veto compares **intersection over the pothole box's own area**, not IoU. A pothole box inside
  a car box has an IoA of 1.0 but an IoU of about 0.004 — an IoU version of this test would never
  fire while reading as perfectly reasonable code.
- The COCO model is fed the **native frame**, not the stretched square, because it was trained on
  undistorted images. The two detectors get different inputs on purpose and meet again in native
  coordinates.

Area and aspect-ratio envelopes were measured and **deliberately rejected**: on top of persistence
they remove 0 and 8 detections respectively, and an envelope drawn from 54 training boxes will
eventually throw away a real pothole of an unusual shape. Don't add them back without new numbers.

Unlike `--hold-frames`, these rules change the detections, so the reported counts change with them.
The summary always prints the funnel rather than a quietly smaller total, and `--no-filters` turns
all three off for an honest before/after:

```
detections : 1368 raw -> 537 kept
  dropped, track seen < 3 frames : 772
  dropped, on a vehicle/person   :  58
  dropped, above the road        :   1
```

The veto detector runs on **every** frame (`--suppress-every 1`). Caching its boxes for three
frames nearly halves the render time, but it was measured to leak: the boxes go two frames stale,
cars move, and 7 of 544 surviving detections ended up back on a vehicle. Raise `--suppress-every`
only if you can accept that.

### What it draws

Corner brackets on each detection, coloured amber through red by confidence, labelled with the
track id and score. A card top-left counts unique potholes and shows the position in the clip;
a bar along the bottom fills as the video plays and grows a tick each time a new pothole is
first seen. `--no-dashboard` drops the card and the bar, leaving just the boxes.

`--hold-frames` (default 8) keeps a detection on screen, fading, for a few frames after the
model stops emitting it. **This is a display effect and nothing else** — every number the run
summary reports is counted from raw per-frame output, so the smoothing cannot flatter the
results. It exists because a detector that fires on a single frame and drops out strobes badly
on playback. `--hold-frames 0` shows the unsmoothed truth; the reported counts are identical
either way, which is worth re-checking if you ever touch the drawing code.

### Thresholds belong to the model, not to this script

The `--conf` default (0.15) is tuned for the bundled `models/road_damage_v2.pt`, not chosen as a
general-purpose value. Swept over 800 held-out images (601 instances), F1 peaks there:

| `--conf` | boxes per image | F1 |
|---|---|---|
| 0.10 | 1.18 | 0.375 |
| **0.15** | **0.60** | **0.409** |
| 0.30 | 0.27 | 0.323 |
| 0.50 | 0.07 | 0.146 |

**Note how far that moved.** The previous Gaziantep-only model peaked at **0.85**; this one peaks
at **0.15**. Confidence calibration is a property of the trained weights and does not transfer
between models — shipping new weights on an old threshold throws away most of the gain. Section 6
of `notebooks/gaziantep_demo.ipynb` prints the number to use for whatever you have just trained.

`--iou 0.5` is below Ultralytics' 0.7 because a damaged patch of asphalt attracts several
overlapping boxes; at 0.7 they all survive and stack up on screen. Unlike `--hold-frames`, this is
real NMS — it changes the detections and the reported counts with them.

**`--preprocess` must match how the training images were resized.** The default is now `native`,
because `road_damage_v2` was trained on images that kept their source aspect ratio.

The `stretch` option exists for the older `gaziantep_pothole_v1.pt`. A Roboflow export resizes to a
square: our 1280x720 footage became 512x512 training images with no letterbox padding, so that
model learned road damage *stretched*, and handing it an undistorted frame silently cost
detections. Measured on 20 frames it had memorised, at `--conf 0.25`:

| input | detections |
|---|---|
| the 512x512 Roboflow jpg it trained on | 18 |
| same instant from the video, native 720p | 14 |
| same instant from the video, stretched to 512 | **18** |

`build_road_damage_dataset.py` removes the problem at the source: it re-points our labels at the
original 1280x720 frames, which is exact because normalised YOLO coordinates are invariant under an
axis-aligned resize. Pass `--preprocess stretch --stretch-size 512` only when running the old v1
weights.

A box labelled `#7 0.93` is a confirmed track; one labelled just `0.93` was detected but never
tracked, because ByteTrack only confirms a track after it matches the same object across
consecutive frames. The run summary reports what share of detections made it into a track — a
low number means the detector is firing on single frames and dropping out, which is a property
of the model, not of the tracker. (The class name is omitted from the chip for single-class
models, where it distinguishes nothing and only makes clustered labels collide.)

### Which weights

`models/road_damage_v2.pt` is the default: `yolo11s` trained on RDD2022 (Czech/India/Japan/US)
merged with our own Gaziantep frames — 21,087 images and 29,462 instances, class `road_damage`.
Measured on a 2,633-image held-out split:

| weights | trained on | mAP50 |
|---|---|---|
| **road_damage_v2** | RDD2022 + ours, 36.7k instances | **0.559** |
| gaziantep_pothole_v1 | our 40 instances only | 0.007 |
| a public pretrained pothole model (Apache-2.0) | ~880 public images | 0.003 |
| `yolo11s` COCO, untrained on this task | — | 0.000 |

Two things that table settles. **An off-the-shelf public model does not work here** — it scored
*worse* than our own, because it detects holes while our labels mark road damage. And the class is
`road_damage`, not `pothole`: our boxes were always drawn around cracks, patches and degraded
surface (median box area 3.6x RDD2022's pothole class, 1.4x its all-damage median), which is why
the old model appeared to "false positive" on manhole covers — it was doing exactly what the
labels asked. The dashboard heading is read from the weights, so it says ROAD DAMAGE FOUND.

`models/` is gitignored; weights are a build artifact, not source.

Be careful what you conclude from the result. **Running a model over the same recording its
training frames came from is a qualitative check, not a measurement** — the model has
already seen most of what it is being shown. Score on held-out footage.

Run `python predict_video.py --help` for `--hold-frames`, `--every`, `--fourcc`, `--show` and
the rest.

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