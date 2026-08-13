"""Assemble a road-damage training set from RDD2022 plus our own Gaziantep frames.

Why
---
Our detector was trained on 40 boxes from a single 193-second drive. It scores 0.265 mAP50, and its
failure mode is firing on manhole covers, tarmac patches and shadows. No amount of tuning fixes
that; the model has simply never seen enough road. RDD2022 supplies 47,420 vehicle-mounted images
from six countries with ~55,000 annotated damage instances, which is the same viewpoint as our
footage and the right order of magnitude more data.

What we are actually detecting
------------------------------
Not potholes. Comparing our 68 labels against RDD2022's classes shows our boxes are 3.6x the median
area of RDD's D40 (pothole) boxes but only 1.4x its all-class median, and visually they sit on
cracked asphalt, repair patches and degraded surface rather than on holes. The Roboflow export was
even named `gaziantepWayImperfections`. So our convention is ROAD DAMAGE, not potholes.

That makes RDD2022's four damage classes all relevant, and collapsing them into one class is both
truer to how we label and far more data than filtering to potholes would give:

    D00 longitudinal crack + D10 transverse + D20 alligator + D40 pothole -> `road_damage`

On the Czech subset alone that is 1,745 instances instead of 197 — nine times the data, from the
same download, with none of our existing labels needing to be redone.

Two things this script is careful about
---------------------------------------
* **Leakage.** RDD2022 images are sequential drive captures, exactly like ours, so neighbouring
  frames are near-identical. A random split puts frame N in train and N+1 in val and every metric
  comes out inflated — the trap `split_dataset.py` was written to avoid. Whole contiguous blocks of
  each country's sequence are assigned to one side, never split.
* **Our 68 images not being drowned.** Against ~20,000 RDD images they would contribute nothing, so
  they are repeated `--gaziantep-repeat` times in the train split. The Gaziantep test images are
  never repeated and never mixed into train.

Get the data first
------------------
The S3 links in the sekilab README are dead (403). The working mirror is Figshare, which hosts one
13.2 GB bundle of per-country zips — see the README for a range-request recipe that pulls only the
countries you want. Extract them so this script sees:

    datasets/rdd2022_raw/<Country>/train/images/*.jpg
    datasets/rdd2022_raw/<Country>/train/annotations/xmls/*.xml

(RDD2022's own `test/` folders hold images with NO annotations, so they are unusable here and the
held-out split is carved out of `train/` instead.)

Run it
------
    python build_road_damage_dataset.py --rdd .\\datasets\\rdd2022_raw ^
        --gaziantep .\\datasets\\gaziantepPotholesSplited ^
        --native-frames .\\datasets\\gaziantep ^
        --dest .\\datasets\\road_damage

    # see the counts and the warnings without writing anything
    python build_road_damage_dataset.py --rdd .\\datasets\\rdd2022_raw --dest .\\datasets\\rd --dry-run
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import re
import shutil
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

# split_dataset.py sits at the repo root and does nothing at import time. Reused rather than
# copied so the Roboflow filename mangling and the YAML quoting cannot drift between the two tools.
from split_dataset import _FRAME_STEM_RE, IMAGE_EXTENSIONS, clean_stem, yaml_str

logger = logging.getLogger("build_road_damage_dataset")

# RDD2022's four damage types, all collapsed to a single class. See the module docstring for why
# this is the right call for our labelling convention rather than filtering to D40.
RDD_DAMAGE_CLASSES = ("D00", "D10", "D20", "D40")
CLASS_NAME = "road_damage"

# RDD2022 also carries a handful of other tags (D43 crosswalk blur, D44 white line blur, D50
# manhole, ...) that are not damage. Listed explicitly so an unexpected tag is reported rather
# than silently swept into the class.
RDD_KNOWN_OTHER = ("D01", "D11", "D43", "D44", "D50", "Repair", "Block crack")


@dataclass
class Sample:
    """One image plus the boxes that travel with it, in normalised YOLO form."""

    image: Path
    boxes: list[tuple[float, float, float, float]]
    source: str          # "rdd:<Country>" or "gaziantep"
    group: str           # the unit that must not be split across sides
    stem: str

    @property
    def is_background(self) -> bool:
        return not self.boxes


# ---------------------------------------------------------------- pure helpers


def voc_to_yolo(xmin: float, ymin: float, xmax: float, ymax: float,
                width: int, height: int) -> tuple[float, float, float, float] | None:
    """Convert a Pascal VOC pixel box to normalised YOLO cx,cy,w,h.

    Returns None for a degenerate box. RDD2022 contains a few annotations with zero width or
    coordinates a pixel outside the image, and a silently-clamped garbage box is worse than a
    dropped one — YOLO would happily train on it.
    """
    xmin, xmax = sorted((max(0.0, xmin), min(float(width), xmax)))
    ymin, ymax = sorted((max(0.0, ymin), min(float(height), ymax)))
    box_w, box_h = xmax - xmin, ymax - ymin
    if box_w < 1.0 or box_h < 1.0 or width <= 0 or height <= 0:
        return None
    return (
        (xmin + box_w / 2) / width,
        (ymin + box_h / 2) / height,
        box_w / width,
        box_h / height,
    )


def block_group(stem: str, block_size: int) -> str:
    """Group key for an RDD image: contiguous runs of the per-country sequence number.

    RDD filenames look like `Czech_000123.jpg`, numbered along the drive, so images whose numbers
    are close are frames of the same stretch of road. Bucketing by index keeps them on one side of
    the split. Falls back to the stem when a name does not match, which puts that image in a group
    of its own — safe, just less effective.
    """
    match = re.match(r"^(?P<country>.+?)_(?P<index>\d+)$", stem)
    if not match:
        return stem
    return f"{match['country']}#b{int(match['index']) // max(1, block_size):06d}"


def assign_blocks(groups: list[str], val_fraction: float, test_fraction: float,
                  seed: int) -> dict[str, str]:
    """Assign whole groups to train/val/test, deterministically from --seed."""
    rng = random.Random(seed)
    shuffled = sorted(groups)
    rng.shuffle(shuffled)
    n_val = int(round(len(shuffled) * val_fraction))
    n_test = int(round(len(shuffled) * test_fraction))
    assignment = {}
    for i, key in enumerate(shuffled):
        if i < n_test:
            assignment[key] = "test"
        elif i < n_test + n_val:
            assignment[key] = "val"
        else:
            assignment[key] = "train"
    return assignment


# ---------------------------------------------------------------- collect


def read_voc(xml_path: Path) -> tuple[list[tuple[float, float, float, float]], Counter]:
    """Read one RDD2022 annotation, returning damage boxes and a tally of every tag seen."""
    seen: Counter = Counter()
    try:
        root = ET.parse(xml_path).getroot()
    except ET.ParseError as error:
        logger.warning("%s is not parseable XML (%s) — skipped", xml_path.name, error)
        return [], seen

    size = root.find("size")
    if size is None:
        logger.warning("%s has no <size> — skipped", xml_path.name)
        return [], seen
    width = int(float(size.findtext("width", "0")))
    height = int(float(size.findtext("height", "0")))

    boxes = []
    for obj in root.findall("object"):
        name = (obj.findtext("name") or "").strip()
        seen[name] += 1
        if name not in RDD_DAMAGE_CLASSES:
            continue
        bnd = obj.find("bndbox")
        if bnd is None:
            continue
        box = voc_to_yolo(
            float(bnd.findtext("xmin", "0")), float(bnd.findtext("ymin", "0")),
            float(bnd.findtext("xmax", "0")), float(bnd.findtext("ymax", "0")),
            width, height,
        )
        if box is None:
            logger.debug("%s has a degenerate %s box — dropped", xml_path.name, name)
            continue
        boxes.append(box)
    return boxes, seen


def collect_rdd(root: Path, block_size: int) -> tuple[list[Sample], Counter]:
    """Walk every <Country>/train/ folder under --rdd and convert its annotations."""
    countries = sorted(p for p in root.iterdir() if p.is_dir() and (p / "train").is_dir())
    if not countries:
        logger.error("no <Country>/train folders under %s — extract the RDD2022 country zips "
                     "there first (see the module docstring)", root)
        raise SystemExit(2)

    samples: list[Sample] = []
    tags: Counter = Counter()
    for country in countries:
        images_dir = country / "train" / "images"
        xml_dir = country / "train" / "annotations" / "xmls"
        if not images_dir.is_dir() or not xml_dir.is_dir():
            logger.warning("%s has no train/images + train/annotations/xmls — skipped", country.name)
            continue

        found = 0
        for xml_path in sorted(xml_dir.glob("*.xml")):
            image = images_dir / f"{xml_path.stem}.jpg"
            if not image.is_file():
                continue
            boxes, seen = read_voc(xml_path)
            tags.update(seen)
            samples.append(Sample(image=image, boxes=boxes, source=f"rdd:{country.name}",
                                  group=block_group(xml_path.stem, block_size),
                                  stem=xml_path.stem))
            found += 1
        logger.info("  %-16s %5d annotated image(s)", country.name, found)
    return samples, tags


def collect_gaziantep(split_root: Path, native_root: Path | None) -> dict[str, list[Sample]]:
    """Load our own labelled frames, preferring the native-resolution originals.

    The Roboflow export stretched 1280x720 to 512x512. Normalised YOLO coordinates are invariant
    under an axis-aligned resize, so pointing the SAME label file at the original frame is exact —
    no re-annotation, and it removes the aspect distortion that forced predict_video.py's
    `--preprocess stretch` workaround.
    """
    per_split: dict[str, list[Sample]] = {"train": [], "test": []}
    native_by_ts = {}
    if native_root is not None and native_root.is_dir():
        for path in native_root.iterdir():
            if path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            match = _FRAME_STEM_RE.match(path.stem)
            if match:
                native_by_ts[match["ms"]] = path

    missing_native = 0
    for split in ("train", "test"):
        images_dir = split_root / split / "images"
        labels_dir = split_root / split / "labels"
        if not images_dir.is_dir():
            logger.error("%s not found — point --gaziantep at the split_dataset.py output", images_dir)
            raise SystemExit(2)
        for image in sorted(p for p in images_dir.iterdir()
                            if p.suffix.lower() in IMAGE_EXTENSIONS):
            label = labels_dir / f"{image.stem}.txt"
            boxes = []
            if label.is_file():
                for line in label.read_text(encoding="utf-8").splitlines():
                    parts = line.split()
                    if len(parts) >= 5:
                        boxes.append(tuple(float(v) for v in parts[1:5]))

            stem = clean_stem(image.name)
            source_image = image
            match = _FRAME_STEM_RE.match(stem)
            if match and match["ms"] in native_by_ts:
                source_image = native_by_ts[match["ms"]]
            elif native_root is not None:
                missing_native += 1

            per_split[split].append(Sample(image=source_image, boxes=boxes, source="gaziantep",
                                           group=f"gaziantep#{split}#{stem}", stem=stem))

    if missing_native:
        # Loud, because silently falling back to the stretched squares would reintroduce exactly
        # the aspect mismatch this rebuild exists to remove, and nothing downstream would notice.
        logger.error("%d Gaziantep image(s) had no native-resolution twin in --native-frames. "
                     "Point it at the extract_frames.py output folder, or pass --no-native to use "
                     "the stretched 512x512 exports on purpose", missing_native)
        raise SystemExit(2)
    return per_split


# ---------------------------------------------------------------- write


def write_split(samples: list[Sample], split_dir: Path, repeats: dict[str, int]) -> tuple[int, int]:
    """Copy images and write label files. Returns (images, instances)."""
    images_dir = split_dir / "images"
    labels_dir = split_dir / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    written = instances = 0
    for sample in samples:
        for copy_index in range(repeats.get(sample.source, 1)):
            # A suffix only on the duplicates, so the first copy keeps a name that traces straight
            # back to the source file.
            suffix = "" if copy_index == 0 else f"__r{copy_index}"
            name = f"{sample.stem}{suffix}"
            shutil.copy2(sample.image, images_dir / f"{name}{sample.image.suffix}")
            (labels_dir / f"{name}.txt").write_text(
                "".join(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n" for cx, cy, w, h in sample.boxes),
                encoding="utf-8",
            )
            written += 1
            instances += len(sample.boxes)
    return written, instances


def write_data_yaml(dest: Path) -> Path:
    """Ultralytics config. Same conventions as split_dataset.write_data_yaml."""
    path = dest / "data.yaml"
    path.write_text("\n".join([
        "# Generated by build_road_damage_dataset.py — regenerate rather than hand-editing,",
        "# except for `path:` below, which is the one line to change if you move this folder.",
        "",
        f"path: {yaml_str(dest.resolve())}",
        "train: train/images",
        "val: val/images",
        "test: test/images",
        "",
        "nc: 1",
        f"names: [{yaml_str(CLASS_NAME)}]",
        "",
    ]), encoding="utf-8")
    return path


# ---------------------------------------------------------------- run


def run(args: argparse.Namespace) -> int:
    if not 0.0 < args.val_fraction + args.test_fraction < 1.0:
        logger.error("--val-fraction + --test-fraction must be between 0 and 1 (got %s + %s)",
                     args.val_fraction, args.test_fraction)
        raise SystemExit(2)
    if args.gaziantep_repeat < 1:
        logger.error("--gaziantep-repeat must be 1 or greater (got %s)", args.gaziantep_repeat)
        raise SystemExit(2)

    dest = Path(args.dest)
    if dest.exists() and any(dest.iterdir()) and not args.overwrite and not args.dry_run:
        logger.error("%s is not empty. Pick an empty folder, or pass --overwrite", dest)
        raise SystemExit(2)

    logger.info("reading RDD2022 from %s", args.rdd)
    rdd, tags = collect_rdd(Path(args.rdd), args.block_size)
    damage = sum(len(s.boxes) for s in rdd)
    logger.info("rdd   : %d image(s), %d damage instance(s) from %s",
                len(rdd), damage, "+".join(sorted(RDD_DAMAGE_CLASSES)))
    logger.info("tags  : %s", dict(tags.most_common()))
    unexpected = set(tags) - set(RDD_DAMAGE_CLASSES) - set(RDD_KNOWN_OTHER) - {""}
    if unexpected:
        logger.warning("annotation tag(s) %s were not recognised and are NOT counted as damage — "
                       "check whether they should be", sorted(unexpected))

    # Cap backgrounds. RDD2022 is mostly undamaged road, and keeping every empty image would bury
    # the positives; a controlled ratio of them is what teaches the model clean asphalt is clean.
    positives = [s for s in rdd if not s.is_background]
    backgrounds = [s for s in rdd if s.is_background]
    allowed = int(len(positives) * args.max_background_ratio)
    if len(backgrounds) > allowed:
        random.Random(args.seed).shuffle(backgrounds)
        logger.info("bg    : keeping %d of %d background image(s) (--max-background-ratio %s)",
                    allowed, len(backgrounds), args.max_background_ratio)
        backgrounds = backgrounds[:allowed]
    rdd = positives + backgrounds

    assignment = assign_blocks(sorted({s.group for s in rdd}),
                               args.val_fraction, args.test_fraction, args.seed)
    buckets: dict[str, list[Sample]] = {"train": [], "val": [], "test": []}
    for sample in rdd:
        buckets[assignment[sample.group]].append(sample)

    if args.gaziantep:
        logger.info("reading our frames from %s", args.gaziantep)
        native = None if args.no_native else Path(args.native_frames)
        ours = collect_gaziantep(Path(args.gaziantep), native)
        # Our own train frames join train; our test frames stay a separate holdout so every number
        # measured against them so far remains comparable.
        buckets["train"].extend(ours["train"])
        buckets["test"].extend(ours["test"])
        logger.info("ours  : %d train + %d test image(s), repeated %dx in train",
                    len(ours["train"]), len(ours["test"]), args.gaziantep_repeat)

    for name, samples in buckets.items():
        stems = [s.stem for s in samples]
        if len(set(stems)) != len(stems):
            logger.warning("%s contains duplicate stems — copies will overwrite each other", name)
    overlap = ({s.stem for s in buckets["train"]} & {s.stem for s in buckets["test"]}) | \
              ({s.stem for s in buckets["train"]} & {s.stem for s in buckets["val"]})
    if overlap:
        logger.error("%d image(s) appear in more than one split, e.g. %s — this is a bug",
                     len(overlap), sorted(overlap)[:3])
        raise SystemExit(1)

    logger.info("")
    logger.info("%-8s %8s %10s %12s", "split", "images", "instances", "background")
    for name in ("train", "val", "test"):
        samples = buckets[name]
        # Count the oversampled copies, not the distinct samples, so a --dry-run reports the same
        # numbers the real build will write. Only train repeats.
        factor = (lambda s: args.gaziantep_repeat if (name == "train" and s.source == "gaziantep")
                  else 1)
        images = sum(factor(s) for s in samples)
        instances = sum(len(s.boxes) * factor(s) for s in samples)
        n_bg = sum(factor(s) for s in samples if s.is_background)
        logger.info("%-8s %8d %10d %12d", name, images, instances, n_bg)

    if args.dry_run:
        logger.info("")
        logger.info("dry run — nothing written. Drop --dry-run to build %s", dest)
        return 0

    repeats = {"gaziantep": args.gaziantep_repeat}
    dest.mkdir(parents=True, exist_ok=True)
    report_splits = {}
    for name in ("train", "val", "test"):
        # Only the train split repeats our frames; duplicating them into val or test would score
        # the model on the same image several times and quietly weight the metric towards it.
        split_repeats = repeats if name == "train" else {}
        images, instances = write_split(buckets[name], dest / name, split_repeats)
        report_splits[name] = {
            "images": images,
            "instances": instances,
            "sources": dict(Counter(s.source for s in buckets[name])),
        }
        logger.info("wrote %-5s %6d image(s), %6d instance(s)", name, images, instances)

    yaml_path = write_data_yaml(dest)
    (dest / "build_report.json").write_text(json.dumps({
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "class_name": CLASS_NAME,
        "rdd_classes_merged": list(RDD_DAMAGE_CLASSES),
        "rdd_tag_counts": dict(tags),
        "seed": args.seed,
        "block_size": args.block_size,
        "val_fraction": args.val_fraction,
        "test_fraction": args.test_fraction,
        "max_background_ratio": args.max_background_ratio,
        "gaziantep_repeat": args.gaziantep_repeat,
        "gaziantep_native_resolution": not args.no_native,
        "splits": report_splits,
    }, indent=2), encoding="utf-8")

    logger.info("")
    logger.info("done — %s", dest)
    logger.info("config: %s", yaml_path.name)
    return 0


# ---------------------------------------------------------------- cli


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Merge RDD2022 and our Gaziantep frames into one YOLO road-damage dataset.",
    )
    p.add_argument("--rdd", required=True,
                   help="folder holding the extracted RDD2022 <Country>/ directories")
    p.add_argument("--dest", required=True, help="folder the merged dataset is written to")
    p.add_argument("--gaziantep", default=None,
                   help="our split_dataset.py output to merge in (default: RDD only)")
    p.add_argument("--native-frames", default="datasets/gaziantep",
                   help="extract_frames.py output holding the original 1280x720 frames; our labels "
                        "are re-pointed at these instead of the stretched 512x512 Roboflow exports "
                        "(default datasets/gaziantep)")
    p.add_argument("--no-native", action="store_true",
                   help="use the stretched 512x512 Roboflow images instead of the native frames "
                        "(default off)")
    p.add_argument("--gaziantep-repeat", type=int, default=15,
                   help="how many times our frames are repeated in the TRAIN split, so ~68 images "
                        "are not lost among ~20000 (default 15)")
    p.add_argument("--max-background-ratio", type=float, default=0.5,
                   help="cap on background (undamaged) images as a fraction of damaged ones "
                        "(default 0.5)")
    p.add_argument("--block-size", type=int, default=50,
                   help="consecutive RDD frame numbers grouped into one indivisible block, so "
                        "near-identical neighbouring frames cannot straddle the split (default 50)")
    p.add_argument("--val-fraction", type=float, default=0.1,
                   help="share of RDD blocks held out for validation (default 0.1)")
    p.add_argument("--test-fraction", type=float, default=0.1,
                   help="share of RDD blocks held out for testing (default 0.1)")
    p.add_argument("--seed", type=int, default=42,
                   help="shuffle seed; same convention as split_dataset.py (default 42)")
    p.add_argument("--dry-run", action="store_true",
                   help="report the counts and warnings without writing anything (default off)")
    p.add_argument("--overwrite", action="store_true",
                   help="write into a --dest that already has files in it (default off)")
    return p


def main() -> int:
    args = build_arg_parser().parse_args()
    logging.basicConfig(level="INFO", format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
