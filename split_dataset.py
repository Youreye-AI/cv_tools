"""Split a Roboflow "100% train" YOLO export into a train/test dataset.

Why
---
Roboflow can export a dataset as 100% train, which is what you want when the split
should be decided here rather than in the browser. But the export is then unusable for
training: Ultralytics needs a validation set, and `data.yaml` has nowhere to point.

`split_images.py` cannot do this job. It splits a FLAT image folder into `person_N/`
folders for annotator assignment — pointed at a YOLO dataset it copies the images and
silently drops every `.txt` label, which is only noticed after a training run produces
garbage. This tool moves each image and its label together, always.

The split itself is the part that matters. Our images come out of `extract_frames.py`
sampling video at a few frames per second, so consecutive frames are near-identical. A
plain shuffle puts frame N in train and frame N+1 in test, the model scores well on
images it has effectively memorised, and the reported metrics are fiction. So images are
grouped before they are split:

  * all of Roboflow's augmented copies of one source image stay on the same side, and
  * frames from the same video within the same --group-seconds window stay together.

Whole groups are then assigned rarest-class-first, so every class still lands in the
test set and its mAP is defined.

The source export is never modified — the split is copied to --dest, so it can be redone
with a different seed or ratio at any time.

Run it
------
    # 80/20, the usual case
    python split_dataset.py --src .\\datasets\\apron_roboflow --dest .\\datasets\\apron_split

    # see the split and the warnings without writing anything
    python split_dataset.py --src .\\datasets\\apron_roboflow --dest .\\datasets\\apron_split --dry-run

    # 70/30, tighter time blocks, different shuffle
    python split_dataset.py --src .\\datasets\\raw --dest .\\datasets\\split ^
        --val-fraction 0.3 --group-seconds 5 --seed 7
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import re
import shutil
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import yaml

logger = logging.getLogger("split_dataset")

# Same set extract_frames.py:42 and split_images.py:31 scan for, so a folder produced by
# either one round-trips through Roboflow and back into this script unchanged.
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# Roboflow renames `clip_t00012500.jpg` -> `clip_t00012500_jpg.rf.<32 hex>.jpg`, and every
# augmented copy of one source image shares the base stem but gets its own hash. Stripping
# both suffixes collapses those copies onto one key so they cannot straddle the split.
_ROBOFLOW_HASH_RE = re.compile(r"\.rf\.[0-9a-f]+$", re.IGNORECASE)
_ROBOFLOW_EXT_SUFFIX_RE = re.compile(r"_(jpg|jpeg|png|webp|bmp)$", re.IGNORECASE)

# The naming extract_frames.py:93 writes: {video_stem}_t{position_ms:08d}. Matching it is
# what lets us rebuild the video timeline from filenames alone and split on time blocks.
_FRAME_STEM_RE = re.compile(r"^(?P<video>.+)_t(?P<ms>\d{8})$")

_SMALL_DATASET_IMAGES = 50  # below this, holdout metrics are too noisy to quote


@dataclass
class Sample:
    """One image and the label file that must travel with it."""

    image: Path
    label: Path | None
    stem: str  # roboflow mangling removed; the identity used for the leakage check
    group: str
    classes: Counter = field(default_factory=Counter)

    @property
    def is_background(self) -> bool:
        return not self.classes


@dataclass
class Group:
    """The unit that gets assigned to a side. Never split."""

    key: str
    samples: list[Sample] = field(default_factory=list)
    classes: Counter = field(default_factory=Counter)

    @property
    def size(self) -> int:
        return len(self.samples)

    @property
    def is_background(self) -> bool:
        return not self.classes


# ---------------------------------------------------------------- pure helpers


def clean_stem(name: str) -> str:
    """Strip Roboflow's mangling to recover the stem the image had before upload."""
    stem = Path(name).stem
    stem = _ROBOFLOW_HASH_RE.sub("", stem)
    stem = _ROBOFLOW_EXT_SUFFIX_RE.sub("", stem)
    return stem


def group_key(stem: str, group_seconds: float | None) -> str:
    """Key that decides which images are inseparable.

    Falls back one layer at a time: a video timestamp gives a time block, anything else
    gives the cleaned stem, which still keeps augmented copies of one image together.
    """
    if group_seconds:
        match = _FRAME_STEM_RE.match(stem)
        if match:
            block_ms = max(1, int(round(group_seconds * 1000.0)))
            block = int(match["ms"]) // block_ms
            return f"{match['video']}#t{block:06d}"
    return stem


def moves_closer(current: int, size: int, target: float) -> bool:
    """Whether taking this group lands the test set nearer its target than stopping short.

    Groups are indivisible, so a hard "never exceed the target" cap systematically
    undershoots: with 12-image groups and a target of 31.8, it would stop at 24 and call
    it a 30% split. Overshooting to 36 is the honest choice — it is simply closer.
    """
    return abs(current + size - target) < abs(current - target)


def parse_label(path: Path) -> Counter:
    """Count class instances in a YOLO label file.

    An empty file is legitimate — it marks a background/negative image, which Roboflow
    writes for "null" annotations and which YOLO trains on happily.
    """
    counts: Counter = Counter()
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        logger.warning("label %s is not valid utf-8 — treated as empty", path.name)
        return counts
    for number, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            counts[int(float(line.split()[0]))] += 1
        except (ValueError, IndexError):
            logger.warning("label %s line %d is not a YOLO row — skipped", path.name, number)
    return counts


def normalise_names(raw: object) -> list[str]:
    """Return class names as a list, in class-id order.

    Roboflow writes a list; Ultralytics also accepts a {id: name} mapping. The ORDER is
    load-bearing — the index is the class id written in every label file — so it is
    carried through untouched rather than sorted or de-duplicated.
    """
    if isinstance(raw, list):
        return [str(n) for n in raw]
    if isinstance(raw, dict):
        return [str(raw[key]) for key in sorted(raw, key=lambda k: int(k))]
    logger.error("data.yaml has no usable `names:` entry (got %r)", raw)
    raise SystemExit(2)


def yaml_str(value: object) -> str:
    """Single-quote a scalar for YAML — Windows paths carry ':' and '\\'."""
    return "'" + str(value).replace("'", "''") + "'"


# ---------------------------------------------------------------- collect


def collect_samples(images_dir: Path, labels_dir: Path, group_seconds: float | None) -> list[Sample]:
    """Pair every image with its label, and work out which group it belongs to."""
    images = sorted(
        p for p in images_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not images:
        logger.error("no images found in %s (checked %s)", images_dir, sorted(IMAGE_EXTENSIONS))
        raise SystemExit(2)

    samples: list[Sample] = []
    missing_labels = 0
    for image in images:
        label = labels_dir / f"{image.stem}.txt"
        if not label.is_file():
            missing_labels += 1
            if missing_labels <= 5:
                logger.warning("no label for %s — kept as a background image", image.name)
            label = None
        stem = clean_stem(image.name)
        samples.append(Sample(
            image=image,
            label=label,
            stem=stem,
            group=group_key(stem, group_seconds),
            classes=parse_label(label) if label else Counter(),
        ))

    if missing_labels > 5:
        logger.warning("... and %d more images without a label file", missing_labels - 5)

    orphans = sum(
        1 for p in labels_dir.iterdir()
        if p.is_file() and p.suffix.lower() == ".txt"
        and not any((images_dir / f"{p.stem}{ext}").is_file() for ext in IMAGE_EXTENSIONS)
    ) if labels_dir.is_dir() else 0
    if orphans:
        logger.warning("%d label file(s) have no matching image — ignored", orphans)

    return samples


def build_groups(samples: list[Sample]) -> list[Group]:
    """Fold samples into groups, keyed deterministically so runs are reproducible."""
    groups: dict[str, Group] = {}
    for sample in samples:
        group = groups.setdefault(sample.group, Group(key=sample.group))
        group.samples.append(sample)
        group.classes.update(sample.classes)
    return [groups[key] for key in sorted(groups)]


# ---------------------------------------------------------------- split


def assign_groups(
    groups: list[Group], val_fraction: float, seed: int
) -> tuple[list[Group], list[Group]]:
    """Assign whole groups to train/test, rarest class first.

    Rarest first matters on a small dataset: by the time the test set is full, a class
    with six instances has either been placed or lost. Handling those groups while both
    sides are still empty is what keeps every class present in the holdout.
    """
    labelled = [g for g in groups if not g.is_background]
    background = [g for g in groups if g.is_background]

    totals: Counter = Counter()
    for group in labelled:
        totals.update(group.classes)
    targets = {cls: count * val_fraction for cls, count in totals.items()}

    # Two independent targets so background images end up proportionally represented too,
    # instead of being crowded out by whatever order they happen to be visited in.
    labelled_target = sum(g.size for g in labelled) * val_fraction
    background_target = sum(g.size for g in background) * val_fraction

    # One random priority per group, drawn in key order so it depends on --seed and the
    # group names alone, never on the order the filesystem handed us the files. It is also
    # the tie-break below: breaking ties on the group key instead would make the sort a
    # total order, and --seed would silently stop changing anything.
    rng = random.Random(seed)
    priority = {group.key: rng.random() for group in sorted(groups, key=lambda g: g.key)}

    def rarest(group: Group) -> int:
        return min(group.classes, key=lambda cls: (totals[cls], cls))

    labelled.sort(key=lambda g: (totals[rarest(g)], -g.classes[rarest(g)], priority[g.key]))
    background.sort(key=lambda g: priority[g.key])

    train: list[Group] = []
    test: list[Group] = []
    test_images = 0
    test_classes: Counter = Counter()

    for group in labelled:
        # Any under-target class is enough to take the group. Requiring it of the RAREST
        # class instead would stall the test set well short of --val-fraction: once the
        # scarce classes are satisfied, every remaining group would fall through to train
        # even with room to spare. Rare classes are already prioritised by the sort above.
        wanted = any(test_classes[cls] < targets[cls] for cls in group.classes)
        # A group that does not fit is skipped rather than ending the loop — a later,
        # smaller group can still take the remaining room.
        if wanted and moves_closer(test_images, group.size, labelled_target):
            test.append(group)
            test_images += group.size
            test_classes.update(group.classes)
        else:
            train.append(group)

    background_images = 0
    for group in background:
        if moves_closer(background_images, group.size, background_target):
            test.append(group)
            background_images += group.size
        else:
            train.append(group)

    return train, test


def check_split(
    train: list[Group],
    test: list[Group],
    names: list[str],
    val_fraction: float,
    group_seconds: float | None,
) -> None:
    """Say out loud everything about this split that could mislead someone reading metrics."""
    train_images = sum(g.size for g in train)
    test_images = sum(g.size for g in test)
    total = train_images + test_images

    if not test_images:
        # Almost always a grouping problem rather than a ratio problem: not one group was
        # small enough to fit the holdout. Point at the flag that actually fixes it.
        smallest = min((g.size for g in train), default=0)
        room = round(total * val_fraction)
        if group_seconds and smallest > room:
            logger.error(
                "the test split came out empty — the smallest group holds %d image(s) but "
                "the test set only has room for %d. Lower --group-seconds (currently %s) to "
                "break the groups up, or raise --val-fraction (currently %s)",
                smallest, room, group_seconds, val_fraction,
            )
        else:
            logger.error(
                "the test split came out empty — raise --val-fraction (currently %s) or "
                "add images", val_fraction,
            )
        raise SystemExit(2)
    if not train_images:
        logger.error("the train split came out empty — lower --val-fraction (currently %s)",
                     val_fraction)
        raise SystemExit(2)

    # The assertion this whole design exists for: no source image may appear on both sides.
    train_stems = {s.stem for g in train for s in g.samples}
    test_stems = {s.stem for g in test for s in g.samples}
    leaked = train_stems & test_stems
    if leaked:
        logger.error("%d image(s) leaked across the split, e.g. %s — this is a bug in "
                     "split_dataset.py, do not train on this dataset",
                     len(leaked), sorted(leaked)[:3])
        raise SystemExit(1)

    test_classes: Counter = Counter()
    for group in test:
        test_classes.update(group.classes)
    train_classes: Counter = Counter()
    for group in train:
        train_classes.update(group.classes)

    for cls in sorted(set(train_classes) | set(test_classes)):
        name = names[cls] if cls < len(names) else f"<id {cls}>"
        if not test_classes[cls]:
            logger.warning(
                "class '%s' has NO instances in the test split — its mAP will be "
                "meaningless. Try a different --seed or a smaller --group-seconds", name)
        elif not train_classes[cls]:
            logger.warning("class '%s' has no instances in the train split — the model "
                           "cannot learn it", name)

    if total < _SMALL_DATASET_IMAGES:
        logger.warning("only %d image(s) in total — holdout metrics from a test set this "
                       "small are very noisy, treat them as indicative only", total)

    biggest = max((g.size for g in train + test), default=0)
    if total and biggest > total * val_fraction:
        logger.warning("the largest group holds %d of %d image(s), more than the %.0f%% "
                       "test share — the ratio cannot be hit exactly. Lower "
                       "--group-seconds to break it up", biggest, total, val_fraction * 100)

    actual = test_images / total
    logger.info("split : %d train / %d test (%.1f%% test, asked for %.1f%%)",
                train_images, test_images, actual * 100, val_fraction * 100)
    if abs(actual - val_fraction) > 0.05:
        logger.warning("the actual test share drifted from --val-fraction by more than 5 "
                       "points — groups are indivisible, so exact ratios are not always "
                       "reachable")


def log_class_table(train: list[Group], test: list[Group], names: list[str]) -> None:
    """Print the per-class instance counts — the table to show alongside the metrics."""
    train_classes: Counter = Counter()
    for group in train:
        train_classes.update(group.classes)
    test_classes: Counter = Counter()
    for group in test:
        test_classes.update(group.classes)

    logger.info("%-28s %8s %8s", "class", "train", "test")
    for cls in range(len(names)):
        logger.info("%-28s %8d %8d", names[cls][:28], train_classes[cls], test_classes[cls])

    train_bg = sum(1 for g in train for s in g.samples if s.is_background)
    test_bg = sum(1 for g in test for s in g.samples if s.is_background)
    if train_bg or test_bg:
        logger.info("%-28s %8d %8d", "(background images)", train_bg, test_bg)


# ---------------------------------------------------------------- io


def copy_split(groups: list[Group], split_dir: Path) -> int:
    """Copy each image and its label into <split>/images and <split>/labels.

    Filenames are preserved exactly: the Roboflow hash is the only thread back to the
    annotation task when a label turns out to be wrong.
    """
    images_dir = split_dir / "images"
    labels_dir = split_dir / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    for group in groups:
        for sample in group.samples:
            shutil.copy2(sample.image, images_dir / sample.image.name)
            if sample.label is not None:
                shutil.copy2(sample.label, labels_dir / sample.label.name)
            else:
                # YOLO reads a missing label as "no objects" anyway, but an explicit empty
                # file makes the intent visible to whoever opens the folder next.
                (labels_dir / f"{sample.image.stem}.txt").write_text("", encoding="utf-8")
            copied += 1
    return copied


def write_data_yaml(dest: Path, test_name: str, names: list[str], src: Path) -> Path:
    """Write the Ultralytics dataset config.

    `val:` and `test:` deliberately point at the same folder. There are only two splits
    on disk, but Ultralytics validates against `val:` during training — a config with
    only `train:` and `test:` fails the run — while `yolo val split=test` reads `test:`.
    """
    path = dest / "data.yaml"
    lines = [
        "# Generated by split_dataset.py — regenerate rather than hand-editing,",
        "# except for `path:` below, which is the one line to change if you move",
        "# this folder (e.g. copying it to the GPU server).",
        f"# source export: {src}",
        "",
        f"path: {yaml_str(dest.resolve())}",
        "train: train/images",
        f"val: {test_name}/images",
        f"test: {test_name}/images",
        "",
        f"nc: {len(names)}",
        "names: [" + ", ".join(yaml_str(n) for n in names) + "]",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_report(dest: Path, report: dict) -> Path:
    """Drop a split_report.json so the dataset stays self-describing.

    Same convention as extract_frames.py:171 and data_collector/session.py — the run that
    produced a folder should be readable from inside the folder months later.
    """
    path = dest / "split_report.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


# ---------------------------------------------------------------- preflight


def validate_args(args: argparse.Namespace) -> None:
    """Reject impossible arguments before touching the filesystem."""
    if not 0.0 < args.val_fraction < 1.0:
        logger.error("--val-fraction must be between 0 and 1, exclusive (got %s)",
                     args.val_fraction)
        raise SystemExit(2)
    if args.group_seconds <= 0:
        logger.error("--group-seconds must be greater than 0 (got %s); pass --no-group to "
                     "turn time-block grouping off", args.group_seconds)
        raise SystemExit(2)
    if not args.test_name or "/" in args.test_name or "\\" in args.test_name:
        logger.error("--test-name must be a plain folder name (got %r)", args.test_name)
        raise SystemExit(2)


def resolve_source(src: Path) -> tuple[Path, Path, Path]:
    """Locate train/images, train/labels and data.yaml inside the Roboflow export."""
    if not src.is_dir():
        logger.error("--src folder not found: %s", src)
        raise SystemExit(2)

    images_dir = src / "train" / "images"
    labels_dir = src / "train" / "labels"
    if not images_dir.is_dir():
        logger.error("%s has no train\\images folder — point --src at the folder that "
                     "contains data.yaml and train\\, not at train\\ itself", src)
        raise SystemExit(2)
    if not labels_dir.is_dir():
        logger.error("%s has no train\\labels folder — this export has no annotations", src)
        raise SystemExit(2)

    yaml_path = src / "data.yaml"
    if not yaml_path.is_file():
        logger.error("no data.yaml in %s — the class names live there and cannot be "
                     "guessed from the labels", src)
        raise SystemExit(2)
    return images_dir, labels_dir, yaml_path


def read_names(yaml_path: Path) -> list[str]:
    """Read the class names out of the source data.yaml, order untouched."""
    try:
        data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        logger.error("could not parse %s: %s", yaml_path, error)
        raise SystemExit(2) from error
    if not isinstance(data, dict):
        logger.error("%s is not a YAML mapping", yaml_path)
        raise SystemExit(2)

    names = normalise_names(data.get("names"))
    declared = data.get("nc")
    if isinstance(declared, int) and declared != len(names):
        logger.warning("data.yaml says nc: %d but lists %d name(s) — using the names",
                       declared, len(names))
    return names


def validate_dest(dest: Path, src: Path, overwrite: bool) -> None:
    """Refuse any --dest that could damage the export, before any work is done.

    Checked up front rather than at copy time: being told the destination is unusable
    only after the whole dataset has been scanned and split is needless.
    """
    src_resolved = src.resolve()
    dest_resolved = dest.resolve()
    if dest_resolved == src_resolved:
        logger.error("--dest is the same folder as --src; this tool never writes into the "
                     "export it reads. Pick a different --dest")
        raise SystemExit(2)
    if src_resolved in dest_resolved.parents:
        logger.error("--dest (%s) is inside --src (%s); pick a --dest outside the export",
                     dest, src)
        raise SystemExit(2)
    if dest_resolved in src_resolved.parents:
        logger.error("--src (%s) is inside --dest (%s); pick a --dest that does not contain "
                     "the export", src, dest)
        raise SystemExit(2)

    if dest.exists() and not dest.is_dir():
        logger.error("--dest exists but is not a folder: %s", dest)
        raise SystemExit(2)
    if dest.is_dir() and any(dest.iterdir()) and not overwrite:
        logger.error("%s is not empty. Pick an empty folder, or pass --overwrite to write "
                     "into this one on purpose", dest)
        raise SystemExit(2)


# ---------------------------------------------------------------- run


def run(args: argparse.Namespace) -> int:
    validate_args(args)

    src = Path(args.src)
    dest = Path(args.dest)
    group_seconds = None if args.no_group else args.group_seconds

    images_dir, labels_dir, yaml_path = resolve_source(src)
    if not args.dry_run:
        validate_dest(dest, src, args.overwrite)
    names = read_names(yaml_path)

    samples = collect_samples(images_dir, labels_dir, group_seconds)
    groups = build_groups(samples)

    unknown = {c for s in samples for c in s.classes if c >= len(names)}
    if unknown:
        logger.error("labels reference class id(s) %s but data.yaml only lists %d name(s) "
                     "— the export is inconsistent, re-download it",
                     sorted(unknown), len(names))
        raise SystemExit(2)

    logger.info("source: %s", src)
    logger.info("input : %d image(s), %d class(es): %s", len(samples), len(names), names)
    if group_seconds:
        logger.info("group : %d group(s) — augmented copies together, plus %.1fs video "
                    "blocks", len(groups), group_seconds)
    else:
        logger.info("group : %d group(s) — augmented copies together, no time blocks "
                    "(--no-group)", len(groups))
    if group_seconds and len(groups) == len(samples):
        logger.warning("every image ended up in its own group — no filename matched the "
                       "extract_frames.py `<video>_t<ms>` pattern, so adjacent video "
                       "frames may still land on opposite sides of the split")

    train, test = assign_groups(groups, args.val_fraction, args.seed)
    check_split(train, test, names, args.val_fraction, group_seconds)
    log_class_table(train, test, names)

    if args.dry_run:
        logger.info("dry run — nothing was written. Drop --dry-run to create %s", dest)
        return 0

    dest.mkdir(parents=True, exist_ok=True)
    train_copied = copy_split(train, dest / "train")
    test_copied = copy_split(test, dest / args.test_name)

    written_yaml = write_data_yaml(dest, args.test_name, names, src)

    train_classes: Counter = Counter()
    for group in train:
        train_classes.update(group.classes)
    test_classes: Counter = Counter()
    for group in test:
        test_classes.update(group.classes)

    report_path = write_report(dest, {
        "source_dataset": str(src.resolve()),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "seed": args.seed,
        "val_fraction": args.val_fraction,
        "grouping": "augmented+time-blocks" if group_seconds else "augmented-only",
        "group_seconds": group_seconds,
        "group_count": len(groups),
        "class_names": names,
        "splits": {
            "train": {
                "images": train_copied,
                "background_images": sum(
                    1 for g in train for s in g.samples if s.is_background),
                "groups": len(train),
                "instances_per_class": {
                    names[c]: train_classes[c] for c in range(len(names))},
            },
            args.test_name: {
                "images": test_copied,
                "background_images": sum(
                    1 for g in test for s in g.samples if s.is_background),
                "groups": len(test),
                "instances_per_class": {
                    names[c]: test_classes[c] for c in range(len(names))},
            },
        },
    })

    logger.info("done — %d train + %d %s image(s) in %s",
                train_copied, test_copied, args.test_name, dest)
    logger.info("config: %s (edit the `path:` line if you move this folder)",
                written_yaml.name)
    logger.info("report: %s", report_path.name)
    return 0


# ---------------------------------------------------------------- cli


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Split a Roboflow 100%-train YOLO export into train/test, keeping "
                    "near-duplicate video frames on the same side.",
    )
    p.add_argument("--src", required=True,
                   help="the Roboflow export folder (the one holding data.yaml and train\\)")
    p.add_argument("--dest", required=True,
                   help="folder the split dataset is written to (created if missing); the "
                        "export itself is never modified")
    p.add_argument("--val-fraction", type=float, default=0.2,
                   help="share of the images to hold out for testing (default 0.2)")
    p.add_argument("--group-seconds", type=float, default=10.0,
                   help="frames from one video within this many seconds of each other stay "
                        "on the same side of the split (default 10.0)")
    p.add_argument("--no-group", action="store_true",
                   help="turn off time-block grouping, keeping only Roboflow's augmented "
                        "copies together; for datasets not taken from video (default off)")
    p.add_argument("--seed", type=int, default=42,
                   help="shuffle seed; the same seed always gives the same split "
                        "(default 42)")
    p.add_argument("--test-name", default="test",
                   help="name of the holdout folder (default test)")
    p.add_argument("--dry-run", action="store_true",
                   help="report the split and its warnings without copying anything, so "
                        "--group-seconds and --seed can be tuned first (default off)")
    p.add_argument("--overwrite", action="store_true",
                   help="write into a --dest folder that already has files in it instead "
                        "of refusing (default off)")
    return p


def main() -> int:
    args = build_arg_parser().parse_args()
    logging.basicConfig(level="INFO", format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
