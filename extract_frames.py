"""Sample whole frames out of a recorded video file, for labelling later.

Why
---
Every other collector in this repo pulls from a LIVE RTSP stream (`crops.py`,
`gstreamer_opt.py`, `data_collector/`). Footage that was already recorded — a site
visit, an NVR export — had no way into a dataset without re-streaming it. This tool
decodes a video file and writes one image per sampled frame into a flat folder, at a
rate you choose (frames per second of video, fractions allowed).

The output folder is exactly what the rest of the pipeline expects: feed it straight to
`split_images.py` to divide it among annotators, or upload it to Roboflow/CVAT.

Frames are saved WHOLE and unmodified — no detection, no cropping. Bounding boxes get
drawn downstream in the annotation tool.

Run it
------
    # 2 frames per second of video
    python extract_frames.py --video clip.mp4 --out .\\datasets\\apron_batch_2 --crops-per-second 2

    # one frame every 5 seconds, only 1:00 -> 4:30, capped at 500 images
    python extract_frames.py --video clip.mp4 --out .\\out -r 0.2 --start 60 --end 270 --max-frames 500
"""
from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger("extract_frames")

IMAGE_FORMATS = ("jpg", "png")

# Same set split_images.py scans for, so a folder produced here is directly consumable.
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# Only used to warn on an odd input — NVR exports are often .dav/.264/.ts, and OpenCV
# can still open plenty of those, so an unknown extension must not be a hard error.
VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".m4v", ".mpg", ".mpeg", ".wmv", ".ts"}

# Containers routinely lie about fps (0, NaN, or a nonsense value on NVR exports).
# Anything outside this range is treated as unusable rather than silently trusted.
_MIN_PLAUSIBLE_FPS = 0.1
_MAX_PLAUSIBLE_FPS = 1000.0

_PNG_COMPRESSION = 3  # png is only for lossless runs; 3 keeps writes fast
_PROGRESS_EVERY = 25  # log a progress line every N saved images


# ---------------------------------------------------------------- pure helpers


def resolve_source_fps(container_fps: float, override: float | None) -> float:
    """Return the fps to sample against, or exit if neither source is trustworthy.

    Guessing here would silently corrupt the whole batch: a wrong fps makes
    --crops-per-second, --start and --end all mean something other than what the user
    asked for. So a bad container fps is a hard error that names the override flag.
    """
    if override is not None:
        return override
    if (
        container_fps is None
        or not np.isfinite(container_fps)
        or not (_MIN_PLAUSIBLE_FPS <= container_fps <= _MAX_PLAUSIBLE_FPS)
    ):
        logger.error(
            "the video does not report a usable frame rate (got %r). Pass the real one "
            "with --source-fps (e.g. --source-fps 25)",
            container_fps,
        )
        raise SystemExit(2)
    return float(container_fps)


def frame_step(source_fps: float, crops_per_second: float) -> float:
    """Frames of video to advance between two saved images.

    Kept as a float so fractional rates stay exact over a long video — advancing the
    target by `step` (rather than rounding each hop) means 12.5 averages out to 12.5
    instead of drifting to 12 or 13 over thousands of frames.
    """
    return max(1.0, source_fps / crops_per_second)


def build_filename(stem: str, position_ms: float, image_format: str) -> str:
    """Name an output image after its position IN THE VIDEO, not wall-clock time.

    Wall-clock (what data_collector/session.py uses for live streams) is meaningless
    for a recording. The video timestamp sorts chronologically, cannot collide within
    a run, and lets anyone scrub straight to that moment in the source clip when a
    label looks wrong months later.
    """
    return f"{stem}_t{int(round(position_ms)):08d}.{image_format}"


def resize_frame(frame: np.ndarray, max_width: int | None) -> np.ndarray:
    """Downscale wide frames; never upscale (that would invent detail annotators trust)."""
    if max_width is None:
        return frame
    height, width = frame.shape[:2]
    if width <= max_width:
        return frame
    new_height = max(1, round(height * max_width / width))
    return cv2.resize(frame, (max_width, new_height), interpolation=cv2.INTER_AREA)


# ---------------------------------------------------------------- io


def save_image(frame: np.ndarray, path: Path, image_format: str, quality: int) -> bool:
    """Encode in memory and write the bytes ourselves, instead of cv2.imwrite().

    cv2.imwrite() silently returns False on Windows when the path contains non-ASCII
    characters — a Turkish folder name would produce an empty dataset with no error.
    imencode + Path.write_bytes goes through Python's own (unicode-safe) file layer.
    """
    if image_format == "jpg":
        params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    else:
        params = [cv2.IMWRITE_PNG_COMPRESSION, _PNG_COMPRESSION]
    ok, buffer = cv2.imencode(f".{image_format}", frame, params)
    if not ok:
        return False
    path.write_bytes(buffer.tobytes())
    return True


def iter_sampled_frames(
    cap: cv2.VideoCapture,
    step: float,
    start_frame: int,
    end_frame: int | None,
    max_frames: int | None,
) -> Iterator[tuple[int, np.ndarray]]:
    """Yield (frame_index, frame) for the sampled frames only.

    Uses grab() to skip and retrieve() only on the frames we keep: grab advances the
    decoder without colour-converting or allocating an array, so skipping 29 of every
    30 frames costs almost nothing. Deliberately no CAP_PROP_POS_FRAMES seeking —
    seeking is unreliable on B-frame and variable-frame-rate footage, where it lands
    on the wrong frame or fails outright.
    """
    index = 0
    saved = 0
    next_target = float(start_frame)
    while True:
        if end_frame is not None and index >= end_frame:
            break
        if max_frames is not None and saved >= max_frames:
            break
        if not cap.grab():
            break  # end of file, or a truncated/corrupt tail — stop cleanly
        if index >= next_target:
            ok, frame = cap.retrieve()
            if not ok:
                break
            yield index, frame
            saved += 1
            next_target += step
        index += 1


def write_metadata(out_dir: Path, meta: dict) -> Path:
    """Drop a metadata.json beside the images so a batch stays self-describing.

    Same idea as data_collector/session.py:write_metadata(); that module can't be
    imported from here (data_collector/ has no __init__.py and its modules import each
    other by bare name) and its fields are RTSP-specific anyway.
    """
    path = out_dir / "metadata.json"
    if path.exists():
        # An --append run must not erase the record of the run that came before it.
        path = out_dir / f"metadata_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


# ---------------------------------------------------------------- preflight


def validate_args(args: argparse.Namespace) -> None:
    """Reject impossible arguments before opening the video, with the fix in the message."""
    if args.crops_per_second <= 0:
        logger.error("--crops-per-second must be greater than 0 (got %s)", args.crops_per_second)
        raise SystemExit(2)
    if args.source_fps is not None and not (
        _MIN_PLAUSIBLE_FPS <= args.source_fps <= _MAX_PLAUSIBLE_FPS
    ):
        logger.error("--source-fps must be between %s and %s (got %s)",
                     _MIN_PLAUSIBLE_FPS, _MAX_PLAUSIBLE_FPS, args.source_fps)
        raise SystemExit(2)
    if args.start < 0:
        logger.error("--start must be 0 or greater (got %s)", args.start)
        raise SystemExit(2)
    if args.end is not None and args.end <= args.start:
        logger.error("--end (%s) must be greater than --start (%s)", args.end, args.start)
        raise SystemExit(2)
    if args.max_frames is not None and args.max_frames <= 0:
        logger.error("--max-frames must be greater than 0 (got %s)", args.max_frames)
        raise SystemExit(2)
    if args.max_width is not None and args.max_width <= 0:
        logger.error("--max-width must be greater than 0 (got %s)", args.max_width)
        raise SystemExit(2)
    if not (1 <= args.quality <= 100):
        logger.error("--quality must be between 1 and 100 (got %s)", args.quality)
        raise SystemExit(2)


def open_video(video_path: Path) -> cv2.VideoCapture:
    """Open the file for decoding, or exit with a readable reason."""
    if not video_path.exists():
        logger.error("video not found: %s", video_path)
        raise SystemExit(2)
    if not video_path.is_file():
        logger.error("--video must be a file, not a folder: %s", video_path)
        raise SystemExit(2)
    if video_path.suffix.lower() not in VIDEO_EXTENSIONS:
        logger.warning("unusual video extension %s — trying to open it anyway",
                       video_path.suffix or "(none)")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        logger.error(
            "could not open %s — the file may be corrupt, or this OpenCV build lacks a "
            "decoder for its codec. Try remuxing it: ffmpeg -i \"%s\" -c copy out.mp4",
            video_path, video_path,
        )
        raise SystemExit(2)
    return cap


def prepare_output_dir(out_dir: Path, append: bool) -> None:
    """Create the output folder, refusing to mix a new batch into an old one.

    Silently adding images to a folder that already holds a labelled batch is the kind
    of mistake that is only noticed after annotation has started, so it takes an
    explicit --append.
    """
    if out_dir.exists() and not out_dir.is_dir():
        logger.error("--out exists but is not a folder: %s", out_dir)
        raise SystemExit(2)
    if out_dir.is_dir():
        existing = sum(
            1 for p in out_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
        if existing and not append:
            logger.error(
                "%s already contains %d image(s). Pick an empty folder, or pass --append "
                "to add to this batch on purpose",
                out_dir, existing,
            )
            raise SystemExit(2)
    out_dir.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------- run


def run(args: argparse.Namespace) -> int:
    validate_args(args)

    video_path = Path(args.video)
    out_dir = Path(args.out)

    cap = open_video(video_path)
    try:
        source_fps = resolve_source_fps(cap.get(cv2.CAP_PROP_FPS), args.source_fps)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            total_frames = 0  # unknown (common on streams remuxed without an index)

        if args.crops_per_second > source_fps:
            logger.warning(
                "--crops-per-second %s is higher than the video's %.3f fps — saving every "
                "frame instead (frames cannot be invented)",
                args.crops_per_second, source_fps,
            )

        step = frame_step(source_fps, args.crops_per_second)
        start_frame = int(args.start * source_fps)
        end_frame = int(args.end * source_fps) if args.end is not None else None

        if total_frames and start_frame >= total_frames:
            logger.error(
                "--start %.3fs is past the end of the video (%.3fs)",
                args.start, total_frames / source_fps,
            )
            raise SystemExit(2)

        prepare_output_dir(out_dir, args.append)

        duration = f"{total_frames / source_fps:.2f}s" if total_frames else "unknown"
        logger.info("video : %s", video_path)
        logger.info("source: %dx%d @ %.3f fps, %s, %s frames",
                    width, height, source_fps, duration, total_frames or "?")
        logger.info("sample: %s frame(s)/s -> every %.2f frame(s)", args.crops_per_second, step)
        logger.info("output: %s (%s, quality %d)", out_dir, args.format, args.quality)

        stem = video_path.stem
        saved = 0
        failed = 0
        last_ms = 0.0
        interrupted = False

        try:
            for index, frame in iter_sampled_frames(
                cap, step, start_frame, end_frame, args.max_frames
            ):
                # Derived from the fps we actually sampled with, not CAP_PROP_POS_MSEC:
                # deterministic, honours --source-fps, and consistent with --start/--end.
                position_ms = index / source_fps * 1000.0
                image = resize_frame(frame, args.max_width)
                path = out_dir / build_filename(stem, position_ms, args.format)
                if save_image(image, path, args.format, args.quality):
                    saved += 1
                    last_ms = position_ms
                else:
                    failed += 1
                    logger.warning("could not encode frame %d — skipped", index)

                if saved and saved % _PROGRESS_EVERY == 0:
                    if total_frames:
                        logger.info("saved %d images (%.1f%% of the video)",
                                    saved, 100.0 * index / total_frames)
                    else:
                        logger.info("saved %d images (at %.2fs)", saved, position_ms / 1000.0)
        except KeyboardInterrupt:
            interrupted = True
            logger.warning("interrupted — keeping the %d image(s) already written", saved)

        if not saved:
            logger.error("no frames were saved — check --start/--end against the video length")
            return 1

        meta_path = write_metadata(out_dir, {
            "source_video": str(video_path.resolve()),
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "source_fps": source_fps,
            "source_fps_overridden": args.source_fps is not None,
            "resolution": {"width": width, "height": height},
            "total_frames": total_frames or None,
            "crops_per_second": args.crops_per_second,
            "frame_step": step,
            "start_seconds": args.start,
            "end_seconds": args.end,
            "max_frames": args.max_frames,
            "image_format": args.format,
            "quality": args.quality if args.format == "jpg" else None,
            "max_width": args.max_width,
            "saved_count": saved,
            "failed_count": failed,
            "last_position_seconds": round(last_ms / 1000.0, 3),
            "interrupted": interrupted,
        })

        logger.info("done — %d image(s) in %s (metadata: %s)", saved, out_dir, meta_path.name)
        if failed:
            logger.warning("%d frame(s) failed to encode", failed)
        return 0
    finally:
        cap.release()


# ---------------------------------------------------------------- cli


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Sample whole frames from a recorded video into a flat folder for labelling.",
    )
    p.add_argument("--video", required=True, help="path to the recorded video file")
    p.add_argument("--out", required=True,
                   help="folder the images are written to (created if missing)")
    p.add_argument("--crops-per-second", "-r", type=float, default=1.0,
                   help="how many frames to save per second of video; fractions are fine, "
                        "e.g. 0.2 saves one frame every 5 seconds (default 1.0)")
    p.add_argument("--source-fps", type=float, default=None,
                   help="override the video's frame rate; needed when the container "
                        "reports a wrong or missing fps (default: read from the file)")
    p.add_argument("--start", type=float, default=0.0,
                   help="skip the first N seconds of the video (default 0)")
    p.add_argument("--end", type=float, default=None,
                   help="stop at this many seconds into the video (default: end of file)")
    p.add_argument("--max-frames", type=int, default=None,
                   help="stop after saving this many images, whatever the rate "
                        "(default: no cap)")
    p.add_argument("--format", choices=IMAGE_FORMATS, default="jpg",
                   help="output image format (default jpg)")
    p.add_argument("--quality", type=int, default=95,
                   help="jpeg quality 1-100, ignored for png (default 95)")
    p.add_argument("--max-width", type=int, default=None,
                   help="downscale frames wider than this, keeping aspect ratio; never "
                        "upscales (default: keep original size)")
    p.add_argument("--append", action="store_true",
                   help="write into an output folder that already contains images "
                        "instead of refusing (default off)")
    return p


def main() -> int:
    args = build_arg_parser().parse_args()
    logging.basicConfig(level="INFO", format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
