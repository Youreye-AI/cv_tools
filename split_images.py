"""
split_images.py

Splits a flat folder of images into N equal, randomly shuffled subfolders.
Built for dividing a dataset among multiple annotators (e.g., for Roboflow
batch upload + assignment).

Usage:
    python split_images.py --src "C:\path\to\images" --dest "C:\path\to\output" --n 5

What it does:
    - Reads all image files in --src (non-recursive, flat folder)
    - Shuffles them randomly (seeded, so it's reproducible if you re-run it)
    - Splits them into N nearly-equal folders: person_1, person_2, ... person_N
    - Copies files (originals are left untouched) — change COPY_MODE below
      to "move" if you'd rather not duplicate 14k images on disk

Notes:
    - If the total doesn't divide evenly, the first folders get one extra
      image each (e.g., 14000 / 5 = 2800 exactly, but this handles odd
      numbers too, e.g. 14003 / 5 -> 2801, 2801, 2801, 2800, 2800)
    - Supported extensions are listed in IMAGE_EXTENSIONS below — add more
      if needed (e.g. ".bmp", ".tiff")
"""

import argparse
import random
import shutil
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
COPY_MODE = "copy"  # "copy" or "move"
RANDOM_SEED = 42  # change or remove for a different shuffle each run


def split_images(src_dir: Path, dest_dir: Path, n: int):
    if not src_dir.is_dir():
        raise SystemExit(f"Source folder not found: {src_dir}")

    images = [
        f for f in src_dir.iterdir()
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
    ]

    if not images:
        raise SystemExit(f"No images found in {src_dir} (checked extensions: {IMAGE_EXTENSIONS})")

    total = len(images)
    print(f"Found {total} images in {src_dir}")

    random.seed(RANDOM_SEED)
    random.shuffle(images)

    # Compute split sizes (extras go to the first few groups)
    base = total // n
    remainder = total % n
    sizes = [base + 1 if i < remainder else base for i in range(n)]

    dest_dir.mkdir(parents=True, exist_ok=True)

    idx = 0
    for person_num, size in enumerate(sizes, start=1):
        person_folder = dest_dir / f"person_{person_num}"
        person_folder.mkdir(parents=True, exist_ok=True)

        chunk = images[idx: idx + size]
        idx += size

        for img_path in chunk:
            target = person_folder / img_path.name
            if COPY_MODE == "move":
                shutil.move(str(img_path), str(target))
            else:
                shutil.copy2(str(img_path), str(target))

        print(f"  person_{person_num}: {len(chunk)} images -> {person_folder}")

    print(f"\nDone. {total} images split into {n} folders under {dest_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Shuffle and split a flat image folder into N subfolders.")
    parser.add_argument("--src", required=True, help="Path to the folder containing all images")
    parser.add_argument("--dest", required=True, help="Path where person_1..person_N folders will be created")
    parser.add_argument("--n", type=int, default=5, help="Number of people to split images between (default 5)")
    args = parser.parse_args()

    split_images(Path(args.src), Path(args.dest), args.n)