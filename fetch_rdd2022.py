"""Download selected RDD2022 country archives without pulling the whole 13.2 GB bundle.

Why
---
The per-country S3 links printed in the sekilab/RoadDamageDetector README are dead — the bucket
now answers every request with `403 AccessDenied`. The dataset survives on Figshare, but only as a
single 13.26 GB archive, and 10.6 GB of that is Norway alone.

That archive happens to contain one nested zip per country, each occupying a contiguous byte span,
and Figshare's CDN honours HTTP range requests. So the whole bundle never has to be transferred:
read the central directory off the tail, then fetch only the countries asked for. Czech + India +
Japan + United_States is about 2.3 GB instead of 13.26 GB.

Note the licence differs by source: the GitHub README says CC BY-SA 4.0, while the Figshare record
this script downloads from states CC BY 4.0. Attribute the RDD2022 authors either way.

Run it
------
    # the four annotated countries worth having, ~2.3 GB
    python fetch_rdd2022.py --countries Czech,India,Japan,United_States --dest .\\datasets\\rdd2022_raw

    # just the smallest one, to try the pipeline end to end first
    python fetch_rdd2022.py --countries Czech --dest .\\datasets\\rdd2022_raw

    # see what is in the archive and how big each country is, without downloading any of it
    python fetch_rdd2022.py --list

Then build a training set from what lands:

    python build_road_damage_dataset.py --rdd .\\datasets\\rdd2022_raw ^
        --gaziantep .\\datasets\\gaziantepPotholesSplited --dest .\\datasets\\road_damage
"""
from __future__ import annotations

import argparse
import io
import logging
import shutil
import time
import urllib.request
import zipfile
from pathlib import Path

logger = logging.getLogger("fetch_rdd2022")

# Figshare record 21431547, file "RDD2022_released_through_CRDDC2022.zip".
ARCHIVE_URL = "https://ndownloader.figshare.com/files/38030910"

# Norway is 10.6 GB of the 13.26 GB and adds one more country's worth of variety; left out of the
# default on purpose. China_Drone has no test split and a drone viewpoint unlike our footage.
DEFAULT_COUNTRIES = "Czech,India,Japan,United_States"

# Big enough that a country archive costs tens of range requests rather than tens of thousands.
_CHUNK = 8 * 1024 * 1024
_RETRIES = 5


class _RangeReader(io.RawIOBase):
    """Seekable file-like object over an HTTP resource, using range requests."""

    def __init__(self, url: str) -> None:
        self.url = url
        self._pos = 0
        with urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=60) as r:
            self.size = int(r.headers["Content-Length"])

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._pos

    def seek(self, offset: int, whence: int = 0) -> int:
        self._pos = (offset if whence == 0 else
                     self._pos + offset if whence == 1 else
                     self.size + offset)
        return self._pos

    def readinto(self, buffer) -> int:
        # RawIOBase derives read() from readinto(), not the reverse, and io.BufferedReader calls
        # readinto — without this the whole thing raises NotImplementedError on the first read.
        data = self._fetch(len(buffer))
        buffer[:len(data)] = data
        return len(data)

    def _fetch(self, n: int) -> bytes:
        if n <= 0 or self._pos >= self.size:
            return b""
        end = min(self.size - 1, self._pos + n - 1)
        last: Exception | None = None
        for attempt in range(_RETRIES):
            try:
                # Always go through the stable Figshare URL rather than caching the redirect: it
                # resolves to a PRESIGNED S3 link that expires after a few minutes, so a cached
                # target starts returning 403 partway through a multi-GB transfer.
                request = urllib.request.Request(
                    self.url, headers={"Range": f"bytes={self._pos}-{end}"})
                with urllib.request.urlopen(request, timeout=120) as response:
                    data = response.read()
                self._pos += len(data)
                return data
            except Exception as error:
                last = error
                logger.debug("range %d-%d failed (%s), retrying", self._pos, end, error)
                time.sleep(2 * (attempt + 1))
        raise RuntimeError(f"gave up fetching bytes {self._pos}-{end}: {last}")


def open_archive() -> zipfile.ZipFile:
    logger.info("reading the archive index over HTTP (no bulk download yet)")
    reader = _RangeReader(ARCHIVE_URL)
    logger.info("archive: %.2f GB", reader.size / 1e9)
    return zipfile.ZipFile(io.BufferedReader(reader, buffer_size=_CHUNK))


def run(args: argparse.Namespace) -> int:
    archive = open_archive()
    entries = {Path(i.filename).stem: i for i in archive.infolist()
               if i.filename.lower().endswith(".zip")}

    if args.list:
        logger.info("%-20s %10s", "country", "MB")
        for name, info in sorted(entries.items(), key=lambda kv: kv[1].header_offset):
            logger.info("%-20s %10.0f", name, info.compress_size / 1e6)
        return 0

    wanted = [c.strip() for c in args.countries.split(",") if c.strip()]
    unknown = [c for c in wanted if c not in entries]
    if unknown:
        logger.error("unknown country/countries %s. Available: %s",
                     ", ".join(unknown), ", ".join(sorted(entries)))
        raise SystemExit(2)

    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)

    for country in wanted:
        target = dest / country
        if target.is_dir() and any(target.rglob("*.xml")) and not args.overwrite:
            logger.info("%-16s already extracted — skipping (pass --overwrite to redo)", country)
            continue

        info = entries[country]
        logger.info("%-16s fetching %.0f MB", country, info.compress_size / 1e6)
        temp = dest / f"{country}.zip.part"
        started = time.perf_counter()
        try:
            with archive.open(info) as src, open(temp, "wb") as dst:
                shutil.copyfileobj(src, dst, length=_CHUNK)
            elapsed = time.perf_counter() - started
            logger.info("%-16s downloaded in %.0fs (%.1f MB/s)", country, elapsed,
                        temp.stat().st_size / 1e6 / max(elapsed, 1e-6))
            with zipfile.ZipFile(temp) as inner:
                inner.extractall(dest)
        finally:
            # A half-written .part left behind would be extracted as a corrupt archive next run.
            temp.unlink(missing_ok=True)

        images = sum(1 for _ in (dest / country).rglob("*.jpg"))
        xmls = sum(1 for _ in (dest / country).rglob("*.xml"))
        logger.info("%-16s %d image(s), %d annotation(s)", country, images, xmls)

    logger.info("")
    logger.info("done — %s", dest)
    logger.info("RDD2022's own test/ folders hold images with NO annotations; only train/ is "
                "usable, which build_road_damage_dataset.py already accounts for.")
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Fetch selected RDD2022 country archives from Figshare via HTTP range requests.",
    )
    p.add_argument("--countries", default=DEFAULT_COUNTRIES,
                   help=f"comma-separated country archives to fetch (default {DEFAULT_COUNTRIES}; "
                        f"Norway is excluded because it is 10.6 GB on its own)")
    p.add_argument("--dest", default="datasets/rdd2022_raw",
                   help="folder the country folders are extracted into (default datasets/rdd2022_raw)")
    p.add_argument("--list", action="store_true",
                   help="list the countries in the archive and their sizes, then exit without "
                        "downloading (default off)")
    p.add_argument("--overwrite", action="store_true",
                   help="re-fetch a country that is already extracted (default off)")
    return p


def main() -> int:
    args = build_arg_parser().parse_args()
    logging.basicConfig(level="INFO", format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
