"""
RTSP Human Crop Collector
=========================
Goes back to past dates and collects human crops from an RTSP stream.
Runs person detection with YOLO11l, saving a crop every N frames.
Automatically uses the GPU when CUDA is available.

All settings are made in the CONFIG block below — no command line needed.

Capture modes (CONFIG["CAPTURE_MODE"])
---------------------------------------
There are three ways this script can acquire frames, selected via the
"CAPTURE_MODE" key in CONFIG:

  "direct"      - Default. cv2.VideoCapture opens the camera's RTSP URL
                  directly via its FFMPEG backend. Decoding happens on
                  CPU.
  "local_relay" - Fallback for machines without an NVIDIA GPU/nvcodec
                  plugin. See "Running with GStreamer (local_relay)"
                  below.
  "nvdec"       - Recommended when available. This script spawns and
                  manages its own GStreamer subprocess per segment,
                  decoding via the NVIDIA GPU (NVDEC). See "Running with
                  NVIDIA GPU decode (nvdec)" below.

Requirements for "local_relay" and "nvdec": GStreamer for Windows —
download the "runtime" AND "development" MSVC 64-bit installers from
https://gstreamer.freedesktop.org/download/ and pick "Complete" during
setup. "local_relay" needs gst-launch-1.0 on PATH in whatever terminal
you run the external relay command from. "nvdec" is more forgiving:
this script auto-detects gst-launch-1.0.exe (checking PATH first, then
the default install path `C:\\Program Files\\gstreamer\\1.0\\msvc_x86_64\\bin`),
so it works even if PATH isn't set in the terminal/IDE you launch
crops.py from — set CONFIG["GST_LAUNCH_EXE"] only if GStreamer was
installed somewhere non-standard.

Both modes need to know the camera's codec first:
    gst-launch-1.0 rtspsrc location="<rtsp url>" ! decodebin ! fakesink -v | Select-String "x-h264|x-h265"
If the caps show "x-h265", use the H.265 elements below; if "x-h264",
use the H.264 ones instead.

Running with NVIDIA GPU decode (CAPTURE_MODE="nvdec")
------------------------------------------------------
1) Requires the GStreamer "nvcodec" plugin (NVIDIA's own NVDEC decoder).
   Confirm it's present:
       gst-inspect-1.0 nvh265dec      (or nvh264dec for H.264)

2) Unlike "local_relay", you don't run anything externally — this script
   builds and spawns the GStreamer pipeline itself per segment (so it can
   restart it automatically with the correct starttime each time),
   reading raw decoded BGR frames straight from the subprocess's stdout
   pipe. No video decoding happens in OpenCV/FFMPEG for this mode.

3) Set these in CONFIG below:
       "CAPTURE_MODE": "nvdec",
       "NVDEC_CODEC": "h265",     # or "h264" — must match the camera
       "NVDEC_WIDTH": 2592,       # camera's native decoded frame width
       "NVDEC_HEIGHT": 1944,      # camera's native decoded frame height
   Width/height must be exact — they're used to size the raw-byte reads
   from the pipe. Get them from the caps line in the codec-check command
   above (e.g. "width=(int)2592, height=(int)1944").

4) If your camera/NVR rejects "?starttime=" (live-only devices reply
   with RTSP 400 Bad Request — test with the codec-check command above,
   with and without a starttime query string, to find out), set
       "APPEND_STARTTIME": False
   so the URL is used exactly as given, with no per-segment time-shifting.

5) Run this script normally: python crops.py

Running with GStreamer (CAPTURE_MODE="local_relay")
-----------------------------------------------------
This script itself doesn't call GStreamer in this mode — OpenCV's FFMPEG
backend can already read "udp://" addresses. Instead, you pre-process
the RTSP connection yourself with an external GStreamer pipeline
(forcing TCP transport, tuning the jitter buffer, etc.) and point that
pipeline's output at a local UDP address; this script then connects to
that fixed address instead of the camera directly. Decoding still
happens later, on CPU, inside OpenCV/FFMPEG — use "nvdec" instead if you
want GPU decode.

1) Start the relay in a separate terminal (bake the starttime you want
   into the URL yourself):
       gst-launch-1.0 -e rtspsrc location="rtsp://your2eye5td:hkYLb-L5dy3Z4T@31.96.64.122:554/ch17/0" protocols=udp latency=150 ! rtph265depay ! h265parse config-interval=-1 ! mpegtsmux ! udpsink host=127.0.0.1 port=5000 sync=false

   This pipeline doesn't decode anything — it just re-packages the
   transport (RTP → MPEG-TS/UDP), so there's no quality or CPU cost on
   the GStreamer side. To switch time ranges, stop this command
   (Ctrl+C), change starttime, and restart it.

2) Set these in CONFIG below:
       "RTSP_BASE_URL": "udp://127.0.0.1:5000",
       "CAPTURE_MODE": "local_relay",
   (In this mode, the script no longer appends starttime to the URL and
   always reconnects to the same local address between segments — the
   time range is controlled by the external command from step 1.)

3) Run this script normally: python crops.py
"""

import time
import threading
import queue
import subprocess
import shutil
from datetime import datetime, timezone, timedelta
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

# ╔══════════════════════════════════════════════════════════════╗
# ║                        CONFIG                               ║
# ║   Make all your settings here, don't touch anything else.    ║
# ╚══════════════════════════════════════════════════════════════╝

CONFIG = {
    # ── RTSP ──────────────────────────────────────────────────
    # Base URL — write it without the starttime parameter.
    # The code automatically appends the ?starttime=... parameter.
    # NOTE: for CAPTURE_MODE="nvdec" or "direct", this must be the
    # camera's actual rtsp:// address. For CAPTURE_MODE="local_relay",
    # it must be the external relay's local udp:// address (see the
    # note at the top of this file).
    "RTSP_BASE_URL": "rtsp://your2eye5td:hkYLb-L5dy3Z4T@31.96.64.122:554/ch17/0",

    # False → ?starttime= is not appended to RTSP_BASE_URL (some
    # live-only cameras/NVRs reject this parameter with RTSP 400 Bad
    # Request — this camera (/ch1/0) is one of those, confirmed). When
    # True, each segment connects with ?starttime= starting from
    # START_TIME below (for NVRs that support playback, e.g. Hikvision
    # "Streaming/tracks").
    "APPEND_STARTTIME": False,

    # Start and end time (UTC, ISO 8601 basic format)
    # Format: YYYYMMDDTHHMMSSZ
    "START_TIME": "20260620T090000Z",
    "END_TIME":   "20260620T200000Z",

    # ── Detection ─────────────────────────────────────────────
    # YOLO model file (yolo11l.pt / yolov8l.pt etc.)
    "MODEL": "yolo11l.pt",

    # Minimum confidence threshold (0.0-1.0)
    "CONF": 0.5,

    # Run YOLO and take a crop every how many frames?
    # 10 = every 10th frame is processed (others are skipped)
    "FRAME_SKIP": 5,

    # ── Output ────────────────────────────────────────────────
    # Folder where crops will be saved
    "OUTPUT_DIR": "./waiter_waitress_mysa",

    # ── Connection ────────────────────────────────────────────
    # How many seconds of time each RTSP session should cover.
    # Set according to the NVR/camera's max playback duration (default: 5 min)
    "SEGMENT_SECONDS": 300,

    # How frames will be acquired:
    #   "direct"      → cv2.VideoCapture connects directly to the
    #                    camera's RTSP URL (FFMPEG backend, CPU decode).
    #   "local_relay" → RTSP_BASE_URL is the local output of an external
    #                    GStreamer relay (e.g. "udp://127.0.0.1:5000").
    #                    starttime is not appended, the URL doesn't
    #                    change between segments — you manage the time
    #                    range yourself via the gst-launch command you
    #                    run externally (decode still on CPU).
    #   "nvdec"       → This script spawns its own GStreamer subprocess,
    #                    decodes via the NVIDIA GPU (NVDEC), and reads
    #                    raw BGR frames from the pipe. See the NVDEC_*
    #                    settings below.
    "CAPTURE_MODE": "nvdec",

    # ── NVDEC (for CAPTURE_MODE="nvdec") ─────────────────────────
    # The camera's codec: "h264" or "h265"
    "NVDEC_CODEC": "h265",

    # Decoded frame size — must be exact, since it's used to determine
    # the number of bytes to read from the pipe (width*height*3).
    "NVDEC_WIDTH": 2592,
    "NVDEC_HEIGHT": 1944,

    # rtspsrc jitter-buffer latency (ms) and transport protocol
    "NVDEC_LATENCY_MS": 150,
    "NVDEC_PROTOCOLS": "udp",

    # Optional: full path to gst-launch-1.0.exe. Leave as None to
    # auto-detect (checks PATH first, then the default GStreamer install
    # location). Only set this if GStreamer was installed somewhere
    # non-standard and isn't on PATH.
    "GST_LAUNCH_EXE": None,

    # ── Visual / Log ──────────────────────────────────────────
    # True → open an OpenCV window (set to False on a server)
    "SHOW_WINDOW": True,

    # Maximum width to display on screen (pixels).
    # Regardless of the camera's resolution, it's scaled to fit this width.
    "DISPLAY_WIDTH": 1280,

    # True → print a log line for every saved crop
    "VERBOSE": False,
}

# ══════════════════════════════════════════════════════════════


# ─────────────────────────────────────────────
# Device selection
# ─────────────────────────────────────────────

def select_device() -> str:
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        print(f"[GPU] CUDA active → {name}")
        return "cuda"
    else:
        print("[CPU] CUDA not found, using CPU.")
        return "cpu"


# ─────────────────────────────────────────────
# Helper: RTSP URL builder
# ─────────────────────────────────────────────

def build_rtsp_url(base_url: str, starttime: str) -> str:
    url = base_url.rstrip("?& ")
    return f"{url}?starttime={starttime}"


def dt_to_rtsp_time(dt: datetime) -> str:
    return dt.strftime("%Y%m%dT%H%M%SZ")


def rtsp_time_to_dt(s: str) -> datetime:
    return datetime.strptime(s, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)


# ─────────────────────────────────────────────
# Frame reader (separate thread)
# ─────────────────────────────────────────────

class RTSPReader(threading.Thread):
    def __init__(self, url: str, frame_queue: queue.Queue, max_queue: int = 64):
        super().__init__(daemon=True)
        self.url = url
        self.q = frame_queue
        self.max_queue = max_queue
        self.running = False
        self.cap = None
        self.error: str | None = None

    def run(self):
        self.running = True
        self.cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)

        if not self.cap.isOpened():
            self.error = f"[ERROR] Could not open stream: {self.url}"
            self.running = False
            return

        print(f"[OK] Stream connection established.")

        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                self.error = "[WARNING] Could not read frame – stream may have ended."
                break
            if self.q.qsize() < self.max_queue:
                self.q.put(frame)

        self.cap.release()
        self.running = False

    def stop(self):
        self.running = False


# ─────────────────────────────────────────────
# Frame reader via NVIDIA GPU (NVDEC) decode
# ─────────────────────────────────────────────

# Default install location of the GStreamer Windows MSVC 64-bit package.
# Used as a fallback when "gst-launch-1.0.exe" isn't on PATH in the
# current shell/environment (a recurring issue across terminals/IDEs).
_GST_LAUNCH_FALLBACK = r"C:\Program Files\gstreamer\1.0\msvc_x86_64\bin\gst-launch-1.0.exe"


def resolve_gst_launch_exe(cfg: dict) -> str:
    override = cfg.get("GST_LAUNCH_EXE")
    if override:
        return override

    found = shutil.which("gst-launch-1.0.exe") or shutil.which("gst-launch-1.0")
    if found:
        return found

    if Path(_GST_LAUNCH_FALLBACK).is_file():
        return _GST_LAUNCH_FALLBACK

    raise FileNotFoundError(
        "gst-launch-1.0.exe not found on PATH or at the default GStreamer "
        f"install location ({_GST_LAUNCH_FALLBACK}). Install GStreamer for "
        "Windows (see the module docstring) or set CONFIG[\"GST_LAUNCH_EXE\"] "
        "to its full path."
    )


def build_nvdec_gst_args(url: str, codec: str, width: int, height: int,
                          latency_ms: int, protocols: str, gst_launch_exe: str) -> list[str]:
    if codec == "h265":
        depay, parse, decoder = "rtph265depay", "h265parse", "nvh265dec"
    elif codec == "h264":
        depay, parse, decoder = "rtph264depay", "h264parse", "nvh264dec"
    else:
        raise ValueError(f"Unsupported NVDEC_CODEC: {codec!r} (must be h264 or h265)")

    return [
        gst_launch_exe, "-q",
        "rtspsrc", f"location={url}", f"protocols={protocols}", f"latency={latency_ms}",
        "!", depay,
        "!", parse,
        "!", decoder,
        "!", "videoconvert",
        "!", f"video/x-raw,format=BGR,width={width},height={height}",
        "!", "fdsink", "fd=1",
    ]


class GstNvDecodeReader(threading.Thread):
    """Same interface as RTSPReader (running/error/is_alive/stop), but frames
    are read as raw BGR bytes from the stdout of a gst-launch-1.0 subprocess
    that decodes via the NVIDIA GPU (NVDEC), instead of through OpenCV."""

    def __init__(self, url: str, frame_queue: queue.Queue, cfg: dict, max_queue: int = 64):
        super().__init__(daemon=True)
        self.url = url
        self.q = frame_queue
        self.max_queue = max_queue
        self.width = cfg["NVDEC_WIDTH"]
        self.height = cfg["NVDEC_HEIGHT"]
        self.frame_bytes = self.width * self.height * 3
        self.gst_args = build_nvdec_gst_args(
            url,
            cfg["NVDEC_CODEC"],
            self.width,
            self.height,
            cfg.get("NVDEC_LATENCY_MS", 150),
            cfg.get("NVDEC_PROTOCOLS", "tcp"),
            resolve_gst_launch_exe(cfg),
        )
        self.running = False
        self.proc: subprocess.Popen | None = None
        self.error: str | None = None

    def run(self):
        self.running = True
        self.proc = subprocess.Popen(
            self.gst_args, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
        )

        print(f"[OK] NVDEC pipeline started (pid={self.proc.pid}).")

        while self.running:
            buf = b""
            while len(buf) < self.frame_bytes:
                chunk = self.proc.stdout.read(self.frame_bytes - len(buf))
                if not chunk:
                    self.error = "[WARNING] Could not read frame from NVDEC pipeline – stream may have ended."
                    self.running = False
                    break
                buf += chunk

            if len(buf) < self.frame_bytes:
                break

            frame = np.frombuffer(buf, dtype=np.uint8).reshape((self.height, self.width, 3))
            if self.q.qsize() < self.max_queue:
                self.q.put(frame)

        self._terminate_proc()
        self.running = False

    def _terminate_proc(self):
        if self.proc is None:
            return
        self.proc.terminate()
        try:
            self.proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            self.proc.kill()

    def stop(self):
        self.running = False
        self._terminate_proc()


# ─────────────────────────────────────────────
# Crop saver
# ─────────────────────────────────────────────

class CropSaver:
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.count = 0

    def save(self, frame, box, label: str = "person") -> str:
        x1, y1, x2, y2 = map(int, box)
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return ""

        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = self.output_dir / f"{label}{ts}{self.count:06d}.jpg"
        cv2.imwrite(str(filename), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
        self.count += 1
        return str(filename)


# ─────────────────────────────────────────────
# Main collector
# ─────────────────────────────────────────────

class HumanCropCollector:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.device = select_device()
        self.model = YOLO(cfg["MODEL"])
        self.model.to(self.device)          # move the model to GPU/CPU
        self.saver = CropSaver(cfg["OUTPUT_DIR"])
        self.frame_q: queue.Queue = queue.Queue(maxsize=128)
        self.reader: RTSPReader | GstNvDecodeReader | None = None

        # Statistics
        self.total_frames = 0
        self.total_crops  = 0

    # ── Segment connection ───────────────────

    def _build_segment_url(self, starttime: str) -> str:
        # Some cameras/NVRs don't accept the ?starttime= parameter and
        # reject the connection with RTSP 400 Bad Request (live-only
        # devices). With APPEND_STARTTIME=False, this parameter isn't
        # appended.
        if self.cfg.get("APPEND_STARTTIME", True):
            return build_rtsp_url(self.cfg["RTSP_BASE_URL"], starttime)
        return self.cfg["RTSP_BASE_URL"]

    def _connect_segment(self, starttime: str):
        if self.reader and self.reader.is_alive():
            self.reader.stop()
            self.reader.join(timeout=3)

        while not self.frame_q.empty():
            try:
                self.frame_q.get_nowait()
            except queue.Empty:
                break

        mode = self.cfg.get("CAPTURE_MODE", "direct")

        if mode == "local_relay":
            # The time range is managed by the external gst-launch
            # command, starttime isn't appended here — connect to the
            # fixed local address.
            url = self.cfg["RTSP_BASE_URL"]
            print(f"  URL (local relay): {url}")
            self.reader = RTSPReader(url, self.frame_q)

        elif mode == "nvdec":
            url = self._build_segment_url(starttime)
            print(f"  URL (NVDEC): {url}")
            self.reader = GstNvDecodeReader(url, self.frame_q, self.cfg)

        else:
            url = self._build_segment_url(starttime)
            print(f"  URL: {url}")
            self.reader = RTSPReader(url, self.frame_q)

        self.reader.start()
        time.sleep(1.5)

    # ── Visualization ────────────────────────

    def _resize_for_display(self, frame):
        max_w = self.cfg.get("DISPLAY_WIDTH", 1280)
        h, w = frame.shape[:2]
        if w <= max_w:
            return frame
        scale = max_w / w
        return cv2.resize(frame, (max_w, int(h * scale)), interpolation=cv2.INTER_AREA)

    def _draw_boxes(self, frame, results):
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf_val = float(box.conf[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"person {conf_val:.2f}",
                    (x1, max(y1 - 8, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 0),
                    2,
                )
        return frame

    # ── Segment processing ───────────────────

    def _process_segment(self, seg_start: datetime, seg_end: datetime) -> bool:
        starttime_str = dt_to_rtsp_time(seg_start)
        print(f"\n[SEGMENT] {starttime_str}  →  {dt_to_rtsp_time(seg_end)}")
        self._connect_segment(starttime_str)

        seg_duration   = (seg_end - seg_start).total_seconds()
        seg_start_wall = time.time()
        frame_skip     = self.cfg["FRAME_SKIP"]
        frame_counter  = 0

        while True:
            elapsed = time.time() - seg_start_wall
            if elapsed >= seg_duration:
                break

            if not self.reader.is_alive() and self.frame_q.empty():
                if self.reader.error:
                    print(f"  {self.reader.error}")
                break

            try:
                frame = self.frame_q.get(timeout=2.0)
            except queue.Empty:
                continue

            self.total_frames += 1
            frame_counter += 1

            if frame_counter % frame_skip != 0:
                continue

            # ── YOLO detection (send to GPU via device=self.device) ──
            results = self.model(
                frame,
                classes=[0], # person class only --- can be changed to detect other classes if needed
                conf=self.cfg["CONF"],
                device=self.device,
                verbose=False,
            )

            n_crops = 0
            for r in results:
                for box in r.boxes:
                    path = self.saver.save(frame, box.xyxy[0].cpu().numpy())
                    if path:
                        n_crops += 1
                        if self.cfg["VERBOSE"]:
                            print(f"  [CROP] {path}  (conf={float(box.conf[0]):.2f})")

            self.total_crops += n_crops

            if n_crops:
                print(
                    f"  frame#{frame_counter:6d} | "
                    f"elapsed {elapsed:5.0f}s | "
                    f"+{n_crops} crop → total {self.total_crops}"
                )

            # ── Visual window ──
            if self.cfg["SHOW_WINDOW"]:
                vis = self._draw_boxes(frame.copy(), results)
                overlay = (
                    f"[{self.device.upper()}] {starttime_str}  |  "
                    f"{elapsed:.0f}s / {seg_duration:.0f}s  |  "
                    f"frame#{frame_counter}  |  crops: {self.total_crops}"
                )
                cv2.putText(vis, overlay, (10, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 2)
                vis = self._resize_for_display(vis)
                cv2.namedWindow("RTSP Human Collector", cv2.WINDOW_NORMAL)
                cv2.imshow("RTSP Human Collector", vis)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    return False

        return True

    # ── Main loop ────────────────────────────

    def run(self):
        start_dt = rtsp_time_to_dt(self.cfg["START_TIME"])
        end_dt   = rtsp_time_to_dt(self.cfg["END_TIME"])
        seg_dur  = timedelta(seconds=self.cfg["SEGMENT_SECONDS"])

        print("=" * 62)
        print(f"  Device     : {self.device.upper()}")
        print(f"  Model      : {self.cfg['MODEL']}")
        print(f"  RTSP Base  : {self.cfg['RTSP_BASE_URL']}")
        print(f"  Start      : {self.cfg['START_TIME']}")
        print(f"  End        : {self.cfg['END_TIME']}")
        print(f"  Frame skip : process every {self.cfg['FRAME_SKIP']} frame(s)")
        print(f"  Conf       : {self.cfg['CONF']}")
        print(f"  Output     : {self.cfg['OUTPUT_DIR']}")
        mode = self.cfg.get("CAPTURE_MODE", "direct")
        if mode == "local_relay":
            print(f"  Mode       : LOCAL RELAY (external GStreamer → {self.cfg['RTSP_BASE_URL']})")
        elif mode == "nvdec":
            print(f"  Mode       : NVDEC ({self.cfg.get('NVDEC_CODEC')}, "
                  f"{self.cfg.get('NVDEC_WIDTH')}x{self.cfg.get('NVDEC_HEIGHT')}, GPU decode)")
        print("=" * 62)

        current = start_dt
        try:
            while current < end_dt:
                seg_end = min(current + seg_dur, end_dt)
                ok = self._process_segment(current, seg_end)
                if not ok:
                    print("\n[STOPPED] User exited (Q key).")
                    break
                current = seg_end

        except KeyboardInterrupt:
            print("\n[STOPPED] Exited via Ctrl+C.")

        finally:
            if self.reader:
                self.reader.stop()
            if self.cfg["SHOW_WINDOW"]:
                cv2.destroyAllWindows()

        print("\n" + "=" * 62)
        print(f"  Total frames read    : {self.total_frames}")
        print(f"  Crops saved          : {self.total_crops}")
        print(f"  Output folder        : {self.cfg['OUTPUT_DIR']}")
        print("=" * 62)


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    collector = HumanCropCollector(CONFIG)
    collector.run()