"""Where frames come from.

One ``--source`` flag covers everything the user asked for:

======================== =========================================
``0``, ``1`` ...         a local camera by index
``clip.mp4``             a single video file
``clips/``               every video in a folder, played in order
``clips/*.mp4``          a glob of videos
``rtsp://`` / ``http://`` a live network stream
``frames/`` (images)     a folder of stills treated as a sequence
======================== =========================================

Live sources (camera and network streams) can be read on a background thread
so the recogniser always works on the newest frame instead of falling behind a
growing buffer.
"""

from __future__ import annotations

import glob
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .utils import IMAGE_SUFFIXES, LOGGER, imread_unicode, is_video_file


@dataclass
class SourceInfo:
    """Static description of one opened source."""

    name: str
    kind: str                 # 'camera' | 'video' | 'stream' | 'images'
    fps: float = 0.0
    width: int = 0
    height: int = 0
    frame_count: int = 0

    @property
    def is_live(self) -> bool:
        return self.kind in ("camera", "stream")


class FrameSource:
    """Iterable wrapper over one input, yielding ``(index, frame, timestamp)``."""

    def __init__(self, spec: str, realtime: bool = False) -> None:
        self.spec = spec
        self.realtime = realtime
        self._capture: Optional[cv2.VideoCapture] = None
        self._images: List[Path] = []
        self._thread: Optional[threading.Thread] = None
        self._latest: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self.info = SourceInfo(name=str(spec), kind="video")

    # ------------------------------------------------------------------ open
    def open(self) -> "FrameSource":
        spec = self.spec
        path = Path(str(spec))

        if isinstance(spec, str) and spec.isdigit():
            self._open_capture(int(spec), kind="camera")
            self.info.name = f"camera:{spec}"
        elif isinstance(spec, str) and "://" in spec:
            self._open_capture(spec, kind="stream")
            self.info.name = spec
        elif path.is_dir():
            images = sorted(p for p in path.iterdir()
                            if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
            if not images:
                raise ValueError(f"No images found in folder '{path}'.")
            self._images = images
            self.info = SourceInfo(name=path.name, kind="images", fps=25.0,
                                   frame_count=len(images))
        elif path.is_file():
            self._open_capture(str(path), kind="video")
            self.info.name = path.name
        else:
            raise FileNotFoundError(f"Source '{spec}' is not a camera index, file, folder or URL.")

        if self.realtime and self.info.is_live and self._capture is not None:
            self._start_reader_thread()
        return self

    def _open_capture(self, target, kind: str) -> None:
        capture = cv2.VideoCapture(target)
        if not capture.isOpened():
            hint = ""
            if kind == "camera":
                hint = " Is another program using the camera, or is the index wrong?"
            raise RuntimeError(f"Could not open source '{target}'.{hint}")

        if kind == "camera":
            # A small buffer keeps webcam latency down.
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self._capture = capture
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        self.info = SourceInfo(
            name=str(target),
            kind=kind,
            fps=fps if 0 < fps < 240 else 0.0,
            width=int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
            height=int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0),
            frame_count=int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0),
        )

    # ------------------------------------------------------- realtime reading
    def _start_reader_thread(self) -> None:
        """Keep only the newest frame, so live output never lags behind."""
        def reader() -> None:
            while not self._stop.is_set():
                ok, frame = self._capture.read()
                if not ok:
                    time.sleep(0.01)
                    continue
                with self._lock:
                    self._latest = frame

        self._thread = threading.Thread(target=reader, daemon=True, name="frame-reader")
        self._thread.start()
        LOGGER.debug("Realtime reader thread started for %s", self.info.name)

    # ------------------------------------------------------------- iteration
    def frames(self, max_frames: Optional[int] = None) -> Iterator[Tuple[int, np.ndarray, float]]:
        """Yield ``(frame_index, frame, timestamp_seconds)`` until exhausted."""
        index = 0
        started = time.time()

        if self._images:
            fps = self.info.fps or 25.0
            for path in self._images:
                if max_frames and index >= max_frames:
                    break
                frame = imread_unicode(path)
                if frame is None:
                    continue
                yield index, frame, index / fps
                index += 1
            return

        assert self._capture is not None
        misses = 0
        while True:
            if max_frames and index >= max_frames:
                break

            if self._thread is not None:
                with self._lock:
                    frame = None if self._latest is None else self._latest.copy()
                if frame is None:
                    if self._stop.is_set():
                        break
                    time.sleep(0.005)
                    misses += 1
                    if misses > 2000:      # ~10s with no frame at all
                        LOGGER.warning("No frames from %s; giving up.", self.info.name)
                        break
                    continue
                misses = 0
                timestamp = time.time() - started
            else:
                ok, frame = self._capture.read()
                if not ok or frame is None:
                    break
                position = self._capture.get(cv2.CAP_PROP_POS_MSEC) or 0.0
                timestamp = position / 1000.0 if position > 0 else (
                    index / self.info.fps if self.info.fps else time.time() - started
                )

            yield index, frame, timestamp
            index += 1

    # ---------------------------------------------------------------- closing
    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def __enter__(self) -> "FrameSource":
        return self.open()

    def __exit__(self, *exc) -> None:
        self.close()


def expand_sources(specs: Sequence[str]) -> List[str]:
    """Turn user input into a flat, ordered list of things to play.

    A folder of videos becomes one entry per video; a glob is expanded; a
    folder of images stays a single entry (it is one sequence, not many).
    """
    expanded: List[str] = []

    for spec in specs:
        text = str(spec)

        if text.isdigit() or "://" in text:
            expanded.append(text)
            continue

        if any(ch in text for ch in "*?["):
            matches = sorted(glob.glob(text))
            if not matches:
                LOGGER.warning("Pattern '%s' matched nothing.", text)
            expanded.extend(matches)
            continue

        path = Path(text)
        if path.is_dir():
            videos = sorted(p for p in path.rglob("*") if p.is_file() and is_video_file(p))
            if videos:
                expanded.extend(str(p) for p in videos)
                continue
            images = [p for p in path.iterdir()
                      if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES]
            if images:
                expanded.append(str(path))
                continue
            LOGGER.warning("Folder '%s' contains no videos or images.", path)
            continue

        expanded.append(text)

    return expanded


def describe_sources(specs: Sequence[str]) -> str:
    if not specs:
        return "no sources"
    if len(specs) == 1:
        return specs[0]
    return f"{len(specs)} sources ({', '.join(Path(s).name for s in specs[:3])}" + (
        ", ...)" if len(specs) > 3 else ")"
    )


class VideoWriter:
    """Lazy MP4 writer - the codec and size are taken from the first frame."""

    def __init__(self, path: Path, fps: float = 25.0) -> None:
        self.path = Path(path)
        self.fps = fps if 0 < fps < 240 else 25.0
        self._writer: Optional[cv2.VideoWriter] = None

    def write(self, frame: np.ndarray) -> None:
        if self._writer is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            height, width = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self._writer = cv2.VideoWriter(str(self.path), fourcc, self.fps, (width, height))
            if not self._writer.isOpened():
                LOGGER.error("Could not open '%s' for writing.", self.path)
                self._writer = None
                return
            LOGGER.info("Writing annotated video to %s", self.path)
        self._writer.write(frame)

    def close(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None
