"""The recognition pipeline: frames in, named and highlighted people out."""

from __future__ import annotations

import csv
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Deque, Dict, List, Optional, Sequence

import cv2
import numpy as np

from .bodies import BodyDetector
from .config import AppConfig
from .drawing import Renderer
from .faces import FaceEngine
from .gallery import Gallery
from .tracking import Track, Tracker
from .utils import LOGGER, format_timestamp, resize_to_width


@dataclass
class FrameResult:
    """Everything the pipeline knows about one processed frame."""

    frame: np.ndarray                    # the annotated frame
    tracks: List[Track] = field(default_factory=list)
    index: int = 0
    timestamp: float = 0.0
    fps: float = 0.0
    detected: bool = False               # did the heavy models run this frame?

    @property
    def names(self) -> List[str]:
        return sorted({t.label for t in self.tracks})


@dataclass
class Appearance:
    """A continuous stretch of time in which one person was on screen."""

    name: str
    track_id: int
    source: str
    start_time: float
    end_time: float
    best_score: float
    frames: int


class Pipeline:
    """Recognise known people in frames and draw the live highlighting.

    Typical use::

        pipeline = Pipeline(config).prepare()
        for result in pipeline.run_source("clip.mp4"):
            cv2.imshow("out", result.frame)
    """

    def __init__(self, config: Optional[AppConfig] = None) -> None:
        self.config = config or AppConfig()
        self.engine = FaceEngine(self.config.face)
        self.gallery = Gallery(self.config.gallery)
        self.bodies = BodyDetector(self.config.body)
        self.tracker = Tracker(self.config.tracking, self.config.recognition)
        self.renderer = Renderer(self.config.draw, self.config.recognition)

        self._fps_window: Deque[float] = deque(maxlen=30)
        self._current_source = ""
        #: Description of the source currently being read, set by run_source.
        self.source_info = None
        self._appearances: List[Appearance] = []
        self._open_tracks: Dict[int, Appearance] = {}
        self._csv_file = None
        self._csv_writer = None

    # --------------------------------------------------------------- startup
    def prepare(self) -> "Pipeline":
        """Load the models and enrol everyone in the input folder."""
        self.engine.load()
        self.gallery.build(self.engine)
        self.bodies.load()

        if self.gallery.is_empty:
            LOGGER.warning("The gallery is empty - every face will be labelled unknown.")
        else:
            LOGGER.info("Known people: %s", ", ".join(self.gallery.names))

        if self.config.runtime.log_csv:
            self._open_csv(Path(self.config.runtime.log_csv))
        return self

    def _open_csv(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        is_new = not path.exists() or path.stat().st_size == 0
        self._csv_file = path.open("a", newline="", encoding="utf-8")
        self._csv_writer = csv.writer(self._csv_file)
        if is_new:
            self._csv_writer.writerow(
                ["source", "name", "track_id", "start", "end",
                 "duration_s", "best_score", "frames"]
            )
        LOGGER.info("Logging recognition events to %s", path)

    # ------------------------------------------------------------ per frame
    def process_frame(self, frame: np.ndarray, index: int = 0, timestamp: float = 0.0,
                      annotate: bool = True) -> FrameResult:
        """Run one frame end to end and return the annotated result."""
        runtime = self.config.runtime
        started = time.time()

        work = resize_to_width(frame, runtime.max_width)
        if runtime.mirror:
            work = cv2.flip(work, 1)

        # With --detect-every N the heavy models run on every Nth frame and the
        # tracker carries the boxes through the frames in between.
        should_detect = (
            not self.config.tracking.enabled
            or runtime.detect_every <= 1
            or index % runtime.detect_every == 0
        )

        tracks: List[Track]
        if should_detect:
            detections = self._detect(work)
            if self.config.tracking.enabled:
                tracks = self.tracker.update(detections, timestamp)
            else:
                tracks = self._as_tracks(detections)
        else:
            tracks = self.tracker.update([], timestamp)

        self._record(tracks, timestamp)

        elapsed = max(1e-6, time.time() - started)
        self._fps_window.append(1.0 / elapsed)
        fps = float(np.mean(self._fps_window))

        output = work
        if annotate:
            output = self.renderer.render(
                work, tracks, fps=fps, source_name=self._current_source, frame_index=index
            )

        return FrameResult(frame=output, tracks=tracks, index=index,
                           timestamp=timestamp, fps=fps, detected=should_detect)

    def _detect(self, frame: np.ndarray) -> List[dict]:
        """Faces -> embeddings -> names -> matching body boxes."""
        faces = self.engine.detect(frame)
        if not faces:
            return []

        recognition = self.config.recognition
        matches = self.gallery.identify_batch(
            [face.embedding for face in faces],
            recognition.threshold,
            recognition.margin,
            recognition.unknown_label,
        )

        body_boxes: List[Optional[Sequence[int]]] = [None] * len(faces)
        if self.bodies.enabled:
            found = self.bodies.bodies_for_faces(frame, [face.box for face in faces])
            for position, body in enumerate(found):
                body_boxes[position] = body.box

        return [
            {
                "box": face.box,
                "score": face.score,
                "landmarks": face.landmarks,
                "body_box": body_boxes[position],
                "name": match.name,
                "similarity": match.score,
            }
            for position, (face, match) in enumerate(zip(faces, matches))
        ]

    def _as_tracks(self, detections: List[dict]) -> List[Track]:
        """Build throwaway tracks when tracking is switched off."""
        tracks: List[Track] = []
        for position, detection in enumerate(detections, start=1):
            track = Track(
                track_id=position,
                box=detection["box"],
                score=detection.get("score", 0.0),
                body_box=detection.get("body_box"),
                landmarks=detection.get("landmarks"),
                hits=self.config.tracking.min_hits,
            )
            track.label = detection.get("name", self.config.recognition.unknown_label)
            track.label_score = float(detection.get("similarity", 0.0))
            track.best_score = track.label_score
            tracks.append(track)
        return tracks

    # ----------------------------------------------------------- run sources
    def run_source(self, spec: str):
        """Yield a :class:`FrameResult` for every frame of one source."""
        from .sources import FrameSource

        runtime = self.config.runtime
        self.tracker.reset()
        self._fps_window.clear()

        with FrameSource(spec, realtime=runtime.realtime) as source:
            self._current_source = source.info.name
            self.source_info = source.info
            LOGGER.info(
                "Reading %s (%s%s)",
                source.info.name,
                source.info.kind,
                f", {source.info.frame_count} frames" if source.info.frame_count > 0 else "",
            )
            try:
                for index, frame, timestamp in source.frames(runtime.max_frames):
                    yield self.process_frame(frame, index, timestamp)
            finally:
                self._flush_open_tracks()

    def run(self):
        """Yield results for every configured source, one after another."""
        from .sources import expand_sources

        specs = expand_sources(self.config.runtime.sources)
        if not specs:
            raise ValueError("No usable source was given. Try --source 0 for a webcam.")

        for position, spec in enumerate(specs, start=1):
            if len(specs) > 1:
                LOGGER.info("[%d/%d] %s", position, len(specs), spec)
            try:
                for result in self.run_source(spec):
                    yield result
            except (RuntimeError, FileNotFoundError, ValueError) as exc:
                LOGGER.error("Skipping '%s': %s", spec, exc)

    # ---------------------------------------------------------- bookkeeping
    def _record(self, tracks: List[Track], timestamp: float) -> None:
        """Track when each person enters and leaves, for the CSV and summary."""
        live_ids = set()
        for track in tracks:
            if track.label == self.config.recognition.unknown_label:
                continue
            live_ids.add(track.track_id)
            appearance = self._open_tracks.get(track.track_id)
            if appearance is None:
                self._open_tracks[track.track_id] = Appearance(
                    name=track.label, track_id=track.track_id, source=self._current_source,
                    start_time=timestamp, end_time=timestamp,
                    best_score=track.best_score, frames=1,
                )
            else:
                appearance.end_time = timestamp
                appearance.frames += 1
                appearance.best_score = max(appearance.best_score, track.best_score)
                # A track can be renamed as the vote settles; keep the final name.
                appearance.name = track.label

        for track_id in [tid for tid in self._open_tracks if tid not in live_ids]:
            self._close_appearance(self._open_tracks.pop(track_id))

    def _close_appearance(self, appearance: Appearance) -> None:
        self._appearances.append(appearance)
        if self._csv_writer is None:
            return
        self._csv_writer.writerow([
            appearance.source,
            appearance.name,
            appearance.track_id,
            format_timestamp(appearance.start_time),
            format_timestamp(appearance.end_time),
            f"{max(0.0, appearance.end_time - appearance.start_time):.2f}",
            f"{appearance.best_score:.3f}",
            appearance.frames,
        ])
        if self._csv_file is not None:
            self._csv_file.flush()

    def _flush_open_tracks(self) -> None:
        for appearance in list(self._open_tracks.values()):
            self._close_appearance(appearance)
        self._open_tracks.clear()

    @property
    def appearances(self) -> List[Appearance]:
        return list(self._appearances)

    def summary(self) -> str:
        """A short 'who was seen, and for how long' report."""
        self._flush_open_tracks()
        if not self._appearances:
            return "No known person was recognised."

        totals: Dict[str, List[float]] = {}
        for appearance in self._appearances:
            duration = max(0.0, appearance.end_time - appearance.start_time)
            entry = totals.setdefault(appearance.name, [0.0, 0.0, 0.0])
            entry[0] += duration
            entry[1] += 1
            entry[2] = max(entry[2], appearance.best_score)

        lines = ["Recognised people:"]
        for name, (duration, count, best) in sorted(totals.items(), key=lambda kv: -kv[1][0]):
            lines.append(
                f"  {name:<22} {duration:7.1f}s over {int(count):3d} appearance(s), "
                f"best score {best:.3f}"
            )
        return "\n".join(lines)

    def close(self) -> None:
        self._flush_open_tracks()
        if self._csv_file is not None:
            self._csv_file.close()
            self._csv_file = None
            self._csv_writer = None

    def __enter__(self) -> "Pipeline":
        return self.prepare()

    def __exit__(self, *exc) -> None:
        self.close()
