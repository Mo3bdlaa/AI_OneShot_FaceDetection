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
from .quality import assess
from .reid import AppearanceBank
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
    #: The same frame before anything was drawn on it, in the coordinates the
    #: track boxes use. Enrolling somebody from what is on screen needs a clean
    #: crop, not one with a label box across their forehead.
    clean: Optional[np.ndarray] = None

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
        body = self.config.body
        self.appearance_bank = AppearanceBank(
            threshold=body.reid_threshold, margin=body.reid_margin, memory=body.reid_memory,
        )
        self.renderer = Renderer(self.config.draw, self.config.recognition)

        self._fps_window: Deque[float] = deque(maxlen=30)
        self._current_source = ""
        #: Description of the source currently being read, set by run_source.
        self.source_info = None
        self._appearances: List[Appearance] = []
        self._open_tracks: Dict[int, Appearance] = {}
        self._csv_file = None
        self._csv_writer = None
        #: Counters behind the "embeddings skipped" line in the run summary.
        self._embeddings = 0
        self._skipped_embeddings = 0
        self._low_quality = 0
        self._body_holds = 0

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
        if should_detect and self.config.tracking.enabled:
            tracks = self._detect_and_track(work, timestamp)
        elif should_detect:
            tracks = self._as_tracks(self._detect(work))
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
                           timestamp=timestamp, fps=fps, detected=should_detect,
                           clean=work)

    def _detect_and_track(self, frame: np.ndarray, timestamp: float) -> List[Track]:
        """Detect, recognise and track - skipping work the tracker makes needless.

        Locating faces costs about a tenth of what embedding them does, so the
        association is worked out first and only the faces that still need an
        answer are embedded. A track that has just named the same person
        several frames running can be taken at its word for a few more.
        """
        recognition = self.config.recognition
        self.tracker.begin_frame()

        faces = self.engine.locate(frame)
        assignment = self.tracker.associate([face.box for face in faces])

        names: List[str] = []
        scores: List[float] = []
        for position, face in enumerate(faces):
            track = self._settled_track(assignment[position])
            if track is not None:
                # Reuse the name this track already earned.
                names.append(track.label)
                scores.append(track.label_score)
                self._skipped_embeddings += 1
                continue
            if self._too_poor_to_judge(frame, face):
                # Better an honest Unknown than a confident mistake.
                names.append(recognition.unknown_label)
                scores.append(0.0)
                continue

            self.engine.embed(frame, face)
            match = self.gallery.identify(
                face.embedding, recognition.threshold, recognition.margin,
                recognition.unknown_label,
            )
            names.append(match.name)
            scores.append(match.score)
            self._embeddings += 1

        detections = self._build_detections(frame, faces, names, scores)
        if self._reid_active:
            self._apply_reid(frame, detections, assignment)
        return self.tracker.commit(
            detections, self._extend_assignment(detections, assignment), timestamp
        )

    # ------------------------------------------------------------------- reid
    @property
    def _reid_active(self) -> bool:
        """ReID needs real person boxes, which only the YOLO backend provides."""
        return self.config.body.reid and self.bodies.mode == "yolo"

    def _apply_reid(self, frame: np.ndarray, detections: List[dict],
                    assignment: Sequence[Optional[int]]) -> None:
        """Carry a name across the moments a face stops being readable.

        Faces are the only thing that can create an identity here, and the
        tracker is the only thing that decides which body is whose. Clothing
        does one job: confirming that a track whose face just became
        unreadable is still on the same body it was a moment ago.

        It deliberately does *not* search the remembered appearances for who a
        body might be. Measured on six real people, two different ones reach
        0.81 while the same person under a shifted light falls to 0.29 - there
        is no threshold in there. Constrained to "still the same body, moments
        later, before the light could change", the same measurement gives 0.97
        against that 0.81, which is answerable.
        """
        frame_index = self.tracker.frame_index
        unknown = self.config.recognition.unknown_label

        # Teach the bank from every body a face just vouched for.
        for detection in detections:
            name = detection.get("name")
            body_box = detection.get("body_box")
            if name and name != unknown and body_box is not None:
                self.appearance_bank.observe(frame, body_box, name, frame_index)

        # A face that is present but no longer recognisable keeps the name its
        # own track already earned - if the body agrees it is still them.
        for position, detection in enumerate(detections):
            if detection.get("name") != unknown or detection.get("body_box") is None:
                continue
            track = self._track_for(assignment, position)
            if track is None or track.label == unknown:
                continue
            held, score = self.appearance_bank.confirms(
                frame, detection["body_box"], track.label, frame_index
            )
            if not held:
                continue
            detection["name"] = track.label
            detection["similarity"] = score
            detection["by_body"] = True
            self._body_holds += 1

        self.appearance_bank.forget_stale(frame_index)

    def _track_for(self, assignment: Sequence[Optional[int]], position: int):
        """The existing track a detection was matched to, if any."""
        if position >= len(assignment):
            return None
        index = assignment[position]
        if index is None or not 0 <= index < len(self.tracker.tracks):
            return None
        return self.tracker.tracks[index]


    @staticmethod
    def _extend_assignment(detections: List[dict], assignment: List) -> List:
        """Pad the association so the body-only detections start their own tracks."""
        return list(assignment) + [None] * (len(detections) - len(assignment))

    def _too_poor_to_judge(self, frame: np.ndarray, face) -> bool:
        """True when a face is too degraded for its embedding to mean anything."""
        minimum = self.config.recognition.min_quality
        if minimum <= 0:
            return False
        quality = assess(frame, face.box, face.landmarks, face.score)
        if quality.score >= minimum:
            return False
        self._low_quality += 1
        return True

    def _settled_track(self, track_index: Optional[int]) -> Optional[Track]:
        """The track at ``track_index``, if it may skip this frame's embedding."""
        every = self.config.recognition.reverify_every
        if every <= 0 or track_index is None:
            return None
        if not 0 <= track_index < len(self.tracker.tracks):
            return None
        track = self.tracker.tracks[track_index]
        if not track.is_settled(self.config.recognition.unknown_label):
            return None
        if self.tracker.frame_index - track.verified_at >= every:
            return None
        return track

    def _build_detections(self, frame: np.ndarray, faces: List, names: List[str],
                          scores: List[float]) -> List[dict]:
        """Attach body boxes and package everything the tracker needs."""
        body_boxes: List[Optional[Sequence[int]]] = [None] * len(faces)
        if self.bodies.enabled and faces:
            found = self.bodies.bodies_for_faces(frame, [face.box for face in faces])
            for position, body in enumerate(found):
                body_boxes[position] = body.box

        return [
            {
                "box": face.box,
                "score": face.score,
                "landmarks": face.landmarks,
                "body_box": body_boxes[position],
                "name": names[position],
                "similarity": scores[position],
                "age": face.age,
                "gender": face.gender,
            }
            for position, face in enumerate(faces)
        ]

    def _detect(self, frame: np.ndarray) -> List[dict]:
        """Faces -> embeddings -> names -> matching body boxes, with no shortcuts."""
        faces = self.engine.detect(frame)
        if not faces:
            return []
        self._embeddings += len(faces)

        recognition = self.config.recognition
        matches = self.gallery.identify_batch(
            [face.embedding for face in faces],
            recognition.threshold,
            recognition.margin,
            recognition.unknown_label,
        )
        return self._build_detections(
            frame, faces, [m.name for m in matches], [m.score for m in matches]
        )

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
        self.appearance_bank.clear()
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
        """Appearances that have ended.

        Somebody still on screen is not here yet - their appearance has no end
        time. Use :meth:`current_appearances` for a report taken mid-run.
        """
        return list(self._appearances)

    def current_appearances(self) -> List[Appearance]:
        """Everything seen so far, including the people still on screen.

        Read-only: a report downloaded while a session is running must not
        change what that session goes on to record.
        """
        return list(self._appearances) + list(self._open_tracks.values())

    def reset_results(self) -> None:
        """Forget what was seen, for a fresh run on the same pipeline.

        The CLI keeps accumulating across the sources of one run, which is what
        its closing summary should cover. A new job - the web UI starting a
        different clip - is a different question, and mixing the two would put
        the last run's people in this run's report.
        """
        self._flush_open_tracks()
        self._appearances.clear()
        self._embeddings = 0
        self._skipped_embeddings = 0
        self._low_quality = 0
        self._body_holds = 0

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

        lines = []
        if self._low_quality:
            lines.append(
                f"Quality gate: {self._low_quality} face(s) were too small, soft or "
                "turned away to identify, and were left Unknown."
            )
            lines.append("")
        if self._body_holds:
            lines.append(
                f"Body ReID: {self._body_holds} frame(s) where somebody kept their name "
                "from their clothing after their face went out of view."
            )
            lines.append("")
        total = self._embeddings + self._skipped_embeddings
        if self._skipped_embeddings and total:
            lines.append(
                f"Recognition throttle: {self._skipped_embeddings}/{total} face embeddings "
                f"skipped ({self._skipped_embeddings / total:.0%}) because their tracks "
                "had already settled."
            )
            lines.append("")
        lines.append("Recognised people:")
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
