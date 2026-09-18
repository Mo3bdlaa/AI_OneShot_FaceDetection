"""The live recognition session behind the web UI.

The browser is not in the processing loop. A worker thread pulls frames,
recognises them and keeps the most recent annotated frame plus a rolling event
log; HTTP handlers only ever read that state. This keeps the recogniser running
at its own pace whether one person is watching, ten are, or nobody is - and
lets settings be changed without restarting anything.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional

import cv2
import numpy as np

from ..config import AppConfig
from ..pipeline import Pipeline
from ..utils import LOGGER, format_timestamp


@dataclass
class Event:
    """One thing worth telling the browser about."""

    kind: str                     # 'seen' | 'left' | 'info' | 'error'
    name: str
    at: float                     # seconds into the source
    wall: float = field(default_factory=time.time)
    detail: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "name": self.name,
            "at": round(self.at, 2),
            "at_text": format_timestamp(self.at),
            "wall": self.wall,
            "detail": self.detail,
        }


class Session:
    """A running (or stopped) recognition job, driven by the web UI."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.pipeline: Optional[Pipeline] = None

        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._lock = threading.Lock()

        self._frame: Optional[np.ndarray] = None
        self._clean: Optional[np.ndarray] = None
        self._jpeg: Optional[bytes] = None
        self._frame_number = 0
        self._last_tracks: List = []
        self._progress: Optional[float] = None

        self.status = "idle"          # idle | starting | running | stopped | error
        self.message = ""
        self.source = ""
        #: Where the annotated video is being written, when recording is on.
        self.recording_path: Optional[Path] = None
        self.fps = 0.0
        self.started_at = 0.0

        #: True while the browser is posting frames rather than the server
        #: pulling them from a camera or file.
        self._pushed = False

        self.events: Deque[Event] = deque(maxlen=400)
        self._on_screen: Dict[str, float] = {}
        self._seen_names: set = set()

    # ------------------------------------------------------------------ state
    def snapshot(self) -> Dict[str, Any]:
        """Everything the UI needs to render one poll, as plain JSON."""
        with self._lock:
            tracks = self._tracks_json()
            return {
                "status": self.status,
                "message": self.message,
                "source": self.source,
                "fps": round(self.fps, 1),
                "frame": self._frame_number,
                "elapsed": round(time.time() - self.started_at, 1) if self.started_at else 0.0,
                "tracks": tracks,
                "on_screen": sorted(self._on_screen),
                "seen": sorted(self._seen_names),
                "gallery": self._gallery_json(),
                "settings": self._settings_json(),
                "recording": self.recording_path.name if self.recording_path else None,
                "pushed": self._pushed,
                "progress": self._progress,
                "has_results": bool(self.pipeline and self.pipeline.current_appearances()),
            }

    def _tracks_json(self) -> List[Dict[str, Any]]:
        # The tracker keeps its tracks after a run ends; showing them as "on
        # screen now" would be a lie once nothing is being processed.
        if self.pipeline is None or self.status != "running":
            return []
        unknown = self.config.recognition.unknown_label
        return [
            {
                "id": track.track_id,
                "name": track.label,
                "score": round(track.label_score, 3),
                "known": track.label != unknown,
                "by_body": bool(track.by_body),
                "box": [int(v) for v in track.box],
                "body": [int(v) for v in track.body_box] if track.body_box else None,
                "age": track.estimated_age,
                "gender": track.estimated_gender,
            }
            for track in self.pipeline.tracker.visible()
        ]

    def _gallery_json(self) -> List[Dict[str, Any]]:
        if self.pipeline is None or self.pipeline.gallery.is_empty:
            return []
        return [
            {
                "name": person.name,
                "embeddings": int(len(person.embeddings)),
                "photos": len({Path(src.split(" (")[0]).name for src in person.sources}),
                "warnings": list(person.warnings or []),
            }
            for person in self.pipeline.gallery.people
        ]

    def _settings_json(self) -> Dict[str, Any]:
        recognition = self.config.recognition
        draw = self.config.draw
        return {
            "threshold": round(recognition.threshold, 3),
            "margin": round(recognition.margin, 3),
            "min_quality": round(recognition.min_quality, 3),
            "reverify_every": recognition.reverify_every,
            "show_body": draw.show_body,
            "show_face": draw.show_face,
            "show_attributes": draw.show_attributes,
            # Toggling the display does nothing unless the model was loaded at
            # startup, so the UI needs to know whether to offer it at all.
            "attributes_available": self.config.face.attributes,
            "blur_unknown": draw.blur_unknown,
            "reid": self.config.body.reid,
            "body_mode": self.config.body.mode,
        }

    # --------------------------------------------------------------- settings
    def apply_settings(self, values: Dict[str, Any]) -> Dict[str, Any]:
        """Change what can safely be changed while the job is running.

        Thresholds and overlay switches are read fresh every frame, so they
        take effect immediately. Anything that would need models reloading is
        rejected rather than silently ignored.
        """
        recognition = self.config.recognition
        draw = self.config.draw
        changed: List[str] = []

        for key, setter in (
            ("threshold", lambda v: setattr(recognition, "threshold", float(v))),
            ("margin", lambda v: setattr(recognition, "margin", float(v))),
            ("min_quality", lambda v: setattr(recognition, "min_quality", float(v))),
            ("reverify_every", lambda v: setattr(recognition, "reverify_every", max(0, int(v)))),
            ("show_body", lambda v: setattr(draw, "show_body", bool(v))),
            ("show_face", lambda v: setattr(draw, "show_face", bool(v))),
            ("show_attributes", lambda v: setattr(draw, "show_attributes", bool(v))),
            ("blur_unknown", lambda v: setattr(draw, "blur_unknown", bool(v))),
        ):
            if key in values and values[key] is not None:
                setter(values[key])
                changed.append(key)

        if changed:
            LOGGER.info("Settings changed: %s", ", ".join(changed))
        return {"changed": changed, "settings": self._settings_json()}

    # ------------------------------------------------------------------ frames
    def latest_jpeg(self) -> Optional[bytes]:
        with self._lock:
            return self._jpeg

    def _publish(self, result) -> None:
        """Make one processed frame the current one for every reader."""
        ok, buffer = cv2.imencode(".jpg", result.frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        with self._lock:
            self._frame = result.frame
            self._clean = result.clean
            self._last_tracks = list(result.tracks)
            self._frame_number = result.index
            self.fps = result.fps
            if ok:
                self._jpeg = buffer.tobytes()

            info = self.pipeline.source_info if self.pipeline else None
            total = getattr(info, "frame_count", 0) or 0
            # Live sources have no end, so they have no progress either.
            self._progress = (
                min(1.0, result.index / total)
                if total > 0 and not getattr(info, "is_live", False) else None
            )

    # ------------------------------------------------- enrolling from the video
    def enrol_from_track(self, track_id: int, name: str, margin: float = 0.45) -> Path:
        """Save the face of a track on screen as a new reference photo.

        The crop comes from the clean frame rather than the annotated one, and
        is padded outwards, because ArcFace wants some room around the face and
        a label box across the forehead is not a feature of anybody's face.
        """
        with self._lock:
            frame = None if self._clean is None else self._clean.copy()
            tracks = list(self._last_tracks)

        if frame is None:
            raise RuntimeError("There is no frame on screen to enrol from.")

        track = next((t for t in tracks if t.track_id == int(track_id)), None)
        if track is None:
            raise LookupError(f"Track #{track_id} is no longer on screen.")

        x1, y1, x2, y2 = (int(v) for v in track.box)
        pad_x = int((x2 - x1) * margin)
        pad_y = int((y2 - y1) * margin)
        height, width = frame.shape[:2]
        crop = frame[max(0, y1 - pad_y):min(height, y2 + pad_y),
                     max(0, x1 - pad_x):min(width, x2 + pad_x)]
        if crop.size == 0:
            raise RuntimeError("That face is outside the frame.")

        folder = Path(self.config.gallery.path)
        folder.mkdir(parents=True, exist_ok=True)
        target = folder / f"{name}.jpg"
        counter = 2
        while target.exists():
            target = folder / f"{name}_{counter}.jpg"
            counter += 1

        ok, buffer = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if not ok:
            raise RuntimeError("Could not encode that face.")
        target.write_bytes(buffer.tobytes())
        LOGGER.info("Enrolled %s from track #%s into %s", name, track_id, target.name)
        return target

    # ----------------------------------------------------------------- control
    # ------------------------------------------------- frames from a browser
    def begin_pushed(self, label: str = "browser camera") -> None:
        """Start a session fed one frame at a time by the client.

        Nothing is pulled here: the browser grabs from its own camera and posts
        each frame, so there is no worker thread and no source to open. The
        rest - the tracker, the roster, the event log - is the same, so the
        page behaves identically whichever camera is in use.
        """
        if self.is_running:
            raise RuntimeError("A session is already running. Stop it first.")

        self.ensure_pipeline().reset_results()
        self.source = label
        self.recording_path = None
        self.status = "running"
        self.message = ""
        self.events.clear()
        self._on_screen.clear()
        self._seen_names.clear()
        self._frame_number = 0
        self._last_tracks = []
        self._progress = None
        self._pushed = True
        self._stop.clear()
        self.started_at = time.time()
        self._note("info", "", 0.0, f"Started on the {label}")

    def push_frame(self, jpeg: bytes) -> Optional[bytes]:
        """Recognise one pushed frame and give back the annotated JPEG."""
        if not self._pushed:
            raise RuntimeError("No pushed session is running. Start one first.")

        buffer = np.frombuffer(jpeg, dtype=np.uint8)
        frame = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("That is not a readable image.")

        index = self._frame_number + 1
        result = self.ensure_pipeline().process_frame(
            frame, index, time.time() - self.started_at
        )
        self._publish(result)
        self._track_names(result)
        return self.latest_jpeg()

    def end_pushed(self) -> None:
        if not self._pushed:
            return
        self._pushed = False
        self._stop.set()
        self.status = "stopped"
        self._note("info", "", 0.0, "Camera stopped")

    @property
    def is_pushed(self) -> bool:
        return self._pushed

    def start(self, source: str, record_to: Optional[Path] = None) -> None:
        """Begin processing ``source`` on a worker thread.

        With ``record_to``, the annotated frames are also written to that file,
        which the UI offers for download once the run ends.
        """
        if self.is_running:
            raise RuntimeError("A session is already running. Stop it first.")

        self.source = source
        self.recording_path = record_to
        self.status = "starting"
        self.message = ""
        self.events.clear()
        self._on_screen.clear()
        self._seen_names.clear()
        self._frame_number = 0
        self._last_tracks = []
        self._progress = None
        self._stop.clear()
        self.started_at = time.time()
        if self.pipeline is not None:
            # This run's report should be about this run.
            self.pipeline.reset_results()

        self._thread = threading.Thread(target=self._run, args=(source, record_to),
                                        daemon=True, name="recognition")
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        if self._pushed:
            self.end_pushed()
            return
        self._stop.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=timeout)
        self._thread = None
        if self.status == "running":
            self.status = "stopped"

    @property
    def is_running(self) -> bool:
        if self._pushed:
            return True
        return self._thread is not None and self._thread.is_alive()

    def ensure_pipeline(self) -> Pipeline:
        """Load models and build the gallery, reusing them across sessions."""
        if self.pipeline is None:
            self.pipeline = Pipeline(self.config).prepare()
        return self.pipeline

    def reload_gallery(self) -> Dict[str, Any]:
        """Re-enrol the input folder after photos were added or removed.

        A photo with no usable face in it is a normal thing for somebody to
        drop into a browser, so it is reported rather than raised: the upload
        already succeeded, and losing the whole gallery over one bad file would
        be a poor trade.
        """
        pipeline = self.ensure_pipeline()
        pipeline.gallery.config.force_rebuild = True
        problem = ""
        try:
            pipeline.gallery.build(pipeline.engine)
        except (ValueError, FileNotFoundError) as exc:
            problem = str(exc)
            LOGGER.warning("Could not rebuild the gallery: %s", exc)
        finally:
            pipeline.gallery.config.force_rebuild = False
        pipeline.appearance_bank.clear()

        result: Dict[str, Any] = {"people": self._gallery_json()}
        if problem:
            result["problem"] = problem
        return result

    # -------------------------------------------------------------- the worker
    def _run(self, source: str, record_to: Optional[Path] = None) -> None:
        from ..sources import VideoWriter

        writer: Optional[VideoWriter] = None
        try:
            pipeline = self.ensure_pipeline()
            self.status = "running"
            self._note("info", "", 0.0, f"Started on {source}")

            for result in pipeline.run_source(source):
                if self._stop.is_set():
                    break
                self._publish(result)
                self._track_names(result)

                if record_to is not None:
                    if writer is None:
                        info = pipeline.source_info
                        writer = VideoWriter(record_to, fps=(info.fps if info else 0.0) or 25.0)
                        self._note("info", "", result.timestamp,
                                   f"Recording to {record_to.name}")
                    writer.write(result.frame)

            if not self._stop.is_set():
                self._note("info", "", 0.0, "Source finished")
                self.status = "stopped"
        except Exception as exc:                      # the UI must hear about this
            LOGGER.exception("Session failed")
            self.status = "error"
            self.message = str(exc)
            self._note("error", "", 0.0, str(exc))
        finally:
            if writer is not None:
                writer.close()
            self._stop.set()

    def _track_names(self, result) -> None:
        """Turn per-frame tracks into arrive/leave events for the UI."""
        unknown = self.config.recognition.unknown_label
        present = {t.label for t in result.tracks if t.label != unknown}

        for name in present - set(self._on_screen):
            self._note("seen", name, result.timestamp)
            self._seen_names.add(name)
        for name in set(self._on_screen) - present:
            self._note("left", name, result.timestamp)

        self._on_screen = {name: result.timestamp for name in present}

    def _note(self, kind: str, name: str, at: float, detail: str = "") -> None:
        self.events.append(Event(kind=kind, name=name, at=at, detail=detail))

    def events_since(self, wall: float) -> List[Dict[str, Any]]:
        return [event.as_dict() for event in self.events if event.wall > wall]
