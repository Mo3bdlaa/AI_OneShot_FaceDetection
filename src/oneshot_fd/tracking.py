"""Frame-to-frame tracking and identity smoothing.

Recognition on a single frame flickers: a blink, a blur or a half-turn can
drop a face below the threshold for one frame. Tracking fixes that. Each face
gets a track, every track votes on its identity over a sliding window, and the
name shown on screen is the winner of that vote - so labels stay put while a
person moves through the shot.

Tracks also make ``--detect-every N`` possible: run the heavy models on every
Nth frame and carry the boxes forward in between.
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .config import RecognitionConfig, TrackingConfig
from .utils import box_center, box_iou


@dataclass
class Track:
    """A single person followed across frames."""

    track_id: int
    box: Sequence[int]
    score: float = 0.0
    body_box: Optional[Sequence[int]] = None
    landmarks: Optional[np.ndarray] = None

    hits: int = 1               # total detections assigned to this track
    age: int = 0                # frames since the last detection
    frames_seen: int = 0        # total frames this track has existed

    #: Rolling (name, similarity) votes used to smooth the displayed label.
    votes: Deque[Tuple[str, float]] = field(default_factory=lambda: deque(maxlen=12))
    label: str = "Unknown"
    label_score: float = 0.0
    best_score: float = 0.0

    velocity: Tuple[float, float] = (0.0, 0.0)
    first_frame: int = 0
    first_time: float = 0.0
    last_time: float = 0.0

    @property
    def confirmed(self) -> bool:
        return self.label != "Unknown"

    def predict(self) -> None:
        """Nudge the box along its recent motion when a frame has no detection."""
        dx, dy = self.velocity
        if abs(dx) < 0.5 and abs(dy) < 0.5:
            return
        x1, y1, x2, y2 = self.box
        self.box = (int(x1 + dx), int(y1 + dy), int(x2 + dx), int(y2 + dy))
        if self.body_box is not None:
            bx1, by1, bx2, by2 = self.body_box
            self.body_box = (int(bx1 + dx), int(by1 + dy), int(bx2 + dx), int(by2 + dy))

    def update_box(self, box: Sequence[int], gap: int = 1) -> None:
        """Move the track onto a new detection ``gap`` frames after the last one.

        The velocity is stored per frame, so ``--detect-every N`` does not make
        the prediction overshoot by a factor of N between detections.
        """
        old_cx, old_cy = box_center(self.box)
        new_cx, new_cy = box_center(box)
        gap = max(1, gap)
        # Smoothed velocity: responsive, but not jumpy on a noisy detection.
        self.velocity = (
            0.6 * (new_cx - old_cx) / gap + 0.4 * self.velocity[0],
            0.6 * (new_cy - old_cy) / gap + 0.4 * self.velocity[1],
        )
        self.box = box

    def vote(self, name: str, score: float, unknown_label: str,
             vote_ratio: float = 0.5) -> None:
        """Record one frame's opinion and recompute the displayed label."""
        self.votes.append((name, score))
        if score > self.best_score:
            self.best_score = score

        tally: Counter = Counter()
        totals: Dict[str, float] = {}
        for voted_name, voted_score in self.votes:
            if voted_name == unknown_label:
                continue
            tally[voted_name] += 1
            totals[voted_name] = totals.get(voted_name, 0.0) + voted_score

        if not tally:
            self.label = unknown_label
            self.label_score = max((s for _, s in self.votes), default=0.0)
            return

        winner, count = tally.most_common(1)[0]
        # The winner must own a real share of the window, otherwise a couple of
        # lucky frames could rename a person mid-shot.
        if count / max(1, len(self.votes)) >= vote_ratio or self.label == unknown_label:
            self.label = winner
            self.label_score = totals[winner] / count
        elif self.label in totals:
            self.label_score = totals[self.label] / tally[self.label]


class Tracker:
    """Greedy IoU tracker - small, fast and plenty for face-sized targets."""

    def __init__(self, tracking: Optional[TrackingConfig] = None,
                 recognition: Optional[RecognitionConfig] = None) -> None:
        self.config = tracking or TrackingConfig()
        self.recognition = recognition or RecognitionConfig()
        self.tracks: List[Track] = []
        self._next_id = 1
        self._frame_index = 0

    def reset(self) -> None:
        """Forget everything - called between separate videos."""
        self.tracks.clear()
        self._next_id = 1
        self._frame_index = 0

    def update(self, detections: Sequence[dict], timestamp: float = 0.0) -> List[Track]:
        """Associate detections with tracks and return the visible ones.

        Each detection is a dict with ``box`` and optionally ``score``,
        ``body_box``, ``landmarks``, ``name`` and ``similarity``.
        """
        self._frame_index += 1

        for track in self.tracks:
            track.age += 1
            track.frames_seen += 1
            track.predict()

        matched_detections: set = set()
        matched_tracks: set = set()

        # Stage 1: overlap. Reliable whenever the person barely moved.
        pairs: List[Tuple[float, int, int]] = []
        for det_index, detection in enumerate(detections):
            for track_index, track in enumerate(self.tracks):
                iou = box_iou(detection["box"], track.box)
                if iou >= self.config.iou_threshold:
                    pairs.append((iou, det_index, track_index))
        pairs.sort(reverse=True)

        for _, det_index, track_index in pairs:
            if det_index in matched_detections or track_index in matched_tracks:
                continue
            matched_detections.add(det_index)
            matched_tracks.add(track_index)
            self._apply(self.tracks[track_index], detections[det_index], timestamp)

        # Stage 2: centre distance. Fast motion - or --detect-every N, where
        # consecutive detections are N frames apart - can move a face clean off
        # its own previous box, leaving no overlap to match on. Distance
        # relative to the face size still links them, and the size check keeps
        # a nearby second face from stealing the track.
        leftovers = [
            (det_index, detection) for det_index, detection in enumerate(detections)
            if det_index not in matched_detections
        ]
        if leftovers:
            candidates: List[Tuple[float, int, int]] = []
            for det_index, detection in leftovers:
                det_cx, det_cy = box_center(detection["box"])
                det_size = max(1.0, detection["box"][2] - detection["box"][0])
                for track_index, track in enumerate(self.tracks):
                    if track_index in matched_tracks:
                        continue
                    track_cx, track_cy = box_center(track.box)
                    track_size = max(1.0, track.box[2] - track.box[0])
                    distance = np.hypot(det_cx - track_cx, det_cy - track_cy)
                    reach = max(det_size, track_size) * self.config.search_radius
                    ratio = min(det_size, track_size) / max(det_size, track_size)
                    if distance <= reach and ratio >= 0.5:
                        candidates.append((distance, det_index, track_index))
            candidates.sort()
            for _, det_index, track_index in candidates:
                if det_index in matched_detections or track_index in matched_tracks:
                    continue
                matched_detections.add(det_index)
                matched_tracks.add(track_index)
                self._apply(self.tracks[track_index], detections[det_index], timestamp)

        for det_index, detection in enumerate(detections):
            if det_index in matched_detections:
                continue
            self._spawn(detection, timestamp)

        self.tracks = [t for t in self.tracks if t.age <= self.config.max_age]
        return self.visible()

    def _apply(self, track: Track, detection: dict, timestamp: float) -> None:
        track.update_box(detection["box"], gap=track.age)
        track.age = 0
        track.hits += 1
        track.score = float(detection.get("score", track.score))
        track.last_time = timestamp
        if detection.get("body_box") is not None:
            track.body_box = detection["body_box"]
        if detection.get("landmarks") is not None:
            track.landmarks = detection["landmarks"]
        if detection.get("name") is not None:
            track.vote(
                detection["name"],
                float(detection.get("similarity", 0.0)),
                self.recognition.unknown_label,
                self.recognition.vote_ratio,
            )

    def _spawn(self, detection: dict, timestamp: float) -> Track:
        track = Track(
            track_id=self._next_id,
            box=detection["box"],
            score=float(detection.get("score", 0.0)),
            body_box=detection.get("body_box"),
            landmarks=detection.get("landmarks"),
            first_frame=self._frame_index,
            first_time=timestamp,
            last_time=timestamp,
        )
        track.votes = deque(maxlen=max(1, self.recognition.vote_window))
        self._next_id += 1
        if detection.get("name") is not None:
            track.vote(
                detection["name"],
                float(detection.get("similarity", 0.0)),
                self.recognition.unknown_label,
                self.recognition.vote_ratio,
            )
        self.tracks.append(track)
        return track

    def visible(self) -> List[Track]:
        """Tracks worth drawing: seen often enough and not long gone."""
        min_hits = self.config.min_hits
        return [
            track for track in self.tracks
            if track.hits >= min_hits and track.age <= max(1, self.config.max_age // 3)
        ]
