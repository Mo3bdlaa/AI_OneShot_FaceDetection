"""Body (person) detection and face-to-body association.

Two backends, chosen automatically:

``yolo``
    Ultralytics YOLO person boxes. Accurate, follows the real silhouette, and
    keeps working when someone turns away from the camera.
``estimate``
    A geometric body box derived from the face box. No extra dependency, no
    extra inference cost, and good enough to highlight who is who.

The estimator is always available, so body highlighting never silently
disappears just because YOLO is not installed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .config import BodyConfig
from .utils import LOGGER, box_area, clip_box, containment


@dataclass
class DetectedBody:
    """A person box, either detected or estimated from the face."""

    box: Sequence[int]
    score: float
    estimated: bool = False


class BodyDetector:
    """Finds person boxes and links each one to the face it belongs to."""

    def __init__(self, config: Optional[BodyConfig] = None) -> None:
        self.config = config or BodyConfig()
        self._model = None
        self._mode = self.config.mode
        self._loaded = False

    # ------------------------------------------------------------------ setup
    def load(self) -> "BodyDetector":
        if self._loaded:
            return self
        self._loaded = True

        if self._mode == "off":
            return self
        if self._mode == "estimate":
            LOGGER.info("Body highlighting: geometric estimate from the face box.")
            return self

        try:
            from ultralytics import YOLO

            LOGGER.info("Body highlighting: YOLO (%s)", self.config.yolo_model)
            self._model = YOLO(self.config.yolo_model)
            self._mode = "yolo"
        except Exception as exc:
            if self._mode == "yolo":
                LOGGER.warning(
                    "YOLO backend requested but unavailable (%s). "
                    "Install it with 'pip install ultralytics'. "
                    "Falling back to geometric body estimation.", exc,
                )
            else:
                LOGGER.info(
                    "Body highlighting: geometric estimate "
                    "(install 'ultralytics' for real person boxes)."
                )
            self._mode = "estimate"
        return self

    @property
    def mode(self) -> str:
        if not self._loaded:
            self.load()
        return self._mode

    @property
    def enabled(self) -> bool:
        return self.mode != "off"

    # -------------------------------------------------------------- detection
    def detect(self, frame: np.ndarray) -> List[DetectedBody]:
        """Person boxes in the frame. Empty for the estimate/off backends."""
        if self.mode != "yolo" or self._model is None or frame is None:
            return []

        height, width = frame.shape[:2]
        try:
            results = self._model.predict(
                frame, classes=[0], conf=self.config.conf, verbose=False
            )
        except Exception as exc:
            LOGGER.warning("YOLO inference failed (%s); switching to estimated bodies.", exc)
            self._mode = "estimate"
            self._model = None
            return []

        bodies: List[DetectedBody] = []
        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            for xyxy, conf in zip(boxes.xyxy.tolist(), boxes.conf.tolist()):
                bodies.append(DetectedBody(clip_box(xyxy, width, height), float(conf)))
        return bodies

    # ------------------------------------------------------------ association
    def estimate_from_face(self, face_box: Sequence[float], width: int, height: int) -> DetectedBody:
        """Grow a plausible torso/body box downwards from a face box.

        Anthropometry gives a decent prior: a standing adult is roughly seven
        to eight head-heights tall and about three head-widths across at the
        shoulders, centred on the face.
        """
        x1, y1, x2, y2 = face_box
        face_w = max(1.0, x2 - x1)
        face_h = max(1.0, y2 - y1)
        cx = (x1 + x2) / 2.0

        body_w = face_w * self.config.estimate_width_factor
        body_top = y1 - face_h * 0.35                       # a little headroom
        body_bottom = y1 + face_h * self.config.estimate_height_factor

        box = clip_box(
            (cx - body_w / 2.0, body_top, cx + body_w / 2.0, body_bottom), width, height
        )
        return DetectedBody(box=box, score=0.0, estimated=True)

    def match(self, face_box: Sequence[float], bodies: Sequence[DetectedBody],
              used: Optional[set] = None) -> Tuple[Optional[int], Optional[DetectedBody]]:
        """Pick the person box that owns ``face_box``.

        A face belongs to the person box that contains it; when several do
        (someone standing behind someone else) the smallest one wins, because
        the tighter box is the closer person.
        """
        best_index: Optional[int] = None
        best_key = None

        for index, body in enumerate(bodies):
            if used is not None and index in used:
                continue
            overlap = containment(face_box, body.box)
            if overlap < 0.6:
                continue
            # A face sits in the upper part of its own body box.
            face_cy = (face_box[1] + face_box[3]) / 2.0
            body_top, body_bottom = body.box[1], body.box[3]
            relative = (face_cy - body_top) / max(1.0, body_bottom - body_top)
            if relative > 0.6:
                continue

            key = (-overlap, box_area(body.box))
            if best_key is None or key < best_key:
                best_key = key
                best_index = index

        if best_index is None:
            return None, None
        return best_index, bodies[best_index]

    def bodies_for_faces(self, frame: np.ndarray,
                         face_boxes: Sequence[Sequence[float]]) -> List[DetectedBody]:
        """One body box per face, detected when possible and estimated otherwise."""
        if not self.enabled or not face_boxes:
            return []

        height, width = frame.shape[:2]
        detected = self.detect(frame)
        used: set = set()
        out: List[DetectedBody] = []

        for face_box in face_boxes:
            index, body = self.match(face_box, detected, used)
            if body is not None and index is not None:
                used.add(index)
                out.append(body)
            else:
                out.append(self.estimate_from_face(face_box, width, height))
        return out
