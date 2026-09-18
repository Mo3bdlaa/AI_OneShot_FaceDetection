"""Face detection and embedding.

Built on InsightFace: SCRFD finds the faces, ArcFace turns each one into a
512-d embedding. ArcFace embeddings are what make one-shot recognition work -
a single reference photo lands close enough to every other photo of the same
person that a plain cosine similarity separates people reliably.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

from .config import FaceConfig
from .utils import LOGGER, clip_box, l2_normalize

_INSTALL_HINT = (
    "InsightFace is required for face recognition.\n"
    "Install it with:  pip install -r requirements.txt\n"
    "(or: pip install insightface onnxruntime opencv-python)"
)


@dataclass
class DetectedFace:
    """One face found in one frame."""

    box: Sequence[int]              # x1, y1, x2, y2 in frame pixels
    score: float                    # detector confidence
    embedding: Optional[np.ndarray] # L2-normalised ArcFace vector
    landmarks: Optional[np.ndarray] = None  # 5x2 keypoints, if available
    age: Optional[int] = None       # estimated years, when --attributes is on
    gender: Optional[str] = None    # "M" or "F", when --attributes is on

    @property
    def width(self) -> int:
        return int(self.box[2] - self.box[0])

    @property
    def height(self) -> int:
        return int(self.box[3] - self.box[1])


def _snap_det_size(size: int) -> int:
    """Round the detector input to something SCRFD can actually use.

    SCRFD works on feature maps of stride 8, 16 and 32, so an input that is not
    a multiple of 32 produces mismatched anchor grids and fails deep inside the
    model with an unhelpful broadcast error. Rounding here turns a confusing
    crash into a one-line notice, and it rounds *up* so nobody silently gets
    less detection resolution than they asked for.
    """
    snapped = max(128, -(-int(size) // 32) * 32)
    if snapped != size:
        LOGGER.warning("Detector size %d is not a multiple of 32; using %d instead.",
                       size, snapped)
    return snapped


def _resolve_providers(device: str) -> List[str]:
    """Pick ONNX Runtime providers, preferring the GPU when one is usable."""
    try:
        import onnxruntime as ort

        available = set(ort.get_available_providers())
    except Exception:  # pragma: no cover - onnxruntime always ships with insightface
        available = set()

    want_gpu = device in ("auto", "cuda", "gpu")
    if want_gpu and "CUDAExecutionProvider" in available:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    if device in ("cuda", "gpu"):
        LOGGER.warning("CUDA requested but no CUDAExecutionProvider is available; using CPU.")
    return ["CPUExecutionProvider"]


class FaceEngine:
    """Thin, well-behaved wrapper around ``insightface.app.FaceAnalysis``."""

    def __init__(self, config: Optional[FaceConfig] = None) -> None:
        self.config = config or FaceConfig()
        self._app = None
        self._ctx_id = -1

    # ------------------------------------------------------------------ setup
    def load(self) -> "FaceEngine":
        """Load the models. Called lazily, but safe to call up front."""
        if self._app is not None:
            return self

        try:
            from insightface.app import FaceAnalysis
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError(_INSTALL_HINT) from exc

        providers = _resolve_providers(self.config.device)
        self._ctx_id = 0 if providers[0].startswith("CUDA") else -1

        # FaceAnalysis loads the 68- and 106-point landmark models by default.
        # We never use them and they cost roughly 60% of the runtime, so we ask
        # only for what we need - plus genderage when --attributes is on.
        modules = ["detection", "recognition"]
        if self.config.attributes:
            modules.append("genderage")
        kwargs = {
            "name": self.config.model_name,
            "providers": providers,
            "allowed_modules": modules,
        }
        if self.config.model_root:
            kwargs["root"] = str(Path(self.config.model_root).expanduser())

        LOGGER.info(
            "Loading face model '%s' on %s (first run downloads ~300 MB)",
            self.config.model_name,
            "GPU" if self._ctx_id >= 0 else "CPU",
        )
        app = FaceAnalysis(**kwargs)
        det_size = _snap_det_size(int(self.config.det_size))
        app.prepare(ctx_id=self._ctx_id, det_size=(det_size, det_size),
                    det_thresh=float(self.config.det_threshold))
        self._app = app
        return self

    @property
    def app(self):
        if self._app is None:
            self.load()
        return self._app

    # ------------------------------------------------------------- inference
    def locate(self, frame: np.ndarray, max_faces: Optional[int] = None) -> List[DetectedFace]:
        """Find the faces in a frame *without* embedding them.

        Detection is cheap (~100 ms for a 960px frame on CPU); the ArcFace
        embedding is not (~100 ms per face). Keeping them apart lets the
        pipeline skip the expensive half for faces it has already identified.
        """
        if frame is None or frame.size == 0:
            return []

        height, width = frame.shape[:2]
        boxes, landmarks = self.app.det_model.detect(
            frame, max_num=max_faces or 0, metric="default"
        )
        if boxes is None or len(boxes) == 0:
            return []

        faces = [
            DetectedFace(
                box=clip_box(row[:4], width, height),
                score=float(row[4]) if len(row) > 4 else 0.0,
                embedding=None,
                landmarks=np.asarray(landmarks[index]) if landmarks is not None else None,
            )
            for index, row in enumerate(boxes)
        ]
        faces.sort(key=lambda f: f.width * f.height, reverse=True)
        return faces

    def embed(self, frame: np.ndarray, face: DetectedFace) -> Optional[np.ndarray]:
        """Compute and attach the ArcFace embedding for one located face.

        When ``--attributes`` is on, the age and gender estimates are filled in
        at the same time, since both work from the same aligned crop.
        """
        if face.embedding is not None:
            return face.embedding
        if face.landmarks is None:
            return None

        from insightface.app.common import Face as _Face

        item = _Face(bbox=np.asarray(face.box, dtype=np.float32),
                     kps=np.asarray(face.landmarks, dtype=np.float32),
                     det_score=face.score)
        self.app.models["recognition"].get(frame, item)

        genderage = self.app.models.get("genderage")
        if genderage is not None:
            try:
                genderage.get(frame, item)
                face.age = int(item.age) if item.age is not None else None
                face.gender = {1: "M", 0: "F"}.get(int(item.gender)) \
                    if item.gender is not None else None
            except Exception as exc:          # an estimate is never worth a crash
                LOGGER.debug("Age/gender estimation failed: %s", exc)
        embedding = getattr(item, "normed_embedding", None)
        if embedding is None:
            raw = getattr(item, "embedding", None)
            embedding = None if raw is None else l2_normalize(np.asarray(raw, np.float32))
        face.embedding = None if embedding is None else np.asarray(embedding, dtype=np.float32)
        return face.embedding

    def detect(self, frame: np.ndarray, max_faces: Optional[int] = None) -> List[DetectedFace]:
        """Locate and embed every face in a BGR frame, biggest first."""
        faces = self.locate(frame, max_faces)
        for face in faces:
            self.embed(frame, face)
        return faces

    def embed_reference(self, image: np.ndarray, min_face_size: int = 40) -> Optional[DetectedFace]:
        """Embed the main face of a reference photo.

        Reference photos are usually portraits, so the largest face is the
        subject. Anything smaller than ``min_face_size`` is treated as a
        bystander rather than the person being enrolled.
        """
        faces = self.detect(image)
        if not faces:
            return None
        best = faces[0]
        if max(best.width, best.height) < min_face_size:
            return None
        if best.embedding is None:
            return None
        return best
