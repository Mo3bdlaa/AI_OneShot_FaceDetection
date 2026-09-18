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

        kwargs = {"name": self.config.model_name, "providers": providers}
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
    def detect(self, frame: np.ndarray, max_faces: Optional[int] = None) -> List[DetectedFace]:
        """Detect and embed every face in a BGR frame, biggest first."""
        if frame is None or frame.size == 0:
            return []

        height, width = frame.shape[:2]
        raw = self.app.get(frame)

        faces: List[DetectedFace] = []
        for item in raw:
            box = clip_box(item.bbox, width, height)
            embedding = getattr(item, "normed_embedding", None)
            if embedding is None:
                embedding = getattr(item, "embedding", None)
                if embedding is not None:
                    embedding = l2_normalize(np.asarray(embedding, dtype=np.float32))
            else:
                embedding = np.asarray(embedding, dtype=np.float32)

            landmarks = getattr(item, "kps", None)
            faces.append(
                DetectedFace(
                    box=box,
                    score=float(getattr(item, "det_score", 0.0)),
                    embedding=embedding,
                    landmarks=np.asarray(landmarks) if landmarks is not None else None,
                )
            )

        faces.sort(key=lambda f: f.width * f.height, reverse=True)
        if max_faces:
            faces = faces[:max_faces]
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
