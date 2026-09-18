"""Small shared helpers: geometry, colours, logging and image loading."""

from __future__ import annotations

import hashlib
import logging
import sys
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

import cv2
import numpy as np

LOGGER = logging.getLogger("oneshot_fd")

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".mpg", ".mpeg", ".wmv", ".flv"}

Box = Tuple[int, int, int, int]  # x1, y1, x2, y2


def setup_logging(quiet: bool = False, verbose: bool = False) -> None:
    """Configure the package logger once, with a compact human format."""
    level = logging.WARNING if quiet else (logging.DEBUG if verbose else logging.INFO)
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    LOGGER.handlers.clear()
    LOGGER.addHandler(handler)
    LOGGER.setLevel(level)
    LOGGER.propagate = False


def imread_unicode(path: Path) -> Optional[np.ndarray]:
    """Read an image, tolerating non-ASCII paths that ``cv2.imread`` chokes on."""
    try:
        buffer = np.fromfile(str(path), dtype=np.uint8)
    except OSError as exc:
        LOGGER.warning("Could not read %s: %s", path, exc)
        return None
    if buffer.size == 0:
        return None
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image is None:
        LOGGER.warning("Unsupported or corrupt image: %s", path)
    return image


def iter_images(folder: Path) -> Iterable[Path]:
    """Yield image files inside ``folder``, sorted for reproducible galleries."""
    for path in sorted(folder.iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            yield path


def is_video_file(path: Path) -> bool:
    return path.suffix.lower() in VIDEO_SUFFIXES


def clip_box(box: Sequence[float], width: int, height: int) -> Box:
    """Clamp a float box to integer pixel coordinates inside the frame."""
    x1, y1, x2, y2 = box
    x1 = int(max(0, min(width - 1, round(x1))))
    y1 = int(max(0, min(height - 1, round(y1))))
    x2 = int(max(0, min(width, round(x2))))
    y2 = int(max(0, min(height, round(y2))))
    if x2 <= x1:
        x2 = min(width, x1 + 1)
    if y2 <= y1:
        y2 = min(height, y1 + 1)
    return x1, y1, x2, y2


def box_area(box: Sequence[float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def box_iou(a: Sequence[float], b: Sequence[float]) -> float:
    """Intersection over union of two boxes."""
    inter_x1 = max(a[0], b[0])
    inter_y1 = max(a[1], b[1])
    inter_x2 = min(a[2], b[2])
    inter_y2 = min(a[3], b[3])
    inter = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)
    if inter <= 0:
        return 0.0
    union = box_area(a) + box_area(b) - inter
    return float(inter / union) if union > 0 else 0.0


def containment(inner: Sequence[float], outer: Sequence[float]) -> float:
    """Fraction of ``inner`` that lies inside ``outer``."""
    inter_x1 = max(inner[0], outer[0])
    inter_y1 = max(inner[1], outer[1])
    inter_x2 = min(inner[2], outer[2])
    inter_y2 = min(inner[3], outer[3])
    inter = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)
    area = box_area(inner)
    return float(inter / area) if area > 0 else 0.0


def box_center(box: Sequence[float]) -> Tuple[float, float]:
    return (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0


def l2_normalize(vector: np.ndarray, axis: int = -1) -> np.ndarray:
    """Unit-normalise so a dot product is the cosine similarity."""
    norm = np.linalg.norm(vector, axis=axis, keepdims=True)
    norm = np.maximum(norm, 1e-10)
    return vector / norm


def color_for_label(label: str) -> Tuple[int, int, int]:
    """A stable, readable BGR colour derived from the name itself.

    The same person keeps the same colour across runs and across videos, which
    makes multi-person footage much easier to follow.
    """
    digest = hashlib.md5(label.encode("utf-8")).digest()
    hue = digest[0] % 180
    hsv = np.uint8([[[hue, 200, 255]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def resize_to_width(frame: np.ndarray, max_width: Optional[int]) -> np.ndarray:
    """Downscale (never upscale) so the frame is at most ``max_width`` wide."""
    if not max_width:
        return frame
    height, width = frame.shape[:2]
    if width <= max_width:
        return frame
    scale = max_width / float(width)
    return cv2.resize(frame, (max_width, int(round(height * scale))), interpolation=cv2.INTER_AREA)


def format_timestamp(seconds: float) -> str:
    """``93.5`` -> ``00:01:33.500``."""
    if seconds < 0 or not np.isfinite(seconds):
        seconds = 0.0
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{int(hours):02d}:{int(minutes):02d}:{secs:06.3f}"
