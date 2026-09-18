"""Holding on to a person after their face is gone.

Face recognition stops the moment someone turns around, walks behind a pillar
or dips out of frame. The body is still right there, so this module remembers
what each recognised person *looks like* from the shoulders down and uses that
to keep the label on them.

The descriptor is a colour histogram of the torso, split into an upper and a
lower band, in HSV with the value channel coarsely binned. That choice is
deliberate: it needs no extra model, costs well under a millisecond, survives
the person turning around (clothing looks much the same from behind), and
degrades honestly - two people in the same uniform will not be told apart, and
the code says so rather than pretending otherwise.

What it is not: a person re-identification network. It will not recognise
someone across two cameras, or tomorrow, or after they take their jacket off.
It bridges seconds, not scenes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .utils import LOGGER, clip_box

#: Histogram resolution: hue matters most, value least (it moves with lighting).
HUE_BINS, SAT_BINS, VAL_BINS = 24, 8, 4
#: The torso, as a fraction of the person box, top and bottom.
UPPER_BAND = (0.18, 0.52)
LOWER_BAND = (0.52, 0.85)


def _band(frame: np.ndarray, box: Sequence[int], band: Tuple[float, float]) -> np.ndarray:
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = clip_box(box, width, height)
    box_height = y2 - y1
    top = y1 + int(box_height * band[0])
    bottom = y1 + int(box_height * band[1])
    # Trim the sides: the edges of a person box are mostly background.
    inset = int((x2 - x1) * 0.15)
    return frame[max(y1, top):max(y1 + 1, bottom), x1 + inset:max(x1 + inset + 1, x2 - inset)]


def describe(frame: np.ndarray, box: Sequence[int]) -> Optional[np.ndarray]:
    """Turn the body inside ``box`` into a comparable appearance vector."""
    if frame is None or frame.size == 0:
        return None

    parts: List[np.ndarray] = []
    for band in (UPPER_BAND, LOWER_BAND):
        patch = _band(frame, box, band)
        if patch.size == 0 or patch.shape[0] < 4 or patch.shape[1] < 4:
            return None
        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        histogram = cv2.calcHist([hsv], [0, 1, 2], None,
                                 [HUE_BINS, SAT_BINS, VAL_BINS],
                                 [0, 180, 0, 256, 0, 256])
        histogram = cv2.normalize(histogram, histogram).flatten()
        parts.append(histogram)

    vector = np.concatenate(parts).astype(np.float32)
    norm = float(np.linalg.norm(vector))
    if norm < 1e-6:
        return None
    return vector / norm


def similarity(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> float:
    """Cosine similarity between two appearance vectors, 0 when either is missing."""
    if a is None or b is None or a.shape != b.shape:
        return 0.0
    return float(np.clip(np.dot(a, b), 0.0, 1.0))


@dataclass
class KnownAppearance:
    """What one identity currently looks like, and how sure we are of it.

    Not to be confused with ``pipeline.Appearance``, which is a stretch of time
    somebody was on screen. This is a description of their clothing.
    """

    name: str
    vector: np.ndarray
    #: Frame index this was last confirmed by an actual face recognition.
    updated_at: int = 0
    samples: int = 0

    def blend(self, vector: np.ndarray, frame_index: int, weight: float = 0.3) -> None:
        """Ease the stored appearance towards a fresh observation.

        Clothing does not change, but lighting and pose do, so the descriptor
        follows slowly rather than jumping to whatever the last frame saw.
        """
        mixed = (1.0 - weight) * self.vector + weight * vector
        norm = float(np.linalg.norm(mixed))
        if norm > 1e-6:
            self.vector = (mixed / norm).astype(np.float32)
        self.updated_at = frame_index
        self.samples += 1


class AppearanceBank:
    """Remembers how each recognised person looks, and answers "who is that?".

    Only faces carry authority here: an entry is created or refreshed when the
    face recogniser names someone, and the bank is then consulted for bodies
    that have no face attached to them.
    """

    def __init__(self, threshold: float = 0.72, margin: float = 0.05,
                 memory: int = 150) -> None:
        #: Appearance similarity a bodiless person box must reach to be named.
        self.threshold = threshold
        #: ... and how far it must beat the runner-up, so two similarly dressed
        #: people are left alone rather than swapped.
        self.margin = margin
        #: Frames an appearance stays usable after its last face confirmation.
        self.memory = memory
        self.entries: Dict[str, KnownAppearance] = {}

    def __len__(self) -> int:
        return len(self.entries)

    def clear(self) -> None:
        self.entries.clear()

    def observe(self, frame: np.ndarray, box: Sequence[int], name: str,
                frame_index: int) -> None:
        """Record what ``name`` looks like right now, from a face-confirmed body."""
        vector = describe(frame, box)
        if vector is None:
            return
        entry = self.entries.get(name)
        if entry is None:
            self.entries[name] = KnownAppearance(name=name, vector=vector,
                                                 updated_at=frame_index, samples=1)
        else:
            entry.blend(vector, frame_index)

    def identify(self, frame: np.ndarray, box: Sequence[int],
                 frame_index: int, exclude: Sequence[str] = ()) -> Tuple[Optional[str], float]:
        """Guess who a face-less body belongs to.

        ``exclude`` lists names already claimed by a visible face this frame -
        one person cannot be in two places, and skipping them stops a body
        borrowing the name of someone standing right next to it.
        """
        vector = describe(frame, box)
        if vector is None or not self.entries:
            return None, 0.0

        scored = sorted(
            (
                (similarity(vector, entry.vector), name)
                for name, entry in self.entries.items()
                if name not in exclude and frame_index - entry.updated_at <= self.memory
            ),
            reverse=True,
        )
        if not scored:
            return None, 0.0

        best_score, best_name = scored[0]
        runner_up = scored[1][0] if len(scored) > 1 else 0.0
        if best_score < self.threshold or (best_score - runner_up) < self.margin:
            return None, best_score
        return best_name, best_score

    def forget_stale(self, frame_index: int) -> None:
        """Drop appearances nobody has confirmed in a long time."""
        stale = [
            name for name, entry in self.entries.items()
            if frame_index - entry.updated_at > self.memory * 4
        ]
        for name in stale:
            del self.entries[name]
        if stale:
            LOGGER.debug("Forgot stale appearances: %s", ", ".join(stale))
