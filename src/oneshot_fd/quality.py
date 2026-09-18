"""Judging whether a face is worth trusting.

One-shot recognition lives or dies on the reference photo. A blurry, tiny,
badly lit or sharply turned face still produces a perfectly confident-looking
512-d vector - it is just the wrong one, and every frame afterwards inherits
that mistake. Rather than let a bad photo quietly poison a whole gallery, the
enrolment step scores each reference and says plainly what is wrong with it.

The same scoring runs at inference time behind ``--min-quality``: a face that
is too degraded to identify reliably is better left as ``Unknown`` than
confidently mislabelled.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence, Tuple

import cv2
import numpy as np

from .utils import clip_box

#: Sharpness is measured on a crop resized to this, so a 1500px studio
#: portrait and a 100px face from a video frame are judged on the same terms.
SHARPNESS_CROP = 112

#: Below this, on that fixed-size crop, a face is genuinely mushy. Derived by
#: measuring: ten real photos from a phone and a TV still scored 112-729, while
#: a face blurred past recognition or shrunk to 40px scored 25-61.
SHARPNESS_FLOOR = 80.0
#: ArcFace is trained on 112px crops; smaller than this is guesswork.
SIZE_FLOOR = 60
#: Mean luma outside this range loses the detail embeddings rely on.
BRIGHTNESS_RANGE = (45.0, 215.0)


@dataclass
class FaceQuality:
    """How usable one face is, and why."""

    score: float                                  # 0..1, higher is better
    sharpness: float = 0.0
    size: int = 0
    brightness: float = 0.0
    detection: float = 0.0
    issues: List[str] = field(default_factory=list)

    @property
    def is_usable(self) -> bool:
        return not self.issues

    def describe(self) -> str:
        return "; ".join(self.issues) if self.issues else "good"


# A head-pose check lived here and was taken out again. Two cheap estimates
# built from the five keypoints - the nose's offset between the eyes, and the
# asymmetry of the nose-to-eye distances - were measured against real photos
# that the recogniser handles at 0.96+, and against warped copies of them. The
# first scored those good photos anywhere from 0.07 to 0.89, and the second did
# not move when a face was turned. Neither measures what it claimed to, and a
# pose warning that fires on a perfectly good photo costs more trust than the
# check could ever repay. The checks below are the ones that were verified to
# behave: size, focus, exposure and the detector's own confidence.


def sharpness_of(crop: np.ndarray, size: int = SHARPNESS_CROP) -> float:
    """How much fine detail a face crop carries, independent of its resolution.

    Variance of the Laplacian is the standard cheap focus measure, but taken
    raw it is not comparable between images: it counts detail *per pixel*, so a
    large, smooth, retouched portrait scores lower than a small noisy selfie
    that is objectively worse. Measured that way, two perfectly good photos out
    of four were flagged as out of focus. Resizing to a fixed crop first makes
    the numbers mean the same thing everywhere.

    It is still fooled by heavy JPEG compression, whose blockiness reads as
    detail - a quality-5 copy of a good photo scored higher than the original.
    So this is a hint that a photo is soft, not a verdict on it.
    """
    if crop.size == 0:
        return 0.0
    if crop.shape[0] != size or crop.shape[1] != size:
        crop = cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)
    grey = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    return float(cv2.Laplacian(grey, cv2.CV_64F).var())


def assess(frame: np.ndarray, box: Sequence[int], landmarks=None,
           detection_score: float = 1.0, reference: bool = False) -> FaceQuality:
    """Score one face in a frame.

    ``reference=True`` applies the stricter bar that enrolment photos deserve -
    a live video frame is allowed to be scrappy, a photo you chose is not.
    """
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = clip_box(box, width, height)
    crop = frame[y1:y2, x1:x2]

    size = int(min(x2 - x1, y2 - y1))
    sharpness = sharpness_of(crop)
    brightness = float(crop.mean()) if crop.size else 0.0

    issues: List[str] = []
    size_floor = SIZE_FLOOR * (1.5 if reference else 1.0)
    sharp_floor = SHARPNESS_FLOOR * (1.0 if reference else 0.5)

    if size < size_floor:
        issues.append(f"face is only {size}px across (want {int(size_floor)}px+)")
    if sharpness < sharp_floor:
        # Worth saying, not worth alarm: ArcFace tolerates a great deal of
        # blur. A face blurred until it is barely a face still matched itself
        # at 0.88, so this is advice for a better photo rather than a
        # prediction that this one will fail.
        issues.append(f"is soft; a sharper photo would be better "
                      f"(detail {sharpness:.0f}, typical is 150+)")
    if brightness < BRIGHTNESS_RANGE[0]:
        issues.append(f"too dark (brightness {brightness:.0f})")
    elif brightness > BRIGHTNESS_RANGE[1]:
        issues.append(f"washed out (brightness {brightness:.0f})")
    if detection_score < 0.6:
        issues.append(f"the detector is unsure this is a face ({detection_score:.2f})")

    # A single 0..1 number for sorting and for --min-quality.
    parts = [
        min(1.0, size / (SIZE_FLOOR * 2.0)),
        min(1.0, sharpness / (SHARPNESS_FLOOR * 2.5)),
        1.0 - min(1.0, abs(brightness - 128.0) / 128.0),
        min(1.0, detection_score),
    ]
    score = float(np.mean(parts))

    return FaceQuality(score=score, sharpness=sharpness, size=size, brightness=brightness,
                       detection=detection_score, issues=issues)


def calibrate_threshold(embeddings_by_person: Sequence[Tuple[str, np.ndarray]],
                        floor: float = 0.30, ceiling: float = 0.65,
                        headroom: float = 0.06) -> Tuple[float, float, str]:
    """Suggest a threshold from the gallery itself.

    The people you actually enrolled are the best available evidence of how
    similar two *different* people look to this model. The worst (highest)
    similarity between two different people is the line recognition must stay
    above; the suggestion sits a little above that, clamped to a sane range.

    Returns ``(suggested_threshold, worst_impostor_similarity, explanation)``.
    """
    people = [(name, np.atleast_2d(vectors)) for name, vectors in embeddings_by_person]
    if len(people) < 2:
        return (0.38, 0.0,
                "Only one person is enrolled, so there is nothing to tell them apart from; "
                "keeping the default threshold of 0.38.")

    worst = -1.0
    worst_pair = ("", "")
    for index, (name_a, vectors_a) in enumerate(people):
        for name_b, vectors_b in people[index + 1:]:
            similarity = float((vectors_a @ vectors_b.T).max())
            if similarity > worst:
                worst = similarity
                worst_pair = (name_a, name_b)

    wanted = worst + headroom
    suggested = float(np.clip(wanted, floor, ceiling))
    lead = (f"The two most similar people in the gallery are {worst_pair[0]} and "
            f"{worst_pair[1]} at {worst:.3f}.")

    if wanted > ceiling:
        # Clamping means the suggestion does NOT separate them. Say so plainly
        # rather than reporting headroom that is not there.
        explanation = (
            f"{lead} That is too close to separate: no usable threshold tells them apart, "
            f"so {suggested:.2f} is only a ceiling. Either they are the same person enrolled "
            "under two names, or their reference photos are too alike - use clearer, more "
            "different photos for at least one of them."
        )
    elif wanted < floor:
        explanation = (
            f"{lead} Everyone is comfortably distinct, so the threshold stays at the "
            f"{suggested:.2f} floor rather than dropping somewhere too permissive."
        )
    else:
        explanation = (
            f"{lead} A threshold of {suggested:.2f} keeps them apart with "
            f"{headroom:.2f} to spare."
        )
    return suggested, worst, explanation
