"""Face quality scoring and threshold calibration."""

import cv2
import numpy as np
import pytest

from oneshot_fd.quality import assess, calibrate_threshold, sharpness_of
from oneshot_fd.utils import l2_normalize


def detailed_face(size=200, blur=0):
    """A synthetic patch with enough texture to look like a real face crop."""
    rng = np.random.default_rng(7)
    patch = rng.integers(60, 200, (size, size, 3), dtype=np.uint8)
    cv2.circle(patch, (size // 3, size // 3), size // 12, (30, 30, 30), -1)
    cv2.circle(patch, (2 * size // 3, size // 3), size // 12, (30, 30, 30), -1)
    if blur:
        patch = cv2.GaussianBlur(patch, (blur, blur), 0)
    return patch


STRAIGHT_ON = np.array([[40, 50], [80, 50], [60, 70], [45, 90], [75, 90]], np.float32)


def test_sharpness_drops_when_an_image_is_blurred():
    sharp = detailed_face()
    soft = cv2.GaussianBlur(sharp, (21, 21), 0)
    assert sharpness_of(sharp) > sharpness_of(soft) * 3


def test_head_pose_is_deliberately_not_judged():
    """Two cheap pose estimates were measured and neither worked; see quality.py.

    A sharply angled but otherwise clean photo must pass, because flagging it
    would be a false alarm rather than useful advice.
    """
    frame = np.zeros((400, 400, 3), np.uint8)
    frame[100:300, 100:300] = detailed_face(200)
    angled = (STRAIGHT_ON + 100).copy()
    angled[2][0] = 145                     # nose almost on top of one eye
    quality = assess(frame, (100, 100, 300, 300), angled, 0.95, reference=True)
    assert quality.is_usable, quality.describe()


def test_a_good_face_has_no_complaints():
    frame = np.zeros((400, 400, 3), np.uint8)
    frame[100:300, 100:300] = detailed_face(200)
    quality = assess(frame, (100, 100, 300, 300), STRAIGHT_ON + 100, 0.95, reference=True)
    assert quality.is_usable, quality.describe()
    assert quality.score > 0.6


def test_a_tiny_face_is_flagged():
    frame = np.zeros((400, 400, 3), np.uint8)
    frame[100:130, 100:130] = detailed_face(30)
    quality = assess(frame, (100, 100, 130, 130), STRAIGHT_ON, 0.9, reference=True)
    assert not quality.is_usable
    assert any("px across" in issue for issue in quality.issues)


def test_a_blurred_face_is_flagged():
    frame = np.zeros((400, 400, 3), np.uint8)
    frame[100:300, 100:300] = detailed_face(200, blur=31)
    quality = assess(frame, (100, 100, 300, 300), STRAIGHT_ON + 100, 0.9, reference=True)
    assert any("focus" in issue for issue in quality.issues), quality.describe()


def test_a_dark_face_is_flagged():
    frame = np.zeros((400, 400, 3), np.uint8)
    frame[100:300, 100:300] = (detailed_face(200) * 0.08).astype(np.uint8)
    quality = assess(frame, (100, 100, 300, 300), STRAIGHT_ON + 100, 0.9, reference=True)
    assert any("dark" in issue for issue in quality.issues), quality.describe()


def test_a_washed_out_face_is_flagged():
    frame = np.full((400, 400, 3), 250, np.uint8)
    quality = assess(frame, (100, 100, 300, 300), STRAIGHT_ON + 100, 0.9, reference=True)
    assert any("washed out" in issue for issue in quality.issues), quality.describe()


def test_video_frames_are_judged_more_leniently_than_reference_photos():
    """A scrappy live frame is normal; a scrappy enrolment photo is a choice.

    The 75px face here lands between the two size floors on purpose: too small
    to enrol from, big enough to still be worth recognising in a moving video.
    """
    frame = np.zeros((400, 400, 3), np.uint8)
    frame[100:175, 100:175] = detailed_face(75)
    strict = assess(frame, (100, 100, 175, 175), STRAIGHT_ON + 100, 0.9, reference=True)
    lenient = assess(frame, (100, 100, 175, 175), STRAIGHT_ON + 100, 0.9, reference=False)
    assert not strict.is_usable, strict.describe()
    assert lenient.is_usable, lenient.describe()


# ------------------------------------------------------------------ calibration

def person(*vectors):
    return l2_normalize(np.asarray(vectors, dtype=np.float32), axis=1)


def test_calibration_needs_at_least_two_people():
    threshold, worst, note = calibrate_threshold([("Solo", person([1, 0, 0, 0]))])
    assert threshold == 0.38
    assert "Only one person" in note


def test_calibration_sits_above_the_most_similar_pair():
    people = [
        ("A", person([1, 0, 0, 0])),
        ("B", person([0, 1, 0, 0])),
        ("C", person([0.45, 0.45, 0.771, 0])),  # C is mildly like both
    ]
    threshold, worst, note = calibrate_threshold(people, headroom=0.05)
    assert worst == pytest.approx(0.45, abs=0.02)
    assert threshold == pytest.approx(worst + 0.05, abs=0.01), "sits just above the pair"
    assert "C" in note


def test_calibration_stays_inside_sane_bounds():
    identical = [("A", person([1, 0, 0, 0])), ("B", person([1, 0, 0, 0]))]
    threshold, worst, note = calibrate_threshold(identical)
    assert threshold <= 0.65, "never suggest a threshold nothing can reach"
    assert "same person enrolled under two names" in note
    assert "to spare" not in note, "a clamped suggestion must not claim headroom it lacks"

    distinct = [("A", person([1, 0, 0, 0])), ("B", person([0, 1, 0, 0]))]
    threshold, _, _ = calibrate_threshold(distinct)
    assert threshold >= 0.30, "never suggest a threshold that lets anyone through"
