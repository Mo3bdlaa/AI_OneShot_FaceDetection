"""Body estimation and face-to-body association."""

import numpy as np
import pytest

from oneshot_fd.bodies import BodyDetector, DetectedBody
from oneshot_fd.config import BodyConfig


@pytest.fixture
def detector():
    return BodyDetector(BodyConfig(mode="estimate")).load()


def test_estimated_body_hangs_below_the_face(detector):
    face = (100, 100, 150, 160)          # 50x60 face
    body = detector.estimate_from_face(face, 1920, 1080)
    assert body.estimated
    assert body.box[1] <= face[1], "body starts at or above the top of the head"
    assert body.box[3] > face[3], "body extends below the face"
    assert body.box[0] < face[0] and body.box[2] > face[2], "body is wider than the face"


def test_estimated_body_is_clipped_to_the_frame(detector):
    body = detector.estimate_from_face((10, 10, 60, 70), 200, 200)
    assert 0 <= body.box[0] < body.box[2] <= 200
    assert 0 <= body.box[1] < body.box[3] <= 200


def test_face_is_matched_to_the_person_box_around_it(detector):
    bodies = [
        DetectedBody((0, 0, 100, 400), 0.9),
        DetectedBody((300, 0, 400, 400), 0.9),
    ]
    index, body = detector.match((20, 20, 80, 80), bodies)
    assert index == 0 and body is bodies[0]


def test_the_closer_person_wins_when_boxes_overlap(detector):
    """Two nested person boxes: the tighter one is the person in front."""
    bodies = [
        DetectedBody((0, 0, 500, 500), 0.9),      # far, loose box
        DetectedBody((10, 10, 120, 400), 0.9),    # near, tight box
    ]
    index, _ = detector.match((20, 20, 80, 80), bodies)
    assert index == 1


def test_a_face_below_the_waist_is_not_that_persons_face(detector):
    bodies = [DetectedBody((0, 0, 200, 400), 0.9)]
    index, _ = detector.match((20, 340, 80, 390), bodies)
    assert index is None


def test_already_used_body_boxes_are_not_reused(detector):
    bodies = [DetectedBody((0, 0, 200, 400), 0.9)]
    index, _ = detector.match((20, 20, 80, 80), bodies, used={0})
    assert index is None


def test_every_face_gets_a_body_box(detector):
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    faces = [(100, 100, 160, 170), (600, 200, 660, 270), (1000, 50, 1060, 120)]
    bodies = detector.bodies_for_faces(frame, faces)
    assert len(bodies) == len(faces)
    assert all(b.estimated for b in bodies)


def test_off_mode_produces_nothing():
    detector = BodyDetector(BodyConfig(mode="off")).load()
    assert not detector.enabled
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    assert detector.bodies_for_faces(frame, [(10, 10, 40, 40)]) == []


def test_auto_mode_falls_back_when_yolo_is_missing():
    """Without ultralytics the detector must degrade, not crash."""
    detector = BodyDetector(BodyConfig(mode="auto", yolo_model="definitely-not-a-model.pt"))
    assert detector.mode in {"yolo", "estimate"}
    assert detector.enabled
