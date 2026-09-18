"""Geometry and helper tests - no models, no network, instant."""

import numpy as np
import pytest

from oneshot_fd.utils import (box_iou, clip_box, color_for_label, containment,
                              format_timestamp, l2_normalize, resize_to_width)


def test_iou_identical_boxes_is_one():
    assert box_iou((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)


def test_iou_disjoint_boxes_is_zero():
    assert box_iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0


def test_iou_half_overlap():
    # Two 10x10 boxes sharing a 5x10 strip: 50 / 150.
    assert box_iou((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(50 / 150)


def test_containment_of_face_inside_body():
    assert containment((10, 10, 20, 20), (0, 0, 100, 100)) == pytest.approx(1.0)
    assert containment((0, 0, 10, 10), (5, 5, 15, 15)) == pytest.approx(0.25)


def test_clip_box_stays_inside_frame_and_never_collapses():
    assert clip_box((-50, -50, 40, 40), 100, 100) == (0, 0, 40, 40)
    assert clip_box((10, 10, 500, 500), 100, 100) == (10, 10, 100, 100)
    x1, y1, x2, y2 = clip_box((99, 99, 99, 99), 100, 100)
    assert x2 > x1 and y2 > y1


def test_l2_normalize_gives_unit_vectors():
    vectors = np.array([[3.0, 4.0], [0.0, 0.0]], dtype=np.float32)
    normalised = l2_normalize(vectors, axis=1)
    assert np.linalg.norm(normalised[0]) == pytest.approx(1.0)
    assert np.all(np.isfinite(normalised))          # the zero row must not blow up


def test_color_for_label_is_stable_and_distinct():
    assert color_for_label("Mohammed") == color_for_label("Mohammed")
    assert color_for_label("Mohammed") != color_for_label("Sara")


def test_resize_only_shrinks():
    frame = np.zeros((100, 400, 3), dtype=np.uint8)
    assert resize_to_width(frame, 200).shape[:2] == (50, 200)
    assert resize_to_width(frame, 800).shape[:2] == (100, 400)   # never upscales
    assert resize_to_width(frame, None).shape[:2] == (100, 400)


def test_format_timestamp():
    assert format_timestamp(0) == "00:00:00.000"
    assert format_timestamp(93.5) == "00:01:33.500"
    assert format_timestamp(-5) == "00:00:00.000"
