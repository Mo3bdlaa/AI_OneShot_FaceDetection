"""Overlay rendering - checks the frame is annotated without exploding."""

import numpy as np
import pytest

from oneshot_fd.config import DrawConfig, RecognitionConfig
from oneshot_fd.drawing import Renderer, blur_region, draw_corner_box, draw_label
from oneshot_fd.tracking import Track


def make_track(label="Mohammed", box=(100, 100, 200, 220), body=(50, 80, 260, 700)):
    track = Track(track_id=1, box=box, score=0.95, body_box=body, hits=5)
    track.label = label
    track.label_score = 0.72
    return track


@pytest.fixture
def frame():
    return np.zeros((720, 1280, 3), dtype=np.uint8)


def test_render_draws_something(frame):
    out = Renderer().render(frame, [make_track()], fps=25.0)
    assert out.shape == frame.shape
    assert out.any(), "the overlay must actually change the frame"
    assert not frame.any(), "the input frame must not be modified in place"


def test_render_survives_an_empty_frame_of_tracks(frame):
    out = Renderer().render(frame, [], fps=0.0)
    assert out.shape == frame.shape


def test_known_and_unknown_use_different_colours():
    renderer = Renderer(recognition=RecognitionConfig(unknown_label="Unknown"))
    assert renderer.color_for(make_track("Mohammed")) != renderer.color_for(make_track("Unknown"))


def test_same_name_always_gets_the_same_colour():
    renderer = Renderer()
    assert renderer.color_for(make_track("Sara")) == renderer.color_for(make_track("Sara"))


def test_labels_at_the_frame_edge_stay_inside(frame):
    draw_label(frame, "Somebody with a long name", (1270, 5), (0, 255, 0))
    draw_label(frame, "Bottom left", (-40, 718), (0, 255, 0))
    assert frame.any()


def test_boxes_touching_the_edges_do_not_crash(frame):
    for box in [(0, 0, 50, 50), (1230, 670, 1280, 720), (-10, -10, 40, 40)]:
        draw_corner_box(frame, box, (255, 0, 0))


def test_blur_region_changes_pixels():
    frame = np.random.default_rng(0).integers(0, 255, (200, 200, 3), dtype=np.uint8)
    before = frame.copy()
    blur_region(frame, (50, 50, 150, 150))
    assert not np.array_equal(before[50:150, 50:150], frame[50:150, 50:150])
    assert np.array_equal(before[0:40, 0:40], frame[0:40, 0:40]), "only the region changes"


def test_blur_region_ignores_an_empty_box():
    frame = np.zeros((100, 100, 3), np.uint8)
    blur_region(frame, (50, 50, 50, 50))          # must not raise


def test_toggles_actually_turn_things_off(frame):
    off = DrawConfig(show_face=False, show_body=False, show_fps=False, show_roster=False)
    out = Renderer(off).render(frame, [make_track()], fps=25.0)
    # Only the name caption is left, so far fewer pixels are touched.
    minimal = int((out != frame).any(axis=2).sum())
    full = int((Renderer().render(frame, [make_track()], fps=25.0) != frame).any(axis=2).sum())
    assert 0 < minimal < full


def test_blur_unknown_only_affects_unknown_people(frame):
    noisy = np.random.default_rng(1).integers(0, 255, frame.shape, dtype=np.uint8)
    config = DrawConfig(blur_unknown=True, show_fps=False, show_roster=False)
    known = Renderer(config).render(noisy, [make_track("Mohammed")])
    unknown = Renderer(config).render(noisy, [make_track("Unknown")])
    assert not np.array_equal(known[120:200, 120:190], unknown[120:200, 120:190])
