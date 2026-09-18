"""Tracker behaviour: association, smoothing and identity voting."""

import pytest

from oneshot_fd.config import RecognitionConfig, TrackingConfig
from oneshot_fd.tracking import Tracker


def detection(box, name="Mohammed", similarity=0.8):
    return {"box": box, "score": 0.9, "name": name, "similarity": similarity}


@pytest.fixture
def tracker():
    return Tracker(TrackingConfig(min_hits=1), RecognitionConfig(vote_window=5, vote_ratio=0.5))


def test_a_steady_face_keeps_one_track_id(tracker):
    for step in range(5):
        tracks = tracker.update([detection((10 + step, 10, 60 + step, 60))])
    assert len(tracks) == 1
    assert tracks[0].track_id == 1
    assert tracks[0].hits == 5


def test_two_people_get_separate_tracks(tracker):
    tracks = tracker.update([
        detection((10, 10, 60, 60), "Mohammed"),
        detection((300, 10, 350, 60), "Sara"),
    ])
    assert len({t.track_id for t in tracks}) == 2
    assert {t.label for t in tracks} == {"Mohammed", "Sara"}


def test_a_jump_with_no_overlap_still_continues_the_track(tracker):
    """Fast motion leaves no IoU, so the distance fallback has to catch it."""
    tracker.update([detection((100, 100, 150, 150))])
    tracks = tracker.update([detection((160, 100, 210, 150))])   # moved a whole box over
    assert len(tracks) == 1
    assert tracks[0].track_id == 1


def test_a_far_away_face_starts_a_new_track(tracker):
    tracker.update([detection((100, 100, 150, 150))])
    tracks = tracker.update([detection((900, 600, 950, 650))])
    assert {t.track_id for t in tracks} == {1, 2}


def test_track_survives_missing_frames_then_expires():
    tracker = Tracker(TrackingConfig(min_hits=1, max_age=6), RecognitionConfig())
    tracker.update([detection((10, 10, 60, 60))])
    for _ in range(3):
        tracker.update([])
    assert len(tracker.tracks) == 1, "a short gap must not kill the track"
    for _ in range(10):
        tracker.update([])
    assert tracker.tracks == [], "a long gap must retire the track"


def test_voting_ignores_a_single_bad_frame(tracker):
    for _ in range(4):
        tracker.update([detection((10, 10, 60, 60), "Mohammed", 0.8)])
    tracks = tracker.update([detection((10, 10, 60, 60), "Unknown", 0.1)])
    assert tracks[0].label == "Mohammed", "one unknown frame must not erase the name"


def test_voting_switches_when_the_evidence_does(tracker):
    for _ in range(6):
        tracker.update([detection((10, 10, 60, 60), "Mohammed", 0.8)])
    for _ in range(6):
        tracks = tracker.update([detection((10, 10, 60, 60), "Sara", 0.9)])
    assert tracks[0].label == "Sara"


def test_min_hits_hides_one_frame_noise():
    tracker = Tracker(TrackingConfig(min_hits=3), RecognitionConfig())
    assert tracker.update([detection((10, 10, 60, 60))]) == []
    tracker.update([detection((10, 10, 60, 60))])
    assert len(tracker.update([detection((10, 10, 60, 60))])) == 1


def test_reset_clears_everything(tracker):
    tracker.update([detection((10, 10, 60, 60))])
    tracker.reset()
    assert tracker.tracks == []
    tracks = tracker.update([detection((10, 10, 60, 60))])
    assert tracks[0].track_id == 1, "ids restart for a new video"


def test_best_score_is_remembered(tracker):
    tracker.update([detection((10, 10, 60, 60), "Mohammed", 0.5)])
    tracker.update([detection((10, 10, 60, 60), "Mohammed", 0.91)])
    tracks = tracker.update([detection((10, 10, 60, 60), "Mohammed", 0.6)])
    assert tracks[0].best_score == pytest.approx(0.91)
