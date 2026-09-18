"""The self-check: degrading a gallery and measuring what survives."""

import cv2
import numpy as np
import pytest

from oneshot_fd import selfcheck
from oneshot_fd.config import AppConfig, RecognitionConfig
from oneshot_fd.gallery import Gallery, Person
from oneshot_fd.selfcheck import DEGRADATIONS, Report, Trial
from oneshot_fd.utils import l2_normalize


# ------------------------------------------------------ what a trial means

def trial(person="Sara", matched="Sara", score=0.8, runner_up_score=0.2, **kwargs):
    return Trial(person=person, photo="s.jpg", degradation="blurred", matched=matched,
                 score=score, runner_up_score=runner_up_score, **kwargs)


def test_a_right_answer_is_correct():
    assert trial().correct
    assert not trial().missed
    assert not trial().confused


def test_no_answer_is_a_miss_not_a_mistake():
    """The margin rule declining to choose is the rule working."""
    t = trial(matched="Unknown")
    assert t.missed
    assert not t.confused, "a refusal is not a misidentification"
    assert not t.correct


def test_the_wrong_person_is_a_confusion():
    t = trial(matched="Omar")
    assert t.confused
    assert not t.missed and not t.correct


def test_an_undetected_face_is_none_of_the_three():
    t = trial(matched="", detected=False)
    assert not t.correct and not t.missed and not t.confused


def test_a_custom_unknown_label_still_counts_as_a_miss():
    t = trial(matched="Guest", unknown_label="Guest")
    assert t.missed and not t.confused


# ------------------------------------------------------------- the report

def test_recall_counts_only_right_answers():
    report = Report(threshold=0.4, people=2, trials=[
        trial(), trial(), trial(matched="Unknown"), trial(matched="Omar"),
    ])
    assert report.recall == pytest.approx(0.5)
    assert len(report.missed) == 1
    assert len(report.confused) == 1


def test_the_advice_leads_with_the_worst_problem():
    confused = Report(threshold=0.4, people=2, trials=[trial(matched="Omar")])
    assert "wrong person" in confused.advice().lower()
    assert "Sara" in confused.advice()

    missing = Report(threshold=0.4, people=2,
                     trials=[trial(matched="Unknown", runner_up="Omar")] * 4)
    assert "Unknown" in missing.advice()

    fine = Report(threshold=0.4, people=2, trials=[trial()] * 10)
    assert "separates cleanly" in fine.advice()


def test_a_near_tie_is_named_in_the_advice():
    report = Report(threshold=0.4, people=2, trials=[
        trial(matched="Unknown", best_name="Ahmed", runner_up="Ahmed2",
              score=0.99, runner_up_score=0.99)
    ] * 4)
    assert "Ahmed" in report.advice() and "Ahmed2" in report.advice()
    assert "same person enrolled twice" in report.advice()


def test_the_report_separates_misses_from_wrong_names():
    report = Report(threshold=0.4, people=2, trials=[
        trial(matched="Unknown", best_name="Ahmed", runner_up="Ahmed2"),
        trial(matched="Omar"),
    ])
    text = report.format()
    assert "WRONG PERSON" in text
    assert "Not a wrong answer - no answer" in text


def test_an_empty_report_says_so():
    assert "gallery is empty" in Report(threshold=0.4).format()


def test_the_report_groups_by_person_and_by_degradation():
    report = Report(threshold=0.4, people=2, trials=[
        trial(person="Sara", matched="Sara"),
        trial(person="Omar", matched="Omar"),
    ])
    assert set(report.by_person()) == {"Sara", "Omar"}
    assert set(report.by_degradation()) == {"blurred"}


# ------------------------------------------------- the degradations themselves

@pytest.fixture
def photo():
    rng = np.random.default_rng(5)
    return rng.integers(30, 220, (200, 160, 3), dtype=np.uint8)


@pytest.mark.parametrize("name, degrade", DEGRADATIONS)
def test_every_degradation_keeps_the_image_usable(name, degrade, photo):
    out = degrade(photo)
    assert out is not None, name
    assert out.dtype == np.uint8
    assert out.ndim == 3
    assert out.shape[2] == 3
    assert not np.array_equal(out, photo), f"{name} changed nothing"


def test_the_degradations_are_the_ones_video_actually_does(photo):
    kinds = dict(DEGRADATIONS)
    assert kinds["mirrored"](photo).shape == photo.shape
    assert kinds["dim"](photo).mean() < photo.mean()
    assert kinds["bright"](photo).mean() > photo.mean()
    # Shrinking and blowing back up must lose detail, not just resize.
    from oneshot_fd.quality import sharpness_of

    assert sharpness_of(kinds["quarter size"](photo)) < sharpness_of(photo)


def test_exposure_does_not_wrap_around():
    """Naive int8 arithmetic turns a bright pixel into a black one."""
    white = np.full((10, 10, 3), 250, np.uint8)
    assert dict(DEGRADATIONS)["bright"](white).min() >= 250

    black = np.full((10, 10, 3), 5, np.uint8)
    assert dict(DEGRADATIONS)["dim"](black).max() <= 5


# ------------------------------------------------------------ the run itself

class ScriptedEngine:
    """Returns a chosen embedding for each degradation, so a run is decidable."""

    def __init__(self, embeddings):
        self.embeddings = embeddings
        self.seen = []

    def load(self):
        return self

    def embed_reference(self, image, min_face_size=40):
        from oneshot_fd.faces import DetectedFace

        index = len(self.seen)
        self.seen.append(index)
        vector = self.embeddings[index % len(self.embeddings)]
        if vector is None:
            return None
        return DetectedFace(box=(0, 0, 50, 50), score=0.9, embedding=vector)


def gallery_of(**people):
    gallery = Gallery()
    gallery._set_people([
        Person(name, l2_normalize(np.asarray([vector], np.float32), axis=1), [f"{name}.jpg"])
        for name, vector in people.items()
    ])
    return gallery


def test_a_run_reports_one_trial_per_degradation(tmp_path):
    photo = tmp_path / "Sara.jpg"
    cv2.imwrite(str(photo), np.full((120, 120, 3), 128, np.uint8))

    gallery = gallery_of(Sara=[1, 0, 0, 0], Omar=[0, 1, 0, 0])
    gallery.people[0].sources = [str(photo)]

    sara = np.array([1, 0, 0, 0], np.float32)
    engine = ScriptedEngine([sara])
    config = AppConfig(recognition=RecognitionConfig(threshold=0.4))

    report = selfcheck.run(config, engine, gallery)
    per_photo = [t for t in report.trials if t.person == "Sara"]
    assert len(per_photo) == len(DEGRADATIONS)
    assert all(t.correct for t in per_photo)
    assert report.recall == 1.0


def test_a_run_records_an_undetectable_copy_rather_than_skipping_it(tmp_path):
    photo = tmp_path / "Sara.jpg"
    cv2.imwrite(str(photo), np.full((120, 120, 3), 128, np.uint8))
    gallery = gallery_of(Sara=[1, 0, 0, 0])
    gallery.people[0].sources = [str(photo)]

    report = selfcheck.run(AppConfig(), ScriptedEngine([None]), gallery)
    assert len(report.undetected) == len(DEGRADATIONS)
    assert report.recall == 0.0, "an undetected copy is not a success"


def test_a_missing_photo_file_does_not_stop_the_run(tmp_path):
    gallery = gallery_of(Ghost=[1, 0, 0, 0])
    gallery.people[0].sources = [str(tmp_path / "gone.jpg")]
    report = selfcheck.run(AppConfig(), ScriptedEngine([np.array([1, 0, 0, 0], np.float32)]),
                           gallery)
    assert report.trials == []


def test_mirrored_reference_entries_are_not_treated_as_separate_photos(tmp_path):
    """The gallery records a mirrored copy against the same file; it is not one."""
    photo = tmp_path / "Sara.jpg"
    cv2.imwrite(str(photo), np.full((120, 120, 3), 128, np.uint8))
    gallery = gallery_of(Sara=[1, 0, 0, 0])
    gallery.people[0].sources = [str(photo), f"{photo} (mirrored)"]

    report = selfcheck.run(AppConfig(), ScriptedEngine([np.array([1, 0, 0, 0], np.float32)]),
                           gallery)
    assert len(report.trials) == len(DEGRADATIONS), "the same photo was checked twice"


# ------------------------------------------------------------ over the API

def test_the_endpoint_refuses_while_a_session_is_running():
    pytest.importorskip("fastapi", reason="the web UI is an optional extra")
    from fastapi.testclient import TestClient

    from oneshot_fd.config import BodyConfig, GalleryConfig
    from oneshot_fd.web.server import create_app

    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as folder:
        faces = Path(folder) / "faces"
        faces.mkdir()
        config = AppConfig(gallery=GalleryConfig(path=faces, cache=False),
                           body=BodyConfig(mode="off"))
        app = create_app(config)
        app.state.session._pushed = True          # pretend a camera session is live
        with TestClient(app) as client:
            response = client.post("/api/self-check")
        assert response.status_code == 409
        assert "Stop the running session" in response.json()["detail"]
