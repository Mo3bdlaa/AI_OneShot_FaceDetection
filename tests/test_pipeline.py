"""Pipeline behaviour, driven with stand-in models so no download is needed."""

import numpy as np
import pytest

from oneshot_fd.config import AppConfig, BodyConfig, RuntimeConfig
from oneshot_fd.faces import DetectedFace
from oneshot_fd.gallery import Person
from oneshot_fd.pipeline import Pipeline
from oneshot_fd.utils import l2_normalize

MOHAMMED = np.array([1, 0, 0, 0], np.float32)
SARA = np.array([0, 1, 0, 0], np.float32)
STRANGER = np.array([0, 0, 1, 0], np.float32)


class FakeEngine:
    """Returns a scripted list of faces per frame instead of running a model."""

    def __init__(self, script):
        self.script = script
        self.calls = 0

    def load(self):
        return self

    def detect(self, frame, max_faces=None):
        faces = self.script[min(self.calls, len(self.script) - 1)]
        self.calls += 1
        return faces


def face(box, embedding):
    return DetectedFace(box=box, score=0.95, embedding=embedding)


def make_pipeline(script, **runtime):
    config = AppConfig(body=BodyConfig(mode="estimate"),
                       runtime=RuntimeConfig(display=False, **runtime))
    pipeline = Pipeline(config)
    pipeline.engine = FakeEngine(script)
    pipeline.gallery._set_people([
        Person("Mohammed", l2_normalize(MOHAMMED.reshape(1, -1)), ["m.jpg"]),
        Person("Sara", l2_normalize(SARA.reshape(1, -1)), ["s.jpg"]),
    ])
    pipeline.bodies.load()
    return pipeline


@pytest.fixture
def frame():
    return np.zeros((720, 1280, 3), dtype=np.uint8)


def test_a_known_face_is_named(frame):
    pipeline = make_pipeline([[face((100, 100, 200, 220), MOHAMMED)]])
    for index in range(3):
        result = pipeline.process_frame(frame, index, index / 25.0)
    assert [t.label for t in result.tracks] == ["Mohammed"]


def test_an_unknown_face_is_not_named(frame):
    pipeline = make_pipeline([[face((100, 100, 200, 220), STRANGER)]])
    for index in range(3):
        result = pipeline.process_frame(frame, index, index / 25.0)
    assert [t.label for t in result.tracks] == ["Unknown"]


def test_two_people_are_recognised_at_once(frame):
    script = [[face((100, 100, 200, 220), MOHAMMED), face((600, 100, 700, 220), SARA)]]
    pipeline = make_pipeline(script)
    for index in range(3):
        result = pipeline.process_frame(frame, index, index / 25.0)
    assert {t.label for t in result.tracks} == {"Mohammed", "Sara"}


def test_every_recognised_face_gets_a_body_box(frame):
    pipeline = make_pipeline([[face((100, 100, 200, 220), MOHAMMED)]])
    for index in range(3):
        result = pipeline.process_frame(frame, index, index / 25.0)
    body = result.tracks[0].body_box
    assert body is not None
    assert body[3] > result.tracks[0].box[3], "the body reaches below the face"


def test_detect_every_skips_the_heavy_work(frame):
    pipeline = make_pipeline([[face((100, 100, 200, 220), MOHAMMED)]], detect_every=3)
    flags = [pipeline.process_frame(frame, i, i / 25.0).detected for i in range(6)]
    assert flags == [True, False, False, True, False, False]
    assert pipeline.engine.calls == 2


def test_max_width_downscales_the_output(frame):
    pipeline = make_pipeline([[]], max_width=640)
    result = pipeline.process_frame(frame, 0, 0.0)
    assert result.frame.shape[1] == 640


def test_appearances_and_summary_are_recorded(frame):
    pipeline = make_pipeline([[face((100, 100, 200, 220), MOHAMMED)]])
    for index in range(6):
        pipeline.process_frame(frame, index, index / 25.0)
    summary = pipeline.summary()
    assert "Mohammed" in summary
    assert pipeline.appearances and pipeline.appearances[0].name == "Mohammed"


def test_summary_is_honest_when_nobody_was_seen(frame):
    pipeline = make_pipeline([[]])
    pipeline.process_frame(frame, 0, 0.0)
    assert "No known person" in pipeline.summary()


def test_csv_log_is_written(tmp_path, frame):
    target = tmp_path / "log.csv"
    pipeline = make_pipeline([[face((100, 100, 200, 220), MOHAMMED)]], log_csv=target)
    pipeline._open_csv(target)
    for index in range(4):
        pipeline.process_frame(frame, index, index / 25.0)
    pipeline.close()

    rows = target.read_text().strip().splitlines()
    assert rows[0].startswith("source,name,track_id")
    assert "Mohammed" in rows[1]


def test_frames_with_no_faces_are_fine(frame):
    pipeline = make_pipeline([[]])
    result = pipeline.process_frame(frame, 0, 0.0)
    assert result.tracks == []
    assert result.frame.shape == frame.shape
