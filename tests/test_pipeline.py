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
    """Returns a scripted list of faces per frame instead of running a model.

    It mirrors the real split: ``locate`` hands back faces with no embedding,
    and ``embed`` fills one in - so the tests see exactly how often the
    expensive half would have run.
    """

    def __init__(self, script):
        self.script = script
        self.calls = 0
        self.embeddings = 0

    def load(self):
        return self

    def locate(self, frame, max_faces=None):
        scripted = self.script[min(self.calls, len(self.script) - 1)]
        self.calls += 1
        return [DetectedFace(box=f.box, score=f.score, embedding=None,
                             landmarks=f.embedding) for f in scripted]

    def embed(self, frame, face):
        # The fake smuggles the embedding through ``landmarks`` so that
        # ``locate`` can hand back a genuinely un-embedded face.
        self.embeddings += 1
        face.embedding = face.landmarks
        return face.embedding

    def detect(self, frame, max_faces=None):
        faces = self.locate(frame, max_faces)
        for face in faces:
            self.embed(frame, face)
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


# --------------------------------------------------------- recognition throttle

def test_throttle_reuses_a_settled_name_instead_of_re_embedding(frame):
    """A track that keeps naming the same person should stop paying for it."""
    pipeline = make_pipeline([[face((100, 100, 200, 220), MOHAMMED)]])
    pipeline.config.recognition.reverify_every = 5

    for index in range(12):
        result = pipeline.process_frame(frame, index, index / 25.0)

    assert [t.label for t in result.tracks] == ["Mohammed"]
    assert pipeline._skipped_embeddings > 0, "the throttle never kicked in"
    assert pipeline.engine.embeddings < 12, "every frame still paid for an embedding"


def test_throttle_still_re_verifies_periodically(frame):
    pipeline = make_pipeline([[face((100, 100, 200, 220), MOHAMMED)]])
    pipeline.config.recognition.reverify_every = 3

    for index in range(12):
        pipeline.process_frame(frame, index, index / 25.0)

    # Roughly one embedding in three, never zero.
    assert 3 <= pipeline.engine.embeddings <= 8


def test_throttle_off_by_default_embeds_every_face(frame):
    pipeline = make_pipeline([[face((100, 100, 200, 220), MOHAMMED)]])
    for index in range(6):
        pipeline.process_frame(frame, index, index / 25.0)
    assert pipeline.engine.embeddings == 6
    assert pipeline._skipped_embeddings == 0


def test_an_unsettled_track_is_always_embedded(frame):
    """Unknown faces never qualify for the shortcut."""
    pipeline = make_pipeline([[face((100, 100, 200, 220), STRANGER)]])
    pipeline.config.recognition.reverify_every = 5
    for index in range(10):
        pipeline.process_frame(frame, index, index / 25.0)
    assert pipeline.engine.embeddings == 10
    assert pipeline._skipped_embeddings == 0


def test_a_new_face_is_never_taken_at_another_tracks_word(frame):
    """A second person entering must be embedded, not handed a neighbour's name."""
    script = [[face((100, 100, 200, 220), MOHAMMED)]] * 8 + [
        [face((100, 100, 200, 220), MOHAMMED), face((700, 100, 800, 220), SARA)]
    ] * 4
    pipeline = make_pipeline(script)
    pipeline.config.recognition.reverify_every = 5

    for index in range(12):
        result = pipeline.process_frame(frame, index, index / 25.0)

    assert {t.label for t in result.tracks} == {"Mohammed", "Sara"}


def test_someone_still_on_screen_is_in_the_running_report(frame):
    """A report taken mid-run must include the people you can still see."""
    pipeline = make_pipeline([[face((100, 100, 200, 220), MOHAMMED)]])
    for index in range(6):
        pipeline.process_frame(frame, index, index / 25.0)

    assert pipeline.appearances == [], "their appearance has not ended yet"
    current = pipeline.current_appearances()
    assert [a.name for a in current] == ["Mohammed"]
    # ... and asking must not have ended it.
    assert pipeline.appearances == []


def test_reset_clears_the_report_for_a_fresh_run(frame):
    pipeline = make_pipeline([[face((100, 100, 200, 220), MOHAMMED)]])
    for index in range(6):
        pipeline.process_frame(frame, index, index / 25.0)
    assert pipeline.current_appearances()

    pipeline.reset_results()
    assert pipeline.appearances == []
    assert "No known person" in pipeline.summary()
    assert pipeline._embeddings == 0
