"""A folder of unrelated photos, as opposed to frames of one scene."""

import cv2
import numpy as np
import pytest

from oneshot_fd.config import AppConfig, BodyConfig, GalleryConfig, RuntimeConfig
from oneshot_fd.faces import DetectedFace
from oneshot_fd.gallery import Person
from oneshot_fd.pipeline import Pipeline
from oneshot_fd.utils import l2_normalize

ALICE = np.array([1, 0, 0, 0], np.float32)
BILAL = np.array([0, 1, 0, 0], np.float32)


class PhotoEngine:
    """Returns a different person for each photo, in the order they are read."""

    def __init__(self, per_photo):
        self.per_photo = per_photo
        self.calls = 0

    def load(self):
        return self

    def locate(self, frame, max_faces=None):
        faces = self.per_photo[min(self.calls, len(self.per_photo) - 1)]
        self.calls += 1
        return [DetectedFace(box=box, score=0.95, embedding=None, landmarks=vector)
                for box, vector in faces]

    def embed(self, frame, face):
        face.embedding = face.landmarks
        return face.embedding

    def detect(self, frame, max_faces=None):
        faces = self.locate(frame)
        for face in faces:
            self.embed(frame, face)
        return faces


@pytest.fixture
def photo_folder(tmp_path):
    folder = tmp_path / "photos"
    folder.mkdir()
    for name in ("a.jpg", "b.jpg"):
        cv2.imwrite(str(folder / name), np.full((200, 200, 3), 128, np.uint8))
    return folder


def make_pipeline(per_photo, folder, **runtime):
    config = AppConfig(
        gallery=GalleryConfig(cache=False),
        body=BodyConfig(mode="off"),
        runtime=RuntimeConfig(sources=[str(folder)], display=False, **runtime),
    )
    pipeline = Pipeline(config)
    pipeline.engine = PhotoEngine(per_photo)
    pipeline.gallery._set_people([
        Person("Alice", l2_normalize(ALICE.reshape(1, -1)), ["a.jpg"]),
        Person("Bilal", l2_normalize(BILAL.reshape(1, -1)), ["b.jpg"]),
    ])
    pipeline.bodies.load()
    return pipeline


def test_two_photos_of_two_people_report_both(photo_folder):
    """The bug this fixes: the second photo inherited the first one's name.

    Two different people photographed in roughly the same place in the frame
    were matched to a single track, and the tracker's vote kept whichever name
    it settled on first. A folder of three people reported one.
    """
    # Same box in both photos - the worst case for a tracker.
    script = [[((50, 50, 150, 150), ALICE)], [((50, 50, 150, 150), BILAL)]]
    pipeline = make_pipeline(script, photo_folder)

    seen = [result.names for result in pipeline.run_source(str(photo_folder))]
    assert seen == [["Alice"], ["Bilal"]]


def test_each_photo_is_credited_by_name(photo_folder):
    script = [[((50, 50, 150, 150), ALICE)], [((50, 50, 150, 150), BILAL)]]
    pipeline = make_pipeline(script, photo_folder)
    list(pipeline.run_source(str(photo_folder)))

    found = {(a.source, a.name) for a in pipeline.current_appearances()}
    assert found == {("a.jpg", "Alice"), ("b.jpg", "Bilal")}


def test_a_single_frame_is_enough_to_be_reported(photo_folder):
    """A track is normally shown only after two sightings; a photo has one."""
    script = [[((50, 50, 150, 150), ALICE)], [((50, 50, 150, 150), ALICE)]]
    pipeline = make_pipeline(script, photo_folder)
    results = list(pipeline.run_source(str(photo_folder)))
    assert all(result.tracks for result in results)


def test_the_min_hits_setting_is_left_as_it_was_found(photo_folder):
    """Photo mode relaxes it temporarily; a video afterwards must not inherit that."""
    script = [[((50, 50, 150, 150), ALICE)]]
    pipeline = make_pipeline(script, photo_folder)
    before = pipeline.tracker.config.min_hits
    list(pipeline.run_source(str(photo_folder)))
    assert pipeline.tracker.config.min_hits == before


def test_as_sequence_keeps_the_tracker_running_across_frames(photo_folder):
    """Frames extracted from a video are a sequence, and should behave like one."""
    script = [[((50, 50, 150, 150), ALICE)], [((52, 50, 152, 150), ALICE)]]
    pipeline = make_pipeline(script, photo_folder, photos_are_independent=False)
    results = list(pipeline.run_source(str(photo_folder)))

    ids = {track.track_id for result in results for track in result.tracks}
    assert ids == {1}, "one person walking through two frames is one track"


def test_several_people_in_one_photo_are_all_reported(photo_folder):
    script = [[((20, 20, 90, 90), ALICE), ((110, 20, 180, 90), BILAL)],
              [((20, 20, 90, 90), ALICE)]]
    pipeline = make_pipeline(script, photo_folder)
    results = list(pipeline.run_source(str(photo_folder)))
    assert set(results[0].names) == {"Alice", "Bilal"}
