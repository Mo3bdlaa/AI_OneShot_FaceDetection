"""Gallery matching, driven with hand-made embeddings instead of real models."""

import numpy as np
import pytest

from oneshot_fd.gallery import Gallery, Person
from oneshot_fd.utils import l2_normalize


def make_gallery(vectors: dict) -> Gallery:
    gallery = Gallery()
    people = [
        Person(name, l2_normalize(np.asarray(v, dtype=np.float32).reshape(1, -1)), [f"{name}.jpg"])
        for name, v in vectors.items()
    ]
    gallery._set_people(people)
    return gallery


@pytest.fixture
def two_people():
    return make_gallery({"Mohammed": [1, 0, 0, 0], "Sara": [0, 1, 0, 0]})


def test_exact_match_wins(two_people):
    match = two_people.identify(np.array([1, 0, 0, 0], np.float32), threshold=0.4)
    assert match.name == "Mohammed"
    assert match.score == pytest.approx(1.0)


def test_unrelated_face_is_unknown(two_people):
    match = two_people.identify(np.array([0, 0, 1, 0], np.float32), threshold=0.4)
    assert match.name == "Unknown"


def test_threshold_is_respected(two_people):
    # 45 degrees off: cosine ~0.707, above 0.5 but below 0.8.
    face = l2_normalize(np.array([1, 0, 1, 0], np.float32))
    assert two_people.identify(face, threshold=0.5).name == "Mohammed"
    assert two_people.identify(face, threshold=0.8).name == "Unknown"


def test_margin_rejects_ambiguous_faces():
    # A face sitting exactly between two people should not be claimed by either.
    gallery = make_gallery({"A": [1, 0, 0, 0], "B": [0.99, 0.141, 0, 0]})
    face = l2_normalize(np.array([1, 0.07, 0, 0], np.float32))
    assert gallery.identify(face, threshold=0.3, margin=0.0).name in {"A", "B"}
    assert gallery.identify(face, threshold=0.3, margin=0.2).name == "Unknown"


def test_best_photo_of_a_person_wins_not_the_average():
    """A person with several photos is matched on their closest one."""
    gallery = Gallery()
    embeddings = l2_normalize(np.array([[1, 0, 0, 0], [0, 0, 0, 1]], np.float32), axis=1)
    gallery._set_people([Person("Ahmed", embeddings, ["a.jpg", "b.jpg"])])
    match = gallery.identify(np.array([0, 0, 0, 1], np.float32), threshold=0.9)
    assert match.name == "Ahmed"
    assert match.score == pytest.approx(1.0)


def test_empty_gallery_is_safe():
    gallery = Gallery()
    assert gallery.is_empty
    assert gallery.identify(np.array([1, 0, 0, 0], np.float32), threshold=0.3).name == "Unknown"


def test_missing_input_folder_explains_itself(tmp_path):
    gallery = Gallery()
    gallery.config.path = tmp_path / "does_not_exist"
    with pytest.raises(FileNotFoundError, match="does not exist"):
        gallery.build(engine=None)


def test_grouping_handles_both_folder_layouts(tmp_path):
    (tmp_path / "Mohammed.jpg").write_bytes(b"x")
    person_dir = tmp_path / "Sara"
    person_dir.mkdir()
    (person_dir / "one.png").write_bytes(b"x")
    (person_dir / "two.jpeg").write_bytes(b"x")
    (tmp_path / "notes.txt").write_text("ignored")

    grouped = Gallery._group_images(tmp_path)
    assert set(grouped) == {"Mohammed", "Sara"}
    assert len(grouped["Sara"]) == 2
