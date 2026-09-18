"""Appearance descriptors and the body-based identity bank."""

import numpy as np
import pytest

from oneshot_fd.reid import AppearanceBank, describe, similarity


def person_patch(frame, box, shirt, trousers):
    """Paint a crude person: a coloured shirt above, trousers below."""
    x1, y1, x2, y2 = box
    height = y2 - y1
    frame[y1:y1 + int(height * 0.55), x1:x2] = shirt
    frame[y1 + int(height * 0.55):y2, x1:x2] = trousers
    return frame


@pytest.fixture
def frame():
    return np.full((720, 1280, 3), 30, dtype=np.uint8)


RED_SHIRT = (40, 40, 200)
BLUE_SHIRT = (200, 60, 40)
DARK_JEANS = (90, 60, 40)


def test_describe_returns_a_unit_vector(frame):
    person_patch(frame, (100, 100, 200, 500), RED_SHIRT, DARK_JEANS)
    vector = describe(frame, (100, 100, 200, 500))
    assert vector is not None
    assert np.linalg.norm(vector) == pytest.approx(1.0, abs=1e-5)


def test_the_same_person_matches_themselves(frame):
    person_patch(frame, (100, 100, 200, 500), RED_SHIRT, DARK_JEANS)
    first = describe(frame, (100, 100, 200, 500))
    # Same clothes, slightly different box - as a tracker would give.
    second = describe(frame, (104, 96, 198, 505))
    assert similarity(first, second) > 0.9


def test_differently_dressed_people_do_not_match(frame):
    person_patch(frame, (100, 100, 200, 500), RED_SHIRT, DARK_JEANS)
    person_patch(frame, (600, 100, 700, 500), BLUE_SHIRT, DARK_JEANS)
    red = describe(frame, (100, 100, 200, 500))
    blue = describe(frame, (600, 100, 700, 500))
    assert similarity(red, blue) < 0.7


def test_describe_refuses_a_box_with_nothing_in_it(frame):
    assert describe(frame, (10, 10, 12, 12)) is None
    assert describe(None, (0, 0, 10, 10)) is None


def test_similarity_is_safe_with_missing_vectors():
    assert similarity(None, None) == 0.0
    assert similarity(np.ones(4, np.float32), None) == 0.0
    assert similarity(np.ones(4, np.float32), np.ones(8, np.float32)) == 0.0


# ------------------------------------------------------------------- the bank

def test_a_body_is_named_from_a_face_confirmed_appearance(frame):
    person_patch(frame, (100, 100, 200, 500), RED_SHIRT, DARK_JEANS)
    bank = AppearanceBank()
    bank.observe(frame, (100, 100, 200, 500), "Mohammed", frame_index=1)

    # The same person a few frames later, face no longer visible.
    later = np.full((720, 1280, 3), 30, np.uint8)
    person_patch(later, (300, 110, 400, 510), RED_SHIRT, DARK_JEANS)
    name, score = bank.identify(later, (300, 110, 400, 510), frame_index=5)
    assert name == "Mohammed"
    assert score > 0.7


def test_a_stranger_is_not_named(frame):
    person_patch(frame, (100, 100, 200, 500), RED_SHIRT, DARK_JEANS)
    bank = AppearanceBank()
    bank.observe(frame, (100, 100, 200, 500), "Mohammed", frame_index=1)

    other = np.full((720, 1280, 3), 30, np.uint8)
    person_patch(other, (300, 110, 400, 510), BLUE_SHIRT, DARK_JEANS)
    name, _ = bank.identify(other, (300, 110, 400, 510), frame_index=5)
    assert name is None


def test_a_name_already_on_screen_is_not_handed_out_twice(frame):
    """One person cannot be in two places, so a visible face wins."""
    person_patch(frame, (100, 100, 200, 500), RED_SHIRT, DARK_JEANS)
    bank = AppearanceBank()
    bank.observe(frame, (100, 100, 200, 500), "Mohammed", frame_index=1)

    name, _ = bank.identify(frame, (100, 100, 200, 500), frame_index=2,
                            exclude=["Mohammed"])
    assert name is None


def test_two_similarly_dressed_people_are_left_alone(frame):
    """Without a clear winner the bank must decline rather than guess."""
    person_patch(frame, (100, 100, 200, 500), RED_SHIRT, DARK_JEANS)
    bank = AppearanceBank(margin=0.15)
    bank.observe(frame, (100, 100, 200, 500), "Mohammed", frame_index=1)
    bank.observe(frame, (100, 100, 200, 500), "Ahmed", frame_index=1)

    name, _ = bank.identify(frame, (100, 100, 200, 500), frame_index=2)
    assert name is None, "identical outfits must not be resolved by guessing"


def test_appearances_expire(frame):
    person_patch(frame, (100, 100, 200, 500), RED_SHIRT, DARK_JEANS)
    bank = AppearanceBank(memory=10)
    bank.observe(frame, (100, 100, 200, 500), "Mohammed", frame_index=1)

    assert bank.identify(frame, (100, 100, 200, 500), frame_index=5)[0] == "Mohammed"
    assert bank.identify(frame, (100, 100, 200, 500), frame_index=500)[0] is None


def test_stale_appearances_are_dropped(frame):
    person_patch(frame, (100, 100, 200, 500), RED_SHIRT, DARK_JEANS)
    bank = AppearanceBank(memory=10)
    bank.observe(frame, (100, 100, 200, 500), "Mohammed", frame_index=1)
    assert len(bank) == 1
    bank.forget_stale(frame_index=1000)
    assert len(bank) == 0


def test_observing_again_refines_rather_than_replaces(frame):
    person_patch(frame, (100, 100, 200, 500), RED_SHIRT, DARK_JEANS)
    bank = AppearanceBank()
    bank.observe(frame, (100, 100, 200, 500), "Mohammed", frame_index=1)
    first = bank.entries["Mohammed"].vector.copy()

    lit = np.clip(frame.astype(np.int16) + 25, 0, 255).astype(np.uint8)
    bank.observe(lit, (100, 100, 200, 500), "Mohammed", frame_index=2)
    second = bank.entries["Mohammed"].vector

    assert bank.entries["Mohammed"].samples == 2
    assert similarity(first, second) > 0.8, "the descriptor drifts, it does not jump"
