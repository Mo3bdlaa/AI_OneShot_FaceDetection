"""Appearance descriptors and the body-based identity bank."""

import cv2
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


# ------------------------------------- what the descriptor can and cannot do
#
# These encode measurements taken on six real people in one photograph, which
# is what exposed the original threshold as unusable. The earlier tests in this
# file use flat painted rectangles and so are far kinder than reality; these
# pin down the limits that actually decide the design.

def clothed(frame, box, shirt, trousers, seed=0):
    """A body wearing something with folds and shading in it.

    Flat colour plus random noise is not a stand-in for cloth: blur averages
    the noise away completely, so a motion-blurred flat patch scored 0.61 where
    a real person in a real photograph scored 0.969. Fabric has structure that
    survives blur, and the synthetic version needs it too or the test measures
    the wrong thing.
    """
    rng = np.random.default_rng(seed)
    x1, y1, x2, y2 = box
    split = y1 + int((y2 - y1) * 0.55)
    for (top, bottom), colour in (((y1, split), shirt), ((split, y2), trousers)):
        height, width = bottom - top, x2 - x1
        shading = np.tile(np.linspace(0.75, 1.15, width), (height, 1))
        folds = 0.12 * np.sin(np.linspace(0, 9 * np.pi, width))[None, :]
        patch = np.array(colour)[None, None, :] * (shading + folds)[:, :, None]
        patch = patch + rng.integers(-8, 8, patch.shape)
        frame[top:bottom, x1:x2] = np.clip(patch, 0, 255).astype(np.uint8)
    return frame


def test_the_same_body_a_moment_later_scores_far_above_the_threshold():
    """Box jitter and motion are what the descriptor actually has to survive."""
    frame = np.full((600, 800, 3), 40, np.uint8)
    clothed(frame, (100, 100, 220, 520), (40, 40, 200), (90, 60, 40))

    settled = describe(frame, (100, 100, 220, 520))
    for label, box in [
        ("shifted", (110, 96, 230, 524)),
        ("tighter", (110, 112, 210, 500)),
    ]:
        assert similarity(settled, describe(frame, box)) > 0.85, label

    # Motion blur measured 0.969 on a real person; textured cloth here gives
    # 0.88, which is the same side of the line.
    blurred = cv2.blur(frame, (15, 3))
    assert similarity(settled, describe(blurred, (100, 100, 220, 520))) > 0.85


def test_a_lighting_change_breaks_it_and_that_is_the_documented_limit():
    """It must fail to Unknown, not to somebody else's name.

    Measured on real people: the same person under a shifted light can fall to
    0.29, which is inside the range two *different* people occupy. No threshold
    separates those, so the memory window is kept short instead of pretending
    otherwise.
    """
    frame = np.full((600, 800, 3), 40, np.uint8)
    clothed(frame, (100, 100, 220, 520), (40, 40, 200), (90, 60, 40))
    settled = describe(frame, (100, 100, 220, 520))

    dim = np.clip(frame.astype(np.int16) - 60, 0, 255).astype(np.uint8)
    dropped = similarity(settled, describe(dim, (100, 100, 220, 520)))

    bank = AppearanceBank()
    assert dropped < bank.threshold, "the test assumes a big light change hurts"

    bank.observe(frame, (100, 100, 220, 520), "Mohammed", frame_index=1)
    name, _ = bank.identify(dim, (100, 100, 220, 520), frame_index=2)
    assert name is None, "a light change must lose the name, never reassign it"


def test_the_default_threshold_sits_above_what_strangers_reach():
    """0.85, from measuring six real people at 0.34-0.81 between them."""
    bank = AppearanceBank()
    assert bank.threshold >= 0.82
    assert bank.memory <= 75, "a long memory outlives the lighting it relies on"


def test_an_open_search_cannot_tell_apart_two_people_dressed_alike():
    """Why the pipeline asks `confirms`, not `identify`.

    With one person in the bank there is no runner-up for the margin rule to
    catch, so a stranger in the same outfit simply takes their name. That is
    unfixable with a colour histogram, so the pipeline never asks the question.
    """
    frame = np.full((600, 900, 3), 40, np.uint8)
    clothed(frame, (100, 100, 220, 520), (45, 45, 190), (90, 60, 40), seed=1)
    clothed(frame, (400, 100, 520, 520), (40, 40, 200), (92, 58, 42), seed=2)

    bank = AppearanceBank()
    bank.observe(frame, (100, 100, 220, 520), "Mohammed", frame_index=1)
    name, score = bank.identify(frame, (400, 100, 520, 520), frame_index=2)
    assert name == "Mohammed" and score > 0.85, "documenting the failure, not endorsing it"


def test_confirms_answers_about_one_named_person_only():
    frame = np.full((600, 800, 3), 40, np.uint8)
    clothed(frame, (100, 100, 220, 520), (40, 40, 200), (90, 60, 40))

    bank = AppearanceBank()
    bank.observe(frame, (100, 100, 220, 520), "Mohammed", frame_index=1)

    held, score = bank.confirms(frame, (104, 96, 224, 524), "Mohammed", frame_index=3)
    assert held and score > 0.85

    # Nobody remembered by that name, so nothing to confirm against.
    assert bank.confirms(frame, (104, 96, 224, 524), "Sara", frame_index=3) == (False, 0.0)


def test_confirms_expires_with_the_memory_window():
    frame = np.full((600, 800, 3), 40, np.uint8)
    clothed(frame, (100, 100, 220, 520), (40, 40, 200), (90, 60, 40))
    bank = AppearanceBank(memory=10)
    bank.observe(frame, (100, 100, 220, 520), "Mohammed", frame_index=1)

    assert bank.confirms(frame, (100, 100, 220, 520), "Mohammed", frame_index=5)[0]
    assert not bank.confirms(frame, (100, 100, 220, 520), "Mohammed", frame_index=500)[0]
