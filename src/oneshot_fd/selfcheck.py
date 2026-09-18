"""Will this actually work for *my* people?

Every other test in this project uses stand-in models, because real ones make
tests slow and flaky. That leaves one question unanswered: given this gallery,
on this machine, will these particular people be recognised?

Nothing shipped can answer that, because the answer depends on photos only the
user has. So the check runs on their gallery instead. Each reference photo is
degraded the way a video degrades a face - smaller, softer, darker, brighter,
mirrored, compressed - and the result is matched back against the gallery. Two
things are then true or not:

* **recall** - does the person still match themselves above the threshold?
* **separation** - do they beat everyone else, and by how much?

That is a genuine measurement on real data, and it is the one that predicts
whether a run will work. It is not a benchmark against a public dataset and
does not pretend to be: a gallery of six people is an easier problem than a
gallery of six hundred, and the report says so.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from .config import AppConfig
from .faces import FaceEngine
from .gallery import Gallery, Person
from .utils import LOGGER, imread_unicode


def _scaled(image: np.ndarray, factor: float) -> np.ndarray:
    """Shrink and blow back up: what distance does to a face."""
    height, width = image.shape[:2]
    small = cv2.resize(image, (max(16, int(width * factor)), max(16, int(height * factor))),
                       interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (width, height), interpolation=cv2.INTER_LINEAR)


def _exposure(image: np.ndarray, delta: int) -> np.ndarray:
    return np.clip(image.astype(np.int16) + delta, 0, 255).astype(np.uint8)


def _compressed(image: np.ndarray, quality: int) -> np.ndarray:
    ok, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return cv2.imdecode(buffer, cv2.IMREAD_COLOR) if ok else image


#: Each is (name, what it does to the image). Chosen because these are what
#: actually happens between a portrait and a frame of video.
DEGRADATIONS: List[Tuple[str, Callable[[np.ndarray], np.ndarray]]] = [
    ("mirrored", lambda image: cv2.flip(image, 1)),
    ("half size", lambda image: _scaled(image, 0.5)),
    ("quarter size", lambda image: _scaled(image, 0.25)),
    ("blurred", lambda image: cv2.GaussianBlur(image, (7, 7), 0)),
    ("dim", lambda image: _exposure(image, -45)),
    ("bright", lambda image: _exposure(image, 45)),
    ("heavily compressed", lambda image: _compressed(image, 25)),
]


@dataclass
class Trial:
    """One degraded photo, put back through recognition."""

    person: str
    photo: str
    degradation: str
    matched: str
    score: float
    runner_up_score: float
    #: Who the best score belonged to, even when the match was refused.
    best_name: str = ""
    runner_up: str = ""
    detected: bool = True
    unknown_label: str = "Unknown"

    @property
    def correct(self) -> bool:
        return self.detected and self.matched == self.person

    @property
    def missed(self) -> bool:
        """Recognised nobody. A refusal, not a mistake - a very different thing.

        Usually the margin rule declining to choose between two people who look
        alike, which is the rule working rather than failing.
        """
        return self.detected and self.matched == self.unknown_label

    @property
    def confused(self) -> bool:
        """Confidently named the wrong person. The failure that actually matters."""
        return self.detected and not self.correct and not self.missed

    @property
    def margin(self) -> float:
        return self.score - self.runner_up_score


#: Below this, a missed match is more likely a near-tie than a weak photo.
recognition_margin_hint = 0.15


@dataclass
class HoldOut:
    """One photo identified by a gallery that did not contain it.

    This is the honest question. Degrading an enrolled photo asks whether the
    pipeline is intact; holding a photo out and asking who it is asks whether
    recognition actually generalises to a different day, a different camera, a
    different haircut. The two give very different numbers - about 0.96 for a
    degraded copy of the same photo, and 0.64-0.84 for a genuinely different
    one - so they are reported apart.
    """

    person: str
    photo: str
    matched: str
    score: float
    runner_up: str = ""
    runner_up_score: float = 0.0
    detected: bool = True

    @property
    def correct(self) -> bool:
        return self.detected and self.matched == self.person

    @property
    def margin(self) -> float:
        return self.score - self.runner_up_score


@dataclass
class Report:
    """What the whole check found."""

    threshold: float
    trials: List[Trial] = field(default_factory=list)
    people: int = 0
    #: Results of the stronger test, where it could be run at all.
    holdouts: List[HoldOut] = field(default_factory=list)
    #: People who only have one photo, so could not be held out.
    single_photo: List[str] = field(default_factory=list)

    @property
    def attempted(self) -> int:
        return len(self.trials)

    @property
    def undetected(self) -> List[Trial]:
        return [t for t in self.trials if not t.detected]

    @property
    def confused(self) -> List[Trial]:
        """Copies matched to the wrong *person* - the failure that matters."""
        return [t for t in self.trials if t.confused]

    @property
    def missed(self) -> List[Trial]:
        """Copies nobody claimed. A gap in recall, not a misidentification."""
        return [t for t in self.trials if t.missed]

    @property
    def recall(self) -> float:
        return (sum(t.correct for t in self.trials) / self.attempted) if self.attempted else 0.0

    def by_person(self) -> Dict[str, List[Trial]]:
        grouped: Dict[str, List[Trial]] = {}
        for trial in self.trials:
            grouped.setdefault(trial.person, []).append(trial)
        return grouped

    def by_degradation(self) -> Dict[str, List[Trial]]:
        grouped: Dict[str, List[Trial]] = {}
        for trial in self.trials:
            grouped.setdefault(trial.degradation, []).append(trial)
        return grouped

    def format(self) -> str:
        if not self.trials:
            return "Nothing to check: the gallery is empty."

        lines = [
            f"Self-check: {self.people} people, {self.attempted} degraded copies, "
            f"threshold {self.threshold:.2f}",
            "",
            "By person:",
        ]
        for name, trials in sorted(self.by_person().items()):
            correct = sum(t.correct for t in trials)
            margins = [t.margin for t in trials if t.correct]
            worst = min(margins) if margins else 0.0
            flag = "  <-- weak" if correct < len(trials) else ""
            lines.append(f"  {name:<22} {correct}/{len(trials)} recognised, "
                         f"smallest margin {worst:+.3f}{flag}")

        lines += ["", "By kind of degradation:"]
        for kind, trials in self.by_degradation().items():
            correct = sum(t.correct for t in trials)
            lines.append(f"  {kind:<22} {correct}/{len(trials)}")

        lines += ["", f"Overall: {self.recall:.0%} of degraded copies recognised"]
        if self.undetected:
            lines.append(f"  {len(self.undetected)} copies had no detectable face at all "
                         "(usually the quarter-size ones - that is the resolution floor)")

        # A wrong name and no name are different failures with different fixes,
        # so they are never lumped together.
        for trial in self.confused:
            lines.append(f"  WRONG PERSON: {trial.person}'s {trial.degradation} copy was "
                         f"named {trial.matched} at {trial.score:.3f}")

        if self.missed:
            lines.append(f"  {len(self.missed)} copies were left Unknown rather than "
                         "named. Not a wrong answer - no answer:")
            for trial in self.missed[:8]:
                nearly = f"{trial.best_name} {trial.score:.3f}" if trial.best_name \
                    else f"{trial.score:.3f}"
                blocked_by = (f", too close to {trial.runner_up} "
                              f"{trial.runner_up_score:.3f}"
                              if trial.runner_up and trial.margin < recognition_margin_hint
                              else "")
                lines.append(f"      {trial.person}'s {trial.degradation} copy: "
                             f"nearly {nearly}{blocked_by}")
            if len(self.missed) > 8:
                lines.append(f"      ... and {len(self.missed) - 8} more")

        lines += ["", self.format_holdouts()]
        lines += ["", self.advice()]
        return "\n".join(lines)

    # ------------------------------------------------------ the stronger test
    @property
    def holdout_recall(self) -> float:
        if not self.holdouts:
            return 0.0
        return sum(h.correct for h in self.holdouts) / len(self.holdouts)

    def format_holdouts(self) -> str:
        """The part that answers whether recognition generalises."""
        if not self.holdouts:
            missing = ", ".join(self.single_photo[:6])
            more = f" and {len(self.single_photo) - 6} others" if len(self.single_photo) > 6 \
                else ""
            return (
                "Held-out test: not possible - everyone has only one photo"
                + (f" ({missing}{more})" if missing else "") + ".\n"
                "  The check above degrades the photo you enrolled, which is an easier\n"
                "  question than a different photo taken another day. Add a second photo\n"
                "  of somebody and this becomes a real measurement."
            )

        correct = sum(h.correct for h in self.holdouts)
        lines = [
            f"Held-out test: {correct}/{len(self.holdouts)} recognised from a gallery "
            "that did not contain that photo",
        ]
        scores = [h.score for h in self.holdouts if h.correct]
        if scores:
            lines.append(f"  same person, a different photo: "
                         f"{min(scores):.3f} to {max(scores):.3f}")
        strangers = [h.runner_up_score for h in self.holdouts if h.runner_up]
        if strangers:
            lines.append(f"  nearest other person:            "
                         f"{min(strangers):.3f} to {max(strangers):.3f}")
        for holdout in self.holdouts:
            if holdout.correct:
                continue
            what = "no face found" if not holdout.detected else f"came back {holdout.matched}"
            lines.append(f"  MISSED: {holdout.person}'s {holdout.photo} {what} "
                         f"({holdout.score:.3f})")
        if self.single_photo:
            lines.append(f"  ({len(self.single_photo)} people have only one photo and "
                         "could not be tested this way)")
        return "\n".join(lines)

    def advice(self) -> str:
        """The one sentence worth acting on.

        The held-out result outranks the degraded one wherever it exists,
        because it is the question that matters.
        """
        if self.holdouts and self.holdout_recall < 1.0:
            missed = [h.person for h in self.holdouts if not h.correct]
            return (f"A different photo of {', '.join(sorted(set(missed))[:3])} was not "
                    "recognised. That is the failure that matters - lower --threshold, or "
                    "enrol them from a photo closer to how they look on camera.")
        if self.confused:
            names = sorted({t.person for t in self.confused})
            return (f"Somebody was named as the wrong person ({', '.join(names[:3])}). "
                    "Raise --threshold and --margin, and use clearer photos for them.")
        if len(self.missed) > self.attempted * 0.25:
            ties = sorted({name for trial in self.missed
                           for name in (trial.best_name, trial.runner_up) if name})
            hint = (f" Most of it is {' and '.join(ties[:2])} being too alike to separate - "
                    "check whether that is the same person enrolled twice."
                    if ties else "")
            return ("A lot of copies were left Unknown rather than named." + hint +
                    " Lower --threshold, or give those people clearer photos.")
        if self.recall >= 0.9 and self.holdouts:
            worst = min((h.score for h in self.holdouts), default=0.0)
            return (f"This gallery separates cleanly, including on photos it had never "
                    f"seen (worst {worst:.2f}, threshold {self.threshold:.2f}). More "
                    "people makes the job harder, so re-run after adding more.")
        if self.recall >= 0.9:
            return ("Nothing here is broken, but every person has one photo, so this only "
                    "shows the pipeline works - not that these people are far apart. Add a "
                    "second photo of somebody, taken another day, for a real measurement.")
        if self.recall >= 0.7:
            return ("Workable, but the weak entries above would benefit from another "
                    "photo each - a different angle or day helps most.")
        return ("Recognition is unreliable on this gallery. Add clearer, larger "
                "reference photos, or lower --threshold and watch for wrong matches.")


def _photos_of(person: Person) -> List[str]:
    """The real files behind a person, without the mirrored duplicates."""
    return sorted({source.split(" (")[0] for source in person.sources})


def hold_out(gallery: Gallery, engine: FaceEngine, config: AppConfig) -> List[HoldOut]:
    """Identify each photo using a gallery built without it.

    For anyone with two or more photos, one is removed and the rest are used to
    recognise it. That is the real question - does this generalise to a
    different photo - and it needs nothing but what the user already has.

    A person with a single photo cannot be tested this way, because removing it
    leaves nothing to recognise them by. Those are reported, not skipped
    silently: knowing the test could not run matters as much as its result.
    """
    recognition = config.recognition
    results: List[HoldOut] = []

    for index, person in enumerate(gallery.people):
        photos = _photos_of(person)
        if len(photos) < 2:
            continue

        for held in photos:
            # Rebuild this person from everything except the held-out photo.
            keep = [
                vector for vector, source in zip(person.embeddings, person.sources)
                if source.split(" (")[0] != held
            ]
            if not keep:
                continue

            reduced = Gallery()
            reduced._set_people([
                Person(person.name, np.vstack(keep).astype(np.float32),
                       [s for s in person.sources if s.split(" (")[0] != held])
                if i == index else other
                for i, other in enumerate(gallery.people)
            ])

            image = imread_unicode(Path(held))
            if image is None:
                continue
            face = engine.embed_reference(image, min_face_size=24)
            if face is None or face.embedding is None:
                results.append(HoldOut(person.name, Path(held).name, matched="",
                                       score=0.0, detected=False))
                continue

            match = reduced.identify(face.embedding, recognition.threshold,
                                     recognition.margin, recognition.unknown_label)
            results.append(HoldOut(
                person=person.name, photo=Path(held).name, matched=match.name,
                score=match.score, runner_up=match.runner_up,
                runner_up_score=match.runner_up_score,
            ))

    return results


def run(config: Optional[AppConfig] = None, engine: Optional[FaceEngine] = None,
        gallery: Optional[Gallery] = None,
        degradations: Optional[List[Tuple[str, Callable]]] = None) -> Report:
    """Check a gallery two ways: degraded copies, and photos held out of it."""
    config = config or AppConfig()
    engine = engine or FaceEngine(config.face).load()
    if gallery is None:
        gallery = Gallery(config.gallery).build(engine)

    recognition = config.recognition
    report = Report(threshold=recognition.threshold, people=len(gallery))
    degradations = degradations if degradations is not None else DEGRADATIONS

    for person in gallery.people:
        # ``sources`` records mirrored copies too; only the real files are here.
        photos = _photos_of(person)
        if len(photos) < 2:
            report.single_photo.append(person.name)
        for photo in photos:
            image = imread_unicode(Path(photo))
            if image is None:
                continue
            for kind, degrade in degradations:
                try:
                    altered = degrade(image)
                except Exception as exc:          # a bad filter must not end the run
                    LOGGER.debug("Could not apply '%s': %s", kind, exc)
                    continue

                face = engine.embed_reference(altered, min_face_size=24)
                if face is None or face.embedding is None:
                    report.trials.append(Trial(
                        person.name, Path(photo).name, kind, matched="",
                        score=0.0, runner_up_score=0.0, detected=False,
                        unknown_label=recognition.unknown_label,
                    ))
                    continue

                match = gallery.identify(face.embedding, recognition.threshold,
                                         recognition.margin, recognition.unknown_label)
                report.trials.append(Trial(
                    person=person.name, photo=Path(photo).name, degradation=kind,
                    matched=match.name, score=match.score, best_name=match.best_name,
                    runner_up=match.runner_up, runner_up_score=match.runner_up_score,
                    unknown_label=recognition.unknown_label,
                ))

    report.holdouts = hold_out(gallery, engine, config)
    return report
