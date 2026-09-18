"""Building the reference gallery from a folder of face photos.

Layout of the input folder - both styles work, and they can be mixed:

    input_faces/
        Mohammed.jpg            <- one photo, one person: this is enough
        Sara.png
        Ahmed/                  <- a folder per person for extra photos
            front.jpg
            side.jpg

The folder (or file stem) name becomes the label shown on screen.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

from .config import GalleryConfig
from .faces import FaceEngine
from .utils import IMAGE_SUFFIXES, LOGGER, imread_unicode, iter_images, l2_normalize

CACHE_VERSION = 2


@dataclass
class Person:
    """One enrolled identity and every embedding that represents them."""

    name: str
    embeddings: np.ndarray      # (n, 512), L2-normalised
    sources: List[str]          # the files each embedding came from

    @property
    def centroid(self) -> np.ndarray:
        """Mean embedding - a stable summary when several photos exist."""
        return l2_normalize(self.embeddings.mean(axis=0))


@dataclass
class Match:
    """The outcome of comparing one face embedding against the gallery."""

    name: str
    score: float
    runner_up: str = ""
    runner_up_score: float = 0.0

    @property
    def margin(self) -> float:
        return self.score - self.runner_up_score


class Gallery:
    """Holds every enrolled person and answers "who is this face?"."""

    def __init__(self, config: Optional[GalleryConfig] = None) -> None:
        self.config = config or GalleryConfig()
        self.people: List[Person] = []
        self._matrix: Optional[np.ndarray] = None   # (total_embeddings, 512)
        self._owners: List[int] = []                # row -> index into self.people

    # ---------------------------------------------------------------- basics
    def __len__(self) -> int:
        return len(self.people)

    @property
    def names(self) -> List[str]:
        return [person.name for person in self.people]

    @property
    def is_empty(self) -> bool:
        return not self.people

    # ----------------------------------------------------------------- build
    def build(self, engine: FaceEngine) -> "Gallery":
        """Load from cache when possible, otherwise embed the input folder."""
        folder = Path(self.config.path).expanduser()
        if not folder.exists():
            raise FileNotFoundError(
                f"Input folder '{folder}' does not exist. Create it and drop in one "
                f"photo per person (for example {folder}/Mohammed.jpg)."
            )

        cache_path = self._cache_path(folder)
        if self.config.cache and not self.config.force_rebuild:
            if self._load_cache(cache_path, folder):
                LOGGER.info(
                    "Loaded %d known %s from cache (%s)",
                    len(self.people),
                    "person" if len(self.people) == 1 else "people",
                    cache_path.name,
                )
                return self

        self._build_from_folder(folder, engine)
        if self.config.cache:
            self._save_cache(cache_path, folder)
        return self

    def _build_from_folder(self, folder: Path, engine: FaceEngine) -> None:
        grouped = self._group_images(folder)
        if not grouped:
            raise ValueError(
                f"No usable images found in '{folder}'. Supported types: "
                + ", ".join(sorted(IMAGE_SUFFIXES))
            )

        LOGGER.info("Enrolling %d %s from %s",
                    len(grouped), "person" if len(grouped) == 1 else "people", folder)

        people: List[Person] = []
        skipped: List[str] = []
        started = time.time()

        for name, paths in grouped.items():
            vectors: List[np.ndarray] = []
            sources: List[str] = []
            for path in paths:
                image = imread_unicode(path)
                if image is None:
                    skipped.append(f"{path.name} (unreadable)")
                    continue

                face = engine.embed_reference(image, self.config.min_face_size)
                if face is None or face.embedding is None:
                    skipped.append(f"{path.name} (no clear face)")
                    continue

                vectors.append(face.embedding)
                sources.append(str(path))

                # A mirrored copy costs one extra forward pass and makes a
                # single reference photo far more forgiving about head pose.
                if self.config.use_flip_augmentation:
                    flipped = engine.embed_reference(cv2.flip(image, 1), self.config.min_face_size)
                    if flipped is not None and flipped.embedding is not None:
                        vectors.append(flipped.embedding)
                        sources.append(f"{path} (mirrored)")

            if not vectors:
                LOGGER.warning("Skipping '%s': no face could be embedded.", name)
                continue

            people.append(
                Person(name=name, embeddings=np.vstack(vectors).astype(np.float32), sources=sources)
            )
            LOGGER.info("  %-24s %d embedding(s) from %d photo(s)",
                        name, len(vectors), len(paths))

        if not people:
            raise ValueError(
                f"Could not enrol anybody from '{folder}'. Make sure each photo shows "
                "one clear, reasonably large face."
            )

        for note in skipped:
            LOGGER.debug("  skipped %s", note)

        self._set_people(people)
        LOGGER.info("Gallery ready: %d people in %.1fs", len(people), time.time() - started)

    @staticmethod
    def _group_images(folder: Path) -> Dict[str, List[Path]]:
        """Map person name -> reference photos, for both folder layouts."""
        grouped: Dict[str, List[Path]] = {}

        for entry in sorted(folder.iterdir()):
            if entry.is_dir():
                images = list(iter_images(entry))
                # Also pick up photos nested one level deeper.
                for sub in sorted(entry.iterdir()):
                    if sub.is_dir():
                        images.extend(iter_images(sub))
                if images:
                    grouped.setdefault(entry.name, []).extend(images)
            elif entry.is_file() and entry.suffix.lower() in IMAGE_SUFFIXES:
                grouped.setdefault(entry.stem, []).append(entry)

        return {name: paths for name, paths in grouped.items() if paths}

    def _set_people(self, people: List[Person]) -> None:
        self.people = people
        matrix: List[np.ndarray] = []
        owners: List[int] = []
        for index, person in enumerate(people):
            for vector in person.embeddings:
                matrix.append(vector)
                owners.append(index)
            # The centroid competes alongside the individual photos: it is more
            # robust once a person has several references, and harmless with one.
            matrix.append(person.centroid)
            owners.append(index)
        self._matrix = np.vstack(matrix).astype(np.float32)
        self._owners = owners

    # ----------------------------------------------------------------- query
    def identify(self, embedding: np.ndarray, threshold: float, margin: float = 0.0,
                 unknown_label: str = "Unknown") -> Match:
        """Return the best matching identity for one face embedding.

        Scores are cosine similarities in ``[-1, 1]``. A face is only claimed
        when it clears ``threshold`` *and* beats the second-best person by
        ``margin``, which stops similar-looking people from trading places.
        """
        if self._matrix is None or embedding is None:
            return Match(unknown_label, 0.0)

        vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
        if vector.shape[0] != self._matrix.shape[1]:
            return Match(unknown_label, 0.0)

        scores = self._matrix @ vector

        # Best score per person, not per photo.
        per_person = np.full(len(self.people), -1.0, dtype=np.float32)
        for row, owner in enumerate(self._owners):
            if scores[row] > per_person[owner]:
                per_person[owner] = scores[row]

        order = np.argsort(-per_person)
        best_index = int(order[0])
        best_score = float(per_person[best_index])

        runner_up, runner_up_score = "", 0.0
        if len(order) > 1:
            second = int(order[1])
            runner_up = self.people[second].name
            runner_up_score = float(per_person[second])

        if best_score < threshold or (best_score - runner_up_score) < margin:
            return Match(unknown_label, best_score, runner_up, runner_up_score)

        return Match(self.people[best_index].name, best_score, runner_up, runner_up_score)

    def identify_batch(self, embeddings: List[Optional[np.ndarray]], threshold: float,
                       margin: float = 0.0, unknown_label: str = "Unknown") -> List[Match]:
        return [
            self.identify(embedding, threshold, margin, unknown_label)
            if embedding is not None else Match(unknown_label, 0.0)
            for embedding in embeddings
        ]

    # ----------------------------------------------------------------- cache
    def _cache_path(self, folder: Path) -> Path:
        return folder.parent / f".{folder.name}.gallery.npz"

    def _fingerprint(self, folder: Path) -> str:
        """Cheap signature of the folder contents: name, size and mtime."""
        parts = []
        for path in sorted(folder.rglob("*")):
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
                stat = path.stat()
                parts.append(f"{path.relative_to(folder)}:{stat.st_size}:{int(stat.st_mtime)}")
        signature = {
            "version": CACHE_VERSION,
            "flip": self.config.use_flip_augmentation,
            "min_face": self.config.min_face_size,
            "files": parts,
        }
        return json.dumps(signature, sort_keys=True)

    def _save_cache(self, cache_path: Path, folder: Path) -> None:
        try:
            np.savez_compressed(
                cache_path,
                fingerprint=np.array(self._fingerprint(folder)),
                names=np.array([p.name for p in self.people], dtype=object),
                counts=np.array([len(p.embeddings) for p in self.people], dtype=np.int32),
                embeddings=np.vstack([p.embeddings for p in self.people]).astype(np.float32),
                sources=np.array([json.dumps(p.sources) for p in self.people], dtype=object),
            )
            LOGGER.debug("Wrote gallery cache to %s", cache_path)
        except Exception as exc:  # cache failures must never break a run
            LOGGER.debug("Could not write gallery cache: %s", exc)

    def _load_cache(self, cache_path: Path, folder: Path) -> bool:
        if not cache_path.exists():
            return False
        try:
            data = np.load(cache_path, allow_pickle=True)
            if str(data["fingerprint"]) != self._fingerprint(folder):
                LOGGER.debug("Gallery cache is stale; rebuilding.")
                return False

            names = [str(n) for n in data["names"]]
            counts = [int(c) for c in data["counts"]]
            embeddings = data["embeddings"].astype(np.float32)
            sources = [json.loads(str(s)) for s in data["sources"]]

            people: List[Person] = []
            offset = 0
            for name, count, source in zip(names, counts, sources):
                people.append(Person(name, embeddings[offset:offset + count], source))
                offset += count
            if not people:
                return False
            self._set_people(people)
            return True
        except Exception as exc:
            LOGGER.debug("Could not read gallery cache (%s); rebuilding.", exc)
            return False

    # ------------------------------------------------------------------ misc
    def summary(self) -> str:
        lines = [f"Gallery: {len(self.people)} known people"]
        for person in self.people:
            lines.append(f"  - {person.name} ({len(person.embeddings)} embeddings)")
        return "\n".join(lines)
