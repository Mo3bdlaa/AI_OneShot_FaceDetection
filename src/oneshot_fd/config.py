"""Central configuration for the recognition pipeline.

Every knob the CLI exposes ends up here, so the library can be driven from
Python with the exact same behaviour as the command line.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence


@dataclass
class GalleryConfig:
    """How the reference faces in the input folder are turned into embeddings."""

    #: Folder holding the reference faces (``Name.jpg`` or ``Name/*.jpg``).
    path: Path = Path("input_faces")
    #: Also embed the horizontally mirrored copy of every reference image.
    #: Cheap and noticeably more robust when a person only has one photo.
    use_flip_augmentation: bool = True
    #: Reference faces smaller than this (pixels, longest side) are skipped.
    min_face_size: int = 40
    #: Cache the computed embeddings next to the folder to avoid recomputing.
    cache: bool = True
    #: Recompute the cache even when it looks up to date.
    force_rebuild: bool = False
    #: Report what is wrong with weak reference photos instead of silently
    #: enrolling them.
    check_quality: bool = True
    #: Refuse a reference photo whose quality score falls below this (0..1).
    #: 0 enrols everything and only warns.
    reject_below: float = 0.0


@dataclass
class FaceConfig:
    """Face detection and embedding settings."""

    #: InsightFace model pack. ``buffalo_l`` is the accurate default,
    #: ``buffalo_s`` is a lighter/faster alternative.
    model_name: str = "buffalo_l"
    #: Detector input resolution. Larger finds smaller faces but costs time.
    det_size: int = 640
    #: Minimum detector score for a face to be considered at all.
    det_threshold: float = 0.5
    #: ``cpu``, ``cuda`` or ``auto``.
    device: str = "auto"
    #: Also estimate age and gender for every face and show them on screen.
    #: This loads an extra model and roughly doubles the per-face cost, so it
    #: is off unless you ask for it. Gender was right on all four test photos;
    #: age was not - the same man measured 33, 33, 45 and 46 - so the age is a
    #: decade-wide hint, not a number.
    attributes: bool = False
    #: Directory used to store the downloaded model packs.
    model_root: Optional[Path] = None


@dataclass
class RecognitionConfig:
    """Turning face embeddings into names."""

    #: Cosine similarity a face must reach to be claimed by a known person.
    threshold: float = 0.38
    #: The best match must beat the runner-up by this margin, otherwise the
    #: face stays ``Unknown``. Keeps look-alikes from swapping identities.
    margin: float = 0.03
    #: Number of recent frames a track votes over before it commits to a name.
    vote_window: int = 12
    #: Fraction of the votes the winner needs to own the track.
    vote_ratio: float = 0.5
    #: Faces in the video below this quality score (0..1) are left Unknown
    #: rather than risk a confident mislabel. 0 disables the check.
    min_quality: float = 0.0
    #: Label used when nobody in the gallery matches.
    unknown_label: str = "Unknown"
    #: Once a track has settled on a name, re-run the (expensive) embedding
    #: only every N detections. 0 verifies every face on every detection,
    #: which is the safe default; 3-5 is a large speed win on live video.
    reverify_every: int = 0


@dataclass
class TrackingConfig:
    """Frame-to-frame association of faces."""

    enabled: bool = True
    #: IoU above which a detection continues an existing track.
    iou_threshold: float = 0.3
    #: Fallback association radius, in multiples of the face width, used when
    #: fast motion (or --detect-every N) leaves no overlap to match on.
    search_radius: float = 1.5
    #: How many frames a track survives without a detection.
    max_age: int = 30
    #: Detections needed before a track is drawn (filters one-frame noise).
    min_hits: int = 2


@dataclass
class BodyConfig:
    """Body / person highlighting."""

    #: ``auto`` uses YOLO when ultralytics is installed and falls back to the
    #: geometric estimator. ``yolo``, ``estimate`` and ``off`` force a mode.
    mode: str = "auto"
    #: Ultralytics weights used when the YOLO backend is active.
    yolo_model: str = "yolov8n.pt"
    #: Minimum confidence for a person box.
    conf: float = 0.35
    #: Keep a person labelled from their clothing once their face is no longer
    #: visible. Needs real person boxes, so it only works with the YOLO backend.
    reid: bool = False
    #: Appearance similarity a face-less body must reach to keep a name.
    #: Above the measured 0.81 ceiling for two different people.
    reid_threshold: float = 0.85
    #: ... and how far it must beat the runner-up.
    reid_margin: float = 0.05
    #: Frames an appearance stays usable after its last face confirmation.
    #: Short on purpose: the descriptor only holds up while the light does.
    reid_memory: int = 50
    #: Body boxes are estimated as this many face heights tall.
    estimate_height_factor: float = 7.5
    #: ... and this many face widths wide.
    estimate_width_factor: float = 3.2


@dataclass
class DrawConfig:
    """Overlay appearance."""

    show_face: bool = True
    show_body: bool = True
    show_landmarks: bool = False
    show_score: bool = True
    #: Show the estimated age and gender beside the name. Needs FaceConfig.attributes.
    show_attributes: bool = False
    show_fps: bool = True
    show_roster: bool = True
    box_thickness: int = 2
    font_scale: float = 0.55
    #: Blur the faces of people who are not in the gallery.
    blur_unknown: bool = False


@dataclass
class RuntimeConfig:
    """Everything about where frames come from and where they go."""

    sources: Sequence[str] = field(default_factory=list)
    #: Show a live preview window.
    display: bool = True
    #: Write the annotated video here (file, or directory for many sources).
    save: Optional[Path] = None
    #: Append recognition events (who/when/where) to this CSV.
    log_csv: Optional[Path] = None
    #: Run the heavy detector every N frames and track in between.
    detect_every: int = 1
    #: Downscale frames so the longest side is at most this many pixels.
    max_width: Optional[int] = None
    #: Drop stale frames on live sources so the overlay stays in the present.
    realtime: bool = False
    #: Stop after this many frames (per source). ``None`` means run to the end.
    max_frames: Optional[int] = None
    #: Mirror the frame, which feels natural for a webcam preview.
    mirror: bool = False
    quiet: bool = False


@dataclass
class AppConfig:
    """The full configuration handed to :class:`~oneshot_fd.pipeline.Pipeline`."""

    gallery: GalleryConfig = field(default_factory=GalleryConfig)
    face: FaceConfig = field(default_factory=FaceConfig)
    recognition: RecognitionConfig = field(default_factory=RecognitionConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    body: BodyConfig = field(default_factory=BodyConfig)
    draw: DrawConfig = field(default_factory=DrawConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
