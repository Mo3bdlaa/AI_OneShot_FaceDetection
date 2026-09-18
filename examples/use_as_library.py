#!/usr/bin/env python3
"""Using the recogniser from your own Python code.

    python examples/use_as_library.py input_faces meeting.mp4

Three ways to drive it, from the simplest to the most hands-on.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import cv2  # noqa: E402

from oneshot_fd.config import AppConfig, GalleryConfig, RuntimeConfig  # noqa: E402
from oneshot_fd.pipeline import Pipeline  # noqa: E402
from oneshot_fd.utils import setup_logging  # noqa: E402


def who_is_in_this_video(faces_dir: str, video: str) -> None:
    """1. Run a whole source and print every recognition as it happens."""
    config = AppConfig(
        gallery=GalleryConfig(path=Path(faces_dir)),
        runtime=RuntimeConfig(sources=[video], display=False),
    )
    with Pipeline(config) as pipeline:
        for result in pipeline.run():
            for track in result.tracks:
                if track.label != "Unknown":
                    print(f"{result.timestamp:7.2f}s  {track.label:<16}"
                          f"score={track.label_score:.2f}  "
                          f"face={tuple(track.box)}  body={tuple(track.body_box or ())}")
        print()
        print(pipeline.summary())


def my_own_capture_loop(faces_dir: str, video: str) -> None:
    """2. Keep your own capture loop and hand single frames to the pipeline."""
    config = AppConfig(gallery=GalleryConfig(path=Path(faces_dir)),
                       runtime=RuntimeConfig(display=False))
    pipeline = Pipeline(config).prepare()

    capture = cv2.VideoCapture(video)
    index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        result = pipeline.process_frame(frame, index, index / 25.0)
        # result.frame is the annotated frame; do whatever you like with it.
        if result.names != ["Unknown"] and result.names:
            print(index, result.names)
        index += 1
    capture.release()
    pipeline.close()


def just_the_matching(faces_dir: str, image_path: str) -> None:
    """3. Only the pieces you need: detect a face, ask the gallery who it is."""
    from oneshot_fd.faces import FaceEngine
    from oneshot_fd.gallery import Gallery

    engine = FaceEngine().load()
    gallery = Gallery(GalleryConfig(path=Path(faces_dir))).build(engine)

    image = cv2.imread(image_path)
    for face in engine.detect(image):
        match = gallery.identify(face.embedding, threshold=0.38, margin=0.03)
        print(f"{match.name:<16} score={match.score:.3f}  "
              f"(runner-up {match.runner_up or '-'} {match.runner_up_score:.3f})")


if __name__ == "__main__":
    setup_logging()
    if len(sys.argv) < 3:
        print(__doc__)
        raise SystemExit(1)
    who_is_in_this_video(sys.argv[1], sys.argv[2])
