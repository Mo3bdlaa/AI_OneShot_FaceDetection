"""Driving the pipeline: preview window, video writing and the final report."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import cv2

from .config import AppConfig
from .pipeline import Pipeline
from .sources import VideoWriter, expand_sources
from .utils import LOGGER

WINDOW = "One-Shot Face Recognition"

HELP_KEYS = (
    "keys: [q]/[Esc] quit   [space] pause   [n] next source   "
    "[b] body boxes   [h] HUD   [s] snapshot"
)


def _display_available() -> bool:
    """True when OpenCV can actually open a window here."""
    try:
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.destroyWindow(WINDOW)
        return True
    except cv2.error:
        return False


def _writer_path(save: Path, spec: str, multiple: bool) -> Path:
    """Where the annotated video for one source goes."""
    save = Path(save)
    treat_as_dir = multiple or save.is_dir() or save.suffix == ""
    if not treat_as_dir:
        return save

    if spec.isdigit():
        stem = f"camera{spec}"
    elif "://" in spec:
        stem = "stream"
    else:
        stem = Path(spec).stem or "output"
    return save / f"{stem}_annotated.mp4"


def list_people(config: AppConfig) -> int:
    """Build the gallery, print it and exit - a quick sanity check."""
    pipeline = Pipeline(config)
    pipeline.engine.load()
    pipeline.gallery.build(pipeline.engine)
    print(pipeline.gallery.summary())
    return 0


def self_check(config: AppConfig) -> int:
    """Measure recognition on the user's own gallery and print the report."""
    from . import selfcheck

    pipeline = Pipeline(config)
    pipeline.engine.load()
    pipeline.gallery.build(pipeline.engine)

    LOGGER.info("Degrading %d reference photo set(s) - this takes a moment.",
                len(pipeline.gallery))
    report = selfcheck.run(config, pipeline.engine, pipeline.gallery)
    print()
    print(report.format())
    # A wrong *name* is a failure worth a non-zero exit, so this can gate a
    # script. A missed match is a gap, not a mistake, and does not.
    return 1 if report.confused else 0


def run_app(config: AppConfig) -> int:
    """Run every configured source with preview, saving and reporting."""
    runtime = config.runtime
    specs = expand_sources(runtime.sources)
    if not specs:
        LOGGER.error("No usable source. Try --source 0 for a webcam.")
        return 1

    pipeline = Pipeline(config)
    pipeline.prepare()

    show = runtime.display and _display_available()
    if runtime.display and not show:
        LOGGER.warning("No display available; continuing headless. Use --save to keep the output.")
    if show:
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        LOGGER.info(HELP_KEYS)

    quit_all = False
    frames_total = 0
    started = time.time()

    try:
        for position, spec in enumerate(specs, start=1):
            if quit_all:
                break
            if len(specs) > 1:
                LOGGER.info("[%d/%d] %s", position, len(specs), spec)

            writer: Optional[VideoWriter] = None
            paused = False

            try:
                for result in pipeline.run_source(spec):
                    frames_total += 1

                    if runtime.save and writer is None:
                        target = _writer_path(Path(runtime.save), spec, len(specs) > 1)
                        # The saved file must play back at the speed of the
                        # source, not at whatever rate we managed to process it.
                        info = pipeline.source_info
                        writer = VideoWriter(target, fps=(info.fps if info else 0.0) or 25.0)
                    if writer is not None:
                        writer.write(result.frame)

                    if not show:
                        continue

                    cv2.imshow(WINDOW, result.frame)
                    key = cv2.waitKey(1) & 0xFF
                    action = _handle_key(key, pipeline, result.frame, spec)
                    if action == "quit":
                        quit_all = True
                        break
                    if action == "next":
                        break
                    if action == "pause":
                        paused = True

                    while paused and show:
                        key = cv2.waitKey(50) & 0xFF
                        action = _handle_key(key, pipeline, result.frame, spec)
                        if action == "pause":
                            paused = False
                        elif action == "quit":
                            quit_all = True
                            paused = False
                            break
                        elif action == "next":
                            paused = False
                            break
                    if quit_all:
                        break

            except (RuntimeError, FileNotFoundError, ValueError) as exc:
                LOGGER.error("Skipping '%s': %s", spec, exc)
            finally:
                if writer is not None:
                    writer.close()

    except KeyboardInterrupt:
        LOGGER.info("Interrupted.")
    finally:
        if show:
            cv2.destroyAllWindows()
        pipeline.close()

    elapsed = max(1e-6, time.time() - started)
    LOGGER.info("Processed %d frames in %.1fs (%.1f FPS average).",
                frames_total, elapsed, frames_total / elapsed)
    print()
    print(pipeline.summary())
    return 0


def _handle_key(key: int, pipeline: Pipeline, frame, spec: str) -> Optional[str]:
    """Interpret one keypress from the preview window."""
    if key in (ord("q"), 27):
        return "quit"
    if key == ord("n"):
        return "next"
    if key == ord(" "):
        return "pause"
    if key == ord("b"):
        pipeline.renderer.config.show_body = not pipeline.renderer.config.show_body
        LOGGER.info("Body highlighting %s",
                    "on" if pipeline.renderer.config.show_body else "off")
    elif key == ord("h"):
        config = pipeline.renderer.config
        new_state = not config.show_fps
        config.show_fps = new_state
        config.show_roster = new_state
    elif key == ord("s"):
        stem = "camera" if spec.isdigit() else Path(spec).stem
        target = Path("outputs") / f"snapshot_{stem}_{int(time.time())}.jpg"
        target.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(target), frame)
        LOGGER.info("Saved snapshot to %s", target)
    return None
