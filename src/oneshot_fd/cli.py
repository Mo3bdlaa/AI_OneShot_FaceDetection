"""Command line interface.

    python -m oneshot_fd --faces input_faces --source 0
    python -m oneshot_fd --faces input_faces --source clips/ --save outputs/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from . import __version__
from .config import (AppConfig, BodyConfig, DrawConfig, FaceConfig, GalleryConfig,
                     RecognitionConfig, RuntimeConfig, TrackingConfig)
from .utils import LOGGER, setup_logging

EPILOG = """\
examples:
  # live webcam, highlighting everyone in input_faces/
  python -m oneshot_fd --faces input_faces --source 0

  # one video, saved with the overlay burned in
  python -m oneshot_fd --source party.mp4 --save outputs/party.mp4 --no-display

  # every video in a folder, plus a CSV of who appeared when
  python -m oneshot_fd --source clips/ --save outputs/ --log-csv outputs/log.csv

  # an RTSP camera, staying in the present instead of buffering
  python -m oneshot_fd --source rtsp://user:pass@192.168.1.10/stream --realtime

  # the browser UI, at http://localhost:8000
  python -m oneshot_fd --serve

  # ... reachable from your phone, using the phone's own camera
  python -m oneshot_fd --serve --host 0.0.0.0 --https

  # just check the gallery loads
  python -m oneshot_fd --faces input_faces --list-people

  # ... and check your people can actually be told apart
  python -m oneshot_fd --faces input_faces --self-check
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="oneshot_fd",
        description="Recognise people from a folder of photos in videos, "
                    "streams and live camera - one photo per person is enough.",
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"oneshot_fd {__version__}")

    io_group = parser.add_argument_group("input / output")
    io_group.add_argument(
        "-f", "--faces", default="input_faces", metavar="DIR",
        help="folder of reference photos: Name.jpg, or Name/ with several photos "
             "(default: input_faces)",
    )
    io_group.add_argument(
        "-s", "--source", action="append", default=None, metavar="SRC",
        help="camera index (0), video file, folder of videos, glob or stream URL. "
             "Repeat to process several (default: 0)",
    )
    io_group.add_argument(
        "-o", "--save", metavar="PATH",
        help="write the annotated video here; a directory when there are several sources",
    )
    io_group.add_argument("--log-csv", metavar="FILE",
                          help="append who was seen, when and for how long to this CSV")
    io_group.add_argument("--no-display", dest="display", action="store_false",
                          help="do not open a preview window (for servers and batch runs)")
    io_group.add_argument("--list-people", action="store_true",
                          help="build the gallery, print who is in it and exit")
    io_group.add_argument("--self-check", action="store_true",
                          help="degrade every reference photo the way video degrades a "
                               "face, put it back through recognition, and report whether "
                               "your people are actually told apart")

    web_group = parser.add_argument_group("web interface")
    web_group.add_argument("--serve", action="store_true",
                           help="run the browser UI and REST API instead of the "
                                "desktop window")
    web_group.add_argument("--host", default="127.0.0.1",
                           help="address to serve on; 0.0.0.0 makes it reachable from "
                                "other devices on your network (default: 127.0.0.1)")
    web_group.add_argument("--port", type=int, default=8000,
                           help="port for the web UI (default: 8000)")
    web_group.add_argument("--token", default=None, metavar="SECRET",
                           help="require this token on every request. One is generated "
                                "when serving on a reachable address and none is given")
    web_group.add_argument("--https", action="store_true",
                           help="serve over HTTPS with a self-signed certificate. "
                                "Browsers only share a camera on a secure origin, so "
                                "this is what lets a phone use its own camera")
    web_group.add_argument("--no-token", dest="require_token", action="store_false",
                           help="serve with no token even on a reachable address; only "
                                "sensible behind something else that authenticates")

    rec_group = parser.add_argument_group("recognition")
    rec_group.add_argument("-t", "--threshold", type=float, default=0.38, metavar="F",
                           help="cosine similarity needed to claim a face (default: 0.38)")
    rec_group.add_argument("--margin", type=float, default=0.03, metavar="F",
                           help="how far the best match must beat the runner-up (default: 0.03)")
    rec_group.add_argument("--vote-window", type=int, default=12, metavar="N",
                           help="frames each track votes over before committing (default: 12)")
    rec_group.add_argument("--reverify-every", type=int, default=0, metavar="N",
                           help="once a track has settled on a name, re-run the expensive "
                                "embedding only every N detections; 3-5 is a big speed win "
                                "on live video (default: 0, verify every face)")
    rec_group.add_argument("--unknown-label", default="Unknown",
                           help="label for faces that match nobody (default: Unknown)")
    rec_group.add_argument("--min-quality", type=float, default=0.0, metavar="F",
                           help="leave faces below this quality score (0..1) as Unknown "
                                "instead of risking a confident mislabel; try 0.45")
    rec_group.add_argument("--reject-below", type=float, default=0.0, metavar="F",
                           help="refuse reference photos below this quality score "
                                "instead of enrolling them with a warning")
    rec_group.add_argument("--no-quality-check", dest="quality_check", action="store_false",
                           help="do not inspect reference photos at all")
    rec_group.add_argument("--rebuild-gallery", action="store_true",
                           help="ignore the cached embeddings and re-enrol every photo")
    rec_group.add_argument("--no-flip-augment", dest="flip_augment", action="store_false",
                           help="do not also embed a mirrored copy of each reference photo")

    model_group = parser.add_argument_group("models / speed")
    model_group.add_argument("--model", default="buffalo_l",
                             help="InsightFace pack: buffalo_l (accurate) or buffalo_s (fast)")
    model_group.add_argument("--det-size", type=int, default=640, metavar="N",
                             help="detector input size; lower is faster (default: 640)")
    model_group.add_argument("--det-threshold", type=float, default=0.5, metavar="F",
                             help="minimum face detector confidence (default: 0.5)")
    model_group.add_argument("--attributes", action="store_true",
                             help="also estimate age and gender and show them beside the "
                                  "name (loads an extra model, roughly doubles per-face cost)")
    model_group.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"],
                             help="where to run the models (default: auto)")
    model_group.add_argument("--model-root", metavar="DIR",
                             help="directory to keep the downloaded model packs in")
    model_group.add_argument("--detect-every", type=int, default=1, metavar="N",
                             help="run the detector every N frames and track in between "
                                  "(default: 1)")
    model_group.add_argument("--max-width", type=int, default=None, metavar="N",
                             help="downscale frames to this width before processing")
    model_group.add_argument("--max-frames", type=int, default=None, metavar="N",
                             help="stop after N frames per source")
    model_group.add_argument("--realtime", action="store_true",
                             help="on a camera or stream, always grab the newest frame "
                                  "instead of queuing up")
    model_group.add_argument("--no-tracking", dest="tracking", action="store_false",
                             help="label every frame independently (flickers more)")

    body_group = parser.add_argument_group("body highlighting")
    body_group.add_argument("--body", default="auto",
                            choices=["auto", "yolo", "estimate", "off"],
                            help="auto uses YOLO when installed, else a geometric estimate")
    body_group.add_argument("--yolo-model", default="yolov8n.pt",
                            help="ultralytics weights for the YOLO backend")
    body_group.add_argument("--reid", action="store_true",
                            help="keep a person labelled from their clothing once their "
                                 "face is no longer visible (requires --body yolo)")
    body_group.add_argument("--reid-threshold", type=float, default=0.72, metavar="F",
                            help="appearance similarity a face-less body needs (default: 0.72)")
    body_group.add_argument("--reid-memory", type=int, default=150, metavar="N",
                            help="frames an appearance stays usable after its last face "
                                 "confirmation (default: 150)")
    body_group.add_argument("--body-conf", type=float, default=0.35, metavar="F",
                            help="minimum confidence for a YOLO person box (default: 0.35)")

    draw_group = parser.add_argument_group("overlay")
    draw_group.add_argument("--no-face-box", dest="face_box", action="store_false",
                            help="hide the face rectangle")
    draw_group.add_argument("--no-body-box", dest="body_box", action="store_false",
                            help="hide the body brackets")
    draw_group.add_argument("--landmarks", action="store_true",
                            help="also draw the five facial keypoints")
    draw_group.add_argument("--blur-unknown", action="store_true",
                            help="pixelate the faces of people who are not in the gallery")
    draw_group.add_argument("--no-hud", dest="hud", action="store_false",
                            help="hide the FPS and on-screen roster panel")
    draw_group.add_argument("--mirror", action="store_true",
                            help="mirror the frame, which feels natural for a webcam")
    draw_group.add_argument("--thickness", type=int, default=2, metavar="N",
                            help="box line thickness (default: 2)")

    log_group = parser.add_argument_group("logging")
    log_group.add_argument("-q", "--quiet", action="store_true", help="warnings and errors only")
    log_group.add_argument("-v", "--verbose", action="store_true", help="extra diagnostics")

    return parser


def config_from_args(args: argparse.Namespace) -> AppConfig:
    """Translate parsed arguments into the dataclass configuration."""
    sources: List[str] = args.source if args.source else ["0"]

    return AppConfig(
        gallery=GalleryConfig(
            path=Path(args.faces),
            use_flip_augmentation=args.flip_augment,
            force_rebuild=args.rebuild_gallery,
            check_quality=args.quality_check,
            reject_below=args.reject_below,
        ),
        face=FaceConfig(
            model_name=args.model,
            det_size=args.det_size,
            det_threshold=args.det_threshold,
            attributes=args.attributes,
            device=args.device,
            model_root=Path(args.model_root) if args.model_root else None,
        ),
        recognition=RecognitionConfig(
            threshold=args.threshold,
            margin=args.margin,
            vote_window=max(1, args.vote_window),
            reverify_every=max(0, args.reverify_every),
            min_quality=args.min_quality,
            unknown_label=args.unknown_label,
        ),
        tracking=TrackingConfig(enabled=args.tracking),
        body=BodyConfig(mode=args.body, yolo_model=args.yolo_model, conf=args.body_conf,
                        reid=args.reid, reid_threshold=args.reid_threshold,
                        reid_memory=args.reid_memory),
        draw=DrawConfig(
            show_face=args.face_box,
            show_body=args.body_box,
            show_landmarks=args.landmarks,
            show_attributes=args.attributes,
            show_fps=args.hud,
            show_roster=args.hud,
            box_thickness=max(1, args.thickness),
            blur_unknown=args.blur_unknown,
        ),
        runtime=RuntimeConfig(
            sources=sources,
            display=args.display,
            save=Path(args.save) if args.save else None,
            log_csv=Path(args.log_csv) if args.log_csv else None,
            detect_every=max(1, args.detect_every),
            max_width=args.max_width,
            realtime=args.realtime,
            max_frames=args.max_frames,
            mirror=args.mirror,
            quiet=args.quiet,
        ),
    )


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(quiet=args.quiet, verbose=args.verbose)
    config = config_from_args(args)

    from .runner import list_people, run_app

    try:
        if args.list_people:
            return list_people(config)
        if args.self_check:
            from .runner import self_check

            return self_check(config)
        if args.serve:
            from .web import serve

            serve(config, host=args.host, port=args.port,
                  token=args.token, require_token=args.require_token,
                  https=args.https)
            return 0
        return run_app(config)
    except KeyboardInterrupt:
        LOGGER.info("Stopped.")
        return 130
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        LOGGER.error("%s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
