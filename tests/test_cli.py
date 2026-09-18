"""CLI argument parsing -> configuration."""

from pathlib import Path

import pytest

from oneshot_fd.cli import build_parser, config_from_args


def parse(*argv):
    return config_from_args(build_parser().parse_args(list(argv)))


def test_defaults_are_a_working_webcam_setup():
    config = parse()
    assert list(config.runtime.sources) == ["0"]
    assert config.gallery.path == Path("input_faces")
    assert config.runtime.display is True
    assert config.body.mode == "auto"
    assert config.tracking.enabled is True


def test_several_sources_are_collected():
    config = parse("-s", "0", "-s", "clip.mp4", "-s", "rtsp://cam")
    assert list(config.runtime.sources) == ["0", "clip.mp4", "rtsp://cam"]


def test_recognition_flags():
    config = parse("-t", "0.5", "--margin", "0.1", "--vote-window", "30",
                   "--unknown-label", "Guest")
    assert config.recognition.threshold == 0.5
    assert config.recognition.margin == 0.1
    assert config.recognition.vote_window == 30
    assert config.recognition.unknown_label == "Guest"


def test_output_flags():
    config = parse("--save", "out/x.mp4", "--log-csv", "out/log.csv", "--no-display")
    assert config.runtime.save == Path("out/x.mp4")
    assert config.runtime.log_csv == Path("out/log.csv")
    assert config.runtime.display is False


def test_speed_flags():
    config = parse("--det-size", "320", "--detect-every", "4", "--max-width", "960",
                   "--realtime", "--no-tracking")
    assert config.face.det_size == 320
    assert config.runtime.detect_every == 4
    assert config.runtime.max_width == 960
    assert config.runtime.realtime is True
    assert config.tracking.enabled is False


def test_nonsense_values_are_clamped_not_crashed():
    config = parse("--detect-every", "0", "--vote-window", "0", "--thickness", "-3")
    assert config.runtime.detect_every == 1
    assert config.recognition.vote_window == 1
    assert config.draw.box_thickness == 1


def test_overlay_flags():
    config = parse("--no-face-box", "--no-body-box", "--landmarks", "--blur-unknown",
                   "--no-hud", "--mirror")
    assert config.draw.show_face is False
    assert config.draw.show_body is False
    assert config.draw.show_landmarks is True
    assert config.draw.blur_unknown is True
    assert config.draw.show_fps is False and config.draw.show_roster is False
    assert config.runtime.mirror is True


def test_body_modes():
    assert parse("--body", "off").body.mode == "off"
    assert parse("--body", "yolo", "--body-conf", "0.6").body.conf == 0.6
    with pytest.raises(SystemExit):
        parse("--body", "nonsense")


def test_gallery_flags():
    config = parse("-f", "photos", "--rebuild-gallery", "--no-flip-augment")
    assert config.gallery.path == Path("photos")
    assert config.gallery.force_rebuild is True
    assert config.gallery.use_flip_augmentation is False
