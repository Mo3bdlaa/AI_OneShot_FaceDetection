"""Source expansion and frame reading."""

import cv2
import numpy as np
import pytest

from oneshot_fd.sources import FrameSource, VideoWriter, expand_sources
from oneshot_fd.utils import is_video_file


def write_video(path, frames=5, size=(64, 48)):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, size)
    for i in range(frames):
        frame = np.full((size[1], size[0], 3), i * 20, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return path


def test_camera_index_and_urls_pass_through():
    assert expand_sources(["0"]) == ["0"]
    assert expand_sources(["rtsp://cam/stream"]) == ["rtsp://cam/stream"]


def test_a_folder_of_videos_becomes_one_entry_per_video(tmp_path):
    write_video(tmp_path / "b.mp4")
    write_video(tmp_path / "a.mp4")
    (tmp_path / "notes.txt").write_text("ignored")

    expanded = expand_sources([str(tmp_path)])
    assert [p.split("/")[-1] for p in expanded] == ["a.mp4", "b.mp4"], "played in order"


def test_a_folder_of_images_stays_one_sequence(tmp_path):
    for name in ("001.jpg", "002.jpg"):
        cv2.imwrite(str(tmp_path / name), np.zeros((32, 32, 3), np.uint8))
    assert expand_sources([str(tmp_path)]) == [str(tmp_path)]


def test_globs_are_expanded(tmp_path):
    write_video(tmp_path / "one.mp4")
    write_video(tmp_path / "two.mp4")
    assert len(expand_sources([str(tmp_path / "*.mp4")])) == 2


def test_several_sources_are_kept_in_order(tmp_path):
    write_video(tmp_path / "clip.mp4")
    expanded = expand_sources(["0", str(tmp_path / "clip.mp4")])
    assert expanded[0] == "0" and expanded[1].endswith("clip.mp4")


def test_reading_a_video_yields_every_frame(tmp_path):
    path = write_video(tmp_path / "clip.mp4", frames=5)
    with FrameSource(str(path)) as source:
        assert source.info.kind == "video"
        frames = list(source.frames())
    assert len(frames) == 5
    assert [index for index, _, _ in frames] == [0, 1, 2, 3, 4]


def test_max_frames_stops_early(tmp_path):
    path = write_video(tmp_path / "clip.mp4", frames=10)
    with FrameSource(str(path)) as source:
        assert len(list(source.frames(max_frames=3))) == 3


def test_reading_a_folder_of_images(tmp_path):
    for i in range(3):
        cv2.imwrite(str(tmp_path / f"{i:03d}.png"), np.full((16, 16, 3), i * 50, np.uint8))
    with FrameSource(str(tmp_path)) as source:
        assert source.info.kind == "images"
        assert len(list(source.frames())) == 3


def test_a_missing_source_says_so(tmp_path):
    with pytest.raises(FileNotFoundError):
        FrameSource(str(tmp_path / "nope.mp4")).open()


def test_an_empty_image_folder_says_so(tmp_path):
    (tmp_path / "sub").mkdir()
    with pytest.raises(ValueError):
        FrameSource(str(tmp_path)).open()


def test_video_writer_creates_the_file_and_its_folder(tmp_path):
    target = tmp_path / "nested" / "out.mp4"
    writer = VideoWriter(target, fps=10)
    for _ in range(3):
        writer.write(np.zeros((48, 64, 3), np.uint8))
    writer.close()
    assert target.exists() and target.stat().st_size > 0


def test_video_suffix_detection():
    from pathlib import Path

    assert is_video_file(Path("a.MP4")) and is_video_file(Path("b.mkv"))
    assert not is_video_file(Path("c.jpg"))
