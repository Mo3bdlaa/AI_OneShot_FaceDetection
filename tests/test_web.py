"""The REST API, exercised through FastAPI's test client.

The models are replaced with stand-ins, so these run in milliseconds and test
the plumbing - routing, uploads, validation, settings - rather than the
recogniser, which has its own tests.
"""

import numpy as np
import pytest

pytest.importorskip("fastapi", reason="the web UI is an optional extra")

from fastapi.testclient import TestClient  # noqa: E402

from oneshot_fd.config import AppConfig, BodyConfig, GalleryConfig, RuntimeConfig  # noqa: E402
from oneshot_fd.gallery import Person  # noqa: E402
from oneshot_fd.utils import l2_normalize  # noqa: E402
from oneshot_fd.web.server import _safe_person_name, create_app  # noqa: E402


class StubEngine:
    """Stands in for the face models - loads instantly, finds nothing."""

    def load(self):
        return self

    def locate(self, frame, max_faces=None):
        return []

    def embed(self, frame, face):
        return None

    def detect(self, frame, max_faces=None):
        return []

    def embed_reference(self, image, min_face_size=40):
        return None


@pytest.fixture
def client(tmp_path):
    config = AppConfig(
        gallery=GalleryConfig(path=tmp_path / "faces", cache=False),
        body=BodyConfig(mode="off"),
        runtime=RuntimeConfig(display=False),
    )
    (tmp_path / "faces").mkdir()
    app = create_app(config)
    session = app.state.session

    # Give the session a ready-made pipeline so nothing downloads a model.
    from oneshot_fd.pipeline import Pipeline

    pipeline = Pipeline(config)
    pipeline.engine = StubEngine()
    pipeline.gallery._set_people([
        Person("Mohammed", l2_normalize(np.array([[1, 0, 0, 0]], np.float32), axis=1), ["m.jpg"]),
        Person("Sara", l2_normalize(np.array([[0, 1, 0, 0]], np.float32), axis=1), ["s.jpg"]),
    ])
    pipeline.bodies.load()
    session.pipeline = pipeline

    with TestClient(app) as test_client:
        test_client.app_session = session
        yield test_client


def png_bytes(name="face.png"):
    import cv2

    ok, buffer = cv2.imencode(".png", np.full((120, 120, 3), 128, np.uint8))
    assert ok
    return (name, buffer.tobytes(), "image/png")


# ------------------------------------------------------------------- basics

def test_the_page_is_served(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "One-Shot Face Recognition" in response.text
    assert "/api/stream.mjpg" in response.text, "the page must point at the stream"


def test_health(client):
    body = client.get("/api/health").json()
    assert body["ok"] is True
    assert body["running"] is False


def test_favicon_is_served_so_the_browser_stops_asking(client):
    response = client.get("/favicon.ico")
    assert response.status_code == 200
    assert "svg" in response.headers["content-type"]


def test_state_has_what_the_ui_needs(client):
    state = client.get("/api/state").json()
    assert state["status"] == "idle"
    assert state["tracks"] == []
    assert {"threshold", "margin", "min_quality", "reverify_every"} <= set(state["settings"])
    assert state["settings"]["attributes_available"] is False


def test_no_tracks_are_reported_when_nothing_is_running(client):
    """A stopped session must not keep claiming people are on screen."""
    session = client.app_session
    session.pipeline.tracker.update([{"box": (10, 10, 60, 60), "name": "Mohammed",
                                      "similarity": 0.9}])
    session.pipeline.tracker.update([{"box": (10, 10, 60, 60), "name": "Mohammed",
                                      "similarity": 0.9}])
    assert session.pipeline.tracker.visible(), "the tracker really does hold a track"
    assert client.get("/api/state").json()["tracks"] == []


# ------------------------------------------------------------------ gallery

def test_gallery_lists_enrolled_people(client):
    body = client.get("/api/gallery").json()
    assert {person["name"] for person in body["people"]} == {"Mohammed", "Sara"}


def test_uploading_a_photo_names_the_person_after_the_file(client, tmp_path):
    response = client.post("/api/gallery", files=[("files", png_bytes("Omar.png"))])
    assert response.status_code == 200
    assert response.json()["saved"] == ["Omar.png"]
    assert (tmp_path / "faces" / "Omar.png").exists()


def test_uploading_with_a_name_groups_the_photos(client, tmp_path):
    client.post("/api/gallery", data={"name": "Layla"},
                files=[("files", png_bytes("one.png")), ("files", png_bytes("two.png"))])
    folder = tmp_path / "faces" / "Layla"
    assert folder.is_dir()
    assert len(list(folder.glob("*.png"))) == 2


def test_non_images_are_rejected_with_a_reason(client):
    body = client.post("/api/gallery",
                       files=[("files", ("notes.txt", b"hello", "text/plain"))]).json()
    assert body["saved"] == []
    assert body["rejected"][0]["why"].startswith(".txt")


def test_a_path_cannot_escape_the_gallery_folder(client, tmp_path):
    """Whatever the browser sends, the file lands inside the gallery."""
    client.post("/api/gallery", files=[("files", png_bytes("../../escape.png"))])
    assert not (tmp_path.parent / "escape.png").exists()
    assert not (tmp_path / "escape.png").exists()


@pytest.mark.parametrize("raw, expected", [
    ("Mohammed", "Mohammed"),
    ("../../etc/passwd", "passwd"),
    ("a/b/c.jpg", "c.jpg"),
    ("  spaced  ", "spaced"),
])
def test_person_names_are_sanitised(raw, expected):
    assert _safe_person_name(raw) == expected


@pytest.mark.parametrize("raw", ["", "..", "/", "***"])
def test_impossible_person_names_are_refused(raw):
    with pytest.raises(ValueError):
        _safe_person_name(raw)


def test_deleting_a_person_removes_their_photos(client, tmp_path):
    client.post("/api/gallery", files=[("files", png_bytes("Omar.png"))])
    assert (tmp_path / "faces" / "Omar.png").exists()

    response = client.delete("/api/gallery/Omar")
    assert response.status_code == 200
    assert not (tmp_path / "faces" / "Omar.png").exists()


def test_deleting_somebody_who_is_not_there_is_a_404(client):
    assert client.delete("/api/gallery/Nobody").status_code == 404


def test_calibrate_reports_the_closest_pair(client):
    body = client.get("/api/calibrate").json()
    assert "suggested" in body
    assert body["closest_pairs"][0]["a"] in {"Mohammed", "Sara"}
    assert "explanation" in body


# ----------------------------------------------------------------- settings

def test_settings_can_be_changed_live(client):
    body = client.post("/api/settings", json={"threshold": 0.5, "blur_unknown": True}).json()
    assert set(body["changed"]) == {"threshold", "blur_unknown"}
    assert body["settings"]["threshold"] == 0.5
    assert client.get("/api/state").json()["settings"]["blur_unknown"] is True


def test_unknown_settings_are_ignored_not_fatal(client):
    body = client.post("/api/settings", json={"nonsense": 1, "threshold": 0.44}).json()
    assert body["changed"] == ["threshold"]


def test_negative_reverify_is_clamped(client):
    body = client.post("/api/settings", json={"reverify_every": -5}).json()
    assert body["settings"]["reverify_every"] == 0


# ------------------------------------------------------------------ control

def test_starting_without_a_source_is_rejected(client):
    assert client.post("/api/start", json={"source": "  "}).status_code == 400


def test_starting_a_source_that_does_not_exist_surfaces_the_error(client):
    assert client.post("/api/start", json={"source": "/no/such/clip.mp4"}).status_code == 200
    # The worker reports the failure through the state, not the HTTP response.
    for _ in range(50):
        state = client.get("/api/state").json()
        if state["status"] == "error":
            break
    assert state["status"] == "error"
    assert "clip.mp4" in state["message"]


def test_stopping_when_nothing_runs_is_harmless(client):
    assert client.post("/api/stop").json()["stopped"] is True


def test_no_frame_yet_is_a_404_not_a_crash(client):
    assert client.get("/api/frame.jpg").status_code == 404


def test_events_start_empty_and_carry_a_clock(client):
    body = client.get("/api/events?since=0").json()
    assert body["events"] == []
    assert body["now"] > 0


def test_a_photo_with_no_face_in_it_does_not_break_the_upload(client, tmp_path):
    """Somebody will drop a landscape in. The file saves; the problem is reported."""
    response = client.post("/api/gallery", files=[("files", png_bytes("Scenery.png"))])
    assert response.status_code == 200
    body = response.json()
    assert body["saved"] == ["Scenery.png"], "the file still lands on disk"
    assert "problem" in body, "and the user is told why nothing was enrolled"


# ------------------------------------------------------- uploading a clip to run

def mp4_bytes(name="clip.mp4"):
    return (name, b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 64, "video/mp4")


def test_a_video_upload_comes_back_as_a_path_to_start(client, tmp_path):
    body = client.post("/api/source", files={"file": mp4_bytes("party.mp4")}).json()
    path = __import__("pathlib").Path(body["path"])
    assert path.exists()
    assert path.name.endswith("party.mp4")
    assert body["bytes"] > 0


def test_a_photo_is_not_accepted_as_a_video(client):
    response = client.post("/api/source", files={"file": png_bytes("face.png")})
    assert response.status_code == 400
    assert "not a video" in response.json()["detail"]


def test_old_uploads_are_pruned_so_the_disk_does_not_fill(client, tmp_path):
    from oneshot_fd.web.server import KEEP_FILES

    for index in range(KEEP_FILES + 4):
        client.post("/api/source", files={"file": mp4_bytes(f"clip{index}.mp4")})

    uploads = tmp_path / ".oneshot_work" / "uploads"
    remaining = sorted(path.name for path in uploads.iterdir())
    assert len(remaining) == KEEP_FILES, remaining
    # The newest survives; the first ones uploaded are gone.
    assert any(name.endswith(f"clip{KEEP_FILES + 3}.mp4") for name in remaining)
    assert not any(name.endswith("clip0.mp4") for name in remaining)


@pytest.mark.parametrize("source, expected", [
    ("0", "camera0"),
    ("rtsp://cam/stream", "stream"),
    ("/uploads/1789716306_party.mp4", "party"),   # the upload stamp is not repeated
    ("/clips/meeting.mkv", "meeting"),
])
def test_recording_names_are_readable(source, expected):
    from oneshot_fd.web.server import _recording_stem

    assert _recording_stem(source) == expected


# ---------------------------------------------------------------- the results

def test_the_csv_has_the_same_columns_as_the_cli_log(client):
    from oneshot_fd.pipeline import Appearance

    client.app_session.pipeline._appearances.append(
        Appearance(name="Mohammed", track_id=1, source="clip.mp4",
                   start_time=1.0, end_time=3.5, best_score=0.91, frames=60)
    )
    response = client.get("/api/appearances.csv")
    assert response.status_code == 200
    assert "attachment" in response.headers["content-disposition"]

    rows = response.text.strip().splitlines()
    assert rows[0] == "source,name,track_id,start,end,duration_s,best_score,frames"
    assert "Mohammed" in rows[1] and "00:00:01.000" in rows[1] and "2.50" in rows[1]


def test_asking_for_a_recording_that_was_not_made(client):
    response = client.get("/api/recording.mp4")
    assert response.status_code == 404
    assert "Save annotated video" in response.json()["detail"]


def test_state_says_whether_there_is_anything_to_download(client):
    from oneshot_fd.pipeline import Appearance

    assert client.get("/api/state").json()["has_results"] is False
    client.app_session.pipeline._appearances.append(
        Appearance(name="Sara", track_id=1, source="c.mp4", start_time=0.0,
                   end_time=1.0, best_score=0.9, frames=25)
    )
    assert client.get("/api/state").json()["has_results"] is True


def test_a_new_session_does_not_inherit_the_last_run_s_results(client):
    """The downloaded CSV must describe the run you just did, not every run."""
    from oneshot_fd.pipeline import Appearance

    client.app_session.pipeline._appearances.append(
        Appearance(name="FromLastTime", track_id=1, source="old.mp4", start_time=0.0,
                   end_time=1.0, best_score=0.9, frames=25)
    )
    assert "FromLastTime" in client.get("/api/appearances.csv").text

    client.post("/api/start", json={"source": "/no/such/clip.mp4"})
    for _ in range(50):
        if client.get("/api/state").json()["status"] in ("error", "stopped"):
            break
    assert "FromLastTime" not in client.get("/api/appearances.csv").text
