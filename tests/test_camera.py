"""Frames pushed from the client's own camera, and the certificate that allows it."""

import cv2
import numpy as np
import pytest

pytest.importorskip("fastapi", reason="the web UI is an optional extra")

from fastapi.testclient import TestClient  # noqa: E402

from oneshot_fd.config import AppConfig, BodyConfig, GalleryConfig  # noqa: E402
from oneshot_fd.web.server import create_app  # noqa: E402
from oneshot_fd.web.tls import _san, ensure_certificate, local_addresses  # noqa: E402


class StubEngine:
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
    (tmp_path / "faces").mkdir()
    config = AppConfig(gallery=GalleryConfig(path=tmp_path / "faces", cache=False),
                       body=BodyConfig(mode="off"))
    app = create_app(config)
    from oneshot_fd.pipeline import Pipeline

    pipeline = Pipeline(config)
    pipeline.engine = StubEngine()
    pipeline.bodies.load()
    app.state.session.pipeline = pipeline
    with TestClient(app) as test_client:
        test_client.app_session = app.state.session
        yield test_client


def jpeg(width=320, height=240):
    ok, buffer = cv2.imencode(".jpg", np.full((height, width, 3), 120, np.uint8))
    assert ok
    return buffer.tobytes()


# ------------------------------------------------------------- pushed frames

def test_a_pushed_frame_comes_back_annotated(client):
    client.post("/api/camera/start")
    response = client.post("/api/camera/frame", content=jpeg(),
                           headers={"Content-Type": "image/jpeg"})
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"

    decoded = cv2.imdecode(np.frombuffer(response.content, np.uint8), cv2.IMREAD_COLOR)
    assert decoded is not None, "the answer must be a readable image"
    assert decoded.shape[:2] == (240, 320)


def test_pushing_before_starting_is_refused(client):
    response = client.post("/api/camera/frame", content=jpeg(),
                           headers={"Content-Type": "image/jpeg"})
    assert response.status_code == 409
    assert "Start one first" in response.json()["detail"]


def test_an_empty_body_is_rejected(client):
    client.post("/api/camera/start")
    assert client.post("/api/camera/frame", content=b"").status_code == 400


def test_rubbish_is_rejected_rather_than_crashing(client):
    client.post("/api/camera/start")
    response = client.post("/api/camera/frame", content=b"this is not a jpeg",
                           headers={"Content-Type": "image/jpeg"})
    assert response.status_code == 400
    assert "readable image" in response.json()["detail"]


def test_the_state_reports_a_pushed_session_as_running(client):
    assert client.get("/api/state").json()["pushed"] is False
    client.post("/api/camera/start")
    state = client.get("/api/state").json()
    assert state["status"] == "running"
    assert state["pushed"] is True
    assert state["source"] == "browser camera"


def test_frames_advance_the_counter(client):
    client.post("/api/camera/start")
    for _ in range(3):
        client.post("/api/camera/frame", content=jpeg(),
                    headers={"Content-Type": "image/jpeg"})
    assert client.get("/api/state").json()["frame"] == 3


def test_stopping_ends_the_pushed_session(client):
    client.post("/api/camera/start")
    client.post("/api/stop")
    state = client.get("/api/state").json()
    assert state["pushed"] is False
    assert state["status"] == "stopped"
    # ... and pushing again is refused rather than silently resuming.
    assert client.post("/api/camera/frame", content=jpeg()).status_code == 409


def test_two_sessions_cannot_run_at_once(client):
    client.post("/api/camera/start")
    assert client.post("/api/camera/start").status_code == 409
    assert client.post("/api/start", json={"source": "0"}).status_code == 409


def test_the_last_pushed_frame_is_served_like_any_other(client):
    client.post("/api/camera/start")
    client.post("/api/camera/frame", content=jpeg(),
                headers={"Content-Type": "image/jpeg"})
    assert client.get("/api/frame.jpg").status_code == 200


# --------------------------------------------------------------- certificates

def test_local_addresses_always_include_loopback():
    assert "127.0.0.1" in local_addresses()


def test_the_san_names_both_kinds_of_host():
    san = _san(["127.0.0.1", "192.168.1.5", "my-laptop"])
    assert "DNS:localhost" in san
    assert "IP:127.0.0.1" in san and "IP:192.168.1.5" in san
    assert "DNS:my-laptop" in san


def test_a_certificate_is_made_and_then_reused(tmp_path):
    made = ensure_certificate(tmp_path, ["127.0.0.1"])
    if made is None:
        pytest.skip("openssl is not installed here")

    certificate, key = made
    assert certificate.exists() and key.exists()
    assert oct(key.stat().st_mode)[-3:] == "600", "a private key must not be world-readable"

    fingerprint = certificate.read_bytes()
    again = ensure_certificate(tmp_path, ["127.0.0.1"])
    assert again == made
    assert certificate.read_bytes() == fingerprint, "an existing certificate is not replaced"


def test_the_certificate_names_the_hosts_it_was_asked_for(tmp_path):
    import subprocess

    made = ensure_certificate(tmp_path, ["127.0.0.1", "192.168.1.5"])
    if made is None:
        pytest.skip("openssl is not installed here")

    text = subprocess.run(
        ["openssl", "x509", "-in", str(made[0]), "-noout", "-text"],
        capture_output=True, text=True, check=True,
    ).stdout
    assert "192.168.1.5" in text, "a browser rejects a certificate that omits the host"
