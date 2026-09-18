"""The token gate on the web interface."""

import pytest

pytest.importorskip("fastapi", reason="the web UI is an optional extra")

from fastapi.testclient import TestClient  # noqa: E402

from oneshot_fd.config import AppConfig, BodyConfig, GalleryConfig  # noqa: E402
from oneshot_fd.web.auth import LOCAL_HOSTS, resolve_token, token_matches  # noqa: E402
from oneshot_fd.web.server import create_app  # noqa: E402


# ------------------------------------------------------------ choosing a token

def test_a_local_server_needs_no_token():
    for host in LOCAL_HOSTS:
        assert resolve_token(None, host) is None


def test_an_exposed_server_gets_one_generated():
    token = resolve_token(None, "0.0.0.0")
    assert token and len(token) >= 12


def test_an_explicit_token_is_always_honoured():
    assert resolve_token("hunter2", "127.0.0.1") == "hunter2"
    assert resolve_token("hunter2", "0.0.0.0") == "hunter2"


def test_comparison_rejects_the_wrong_token():
    assert token_matches("abc", "abc")
    assert not token_matches("abc", "abd")
    assert not token_matches("abc", None)
    assert not token_matches("abc", "")
    assert token_matches(None, None), "no token configured means no check"
    assert token_matches(None, "anything")


# -------------------------------------------------------------- the middleware

@pytest.fixture
def guarded(tmp_path):
    (tmp_path / "faces").mkdir()
    config = AppConfig(gallery=GalleryConfig(path=tmp_path / "faces", cache=False),
                       body=BodyConfig(mode="off"))
    with TestClient(create_app(config, token="s3cret")) as client:
        yield client


def test_without_a_token_everything_is_401(guarded):
    for path in ("/", "/api/state", "/api/gallery", "/api/calibrate"):
        assert guarded.get(path).status_code == 401, path
    assert guarded.post("/api/start", json={"source": "0"}).status_code == 401
    assert guarded.delete("/api/gallery/Someone").status_code == 401


def test_the_health_check_stays_open_for_docker(guarded):
    assert guarded.get("/api/health").status_code == 200
    assert guarded.get("/favicon.ico").status_code == 200


def test_a_header_gets_in(guarded):
    assert guarded.get("/api/state", headers={"X-Token": "s3cret"}).status_code == 200


def test_a_query_string_gets_in_and_is_remembered(guarded):
    response = guarded.get("/?token=s3cret")
    assert response.status_code == 200
    assert "oneshot_token" in response.cookies
    # The cookie now carries it, so the plain stream URL works.
    assert guarded.get("/api/state").status_code == 200


def test_a_wrong_token_does_not_get_in(guarded):
    assert guarded.get("/api/state", headers={"X-Token": "s3cre"}).status_code == 401
    assert guarded.get("/api/state?token=wrong").status_code == 401


def test_uploads_are_guarded_too(guarded):
    """The interesting endpoints are the ones that write."""
    response = guarded.post("/api/gallery",
                            files=[("files", ("x.png", b"not really", "image/png"))])
    assert response.status_code == 401


def test_no_token_means_no_gate(tmp_path):
    (tmp_path / "faces").mkdir()
    config = AppConfig(gallery=GalleryConfig(path=tmp_path / "faces", cache=False),
                       body=BodyConfig(mode="off"))
    with TestClient(create_app(config, token=None)) as client:
        assert client.get("/api/state").status_code == 200


# ------------------------------------------------------------ the environment

def test_the_environment_can_supply_the_token(monkeypatch):
    """So a container can be told its token instead of printing a fresh one."""
    from oneshot_fd.web.auth import TOKEN_ENV

    monkeypatch.setenv(TOKEN_ENV, "from-the-environment")
    assert resolve_token(None, "0.0.0.0") == "from-the-environment"
    assert resolve_token(None, "127.0.0.1") == "from-the-environment"


def test_the_command_line_beats_the_environment(monkeypatch):
    from oneshot_fd.web.auth import TOKEN_ENV

    monkeypatch.setenv(TOKEN_ENV, "from-the-environment")
    assert resolve_token("explicit", "0.0.0.0") == "explicit"


def test_a_blank_environment_variable_is_ignored(monkeypatch):
    from oneshot_fd.web.auth import TOKEN_ENV

    monkeypatch.setenv(TOKEN_ENV, "   ")
    assert resolve_token(None, "127.0.0.1") is None


def test_the_generated_warning_formats_without_blowing_up(caplog):
    """A logging call with the wrong number of arguments only fails when logged."""
    import logging

    with caplog.at_level(logging.WARNING, logger="oneshot_fd"):
        token = resolve_token(None, "0.0.0.0")
    assert token in caplog.text
    assert "ONESHOT_TOKEN" in caplog.text
