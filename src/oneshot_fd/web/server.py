"""FastAPI app: the browser UI and the REST API behind it.

Endpoints
---------
``GET  /``                  the single-page UI
``GET  /api/state``         everything the UI polls: status, tracks, roster, settings
``GET  /api/events``        recognition events since a timestamp
``GET  /api/stream.mjpg``   the annotated video as a browser-native MJPEG stream
``GET  /api/frame.jpg``     the latest annotated frame, for a poor connection
``POST /api/start``         begin a session on a camera, file, folder or URL
``POST /api/stop``          end it
``POST /api/settings``      change thresholds and overlays while running
``GET  /api/gallery``       who is enrolled, with any photo warnings
``POST /api/gallery``       upload reference photos
``DELETE /api/gallery/{n}`` remove a person
``POST /api/gallery/reload``re-enrol the folder
``GET  /api/calibrate``     the threshold the gallery itself suggests
``POST /api/analyse``       process an uploaded video and return the results
``GET  /api/health``        liveness, for Docker and load balancers
"""

import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
    from fastapi.responses import HTMLResponse, Response, StreamingResponse
except ImportError as _exc:  # pragma: no cover - depends on the install
    raise RuntimeError(
        "The web UI needs FastAPI. Install it with:\n"
        "  pip install -r requirements-web.txt\n"
        "(or: pip install fastapi uvicorn[standard] python-multipart)"
    ) from _exc

from ..config import AppConfig
from ..utils import IMAGE_SUFFIXES, LOGGER
from .session import Session

BOUNDARY = "oneshotframe"

#: A face-in-a-frame glyph, small enough to keep inline.
_FAVICON = (
    b'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32">'
    b'<rect width="32" height="32" rx="6" fill="#0e1116"/>'
    b'<rect x="8" y="7" width="16" height="18" rx="4" fill="none" '
    b'stroke="#4da3ff" stroke-width="2"/>'
    b'<circle cx="13" cy="14" r="1.6" fill="#4da3ff"/>'
    b'<circle cx="19" cy="14" r="1.6" fill="#4da3ff"/>'
    b'<path d="M12.5 19.5c1.4 1.6 5.6 1.6 7 0" stroke="#4da3ff" '
    b'stroke-width="2" fill="none" stroke-linecap="round"/></svg>'
)


def _safe_person_name(raw: str) -> str:
    """Keep uploads inside the gallery folder, whatever the browser sends."""
    name = Path(str(raw)).name.strip()
    name = "".join(ch for ch in name if ch.isalnum() or ch in " _-.()").strip()
    if not name or name in (".", ".."):
        raise ValueError("That name cannot be used for a folder.")
    return name


def create_app(config: Optional[AppConfig] = None):
    """Build the FastAPI application around one :class:`Session`."""
    config = config or AppConfig()
    session = Session(config)

    app = FastAPI(title="One-Shot Face Recognition", version="1.0.0",
                  docs_url="/api/docs", redoc_url=None)
    app.state.session = session

    # ------------------------------------------------------------------- page
    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return (Path(__file__).parent / "static" / "index.html").read_text(encoding="utf-8")

    @app.get("/favicon.ico")
    def favicon():
        """A tiny inline icon, so the browser stops asking for one."""
        return Response(content=_FAVICON, media_type="image/svg+xml",
                        headers={"Cache-Control": "max-age=86400"})

    @app.get("/api/health")
    def health() -> Dict[str, Any]:
        return {
            "ok": True,
            "status": session.status,
            "running": session.is_running,
            "people": len(session.pipeline.gallery) if session.pipeline else None,
        }

    # ------------------------------------------------------------------ state
    @app.get("/api/state")
    def state() -> Dict[str, Any]:
        return session.snapshot()

    @app.get("/api/events")
    def events(since: float = Query(0.0, description="Unix time of the last event seen")):
        return {"now": time.time(), "events": session.events_since(since)}

    # ------------------------------------------------------------------ video
    @app.get("/api/frame.jpg")
    def frame():
        jpeg = session.latest_jpeg()
        if jpeg is None:
            raise HTTPException(status_code=404, detail="No frame yet.")
        return Response(content=jpeg, media_type="image/jpeg",
                        headers={"Cache-Control": "no-store"})

    @app.get("/api/stream.mjpg")
    def stream():
        """Multipart JPEG - every browser renders it in a plain <img>."""
        def frames():
            last_sent = -1
            idle_since = time.time()
            while True:
                jpeg = session.latest_jpeg()
                number = session._frame_number
                if jpeg is not None and number != last_sent:
                    last_sent = number
                    idle_since = time.time()
                    yield (
                        f"--{BOUNDARY}\r\nContent-Type: image/jpeg\r\n"
                        f"Content-Length: {len(jpeg)}\r\n\r\n".encode() + jpeg + b"\r\n"
                    )
                else:
                    time.sleep(0.02)
                    # Let go once the session has stopped producing frames, so a
                    # forgotten browser tab does not hold a thread forever.
                    if not session.is_running and time.time() - idle_since > 2.0:
                        return

        return StreamingResponse(
            frames(), media_type=f"multipart/x-mixed-replace; boundary={BOUNDARY}",
            headers={"Cache-Control": "no-store"},
        )

    # ---------------------------------------------------------------- control
    @app.post("/api/start")
    def start(payload: Dict[str, Any]):
        source = str(payload.get("source", "")).strip()
        if not source:
            raise HTTPException(status_code=400, detail="Give a camera index, path or URL.")
        if payload.get("settings"):
            session.apply_settings(payload["settings"])
        try:
            session.start(source)
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc))
        return {"started": True, "source": source}

    @app.post("/api/stop")
    def stop():
        session.stop()
        return {"stopped": True, "status": session.status}

    @app.post("/api/settings")
    def settings(payload: Dict[str, Any]):
        return session.apply_settings(payload)

    # ---------------------------------------------------------------- gallery
    @app.get("/api/gallery")
    def gallery():
        pipeline = session.ensure_pipeline()
        return {
            "people": session._gallery_json(),
            "folder": str(Path(config.gallery.path).resolve()),
            "count": len(pipeline.gallery),
        }

    @app.post("/api/gallery")
    async def upload(files: List[UploadFile] = File(...), name: str = Form("")):
        """Add reference photos.

        With a ``name``, every file goes into that person's folder. Without one,
        each file's own name becomes the person - so dragging in `Sara.jpg` and
        `Omar.jpg` enrols two people in one go.
        """
        folder = Path(config.gallery.path)
        folder.mkdir(parents=True, exist_ok=True)

        saved: List[str] = []
        rejected: List[Dict[str, str]] = []
        for upload_file in files:
            suffix = Path(upload_file.filename or "").suffix.lower()
            if suffix not in IMAGE_SUFFIXES:
                rejected.append({"file": upload_file.filename or "?",
                                 "why": f"{suffix or 'no extension'} is not an image"})
                continue
            try:
                if name.strip():
                    person = _safe_person_name(name)
                    target_dir = folder / person
                    target_dir.mkdir(parents=True, exist_ok=True)
                    target = target_dir / _safe_person_name(upload_file.filename or "photo.jpg")
                else:
                    person = _safe_person_name(Path(upload_file.filename or "person.jpg").stem)
                    target = folder / f"{person}{suffix}"
            except ValueError as exc:
                rejected.append({"file": upload_file.filename or "?", "why": str(exc)})
                continue

            with target.open("wb") as handle:
                shutil.copyfileobj(upload_file.file, handle)
            saved.append(str(target.relative_to(folder)))

        result = session.reload_gallery() if saved else {"people": session._gallery_json()}
        return {"saved": saved, "rejected": rejected, **result}

    @app.delete("/api/gallery/{person}")
    def remove(person: str):
        folder = Path(config.gallery.path)
        try:
            safe = _safe_person_name(person)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

        removed: List[str] = []
        directory = folder / safe
        if directory.is_dir():
            shutil.rmtree(directory)
            removed.append(f"{safe}/")
        for candidate in folder.glob(f"{safe}.*"):
            if candidate.suffix.lower() in IMAGE_SUFFIXES:
                candidate.unlink()
                removed.append(candidate.name)

        if not removed:
            raise HTTPException(status_code=404, detail=f"No reference photos for '{safe}'.")
        return {"removed": removed, **session.reload_gallery()}

    @app.post("/api/gallery/reload")
    def reload_gallery():
        return session.reload_gallery()

    @app.get("/api/calibrate")
    def calibrate():
        """What threshold the enrolled people themselves suggest."""
        pipeline = session.ensure_pipeline()
        suggested, worst, explanation = pipeline.gallery.suggest_threshold()
        return {
            "suggested": round(suggested, 3),
            "worst_pair_similarity": round(worst, 3),
            "explanation": explanation,
            "closest_pairs": [
                {"similarity": round(score, 3), "a": a, "b": b}
                for score, a, b in pipeline.gallery.closest_pairs(5)
            ],
        }

    # --------------------------------------------------------------- analysis
    @app.post("/api/analyse")
    async def analyse(file: UploadFile = File(...), max_frames: int = Form(0)):
        """Process an uploaded video and return who was in it, as JSON.

        Runs to completion before answering, so it suits short clips and
        scripted use rather than a live feed - point ``/api/start`` at a path
        for anything long.
        """
        if session.is_running:
            raise HTTPException(status_code=409,
                                detail="A live session is running. Stop it first.")

        suffix = Path(file.filename or "clip.mp4").suffix or ".mp4"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
            shutil.copyfileobj(file.file, handle)
            temporary = Path(handle.name)

        pipeline = session.ensure_pipeline()
        previous = config.runtime.max_frames
        config.runtime.max_frames = int(max_frames) or None
        started = time.time()
        frames = 0
        try:
            for _ in pipeline.run_source(str(temporary)):
                frames += 1
        finally:
            config.runtime.max_frames = previous
            temporary.unlink(missing_ok=True)

        appearances = [
            {
                "name": appearance.name,
                "track_id": appearance.track_id,
                "start": round(appearance.start_time, 2),
                "end": round(appearance.end_time, 2),
                "duration": round(max(0.0, appearance.end_time - appearance.start_time), 2),
                "best_score": round(appearance.best_score, 3),
                "frames": appearance.frames,
            }
            for appearance in pipeline.appearances
        ]
        return {
            "file": file.filename,
            "frames": frames,
            "seconds": round(time.time() - started, 1),
            "people": sorted({a["name"] for a in appearances}),
            "appearances": appearances,
        }

    return app


def serve(config: Optional[AppConfig] = None, host: str = "127.0.0.1",
          port: int = 8000, reload: bool = False) -> None:
    """Run the web UI with uvicorn."""
    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "The web UI needs uvicorn. Install it with:\n"
            "  pip install -r requirements-web.txt"
        ) from exc

    app = create_app(config)
    shown = "localhost" if host in ("127.0.0.1", "0.0.0.0") else host
    LOGGER.info("Web UI on http://%s:%d  (API docs at /api/docs)", shown, port)
    if host == "0.0.0.0":
        LOGGER.info("Listening on all interfaces - reachable from other devices "
                    "on this network.")
    uvicorn.run(app, host=host, port=port, log_level="warning")
