# One-Shot Face Recognition - the web UI and REST API in a container.
#
#   docker build -t oneshot-fd .
#   docker run -p 8000:8000 -v "$PWD/input_faces:/app/input_faces" oneshot-fd
#
# Then open http://localhost:8000
#
# The face models (~300 MB) are downloaded during the build, so the container
# starts ready to work and needs no network at run time.

FROM python:3.11-slim AS base

# OpenCV needs these even in its headless build; ffmpeg gives it the codecs to
# read mp4/mkv and to open RTSP streams.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libgl1 \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    INSIGHTFACE_HOME=/models

WORKDIR /app

# Dependencies first, so editing the source does not re-resolve them.
COPY requirements.txt requirements-web.txt ./
RUN pip install --no-cache-dir -r requirements-web.txt \
    && pip uninstall -y opencv-python \
    && pip install --no-cache-dir opencv-python-headless

COPY pyproject.toml README.md ./
COPY src/ ./src/
RUN pip install --no-cache-dir -e .

# Bake the models into the image so first use is not a 300 MB wait.
RUN python -c "\
from insightface.app import FaceAnalysis; \
app = FaceAnalysis(name='buffalo_l', root='/models', \
                   providers=['CPUExecutionProvider'], \
                   allowed_modules=['detection', 'recognition']); \
app.prepare(ctx_id=-1, det_size=(640, 640)); \
print('models ready')"

# Mount your own photos over this to enrol people.
RUN mkdir -p /app/input_faces /app/outputs
VOLUME ["/app/input_faces", "/app/outputs"]

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request; \
urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=4)"

# A container has to bind 0.0.0.0 for the published port to reach it, which
# means the token gate is always on. Set ONESHOT_TOKEN to choose it; otherwise
# one is generated and printed at startup. `--no-token` turns it off entirely.
#
# 0.0.0.0 so the port mapping actually reaches it from outside the container.
ENTRYPOINT ["python", "-m", "oneshot_fd"]
CMD ["--serve", "--host", "0.0.0.0", "--port", "8000", \
     "--faces", "/app/input_faces", "--model-root", "/models"]
