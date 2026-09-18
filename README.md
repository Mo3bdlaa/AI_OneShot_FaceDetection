# AI One-Shot Face Detection

Give it a folder of face photos — **one photo per person is enough** — and it
recognises those people in a video, a folder of videos, an RTSP/HTTP stream or
a live camera, highlighting each person's **face and body** live, with their
name on screen.

- 🎯 **One-shot**: a single reference photo per person, no training, no fine-tuning
- 👥 **Many people at once**: everyone in frame is recognised in parallel
- 📹 **Any input**: webcam, video file, whole folders of videos, globs, RTSP/HTTP streams, image sequences
- 🖍️ **Live highlighting**: face box + body brackets + name + confidence, one stable colour per person
- 🧠 **Stable names**: tracking and vote smoothing stop labels from flickering frame to frame
- 📝 **A record of what happened**: CSV of who appeared, when, and for how long

Built on **InsightFace** (SCRFD detection + ArcFace embeddings), with optional
**YOLO** person boxes for body highlighting.

---

## Install

```bash
git clone https://github.com/Mo3bdlaa/AI_OneShot_FaceDetection.git
cd AI_OneShot_FaceDetection

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

Optional extras:

```bash
pip install ultralytics            # real person/body boxes instead of estimated ones
pip install onnxruntime-gpu        # NVIDIA GPU (replaces onnxruntime)
```

The face models (~300 MB) download automatically the first time you run it.

## Add the people you want recognised

```
input_faces/
├── Mohammed.jpg          # one photo  ->  person "Mohammed"
├── Sara.png              # one photo  ->  person "Sara"
└── Ahmed/                # or a folder, if you have several photos
    ├── front.jpg
    └── side.jpg
```

The file (or folder) name becomes the name drawn on screen. Extra photos are
optional and always help, but a single clear photo is all it needs.

Check it loaded everyone:

```bash
python run.py --list-people
```

## Run it

```bash
# live webcam
python run.py --source 0

# a video file, with the overlay saved to disk
python run.py --source party.mp4 --save outputs/party.mp4

# every video in a folder, plus a log of who appeared when
python run.py --source clips/ --save outputs/ --log-csv outputs/log.csv

# an IP camera / RTSP stream, always showing the present moment
python run.py --source rtsp://user:pass@192.168.1.10/stream --realtime

# several inputs in one go
python run.py --source 0 --source clips/ --source "recordings/*.mp4"

# a headless server: no window, just write the result
python run.py --source clip.mp4 --no-display --save outputs/clip.mp4
```

After `pip install -e .` the same thing is available as `oneshot-fd …`, or as
`python -m oneshot_fd …`.

### Keys in the preview window

| Key | Action |
|-----|--------|
| `q` / `Esc` | quit |
| `space` | pause / resume |
| `n` | skip to the next source |
| `b` | toggle body highlighting |
| `h` | toggle the FPS / roster panel |
| `s` | save a snapshot to `outputs/` |

## What you get

On screen, per person: a coloured face box, corner brackets around the body,
and a caption with the track id, the name and the match score. The panel in the
corner shows the frame rate and who is currently visible. Each person keeps the
same colour everywhere, so multi-person footage stays readable.

`--log-csv` writes one row per appearance:

```csv
source,name,track_id,start,end,duration_s,best_score,frames
meeting.mp4,Mohammed,1,00:00:00.040,00:00:03.560,3.52,0.969,89
meeting.mp4,Sara,3,00:00:00.040,00:00:03.560,3.52,0.980,89
```

and a summary is printed when the run finishes:

```
Recognised people:
  Mohammed                   3.5s over   1 appearance(s), best score 0.969
  Sara                       3.5s over   1 appearance(s), best score 0.980
```

## Tuning

### Accuracy

| Problem | Fix |
|---|---|
| A known person shows up as `Unknown` | lower `--threshold` (try `0.32`), or add another photo of them |
| Two people get confused with each other | raise `--threshold` (try `0.45`) and `--margin` (try `0.08`) |
| Small or distant faces are missed | raise `--det-size` to `960`, or lower `--det-threshold` to `0.35` |
| The name flickers between frames | raise `--vote-window` (try `24`) |

`--threshold` is a cosine similarity between ArcFace embeddings. Roughly:
`> 0.6` the same photo, `0.4–0.6` clearly the same person, `0.3–0.4` probably
the same person, `< 0.3` different people. The default of `0.38` is a
deliberately cautious middle.

### Speed

Two things are already done for you: only the models that are actually used
are loaded (the default InsightFace setup also loads 68- and 106-point
landmark models that nothing here reads — dropping them measured **2.3× faster**
for identical embeddings), and detection is separated from embedding so the
expensive half can be skipped.

On top of that:

```bash
python run.py --source 0 \
    --reverify-every 5 \    # settled tracks skip the embedding for a few frames
    --det-size 320 \        # smaller detector input
    --max-width 640 \       # downscale before processing
    --detect-every 3 \      # run the models every 3rd frame, track in between
    --model buffalo_s       # the lighter model pack
```

`--reverify-every N` is the biggest single win. Locating a face costs about a
tenth of what embedding it does, so once a track has named the same person
several frames running it is taken at its word for N more detections. On a
four-person clip that skipped **90% of the embeddings and ran 2.9× faster**,
with the same people recognised. It is off by default because it trades a
little identity paranoia for speed; 3–5 is a good setting for live video.

`--detect-every N` skips detection too: the tracker carries the boxes through
the skipped frames, so the overlay still updates every frame.

For real-time work, a GPU (`pip install onnxruntime-gpu`, `--device cuda`)
takes it to comfortably above 30 FPS.

On a live camera or stream, add `--realtime`: frames are read on a background
thread and the recogniser always takes the newest one, so the overlay never
drifts behind what the camera is seeing.

### Body highlighting

`--body auto` (the default) uses YOLO person boxes when `ultralytics` is
installed, and otherwise estimates the body geometrically from the face. The
estimate costs nothing and is good enough to show who is who; YOLO follows the
real silhouette and keeps working when someone turns away from the camera.

```bash
pip install ultralytics
python run.py --source clip.mp4 --body yolo --yolo-model yolov8n.pt
```

The YOLO weights download on first use into the current directory. YOLO is a
second model per frame, so it does cost speed — use `--body estimate` to force
the free path, or `--body off` for faces only.

### Age and gender

`--attributes` also estimates each person's age and gender and shows them
beside the name (`Mohammed 0.92 | M ~29`). The estimates are noisy frame to
frame, so the display uses the median age and majority gender over a rolling
window. It loads one more model and costs roughly 30% throughput, so it is off
unless asked for.

```bash
python run.py --source clip.mp4 --attributes
```

### Privacy

`--blur-unknown` pixelates the face of anyone who is *not* in your input
folder, which is useful when you only have consent for specific people.

## All options

Run `python run.py --help`. The main ones:

```
input / output
  -f, --faces DIR        folder of reference photos (default: input_faces)
  -s, --source SRC       camera index, file, folder, glob or URL; repeatable
  -o, --save PATH        write the annotated video (a directory for many sources)
      --log-csv FILE     append who was seen, when and for how long
      --no-display       do not open a preview window
      --list-people      build the gallery, print it and exit

recognition
  -t, --threshold F      similarity needed to claim a face (default: 0.38)
      --margin F         how far the best match must beat the runner-up (0.03)
      --vote-window N    frames a track votes over before committing (12)
      --reverify-every N settled tracks skip the embedding for N detections
      --rebuild-gallery  re-enrol every photo, ignoring the cache

models / speed
      --model NAME       buffalo_l (accurate) or buffalo_s (fast)
      --attributes       also estimate and show age and gender
      --det-size N       detector input size (640)
      --device           auto | cpu | cuda
      --detect-every N   run the models every N frames and track in between
      --max-width N      downscale frames before processing
      --realtime         on a camera/stream, always take the newest frame

body / overlay
      --body MODE        auto | yolo | estimate | off
      --blur-unknown     pixelate faces that are not in the gallery
      --landmarks        draw the five facial keypoints
      --mirror           mirror the frame (natural for a webcam)
```

## Use it from Python

```python
from pathlib import Path
from oneshot_fd.config import AppConfig, GalleryConfig, RuntimeConfig
from oneshot_fd.pipeline import Pipeline

config = AppConfig(
    gallery=GalleryConfig(path=Path("input_faces")),
    runtime=RuntimeConfig(sources=["meeting.mp4"], display=False),
)

with Pipeline(config) as pipeline:          # loads models + enrols the gallery
    for result in pipeline.run():
        for track in result.tracks:
            if track.label != "Unknown":
                print(f"{result.timestamp:6.2f}s  {track.label}  {track.label_score:.2f}"
                      f"  face={track.box}  body={track.body_box}")
    print(pipeline.summary())
```

`Pipeline.process_frame(frame)` does the same for a single frame if you already
have your own capture loop.

## How it works

```
input_faces/           every photo -> ArcFace embedding (plus a mirrored copy)
      |                the embeddings are cached and rebuilt when photos change
      v
   Gallery  <--- cosine similarity ---  face embeddings from the current frame
      |                                        ^
      |                                        |
      |                                  SCRFD face detection
      v                                        ^
  name + score                            video / camera frame
      |                                        |
      v                                        v
  Tracker (IoU + motion, vote smoothing)   YOLO person boxes
      |                                        |
      +-------------------+--------------------+
                          v
              face box + body box + stable name  ->  overlay / CSV
```

Three details that matter in practice:

**Mirrored references.** Every reference photo is embedded twice, normally and
mirrored. It costs one extra forward pass at startup and makes a single
reference photo noticeably more forgiving about which way the head is turned.

**A margin, not just a threshold.** A face is only claimed when the best match
also beats the runner-up by `--margin`. Without it, two people who look alike
trade identities whenever the lighting shifts.

**Voting instead of per-frame labels.** A blink, a blur or a half-turn can drop
one frame below the threshold. Each track votes over a sliding window and shows
the winner, so the name on screen stays put while the person moves.

## Tests

```bash
pip install pytest
pytest
```

85 tests covering geometry, gallery matching, tracking, body association,
source handling, rendering, CLI parsing and the pipeline. They use stand-in
models, so they run in under a second and need no downloads.

## Requirements

Python 3.8+, and the packages in `requirements.txt`. Works on CPU; a CUDA GPU
is optional and much faster.

## Licence

MIT — see [LICENSE](LICENSE).
