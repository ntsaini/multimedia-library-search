# Implementation Plan — Local Video Library Search Engine

## Stack

| Concern | Tool | Reason |
|---|---|---|
| Language | Python 3.11+ | First-class support across all ML/CV libraries |
| API + HTML server | FastAPI + Jinja2 | Single process serves API and UI; no separate frontend build |
| Face detection + embedding | InsightFace (buffalo_l, ONNX) | Faster and lighter than DeepFace; no TensorFlow dependency; GPU-optional |
| Vector store | ChromaDB (embedded/persistent) | Simple Python API, no server process, persists to disk |
| Relational store | SQLite (`sqlite3` stdlib) | No ORM needed; zero extra dependencies |
| Frame extraction | OpenCV (`cv2`) | Standard, reliable |
| Video cutting + compilation | FFmpeg via `subprocess` | Lossless segment copy; more control than `ffmpeg-python` wrapper |
| Face clustering | DBSCAN (`scikit-learn`) | Discovers cluster count automatically; handles noise |
| Frontend | Vanilla HTML + htmx | No build step; htmx handles dynamic updates without a JS framework |
| CLI | `argparse` (stdlib) | No extra dependency |
| Progress | `tqdm` | Single-dependency progress bars |

---

## Project Structure

```
multimedia-library-search/
├── requirements.txt
├── cli.py                   # CLI entry point
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI app, mounts routes + templates
│   ├── config.py            # Settings (paths, model name, thresholds)
│   ├── database.py          # SQLite connection + table initialisation
│   ├── chroma.py            # ChromaDB client + collection setup
│   ├── indexer.py           # Frame extraction + face detection pipeline
│   ├── clusterer.py         # DBSCAN clustering of face embeddings
│   ├── compiler.py          # Timestamp merging + FFmpeg highlight reel
│   └── api/
│       ├── index.py         # POST /api/index, GET /api/index/status
│       ├── persons.py       # GET/POST /api/persons — label + merge
│       ├── search.py        # GET /api/search, POST /api/search/photo
│       ├── compile.py       # POST /api/compile, GET /api/compile/{job_id}
│       └── video.py         # GET /api/video/{video_id} (range-aware serve)
├── templates/
│   ├── base.html            # Nav + shared layout
│   ├── label.html           # Phase 2: person cluster labeling grid
│   └── search.html          # Phase 3: search form + results
└── static/                  # Served at /static
    └── thumbnails/          # Face crop PNGs — gitignored, created at runtime

# Gitignored (created automatically on first run):
# data/          — SQLite + ChromaDB files
# static/thumbnails/ — face crop PNGs
# output/        — compiled highlight reels
```

---

## Data Models

### SQLite

```sql
CREATE TABLE IF NOT EXISTS videos (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    path         TEXT UNIQUE NOT NULL,
    filename     TEXT NOT NULL,
    duration_sec REAL,
    indexed_at   TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS persons (
    id             TEXT PRIMARY KEY,   -- UUID4
    name           TEXT,               -- NULL until labeled by user
    thumbnail_path TEXT,               -- representative face crop path
    face_count     INTEGER DEFAULT 0,
    created_at     TEXT DEFAULT (datetime('now'))
);
```

### ChromaDB — `faces` collection (cosine distance)

Each document represents one detected face instance.

```
id:        "<video_filename>_<timestamp_sec>_<face_index>"
embedding: [512 floats]  -- InsightFace normed_embedding (ArcFace)
metadata:
  video_id       int    -- FK → videos.id
  timestamp_sec  float
  person_id      str    -- person UUID once clustered; "unlabeled" otherwise
  thumbnail_path str    -- relative path to saved face crop PNG
```

> **ChromaDB null workaround:** ChromaDB has no native `IS NULL` filter. The sentinel
> value `"unlabeled"` is used for `person_id` until a face is assigned to a cluster.
> This is consistent and fully queryable via `{"$eq": "unlabeled"}`.

**Key query patterns:**

```python
# All faces for a named person
collection.get(where={"person_id": person_uuid})

# All unassigned faces
collection.get(where={"person_id": {"$eq": "unlabeled"}})

# Similarity search — photo search
collection.query(query_embeddings=[embedding], n_results=30)

# Similarity search scoped to one person — verification / re-labeling
collection.query(query_embeddings=[embedding], where={"person_id": person_uuid}, n_results=10)
```

---

## Phase 1 — Indexing Pipeline

**Objective:** Build the foundational pipeline to extract faces from videos, store embeddings, and verify the index via CLI.

**CLI:** `python cli.py index <directory> [--interval 1.0] [--gpu]`

### Startup checks

Before any indexing begins, verify external dependencies are available:

```python
import shutil, sys

def check_dependencies():
    if shutil.which("ffmpeg") is None:
        sys.exit("Error: ffmpeg not found. Install it and ensure it is on your PATH.")
```

Call this at the top of `cli.py` before any command runs, not only before compile — FFmpeg is needed in Phase 4 and an early clear error is better than a confusing failure later.

### `indexer.py` — VideoIndexer

1. Walk directory for `.mp4 .avi .mov .mkv` files
2. For each video:
   - Check SQLite `videos` table by path → skip if already indexed
   - Open with `cv2.VideoCapture`, read FPS and total frame count
   - Extract one frame every `interval` seconds
   - Run `FaceAnalysis.get(frame)` → list of detected faces
   - For each face: crop thumbnail → save to `static/thumbnails/` → add to ChromaDB with `person_id="unlabeled"`
   - Insert row into `videos` table on completion
   - Update shared progress state after each video and each frame

### Progress state

A module-level dict in `indexer.py` is updated as indexing runs and read by the API status endpoint — shared between CLI and web UI without any extra machinery:

```python
index_progress = {
    "status": "idle",        # idle | running | done | error
    "videos_total": 0,
    "videos_done": 0,
    "current_video": "",
    "faces_found": 0,
    "started_at": None,
    "eta_sec": None,         # recomputed after each video completes
    "error": None,
}
```

ETA after each video:
```python
elapsed = time.time() - index_progress["started_at"]
rate    = videos_done / elapsed          # videos per second
eta_sec = (videos_total - videos_done) / rate if rate > 0 else None
```

### CLI progress output

Two-level `tqdm` bars so the user sees both overall progress and per-video frame progress:

```
Indexing: 3/12 [████████░░░░░░░░] 25% • vacation.mp4 • 47 faces found • ETA 08:32
  Frames: 450/1800 [████░░░░░░░░░░░░] 25%
```

```python
with tqdm(total=len(videos), desc="Indexing", unit="video") as outer:
    for video_path in videos:
        outer.set_postfix(file=video_path.name, faces=total_faces)
        with tqdm(total=frame_count, desc="  Frames", unit="frame", leave=False) as inner:
            for frame, timestamp in extract_frames(video_path):
                # ... detect, store ...
                inner.update(1)
        outer.update(1)
```

On completion, print a one-line summary:
```
Done. 12 videos | 3,847 faces | 00:12:34
```

### InsightFace initialisation

```python
from insightface.app import FaceAnalysis

fa = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
fa.prepare(ctx_id=0, det_size=(640, 640))
```

GPU is used automatically when available; falls back to CPU transparently.

### API routes (`api/index.py`)

```
POST /api/index
     body: {directory_path, interval_sec=1.0}
     → starts indexing in a background thread, returns immediately

GET  /api/index/status
     → returns the current index_progress dict plus a computed progress float
```

Status response shape:
```json
{
  "status": "running",
  "videos_done": 3,
  "videos_total": 12,
  "current_video": "vacation.mp4",
  "faces_found": 247,
  "eta_sec": 512,
  "progress": 0.25
}
```

### Web UI progress (available once `serve` is running)

Even before the full labeling UI is built in Phase 2, the index page at `/` provides:

- A text input for the video directory path + "Start Indexing" button
- On submit: htmx `POST /api/index`, reveals a progress panel
- htmx polls `GET /api/index/status` every 2 seconds while `status == "running"`
- Displays: progress bar (`videos_done / videos_total`), currently processing filename, total faces found so far, estimated time remaining formatted as `mm:ss`
- On `status == "done"`: shows summary and a "Go to Label →" link

### Key config values (`config.py`)

```python
KEYFRAME_INTERVAL_SEC = 1.0   # frames to extract per second of video
THUMBNAIL_SIZE        = (128, 128)
FACE_DET_SIZE         = (640, 640)
```

**Verification:** Index a small test folder (~5 min of video). Run `python cli.py stats` — confirm video count and face count are non-zero. Re-run indexer on the same folder — confirm no duplicates are added.

---

## Phase 2 — Clustering & Labeling UI

**Objective:** Group detected faces into person identities; provide a web UI to assign names.

**CLI:** `python cli.py cluster [--eps 0.4] [--min-samples 3]`

### `clusterer.py` — FaceClusterer

1. Fetch all face embeddings from ChromaDB: `collection.get(include=["embeddings", "metadatas"])`
2. Stack into a numpy array
3. Run `DBSCAN(eps=0.4, min_samples=3, metric="cosine")`
   - `eps=0.4` is a reasonable starting point for ArcFace cosine distance; expose as a CLI flag for tuning
4. For each cluster (label ≥ 0):
   - Generate a UUID for the person
   - Select the medoid face (closest embedding to cluster mean) as representative thumbnail
   - Insert row into SQLite `persons` table
   - Update `person_id` in ChromaDB for all faces in the cluster
5. Noise points (label = -1) remain `person_id = "unlabeled"`

### API routes (`api/persons.py`)

```
GET  /api/persons                    → list all persons (id, name, thumbnail, face_count)
POST /api/persons/{id}/label         → body: {name: "Alice"} → update name in SQLite
POST /api/persons/merge              → body: {source_id, target_id} → reassign source faces to target, delete source
```

### Web UI — `/label`

- Jinja2 template: one card per person, ordered by face_count descending
- Each card: representative thumbnail + 4-face sample grid + appearance count
- Inline name input + Save button (htmx `hx-post`, swaps card in place — no page reload)
- Merge: checkbox-select two cards → "Merge selected" button
- Unlabeled persons display as "Unknown #1", "Unknown #2" etc.

**Verification:** Index a folder containing known people. Run cluster. Open `/label` — confirm distinct people appear as separate clusters. Assign a name — confirm it persists after restarting the server.

### Future: Label Preservation Across Full Re-clustering

Currently, re-running `cluster` deletes all persons and regenerates UUIDs, which erases any names the user has assigned. A future pass should preserve labels:

1. Before wiping `persons`, snapshot `{old_person_id → name}` from SQLite.
2. After building new clusters, for each new cluster compare its face IDs against the old `person_id` assignments stored in ChromaDB (pre-reset).
3. Find the old `person_id` that contributed the majority of faces to the new cluster (argmax overlap).
4. If that old person had a name and the overlap fraction exceeds a threshold (e.g. > 50%), carry the name forward to the new person record.
5. This handles splits conservatively (only the dominant inheritor gets the name) and merges naturally (the target already has the name; the source's name is discarded with a warning).

This is not built in Phase 2 because the typical workflow is: tune eps → cluster → label (one-shot). The incremental clustering approach below makes full re-clusters rare, reducing this friction further.

---

## Phase 2.5 — Progressive (Incremental) Clustering

**Objective:** As new videos are indexed, assign new faces to existing person clusters without re-running full DBSCAN. Labels survive across all incremental runs. A full re-cluster remains available for initial setup or restructuring.

**CLI:**
```
python cli.py cluster              # full DBSCAN — initial setup or restructure
python cli.py cluster --incremental  # assign new faces only, discover new persons
```

After indexing, `index` automatically triggers `cluster --incremental` so new faces appear in `/label` without a manual step. Pass `--no-cluster` to skip.

### Data model changes

Add `centroid TEXT` column to the `persons` SQLite table — JSON-serialised float array (512 floats, same dimension as InsightFace normed embedding). Stored after every full cluster run and updated in-place as new faces are assigned incrementally.

```sql
ALTER TABLE persons ADD COLUMN centroid TEXT;
```

### Full cluster (existing behaviour, extended)

After DBSCAN completes, compute and store the centroid for each new person:

```python
centroid = cluster_embeddings.mean(axis=0)
centroid /= np.linalg.norm(centroid)   # re-normalise after averaging
persons_row["centroid"] = json.dumps(centroid.tolist())
```

### `run_incremental_clusterer()` — step by step

1. Load all person centroids from SQLite → stack into matrix `C` (shape: `[P, 512]`)
2. If `C` is empty (no persons yet), fall back to full cluster automatically
3. Fetch all `unlabeled` face embeddings from ChromaDB (new faces since last run)
4. If none, exit early — nothing to do
5. For each new face embedding `f`:
   - Compute euclidean distance to every centroid: `dists = ||C - f||`
   - `best_person, best_dist = argmin(dists)`
   - If `best_dist < eps` → assign to `best_person`:
     - Update `person_id` in ChromaDB
     - Update centroid (incremental mean): `new_c = (old_c * N + f) / (N + 1)`, then re-normalise
     - Increment `face_count` and persist updated centroid to SQLite
   - Else → add to `unmatched` list
6. Run DBSCAN only on `unmatched` embeddings → new clusters become new persons (with UUIDs, medoids, samples, centroids stored as in the full cluster path)
7. All label updates are purely additive — no existing person is deleted or renamed

### Centroid update formula

```python
new_centroid = (old_centroid * face_count + new_embedding) / (face_count + 1)
new_centroid /= np.linalg.norm(new_centroid)
```

Batch all SQLite centroid writes at the end of the run (not per-face) to avoid thrashing.

### Merge update

When two persons are merged, recompute the target's centroid as a weighted average:

```python
merged_centroid = (c_target * n_target + c_source * n_source) / (n_target + n_source)
merged_centroid /= np.linalg.norm(merged_centroid)
```

### Auto-trigger after index

In `run_indexer()`, after the final summary line:

```python
if auto_cluster and newly_indexed > 0:
    from app.clusterer import run_incremental_clusterer
    run_incremental_clusterer(eps=eps)
```

`auto_cluster=True` by default; `--no-cluster` on the `index` command sets it to `False`.

### Future: Re-sync Centroids

A lightweight middle ground between incremental (which drifts) and full re-cluster (which loses labels). Re-sync recomputes each person's centroid from their actual assigned faces without changing any `person_id` assignments:

1. For each person in SQLite, query ChromaDB: `collection.get(where={"person_id": pid}, include=["embeddings"])`
2. Compute mean of all embeddings → L2-normalise → write back to `persons.centroid`
3. No person is created, deleted, or renamed — purely a centroid refresh

This corrects drift that accumulates from many incremental runs (where each new face nudges the running-mean centroid slightly). Can be exposed as `python cli.py cluster --resync` and a "Re-sync Centroids" button on `/label`. Run it whenever the incremental clusterer starts producing unexpected new-person false positives.

### Known limitations

- **Centroid drift**: repeated incremental runs shift centroids as running means accumulate. A person's centroid after 1000 faces may differ noticeably from after 10 faces. Periodic full re-cluster (user-triggered) corrects this.
- **No retrospective re-assignment**: incremental never moves already-assigned faces. Mis-assignments from early runs require a manual merge in `/label` or a full re-cluster.
- **New-person false positives**: if a known person appears in an atypical pose/lighting that lands outside `eps` from their centroid, a new cluster is created and the user must merge it in `/label`. This is the same manual-merge UX as today, but the exception rather than the norm.

---

## Phase 3 — Search

**Objective:** Search by name or reference photo; play results in-browser.

**CLI:** `python cli.py serve` then open `http://localhost:8000/search`

### API routes (`api/search.py`)

```
GET /api/search?name=Alice
```
1. Look up person UUID by name in SQLite `persons` table
2. `collection.get(where={"person_id": person_uuid}, include=["metadatas"])`
3. Group by `video_id`, sort by `timestamp_sec`
4. Return `[{video_id, filename, timestamp_sec, thumbnail_path}]`

```
POST /api/search/photo   (multipart/form-data: file)
```
1. Decode uploaded image with OpenCV
2. Run InsightFace → get embedding of first detected face
3. `collection.query(query_embeddings=[embedding], n_results=30)`
4. Filter results by cosine distance threshold (< 0.5); return ranked list

### Video serving (`api/video.py`)

```
GET /api/video/{video_id}
```

Serves the source video file from disk using `StreamingResponse` with HTTP range request support. Range requests are required for the browser `<video>` element to seek to an arbitrary timestamp.

### Web UI — `/search`

- Text input → GET search by name (htmx swaps results list)
- File upload → POST search by photo (htmx swaps results list)
- Results: thumbnail | filename | timestamp | Play button
- Play opens a modal: `<video src="/api/video/{id}#t={timestamp}" controls autoplay>`

**Verification:** Label a person in Phase 2. Search by their name — confirm results include correct videos and timestamps. Click Play — confirm video opens at the right moment.

### Phase 3 implementation notes (actual vs. plan)

- Name search uses `LOWER(name) LIKE LOWER(?)` for case-insensitive partial match; all matching persons are combined into one result set.
- Results are deduplicated server-side to one hit per (video_id, minute) before returning, capping at 200 total — avoids flooding when a person appears every second in a long video.
- Client-side grouping: the flat API list is grouped by `video_id` in JS. Name search sorts groups by most appearances first; photo search sorts by best (lowest) cosine distance first.
- Each result card shows three evenly-sampled video frames (not face crops) via `GET /api/frame/{id}?t=` — on-demand OpenCV extraction, LRU-cached for 500 frames per server session.
- Custom video player replaces native controls: play/pause, click-to-seek scrubber that expands on hover, yellow marker lines at each detected timestamp, tooltip on marker hover, `Space`/`←`/`→`/`Esc` keyboard shortcuts, fullscreen.
- Markers are placed after `loadedmetadata` using `(ts / duration) * 100%` positioning. All timestamps for the active video are stored in a module-level `_activeTimestamps` array and re-placed whenever the video changes.
- `videoEl.removeAttribute('src'); videoEl.load()` is used on modal close (not `src = ''`) to fully reset the element state across browsers.
- InsightFace for photo search is lazy-initialised on first request (module-level singleton); uses `MODEL_NAME_DEFAULT` (`buffalo_sc`) to match the default indexing model.

---

## Phase 4 — Highlight Reel Compilation

**Objective:** Compile all clips for a person into a single downloadable MP4.

### Timestamp merging (before compilation)

Raw search results are per-frame detections. Before cutting clips, consolidate nearby timestamps from the same video into continuous segments:

```python
def merge_timestamps(timestamps, padding_sec=2.0, merge_gap_sec=5.0):
    """
    timestamps: sorted list of floats (seconds)
    Returns: list of (start_sec, end_sec) tuples
    """
    segments = []
    for t in timestamps:
        start, end = t - padding_sec, t + padding_sec
        if segments and start <= segments[-1][1]:
            segments[-1] = (segments[-1][0], max(segments[-1][1], end))
        else:
            segments.append((start, end))
    return segments
```

Example: detections at 0:10, 0:11, 0:12 with 2 sec padding and 5 sec merge gap → one clip from 0:08 to 0:14 instead of three overlapping clips.

### `compiler.py` — HighlightCompiler

Input: `[{video_path, start_sec, end_sec}]` (already merged)

```
1. For each segment:
   ffmpeg -ss {start} -to {end} -i {input} -c copy {tmp_clip_N}.mp4

2. Write concat list:
   file '/tmp/clip_001.mp4'
   file '/tmp/clip_002.mp4'

3. Concatenate:
   ffmpeg -f concat -safe 0 -i concat_list.txt -c copy {output}.mp4

4. Delete temp clips
```

`-c copy` is lossless and fast — no re-encoding. Output: `output/{name}_{job_id[:8]}.mp4`

### API routes (`api/compile.py`)

```
POST /api/compile
     body: {person_id, padding_sec=2, merge_gap_sec=5}
     → launches background thread, returns {job_id}

GET  /api/compile/{job_id}
     → returns {status: "running"|"done"|"error", progress: 0.0–1.0, download_url}
```

Job state held in an in-memory dict — sufficient for a single-user local tool.

### Web UI

- "Create Highlight Reel" button on `/search` results page
- htmx polls `GET /api/compile/{job_id}` every second; renders a progress bar
- "Download" link appears when `status == "done"`

**Verification:** Search for a person who appears across multiple videos. Request a highlight reel. Confirm the downloaded MP4 contains the correct clips in chronological order, with nearby appearances merged into single continuous segments.

---

## Phase 5 — Full Body Re-ID (Future)

Not built in Phases 1–4. Design notes for when this is scoped:

- **Person detector:** YOLOv8n (ultralytics) — fast, CPU-capable, detects person bounding boxes per frame
- **Re-ID model:** OSNet (`torchreid`) — 512-dim body appearance embedding, trained on person Re-ID datasets
- **Storage:** new ChromaDB collection `body_embeddings`, same metadata schema as `faces`
- **Indexer change:** after face detection, also run YOLOv8 → crop persons → OSNet embedding → store in `body_embeddings`
- **Search fusion:** `score = 0.6 * face_similarity + 0.4 * body_similarity`; body-only if no face detected in frame
- **UI:** "Full body matching" toggle on `/search`; enrolls body embeddings from existing labeled faces automatically

Phases 1–4 are unaffected — this is purely additive.

---

## CLI Reference

```
python cli.py index   <directory>  [--interval 1.0] [--gpu]
python cli.py cluster              [--eps 0.4] [--min-samples 3]
python cli.py serve                [--host 0.0.0.0] [--port 8000]
python cli.py stats
python cli.py prune                [--dry-run]
```

All web UI and API available at `http://localhost:8000` after `serve`.

### `prune` command

Removes stale data for videos that no longer exist on disk. Scope covers all three stores together — a partial prune that only cleans thumbnails but leaves ChromaDB or SQLite stale would cause inconsistency:

1. Load all rows from SQLite `videos` table
2. For each row where `path` no longer exists on disk:
   - Delete all ChromaDB face records with that `video_id`
   - Delete all thumbnail PNGs referenced by those face records
   - Delete the SQLite `videos` row
   - If all faces for a `person_id` are gone, delete that person from SQLite `persons`
3. Delete any PNG files in `static/thumbnails/` not referenced by any ChromaDB record (orphan cleanup)
4. `--dry-run` prints what would be deleted without making changes

---

## Requirements

File lives at `requirements.txt` in the project root:

```bash
pip install -r requirements.txt
```

```
fastapi
uvicorn[standard]
jinja2
python-multipart
insightface
onnxruntime              # CPU-only. NVIDIA GPU users: replace with onnxruntime-gpu
                         # for significantly faster indexing (pip install onnxruntime-gpu)
chromadb
opencv-python-headless
numpy
scikit-learn
tqdm
```

No React, no Node, no Docker required.

---

## Key Design Decisions

| Decision | Rationale |
|---|---|
| Single FastAPI process for API + UI | No CORS config, no separate dev server, one command to run |
| SQLite stdlib, no ORM | Zero extra dependencies; schema is simple enough for raw SQL |
| FFmpeg via `subprocess` | More control over flags than `ffmpeg-python`; `-c copy` is lossless and fast |
| ChromaDB sentinel `"unlabeled"` | ChromaDB has no native null filter; sentinel is explicit and queryable |
| DBSCAN over k-means | Number of persons in a library is unknown; DBSCAN discovers it automatically |
| htmx over React | No build step; server-rendered pages with targeted in-place updates |
| Thumbnails on disk, path in metadata | ChromaDB is not a blob store; disk is the right place for image files |
| Timestamp merging before compilation | Prevents duplicate/overlapping clips in the reel; produces cleaner output |
| `prune` spans SQLite + ChromaDB + disk | Cleaning only thumbnails leaves the DB stale; all three stores must stay consistent |
| Progress state as a module-level dict | Simplest shared state between the indexer thread and the API status endpoint; no queue or IPC needed for a single-user tool |
