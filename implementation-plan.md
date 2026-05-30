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
| EXIF + image composition | Pillow | EXIF `DateTimeOriginal` extraction; photo collage grid rendering |

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
    recorded_at  TEXT,                 -- ISO 8601; from ffprobe, filename, or mtime
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

`recorded_at` is populated during indexing via `_extract_recording_date()` in `indexer.py`, which tries in order: ffprobe `creation_time` container tag → filename date patterns (`YYYY-MM-DD HH-MM-SS`, `YYYYMMDD_HHMMSS`) → file mtime. For existing videos indexed before this column was added, run `python cli.py backfill-dates`.

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

### Scene-based timestamp merging

Raw ChromaDB detections are one per indexed frame (typically 1/sec). Naively padding and concatenating each timestamp would produce a reel as long as the source videos for someone present throughout an entire recording. Instead, the compiler uses a scene-based approach:

```python
def merge_timestamps(timestamps, merge_gap_sec=30.0):
    """Group timestamps into (scene_start, scene_end) tuples separated by merge_gap_sec."""
    segments = []
    for t in sorted(timestamps):
        if segments and t <= segments[-1][1] + merge_gap_sec:
            segments[-1] = (segments[-1][0], t)   # extend current scene
        else:
            segments.append((t, t))                # new scene
    return segments
```

For each scene, a single short clip of `clip_duration_sec` is cut, centered on the scene midpoint. A 9-minute continuous-presence scene becomes one 30-second representative clip, not 9 minutes of footage.

### `compiler.py` — `run_compile()`

```
1. Fetch all face detections for person_id from ChromaDB
2. Group by video; merge timestamps into scenes per video
3. If scenes > max_clips_per_video, evenly sample down to the cap
4. For each chosen scene: mid = (start+end)/2; clip = [mid-half, mid+half]
5. ffmpeg -ss {start} -to {end} -i {input} -c copy {tmp_clip_N}.mp4
   (fallback: re-encode with libx264 if -c copy fails)
6. Write concat list → ffmpeg concat -c copy -movflags +faststart {output}.mp4
```

`-c copy` avoids re-encoding for speed. `-movflags +faststart` moves the MP4 index to the front for streaming. Output: `output/{safe_name}_{job_id[:8]}.mp4`.

### Clip ordering

Clips are ordered before concatenation by the `order` parameter:

| Value | Behaviour |
|---|---|
| `asc` (default) | Sort by `recorded_at`, earliest first |
| `desc` | Sort by `recorded_at`, latest first |
| `random` | `random.shuffle()` |

`recorded_at` falls back to `indexed_at` for videos where the recording date could not be determined.

### API routes (`api/compile.py`)

```
POST /api/compile
     body: {
       person_id,
       clip_duration_sec=30,
       merge_gap_sec=30.0,
       max_clips_per_video=5,
       order="asc"
     }
     → launches background thread, returns {job_id}

GET  /api/compile/{job_id}
     → {status, progress, segments_total, segments_done, error, filename}

GET  /api/compile/{job_id}/download
     → FileResponse (video/mp4) when status == "done"
```

Job state held in a module-level dict — sufficient for a single-user local tool.

### Web UI

The Highlight Reel panel appears on `/search` whenever named persons are in the results. It exposes all four compile parameters with plain-language labels, a clip order dropdown (Earliest first / Latest first / Random), a progress bar that polls every second, and a download link on completion.

### Phase 4 implementation notes (actual vs. plan)

- Scene-based clipping avoids jumbo reels. A person present throughout a full video produces one short representative clip, not a copy of the whole video.
- `max_clips_per_video` uses evenly-spaced sampling (not head/tail) so the full chronological spread of a video is represented when scenes are subsampled.
- FFmpeg fast-seek (`-ss` before `-i`) is used for performance; `-c copy` for losslessness. A re-encode fallback (`libx264`/`aac`) handles codec or container mismatches that prevent stream copy.
- Client-side search results mirror the scene logic: consecutive per-minute detection hits are merged into scene chips. A brief appearance shows as `1:23`; a continuous stretch shows as `0:00–5:52`. The displayed end is extended to video duration when the last detection is within 90 s of the real end (compensating for per-minute server deduplication).
- Recording date extraction priority: ffprobe `creation_time` container tag → filename date pattern (`YYYY-MM-DD HH-MM-SS` or `YYYYMMDD_HHMMSS`) → file mtime. `python cli.py backfill-dates` populates `recorded_at` for already-indexed videos without re-indexing.

---

## Phase 5 — Photo Library Extension

**Objective:** Index photos alongside videos; show photo results in a dedicated tab on `/search`; generate downloadable photo collages.

---

### Data Model

**New `photos` table (SQLite)**

```sql
CREATE TABLE IF NOT EXISTS photos (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    path       TEXT UNIQUE NOT NULL,
    filename   TEXT NOT NULL,
    taken_at   TEXT,          -- ISO 8601; from EXIF DateTimeOriginal, fallback to mtime
    indexed_at TEXT DEFAULT (datetime('now'))
);
```

**ChromaDB `faces` collection — new metadata fields on photo face records**

```
media_type  str   -- "photo" (absent on existing video records = treated as "video")
photo_id    int   -- FK → photos.id; set only when media_type == "photo"
```

Old video records require no migration. Absence of `media_type` is treated as `"video"` everywhere.

---

### `app/photo_indexer.py` — PhotoIndexer

1. Walk directory for `.jpg .jpeg .png .heic .heif .webp` files
2. Check `photos` table by path → skip if already indexed
3. Load with `cv2.imread`; for HEIC files fall back to Pillow (`Image.open` → `np.array`)
4. Run `FaceAnalysis.get(frame)` → same pipeline as VideoIndexer
5. For each face: crop 128×128 thumbnail → save to `static/thumbnails/` → upsert to ChromaDB with `media_type="photo"`, `photo_id=N`
6. Insert row into `photos` table on completion
7. Extract `taken_at`: Pillow `Image.open()._getexif()` tag 36867 (`DateTimeOriginal`) → parse `"%Y:%m:%d %H:%M:%S"` → ISO 8601; fallback to `os.path.getmtime()`

Progress state: extend the existing `index_progress` dict in `indexer.py` with `photos_total` and `photos_done` fields alongside the existing video fields.

### Thumbnail cleanup (no changes required)

Photo face thumbnails follow the exact same lifecycle as video face thumbnails:

- **At index time:** each detected face produces a 128×128 PNG in `static/thumbnails/` — identical to the video path.
- **After cluster:** `trim_thumbnails()` runs automatically and removes all but the representative + 4 sample crops per person. It operates on the `persons` table and `static/thumbnails/` directory regardless of the source media type — no modifications needed.
- **On prune:** `prune_stale_media()` (renamed from `prune_stale_videos()` in Phase 5) checks both the `videos` and `photos` tables against disk, removes stale SQLite rows, their ChromaDB face records, and their thumbnail files in a single pass.

### Clustering and labeling (no changes required)

All clustering and labeling paths are media-type-agnostic and work unchanged for photo faces:

**Full cluster (`python cli.py cluster`)** — fetches all embeddings from ChromaDB via `collection.get(include=["embeddings", "metadatas"])`. The query has no `media_type` filter; photo face embeddings are included automatically. DBSCAN groups them by similarity regardless of source. A cluster may contain crops from videos, photos, or both — that is correct and expected.

**Incremental cluster (`python cli.py cluster --incremental`)** — fetches all `unlabeled` embeddings and matches against person centroids. Works identically for photo faces; the embedding vector is identical in format (512-dim ArcFace normed embedding from InsightFace). Auto-trigger fires after photo indexing too — `run_photo_indexer()` calls `run_incremental_clusterer(eps=eps)` using the same logic as `run_indexer()`.

**Centroid updates** — weighted-average formula is embedding-only. Whether the face came from a video frame or a photo is irrelevant; the math is the same.

**Re-sync (`python cli.py cluster --resync`)** — recomputes centroids from all assigned embeddings in ChromaDB, which now includes photo faces. No changes needed.

**Merge** — operates at the person level, reassigning all ChromaDB face records for `source_id` to `target_id` regardless of `media_type`. Centroid is recomputed as a weighted average of all face embeddings. No changes needed.

**Label preservation across full re-cluster** — works by comparing face ID overlap between old and new clusters. Face IDs for photos follow the same format as videos (`{safe_filename}_{face_idx}`); the preservation logic does not inspect media type.

**`/label` UI — no changes**

Photo face crops are 128×128 PNGs in `static/thumbnails/` — visually identical to video face crops. The label page shows them in the same sample grid. `face_count` on each person card reflects faces from both videos and photos as a single number; no source breakdown is shown. The point of `/label` is identifying *who*, not *where they appeared* — that distinction belongs on `/search`. Adding a "from photo / from video" indicator to `/label` would add noise without helping the user assign names.

A person labeled in `/label` automatically surfaces in both the Videos tab and the Photos tab on `/search` — this is the intended behavior.

---

### CLI changes

`python cli.py index <directory>` processes both video and photo files in a single pass:
- Videos: `.mp4 .avi .mov .mkv` (unchanged)
- Photos: `.jpg .jpeg .png .heic .heif .webp` (new)

New flags:
- `--videos-only` — skip photo files
- `--photos-only` — skip video files

Incremental clustering auto-triggers after index as before; photo faces are assigned to persons via the same DBSCAN + centroid-match pipeline.

---

### API routes

**Photo serving (`api/photo.py`)**

```
GET /api/photo/{photo_id}
```

Simple `FileResponse` returning the source image. No range requests needed.

**Search API updates (`api/search.py`)**

Both name search and photo-upload search already query the shared `faces` collection. The only change is splitting the response by `media_type`:

```
GET /api/search?name=Alice
POST /api/search/photo   (multipart/form-data: file)
```

Response shape (both endpoints):
```json
{
  "videos": [
    {"media_type": "video", "video_id": 1, "filename": "...", "timestamp_sec": 12.5, "thumbnail_path": "..."}
  ],
  "photos": [
    {"media_type": "photo", "photo_id": 7, "filename": "...", "taken_at": "2024-08-14T15:30:00", "thumbnail_path": "..."}
  ]
}
```

Photo results are deduplicated by `photo_id` (a person may have multiple faces detected in one photo).

**Photo collage (`api/collage.py`)**

```
POST /api/collage
     body: {
       person_id,
       columns=3,            -- 2 | 3 | 4
       sort="asc",           -- asc | desc | random
       captions=true         -- draw taken_at date under each cell
     }
     → {job_id}

GET  /api/collage/{job_id}
     → {status, progress, photos_total, photos_done, error, filename}

GET  /api/collage/{job_id}/download
     → FileResponse (image/png) when status == "done"
```

Job state held in a module-level dict in `collage.py` — same pattern as compile jobs.

---

### `app/collage.py` — `run_collage()`

```
1. Fetch all photo face records for person_id from ChromaDB where media_type == "photo"
2. Deduplicate by photo_id — one cell per source photo
3. Load taken_at from SQLite photos table; sort per order parameter
4. For each photo: load with Pillow, center-crop to square, resize to COLLAGE_CELL_SIZE (400px)
5. Compose grid: ceil(n / columns) rows × columns columns on a white canvas with COLLAGE_PADDING (8px) gutters
6. If captions=true: draw taken_at date string below each cell using ImageDraw (small font, dark grey)
7. Save as PNG to output/{safe_name}_{job_id[:8]}.png
```

Constants in `config.py`:
```python
COLLAGE_CELL_SIZE = 400   # px — each photo cell is resized to this square
COLLAGE_PADDING   = 8     # px — gutter between cells and canvas border
```

---

### Web UI — `/search` updates

**Two tabs on the results section:**

```
[Videos (14)]  [Photos (31)]
```

Default active tab: whichever count is higher. Tabs swap the result list in place via htmx.

**Video tab** — unchanged from Phase 3/4 (scene chips, custom player, highlight reel panel).

**Photo tab:**
- Each result card: square preview thumbnail + filename + `taken_at` date + "View" button
- "View" opens a lightbox modal: full-size `<img src="/api/photo/{id}">`, `←` / `→` keyboard navigation between photo results, `Esc` to close
- `imgEl.src = ''` on close to release memory (same pattern as video player reset)

**Photo Collage panel** — appears below photo results when a named person is in scope (mirrors Highlight Reel panel):
- Columns selector: 2 / 3 / 4 (default 3)
- Sort dropdown: Earliest first / Latest first / Random
- Captions toggle: on/off
- "Create Collage" button → `POST /api/collage` → reveals progress bar polling every second
- On completion: "Download Collage" link

---

### Project structure additions

```
app/
├── photo_indexer.py    # PhotoIndexer — EXIF extraction + face detection for images
├── collage.py          # run_collage() + job state dict
└── api/
    ├── photo.py        # GET /api/photo/{photo_id}
    └── collage.py      # POST /api/collage, GET /api/collage/{job_id}[/download]
```

---

### Verification

- `python cli.py index /my/photos` processes image files, skips re-runs, triggers incremental cluster
- `/search?name=Alice` returns a Photos tab with photo cards dated correctly from EXIF
- "View" lightbox opens full photo; keyboard navigation cycles through results
- "Create Collage" runs to completion; downloaded PNG shows a correctly laid-out grid with dates
- Delete a source photo from disk; re-run `index` — `prune_stale_media()` removes its SQLite row, ChromaDB face records, and thumbnail crops
- Run `cluster` after indexing photos — `trim_thumbnails()` reduces `static/thumbnails/` to representative + sample crops only, same as after video indexing
- Index a mixed folder (videos + photos), run `cluster` — the same person appears on one `/label` card regardless of whether their faces came from videos, photos, or both
- Assign a name in `/label` — that name returns results in both the Videos tab and Photos tab on `/search`
- Run `cluster --incremental` after adding new photos — new photo faces are assigned to existing persons without disrupting labels

---

## Phase 6 — LLM Tool Layer & MCP Server

**Objective:** Expose the offline multimedia face-search app to AI agents through a clean HTTP API boundary and a thin MCP server. FastAPI remains the business-logic runtime and system of record; MCP becomes an independently runnable local client that calls stable `/api/*` endpoints and returns machine-friendly JSON.

### Architecture

```text
MCP client
  -> MCP server / tool package
  -> HTTP API
  -> FastAPI services
  -> SQLite + ChromaDB
```

MCP runs as a separate process and calls FastAPI over HTTP. This preserves the API boundary, keeps FastAPI as the single source of business behavior, avoids shared in-process singleton assumptions, and makes future auth or remote deployment simpler. The tradeoff: FastAPI must be running before any MCP tool can execute.

### Service extraction (`app/services/`)

Extract search logic from route functions into thin service modules. Route handlers become wrappers that call these services — no behavior change, no new API contract.

- `app/services/search_service.py`: `search_person_by_name(...)` and `search_by_reference_image_bytes(...)`
- `app/services/person_service.py`: list/get person helpers

### New API endpoints

Add the read-oriented endpoints the MCP layer needs. All other existing routes remain unchanged.

```
GET /api/health
    → {"status": "ok", "sqlite": true, "chromadb": true}

GET /api/stats
    → {"videos": N, "photos": N, "faces": N, "persons": N, "labeled_persons": N}

GET /api/persons/{person_id}
    → {id, name, thumbnail_path, face_count, samples: [...]}
      name: null for unnamed clusters; 404 only when person_id does not exist
      samples parsed from JSON string in SQLite persons.samples column

GET /api/video/{video_id}/info
GET /api/photo/{photo_id}/info
    → JSON metadata + local API access paths for that media item
```

Do not add MCP endpoints for label, merge, re-cluster, prune, or force-index in this phase.

### MCP server package (`mcp_server/`)

```
mcp_server/
├── server.py          # FastMCP app + tool registration
├── config.py          # pydantic-settings config
├── client.py          # httpx.AsyncClient wrapper around FastAPI
├── logging_config.py  # structured stderr logging
├── models/            # Pydantic input/output models per tool
└── tools/             # one module per tool group
```

Use `mcp.server.fastmcp.FastMCP`. Default transport is stdio for local agent clients. Optional Streamable HTTP transport for network-accessible deployments — verify transport name against the installed MCP SDK before documenting.

Log structured events to stderr. MCP stdio stdout must carry only protocol frames.

### MCP tools

| Tool | Description |
|---|---|
| `health_check()` | Verify FastAPI is reachable and storage is initialized |
| `get_library_stats()` | Return indexed video/photo/face/person counts |
| `list_people(include_unnamed, limit, name_query)` | List person clusters; `include_unnamed` controls `name IS NULL` rows |
| `get_person(person_id)` | Fetch person record with thumbnail and sample paths |
| `search_by_name(name, limit)` | Name search; returns `{videos, photos}`; `limit` applies per media type |
| `search_by_photo(image_path, limit, distance_threshold)` | Read file from disk, POST to `/api/search/photo`, then apply tool-side per-media-type limit and distance filtering unless the API is extended to accept those parameters |
| `get_media_info(video_id, photo_id)` | Metadata + API access paths for exactly one video or photo; reject calls that pass neither or both IDs |
| `compile_highlight_reel(person_id, clip_duration_sec, merge_gap_sec, max_clips_per_video, order)` | Start compile job; return job_id |
| `check_compile_status(job_id)` | Poll job; include absolute `download_url` when done |
| `create_photo_collage(person_id, columns, sort, captions)` | Start collage job; return job_id |
| `check_collage_status(job_id)` | Poll job; include absolute `download_url` when done |

Tool outputs are Pydantic-validated and JSON-serializable. Search result fields follow existing API keys — do not claim `face_id`, `recorded_at`, or bounding box fields that the current API does not return.

`search_by_photo` requires the MCP server process to share filesystem access with the image file. In the default local deployment this is always satisfied.

### Configuration

Environment variables via `pydantic-settings`:

| Variable | Default | Purpose |
|---|---|---|
| `MULTIMEDIA_API_BASE_URL` | `http://127.0.0.1:8000` | FastAPI base URL |
| `MULTIMEDIA_HTTP_TIMEOUT_SEC` | `30` | httpx request timeout |
| `MULTIMEDIA_SEARCH_LIMIT_MAX` | `200` | Hard cap on tool result counts per media type |
| `MULTIMEDIA_LOG_LEVEL` | `INFO` | MCP server log verbosity |

`MULTIMEDIA_API_KEY` may be reserved for future auth but is not implemented here.

### Dependencies

Add to `requirements.txt`:
```
mcp[cli]
httpx
pydantic-settings
```

### CLI and docs

- `python cli.py mcp` — runs MCP server over stdio (primary entry point)
- `python -m mcp_server.server` — direct alternative
- Update `README.md`: setup steps, FastAPI startup requirement, Claude Desktop JSON config, tool usage examples, privacy/offline notes
- Add `examples/api_calls.py` with runnable HTTP API calls against a live FastAPI instance. This is an API smoke example, not a replacement for MCP Inspector/client verification.

### Project structure additions

```
mcp_server/
├── server.py
├── config.py
├── client.py
├── logging_config.py
├── models/
└── tools/
app/services/
├── search_service.py
└── person_service.py
examples/
└── api_calls.py
```

### Test plan

- **Service tests:** extracted services preserve existing route response shapes; test empty name, no matches, empty Chroma collection, invalid image, no-face image
- **API regression tests:** existing routes (`GET /api/search`, `POST /api/search/photo`, `GET /api/persons`) still pass; new endpoints return typed JSON and correct status codes
- **MCP/tool tests:** validate input bounds; mock HTTP responses for success, 404, 422, 500, and connection failure; verify all tool outputs are Pydantic-validated; verify tool names/descriptions are discoverable; verify logs go to stderr
- **Manual smoke tests:** `python cli.py serve` + `python -m mcp_server.server`; call `health_check`, `list_people`, `search_by_name`, and job status tools through MCP Inspector or another MCP-compatible client; use `examples/api_calls.py` only for HTTP API smoke checks; `python -m compileall app mcp_server cli.py`

### Known constraints

- FastAPI must be running for any MCP tool to execute — document this prominently.
- MCP is localhost/offline by default. Network-accessible deployments require an API auth layer added to FastAPI before exposing either process on a network.
- Redis, Chroma HTTP mode, load balancers, and multi-worker scaling are out of scope for this local single-user tool.
- Destructive or identity-changing tools (label, merge, index, cluster, prune) are intentionally excluded from the initial MCP surface.

**Verification:** Start `python cli.py serve`. Run `python cli.py mcp`. Open MCP Inspector — confirm all tools are listed. Call `health_check` — confirm `status: ok`. Call `search_by_name` with a labeled person — confirm structured JSON results. Start a compile job, poll `check_compile_status` — confirm `download_url` appears when done.

---

## Phase 7 — Full Body Re-ID (Future)

Not built in Phases 1–4. Design notes for when this is scoped:

- **Person detector:** YOLOv8n (ultralytics) — fast, CPU-capable, detects person bounding boxes per frame
- **Re-ID model:** OSNet (`torchreid`) — 512-dim body appearance embedding, trained on person Re-ID datasets
- **Storage:** new ChromaDB collection `body_embeddings`, same metadata schema as `faces`
- **Indexer change:** after face detection, also run YOLOv8 → crop persons → OSNet embedding → store in `body_embeddings`
- **Search fusion:** `score = 0.6 * face_similarity + 0.4 * body_similarity`; body-only if no face detected in frame
- **UI:** "Full body matching" toggle on `/search`; enrolls body embeddings from existing labeled faces automatically

Phases 1–6 are unaffected — this is purely additive.

---

## CLI Reference

```
python cli.py index          <directory>  [--interval 1.0] [--gpu] [--no-cluster] [--eps 0.6] [--videos-only] [--photos-only]
python cli.py cluster                     [--incremental] [--eps 0.6] [--min-samples 3]
python cli.py prune                       [--dry-run]
python cli.py trim-thumbnails             [--dry-run]
python cli.py backfill-dates
python cli.py stats
python cli.py serve                       [--host 0.0.0.0] [--port 8000]
python cli.py mcp                         # run MCP server over stdio (Phase 6)
```

`--videos-only` and `--photos-only` are mutually exclusive; omitting both processes all media types.

All web UI and API available at `http://localhost:8000` after `serve`.

---

## Maintenance

### Automatic pipeline integration

| Step | Triggered by | What it does |
|---|---|---|
| `prune_stale_media()` | Start of `index` | Removes SQLite rows (both `videos` and `photos`), ChromaDB face records, and thumbnail files for any media deleted from disk; also removes persons whose last faces are gone |
| `trim_thumbnails()` | End of `cluster` (full and incremental) | Deletes all face thumbnail PNGs not referenced as a person's representative thumbnail or sample; keeps at most 5 per person. Covers face crops from both videos and photos — they share the same `static/thumbnails/` directory |

Both are also available as manual CLI commands (`prune`, `trim-thumbnails`) with a `--dry-run` flag for safe previewing.

### Thumbnail lifecycle

Each detected face — whether from a video frame or a source photo — gets a 128×128 PNG crop saved to `static/thumbnails/` at index time. After clustering, only 1 representative thumbnail + up to 4 sample thumbnails per person are needed (used by the `/label` UI). Everything else is redundant — `trim_thumbnails()` removes them automatically after each cluster run, keeping the directory small regardless of library size. Photo face crops and video face crops are indistinguishable at this stage; `trim_thumbnails()` requires no changes to handle both.

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
Pillow                   # Phase 5: EXIF date extraction + photo collage composition
mcp[cli]                 # Phase 6: MCP server runtime
httpx                    # Phase 6: async HTTP client for MCP → FastAPI calls
pydantic-settings        # Phase 6: environment-variable configuration for MCP server
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
