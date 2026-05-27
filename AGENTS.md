# AGENTS.md — Multimedia Library Search

Context anchor for AI agents operating on this repository.

---

## 1. Project Mission

Private, **fully offline** video library search engine: scan a local directory of videos, detect and cluster faces, then **search by person name or reference photo** and compile highlight reels—all on the user's machine with zero cloud dependency.

---

## 2. Tech Stack & Core Dependencies

| Layer | Technology |
|---|---|
| Language | **Python 3.11+** |
| API + serving | **FastAPI** + **Jinja2** templates (single process) |
| Frontend | **Vanilla HTML + htmx** (htmx.org 1.9.12 via CDN; no build step) |
| Face detection + embedding | **InsightFace** (buffalo_l / buffalo_sc model; **ONNX Runtime**) |
| Vector database | **ChromaDB** (embedded `PersistentClient`; cosine distance, `faces` collection) |
| Relational database | **SQLite** (stdlib `sqlite3`; no ORM) |
| Frame extraction | **OpenCV** (`cv2`) |
| Video processing | **FFmpeg** via `subprocess` (`-c copy` for lossless, `-movflags +faststart`) |
| Face clustering | **DBSCAN** (`scikit-learn`; `ball_tree` on normed vectors) |
| CLI | **argparse** (stdlib) |
| Progress | **tqdm** |

---

## 3. Architecture & Design Patterns

- **Monolithic single-process app:** One FastAPI process serves the REST API (`/api/*`) and the HTML UI (`/`, `/label`, `/search`) via Jinja2 templates. No separate frontend build or CORS config.
- **Progressive incremental clustering:** After indexing, a **centroid-based** incremental pass assigns new faces to existing persons (running-mean centroid update: `new_c = (c * N + f) / (N + 1)`, renormalised), preserving all user-assigned labels. Falls back to full DBSCAN when no prior clusters exist.
- **Scene-based highlight compilation:** Raw per-frame detections are **merged into scenes** (gap threshold = `merge_gap_sec`); one short clip per scene centered on the scene midpoint. Prevents one long video from dominating the reel.
- **Lazy singleton initialization:** InsightFace (`_get_face_analysis()`) and ChromaDB (`get_collection()`) are module-level singletons — initialized on first use, never recreated.
- **Module-level progress state:** Indexing and compilation jobs broadcast status via module-level dicts (`index_progress`, `_jobs` in `app/indexer.py` and `app/compiler.py`). Simple shared state sufficient for a single-user local tool.
- **HTMX-driven UI:** htmx `hx-post` / `hx-swap` handles in-place DOM updates from FastAPI endpoints — no JS framework, no WebSocket, no polling loop.

---

## 4. Directory Mental Model

```
multimedia-library-search/
├── cli.py                     # CLI entry point (argparse subcommands: index, cluster, serve, etc.)
├── app/
│   ├── main.py                # FastAPI app factory, lifespan (DB init), route mounting, page templates
│   ├── config.py              # Central constants: BASE_DIR, DB_PATH, THUMBNAILS_DIR, OUTPUT_DIR, model names, etc.
│   ├── database.py            # sqlite3 connection + schema creation + backward-compatible ALTER migrations
│   ├── chroma.py              # ChromaDB singleton — PersistentClient + "faces" collection (cosine)
│   ├── indexer.py             # VideoIndexer: OpenCV frame extraction → InsightFace face detection + embedding → ChromaDB upsert; prune_stale_videos()
│   ├── clusterer.py           # FaceClusterer: full DBSCAN + incremental centroid-based clusterer; trim_thumbnails()
│   ├── compiler.py            # Scene merging + FFmpeg highlight reel compilation (background job via _jobs dict)
│   └── api/
│       ├── index.py           # POST /api/index — start; GET /api/index/status — progress
│       ├── cluster.py         # POST /api/cluster — start; GET /api/cluster/status — progress
│       ├── persons.py         # GET/POST /api/persons — list, label, merge
│       ├── search.py          # GET /api/search?name=X — name search; POST /api/search/photo — photo search
│       ├── video.py           # GET /api/video/{id} — HTTP range-aware video streaming; GET /api/frame/{id}?t=X — frame extraction (LRU cached)
│       └── compile.py         # POST /api/compile — start reel job; GET /api/compile/{job_id} — poll; download endpoint
├── templates/
│   ├── base.html              # Shared nav, layout, CSS (system-ui)
│   ├── index.html             # Indexing page (directory input, progress panel, indexed video list)
│   ├── label.html             # Person cluster labeling grid (merge, re-cluster controls)
│   └── search.html            # Search form (name autocomplete + photo upload), custom video player with timestamp markers
├── requirements.txt           # 11 dependencies (see README for install)
└── data/                      # Runtime: SQLite DB + ChromaDB index (gitignored)
└── static/thumbnails/          # Runtime: 128×128 face crop PNGs (gitignored)
└── output/                    # Runtime: compiled MP4 highlight reels (gitignored)
```

**Key runtime paths (from `config.py`):**
- **DB:** `data/library.db`
- **ChromaDB:** `data/chroma/`
- **Thumbnails:** `static/thumbnails/`
- **Output:** `output/`

---

## 5. Development Standards

### Naming conventions
- **Modules:** `snake_case.py` (e.g., `indexer.py`, `clusterer.py`, `compiler.py`)
- **Functions:** `snake_case` (e.g., `run_indexer`, `run_clusterer`, `merge_timestamps`)
- **Classes:** `PascalCase` (no classes exist currently, but follow this pattern if adding)
- **Constants:** `UPPER_SNAKE_CASE` in `config.py` (e.g., `KEYFRAME_INTERVAL_SEC`, `FACE_DET_SIZE`)
- **API routes:** verb + noun, plural (`/api/persons`, `/api/compile/{job_id}`)
- **Frontend templates:** `kebab-case.html`

### Database patterns
- **Raw SQL only.** Never add an ORM. Use parameterized queries (`?` placeholders) to prevent injection.
- `init_db()` (in `database.py`) handles schema creation and **backward-compatible ALTER migrations** (wrapped in try/except to skip already-existing columns).
- **ChromaDB `"unlabeled"` sentinel:** Never use `NULL` as a ChromaDB `person_id` value — ChromaDB has no native `IS NULL` filter. Always use the string `"unlabeled"`.
- **Person table `centroid` column:** JSON-serialised 512-float array, stored after every (full or incremental) cluster run. Used by incremental clustering for O(P×F) centroid distance computation.

### Indexing pipeline
- **Keyframe sampling** (not every-frame): default 1 frame/sec, configurable via `--interval`. Trades precision for speed; padding/merging compensates.
- **Idempotent indexing:** already-indexed video paths are skipped (SQLite UNIQUE check).
- **Auto prune on index start:** `prune_stale_videos()` removes SQLite rows + ChromaDB records + thumbnails for files deleted from disk.
- **Recording date extraction:** ffprobe `creation_time` tag → filename date pattern → file mtime. Fallback chain in `_extract_recording_date()`.

### Clustering patterns
- **Full cluster (`run_clusterer`):** DBSCAN on ALL embeddings; wipes `persons` table. Labels are **lost** — emit a warning.
- **Incremental cluster (`run_incremental_clusterer`):** Assigns `"unlabeled"` faces via centroid distance; mini-DBSCAN only on `unassigned` remainder. Labels are **never lost**.
- **Centroid update formula:** Running mean `(c * N + f) / (N + 1)`, renormalised. Batch all SQLite writes at the end.
- **Auto-trigger:** Every `index` run automatically calls `run_incremental_clusterer()` unless `--no-cluster` is passed.

### Compilation pattern
- **Scene merging first:** consecutive detections within `merge_gap_sec` become one `(start, end)` scene.
- **Clip centering:** clip of `clip_duration_sec` is centered on scene midpoint — prevents jumbo reels from long continuous appearances.
- **Per-video cap:** evenly-sampled across scenes (not head/tail) when `scenes > max_clips_per_video`.
- **FFmpeg flags:** `-ss` before `-i` (fast keyframe seek) + `-c copy` (lossless). Fallback to `libx264`/`aac` re-encode if stream copy fails.
- **Concat:** `ffmpeg -f concat -safe 0` with `+faststart` for streaming.
- **Background jobs:** module-level `_jobs` dict, mutated throughout compilation thread.

### Error handling
- **FFmpeg failures:** catch `returncode != 0`, log truncated stderr (300 chars), return `"error"` status to client.
- **Index errors:** `index_progress["status"] = "error"`, `index_progress["error"] = str(exc)`.
- **IO errors gracefully ignored:** unreadable videos are skipped (not crashed), face crops with invalid bbox return `""` and are silently excluded.

---

## 6. Hard Constraints & Anti-Patterns

- **NEVER add cloud calls, external APIs, or network-dependent code.** Privacy is the product. Every dependency must be installable `pip install -r requirements.txt` offline.
- **NEVER use ChromaDB `IS NULL` queries.** Always use `{"person_id": {"$eq": "unlabeled"}}` sentinel pattern. This is non-negotiable.
- **NEVER add an ORM** (SQLAlchemy, Peewee, etc.). The schema is simple enough for raw `sqlite3`.
- **NEVER remove existing columns from `persons` or `videos` tables.** Migration code in `init_db()` already handles backward-compatible `ALTER TABLE` gracefully.
- **NEVER mix up `MODEL_NAME_DEFAULT` ("buffalo_sc") with `MODEL_NAME_HIGH` ("buffalo_l").** `buffalo_l` is the high-quality model (used in indexer); `buffalo_sc` is the standard/slimmer model (used by default).
- **NEVER rely on `CAP_PROP_FRAME_COUNT` for logic** — it is unreliable for MP4/MKV files. Use it only for display purposes.
- **NEVER commit data/, static/thumbnails/, or output/**. These are runtime artifacts, always gitignored.
- **NEVER use `redis-py`, Celery, or any external messaging system.** This is a single-user local tool — module-level dicts are the correct concurrency model.
- **BEFORE adding any new dependency**, check `requirements.txt` first. All 11 dependencies are pinned in the file.

---

## 7. Operational Commands

### Setup (one-time)

```bash
git clone <repo>
cd multimedia-library-search
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# GPU users (replace after pip install):
# pip uninstall onnxruntime
# pip install onnxruntime-gpu
```

### Core workflow (CLI)

```bash
# Index a video directory (auto-clusters afterwards)
python cli.py index /path/to/videos

# Cluster faces (full — destructive to labels — on first run)
python cli.py cluster

# Cluster faces (incremental — preserves all labels — after new indexing)
python cli.py cluster --incremental

# Maintenance
python cli.py prune                    # remove stale data for deleted videos
python cli.py prune --dry-run          # preview what would be removed
python cli.py trim-thumbnails          # delete redundant face thumbnails
python cli.py trim-thumbnails --dry-run
python cli.py backfill-dates           # populate recorded_at for old videos
python cli.py stats                    # show index statistics

# Start the web server
python cli.py serve                    # defaults: 127.0.0.1:8000
python cli.py serve --host 0.0.0.0 --port 8080
```

### API Quick Reference

```
POST  /api/index          {"directory_path": "/path", "interval_sec": 1.0}
GET   /api/index/status
POST  /api/cluster        {"incremental": false, "eps": 0.6, "min_samples": 3}
GET   /api/cluster/status
GET   /api/persons
POST  /api/persons/{id}/label   {"name": "Alice"}
POST  /api/persons/merge        {"source_id": "uuid", "target_id": "uuid"}
GET   /api/search?name=Alice
POST  /api/search/photo         (multipart: file)
GET   /api/video/{id}           (HTTP range-aware streaming)
GET   /api/frame/{id}?t=47      (frame extraction, LRU cached)
POST  /api/compile              {"person_id": "uuid", ...}
GET   /api/compile/{job_id}     (poll progress)
GET   /api/compile/{job_id}/download  (download MP4)
```

### Web UI

| Page | URL | Purpose |
|---|---|---|
| Index | `/` | Start indexing, watch progress, view indexed files |
| Label | `/label` | Assign names to person clusters, merge duplicates, re-cluster |
| Search | `/search` | Search by name or reference photo; custom video player; highlight reel compilation |
