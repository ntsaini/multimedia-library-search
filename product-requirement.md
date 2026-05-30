# Product Requirements — Local Video Library Search Engine

## Overview

A local, private video library search engine that indexes a directory of videos, identifies and clusters the people found in them, and lets the user search for clips by person name or reference photo. Results can be returned as timestamped clip references or compiled into a downloadable highlight reel. Everything runs on the user's own machine — no data leaves the device.

---

## Goals

- **Private by design:** no cloud calls, no external APIs, no data leaves the machine
- **Simple to operate:** minimal technical knowledge required to index and search
- **Incrementally useful:** each phase ships a working tool, not just a partial one

## Non-Goals

- Cloud sync or remote access
- Multi-user or authenticated access
- Real-time or live video stream analysis
- Audio / speech-based search
- Mobile app
- Editing or deleting source video files
- Facial recognition accuracy guarantees (best-effort; depends on video quality)

---

## Technical Architecture

| Layer | Technology |
|---|---|
| Backend API | Python / FastAPI |
| Web UI | Vanilla HTML/JS + htmx (served by FastAPI, no build step) |
| Face detection + embedding | InsightFace (buffalo_l model, ONNX runtime) |
| Vector database | ChromaDB (embedded/persistent mode) |
| Relational database | SQLite |
| Frame extraction | OpenCV |
| Video processing | FFmpeg |
| Face clustering | DBSCAN (scikit-learn) |
| EXIF + image composition | Pillow |
| Agent tool interface | MCP (Model Context Protocol) via FastMCP |
| MCP HTTP client | httpx (async) |

---

## User Stories

### Phase 1 — Index

**US-01** As a user, I want to point the tool at a folder of videos so it can scan and index all the people found in them.

**US-02** As a user, I want the tool to skip videos it has already indexed so re-running doesn't duplicate work.

**US-03** As a user, I want to see progress while indexing is running so I know it hasn't frozen.

### Phase 2 — Label

**US-04** As a user, I want to see a visual overview of all the distinct people found during indexing, shown as face thumbnails grouped by person.

**US-05** As a user, I want to assign a name to each detected person so I can search for them later.

**US-06** As a user, I want to merge two clusters that represent the same person in case the auto-grouping split them incorrectly.

### Phase 3 — Search

**US-07** As a user, I want to type a person's name and get back a list of video clips they appear in, with timestamps.

**US-08** As a user, I want to upload a photo of a person and find all clips they appear in, even if I haven't labeled them yet.

**US-09** As a user, I want to click a search result and play the video starting at the matched moment directly in the browser.

### Phase 4 — Highlight Reel

**US-10** As a user, I want to generate a single compiled video of all clips a specific person appears in.

**US-11** As a user, I want to configure how much padding (seconds before/after each appearance) is included in the reel.

**US-12** As a user, I want to download the compiled highlight reel as an MP4 file.

### Phase 5 — Photo Library Extension

**US-14** As a user, I want to index a folder of photos alongside my videos so all appearances of a person are searchable in one place.

**US-15** As a user, I want search results to show a Photos tab and a Videos tab separately so I can focus on one media type at a time.

**US-16** As a user, I want to click on a photo result and view it full-size in a lightbox, with keyboard navigation between results.

**US-17** As a user, I want to generate a photo collage of all photos a person appears in so I can get a visual overview at a glance.

**US-18** As a user, I want to configure the collage layout (number of columns, sort order, date captions) before generating it.

**US-19** As a user, I want to download the finished collage as a PNG file.

### Phase 6 — LLM Tool Layer & MCP Server

**US-20** As an AI agent or LLM client, I want to verify the multimedia library service is healthy and storage is initialized before issuing queries.

**US-21** As an AI agent, I want to retrieve library statistics (video, photo, face, and person counts) so I can orient myself before answering questions.

**US-22** As an AI agent, I want to list and look up labeled people in the library so I can answer questions about who is present.

**US-23** As an AI agent, I want to search the library by person name and receive structured results I can reason over.

**US-24** As an AI agent, I want to search by a reference image file path and receive ranked results with distance scores.

**US-25** As an AI agent, I want to retrieve full metadata for a specific video or photo from a prior search result.

**US-26** As an AI agent, I want to kick off a highlight reel or photo collage compilation job and poll its status, then return a download URL to the user.

**US-27** As a developer, I want to connect Claude Desktop (or any MCP-compatible client) to the local multimedia library by adding a single JSON config block.

### Phase 6.5 — Reference-Face Auto-Labeling

**US-28** As a user, I want to place trusted face photos in a local `labeled-faces/` directory so the app can automatically restore names after clustering.

**US-29** As a user, I want full and incremental cluster runs to apply confident reference-face labels without overwriting labels I already assigned manually.

**US-30** As a user, I want to manually trigger the reference-label pass from the CLI or Label page without re-running clustering.

**US-31** As a user, I want ambiguous or invalid reference images skipped rather than risk incorrect automatic labels.

### Phase 7 — Full Body Search (Future)

**US-13** As a user, I want the search to find a person even when their face is partially obscured (cap, glasses, turned away), using their overall body appearance.

---

## Functional Requirements

### Phase 1 — Indexing Pipeline

| ID | Requirement |
|---|---|
| F-01 | Accept a directory path and find all video files (.mp4, .avi, .mov, .mkv) |
| F-02 | Extract frames at a configurable interval (default: 1 frame/sec) |
| F-03 | Detect all faces in each frame and compute a 512-dim face embedding |
| F-04 | Save a cropped face thumbnail for each detected face |
| F-05 | Store video metadata in SQLite and face embeddings + metadata in ChromaDB |
| F-06 | Skip previously indexed videos (idempotent — safe to re-run on same directory) |
| F-07 | Expose CLI command: `python cli.py index <directory>` |
| F-08 | Show a progress bar during indexing |

### Phase 2 — Clustering & Labeling

| ID | Requirement |
|---|---|
| F-09 | Group detected faces into person clusters using DBSCAN on face embeddings |
| F-10 | Expose CLI command: `python cli.py cluster` to (re-)run clustering |
| F-11 | Provide a web page at `/label` showing each cluster as a grid of face thumbnails |
| F-12 | Show appearance count per cluster |
| F-13 | Allow the user to type a name and assign it to a cluster |
| F-14 | Allow the user to merge two clusters into one |
| F-15 | Unlabeled clusters display as "Unknown #N" until named |

### Phase 3 — Search

| ID | Requirement |
|---|---|
| F-16 | Provide a search page at `/search` |
| F-17 | Search by name: return all clips where the named person appears, with timestamps |
| F-18 | Search by photo: detect face in uploaded image, find visually similar faces in index |
| F-19 | Each result shows: video filename, timestamp, face thumbnail |
| F-20 | Clicking a result plays the video in-browser starting at the matched timestamp |
| F-21 | Video files served locally with HTTP range request support (required for browser seeking) |

### Phase 4 — Highlight Reel Compilation

| ID | Requirement |
|---|---|
| F-22 | "Create Highlight Reel" button available on search results |
| F-23 | Merge nearby timestamp matches from the same video into a single continuous clip before compiling (e.g. matches at 0:10 and 0:12 become one clip, not two) |
| F-24 | Compile all clips into a single MP4 using FFmpeg |
| F-25 | Support configurable padding around each appearance (default: 2 sec before/after) |
| F-26 | Run compilation as a background job; show progress in the UI |
| F-27 | Provide a download link for the completed reel |

### Phase 5 — Photo Library Extension

| ID | Requirement |
|---|---|
| F-32 | Accept image files (.jpg, .jpeg, .png, .heic, .heif, .webp) during `index` alongside video files |
| F-33 | Extract EXIF `DateTimeOriginal` from photos; fall back to file mtime |
| F-34 | Skip previously indexed photos (idempotent — safe to re-run on same directory) |
| F-35 | Support `--videos-only` and `--photos-only` flags on the `index` command |
| F-36 | Search results split into Videos and Photos tabs; default tab is whichever has more results |
| F-37 | Photo result cards show a preview thumbnail, filename, and captured date |
| F-38 | Clicking a photo result opens a full-size lightbox with `←` / `→` keyboard navigation and `Esc` to close |
| F-39 | Photo files served locally via `GET /api/photo/{photo_id}` |
| F-40 | Photo Collage panel available below photo results for named-person searches |
| F-41 | Collage supports configurable columns (2/3/4), sort order (asc/desc/random), and optional date captions |
| F-42 | Run collage generation as a background job; show progress in the UI |
| F-43 | Provide a download link for the completed collage PNG |

### Phase 6 — LLM Tool Layer & MCP Server

| ID | Requirement |
|---|---|
| F-48 | Add `GET /api/health` endpoint returning readiness of the app, SQLite, and ChromaDB |
| F-49 | Add `GET /api/stats` endpoint returning counts of indexed videos, photos, faces, persons, and labeled persons |
| F-50 | Add `GET /api/persons/{person_id}` endpoint for clean person lookup by ID; 404 when not found; `name: null` for unnamed clusters |
| F-51 | Add `GET /api/video/{video_id}/info` and `GET /api/photo/{photo_id}/info` endpoints returning JSON metadata and local API access paths |
| F-52 | Extract search logic from route functions into `app/services/search_service.py` and person helpers into `app/services/person_service.py`; existing routes remain unchanged |
| F-53 | Create `mcp_server/` package using `mcp.server.fastmcp.FastMCP`; default transport is stdio; support optional Streamable HTTP transport |
| F-54 | Implement MCP tools: `health_check`, `get_library_stats`, `list_people`, `get_person`, `search_by_name`, `search_by_photo`, `get_media_info`, `compile_highlight_reel`, `check_compile_status`, `create_photo_collage`, `check_collage_status`; `get_media_info` must accept exactly one media ID |
| F-55 | Configure MCP server via environment variables: `MULTIMEDIA_API_BASE_URL` (default `http://127.0.0.1:8000`), `MULTIMEDIA_HTTP_TIMEOUT_SEC`, `MULTIMEDIA_SEARCH_LIMIT_MAX`, `MULTIMEDIA_LOG_LEVEL` |
| F-56 | Add `python cli.py mcp` command to run the MCP server over stdio; keep `python -m mcp_server.server` as a direct alternative |
| F-57 | Log structured events to stderr only; MCP stdio stdout must remain protocol-only |
| F-58 | Job status tools (`check_compile_status`, `check_collage_status`) return an absolute `download_url` when status is `"done"` |
| F-59 | Update README with MCP setup instructions, required FastAPI startup step, Claude Desktop config snippet, tool examples, and privacy/offline notes |
| F-60 | Add `examples/api_calls.py` with runnable HTTP API smoke examples against a live FastAPI instance; MCP transport examples should use MCP Inspector or another MCP-compatible client |
| F-61 | Keep MCP localhost/offline by default; any network-accessible MCP or FastAPI deployment requires a separate auth layer before exposure |

### Phase 6.5 — Reference-Face Auto-Labeling

| ID | Requirement |
|---|---|
| F-62 | Add `LABELED_FACES_DIR = BASE_DIR / "labeled-faces"` and document the flat directory convention: one image per person, filename stem equals person name |
| F-63 | Add `labeled-faces/` to `.gitignore`; reference images are private biometric user data |
| F-64 | Add `app/labeled_faces.py` with a reference loader that scans supported image files, decodes each image locally, runs InsightFace, and returns one normalized embedding per valid filename stem |
| F-65 | Reference images with zero detected faces, multiple detected faces, unreadable files, or duplicate filename stems are skipped and counted in the label-pass summary |
| F-66 | Add an auto-label pass that compares SQLite `persons.centroid` values against reference embeddings using Euclidean distance on normalized vectors |
| F-67 | A cluster is labeled only when `best_distance < threshold` and `second_best_distance - best_distance >= margin`; defaults are threshold `0.6` and margin `0.08` |
| F-68 | Auto-labeling does not overwrite existing non-null `persons.name` values unless explicitly requested with an overwrite option |
| F-69 | Full and incremental cluster runs invoke auto-labeling after clustering when `labeled-faces/` exists and auto-labeling is not disabled |
| F-70 | Cluster API results return auto-label details under a nested `auto_label` key rather than mixing label counts with clustering counts |
| F-71 | Add CLI flags for cluster: `--no-auto-label`, `--label-threshold`, and `--label-margin` |
| F-72 | Add `python cli.py relabel` to run only the reference-label pass; support `--label-threshold`, `--label-margin`, and `--overwrite` |
| F-73 | Add `POST /api/cluster/auto-label` for manual reference-label application and `GET /api/cluster/label-refs` for lightweight reference-directory status |
| F-74 | Add a minimal `/label` page control showing reference-directory status and an "Apply Reference Labels" button with inline result text |

### Phase 7 — Full Body Re-ID (Future)

| ID | Requirement |
|---|---|
| F-44 | Detect full-person bounding boxes alongside face detection during indexing |
| F-45 | Compute body appearance embeddings stored in ChromaDB |
| F-46 | Search fuses face similarity and body similarity scores |
| F-47 | UI provides a toggle to enable full-body matching |

---

## Non-Functional Requirements

| ID | Requirement |
|---|---|
| NF-01 | Runs entirely locally — no network calls during indexing or search |
| NF-02 | All data (embeddings, thumbnails, DBs) stored in a configurable local `data/` directory |
| NF-03 | Works on CPU; GPU acceleration optional and auto-detected |
| NF-04 | Indexing a 1-hour video completes in under 15 minutes on CPU |
| NF-05 | Search results returned in under 2 seconds for a library up to 10,000 indexed faces |
| NF-06 | Setup requires only: `pip install -r requirements.txt` then `python cli.py serve` |

## Performance Considerations

The system uses keyframe sampling (default: 1 frame/sec) rather than processing every frame. This trades some precision on exact clip boundaries for dramatically faster indexing. The tradeoff is acceptable because:
- Highlight reel padding (±2 sec) covers any boundary imprecision
- Clip merging (F-23) handles the case where a person appears across multiple consecutive sampled frames
- Users can raise the sampling rate via `--interval 0.5` if they want finer granularity at the cost of indexing time

---

## Acceptance Criteria by Phase

### Phase 1 Done When:
- `python cli.py index /my/videos` runs without error, shows progress, and populates the database
- Re-running on the same directory skips already-indexed files
- `python cli.py stats` reports correct video and face counts

### Phase 2 Done When:
- `python cli.py cluster` groups faces into person clusters without error
- `/label` page shows face thumbnails grouped by person with appearance counts
- A name assigned in the UI persists across server restarts

### Phase 3 Done When:
- Searching by name returns correct clips with timestamps
- Uploading a photo returns visually similar faces from the index
- Clicking a result plays the video at the correct timestamp in the browser

### Phase 4 Done When:
- "Create Highlight Reel" triggers a background job with visible progress
- The compiled MP4 downloads and contains the correct clips in chronological order
- Clips from the same video that are close together are merged into one continuous segment

### Phase 5 Done When:
- `python cli.py index /my/photos` processes image files, skips already-indexed files, and triggers incremental clustering
- `/search?name=Alice` shows a Photos tab with correctly dated photo cards
- Clicking a photo opens the lightbox; keyboard navigation cycles through results
- "Create Collage" runs as a background job; the downloaded PNG shows a correctly laid-out grid

### Phase 6 Done When:
- `GET /api/health`, `GET /api/stats`, `GET /api/persons/{id}`, `GET /api/video/{id}/info`, and `GET /api/photo/{id}/info` all return typed JSON with correct status codes
- `python cli.py mcp` starts the MCP server over stdio without error
- An MCP client (e.g. Claude Desktop or MCP Inspector) can discover and call all tools
- `health_check`, `list_people`, `search_by_name`, `search_by_photo`, `compile_highlight_reel`, and `check_compile_status` return structured, Pydantic-validated JSON; search tool `limit` behavior is documented as per media type
- Job status tools return an absolute `download_url` when the job is done
- MCP logs go to stderr; stdout carries only MCP protocol frames
- README documents setup, FastAPI startup dependency, Claude Desktop config, and tool usage examples
- MCP setup remains localhost-oriented and clearly warns against exposing the app on a network without adding authentication

### Phase 6.5 Done When:
- `labeled-faces/Alice Smith.jpg` is detected as one reference person named `Alice Smith`
- Reference images with zero faces, multiple faces, unreadable files, or duplicate names are skipped and reported
- `python cli.py cluster` applies reference labels after clustering when `labeled-faces/` exists, without overwriting existing names
- `python cli.py relabel` applies reference labels without re-running clustering
- `/label` shows reference-directory status and can trigger `POST /api/cluster/auto-label`
- Ambiguous cluster/reference matches remain unlabeled rather than receiving a low-confidence name
