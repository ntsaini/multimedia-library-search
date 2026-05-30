# Multimedia Library Search

Search your local video and photo library by face. Point it at a directory, index it, and find every clip or photo a person appears in — entirely offline, no cloud.

## Requirements

- Python 3.11+
- [FFmpeg](https://ffmpeg.org/download.html) available on `PATH`
- (Optional) NVIDIA GPU + CUDA for faster indexing

## Setup

```bash
git clone <repo>
cd multimedia-library-search

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

> **GPU users:** replace `onnxruntime` with `onnxruntime-gpu` in `requirements.txt` before running `pip install`.

## Workflow

```
index → (auto-cluster) → label → search → compile / collage
```

1. `python cli.py index /path/to/media` — extract faces from videos and photos; incremental cluster runs automatically
2. Open `/label` — assign names to person clusters; merge duplicates; re-cluster if needed
3. Open `/search` — search by name or upload a reference photo to find appearances across videos and photos
4. Use the **Highlight Reel** panel to compile video clips into a downloadable MP4, or the **Photo Collage** panel to generate a JPEG grid of photo appearances

For the first run with no existing clusters, the auto-triggered incremental pass falls back to a full cluster automatically.

---

## Pipeline

```mermaid
flowchart LR
    subgraph Input
        V[Videos]
        P[Photos]
    end
    subgraph Indexer
        FA[InsightFace\nface detection]
    end
    subgraph Storage
        DB[(SQLite\nvideos · photos · persons)]
        CH[(ChromaDB\nface embeddings)]
    end
    subgraph Cluster
        CL[DBSCAN /\nIncremental]
    end
    subgraph Output
        HL[Highlight reel\nMP4]
        CO[Photo collage\nJPEG]
    end

    V & P --> FA
    FA --> DB & CH
    CH --> CL --> DB
    DB & CH --> Search
    Search --> HL & CO
```

---

## CLI Reference

### Index a media directory

```bash
python cli.py index /path/to/media
```

Scans for both videos (`.mp4`, `.avi`, `.mov`, `.mkv`) and photos (`.jpg`, `.jpeg`, `.png`, `.heic`, `.heif`, `.webp`). Already-indexed files are skipped automatically.

| Flag | Default | Description |
|---|---|---|
| `--interval 2.0` | `1.0` | Seconds between sampled keyframes (videos only). Higher = faster, lower accuracy. |
| `--gpu` | off | Use CUDA for face inference (requires `onnxruntime-gpu`). |
| `--no-cluster` | off | Skip the automatic incremental cluster pass after indexing. |
| `--eps 0.6` | `0.6` | DBSCAN eps used by the auto-triggered incremental cluster. |

After indexing, stale media (deleted from disk since the last run) is pruned automatically, then an incremental cluster pass runs so new faces appear in `/label` without a separate step. Pass `--no-cluster` when batch-indexing several directories before a single cluster run.

### Cluster faces into person identities

```bash
python cli.py cluster              # full DBSCAN — initial setup or restructure
python cli.py cluster --incremental  # assign new faces only, preserve labels
```

**Full cluster** groups all faces from scratch using DBSCAN. Use this on first run or to tune `eps` and restructure everything. **All existing labels are lost.**

**Incremental cluster** assigns only unlabeled (new) faces to existing persons using centroid distance, then runs a mini-DBSCAN on the remainder to discover new persons. Existing labels are fully preserved.

After every cluster run, redundant face thumbnails are trimmed automatically — keeping only the representative samples shown in the label UI (at most 5 per person).

| Flag | Default | Description |
|---|---|---|
| `--incremental` | off | Run incremental mode instead of full DBSCAN. |
| `--eps 0.6` | `0.6` | Grouping radius (euclidean distance on L2-normed embeddings). Lower = tighter clusters. |
| `--min-samples 3` | `3` | Minimum faces required to form a cluster. |

### Maintenance commands

```bash
python cli.py prune                   # remove stale data for media deleted from disk
python cli.py prune --dry-run         # preview what would be removed

python cli.py trim-thumbnails         # delete redundant face thumbnails (keeps label-page samples)
python cli.py trim-thumbnails --dry-run

python cli.py backfill-dates          # populate recorded_at for already-indexed videos
python cli.py stats                   # video/photo count, face count, labeled/unlabeled persons
```

`prune` and `trim-thumbnails` run automatically as part of `index` and `cluster` respectively; these manual commands are for one-off use or verification with `--dry-run`.

`backfill-dates` is a one-time migration for videos indexed before recording-date extraction was added. For each video it tries, in order: ffprobe container tag → filename date pattern → file mtime.

### Start the web UI

```bash
python cli.py serve
```

Opens at `http://127.0.0.1:8000`. Options: `--host 0.0.0.0 --port 8000`.

### Start the MCP server

The MCP server lets local AI agents call the library through a small tool surface. Start the FastAPI app first, then start MCP in the agent client config.

```bash
python cli.py serve
python cli.py mcp
```

Direct execution is also available:

```bash
python -m mcp_server.server
```

Configuration uses environment variables:

| Variable | Default | Description |
|---|---|---|
| `MULTIMEDIA_API_BASE_URL` | `http://127.0.0.1:8000` | FastAPI base URL MCP should call |
| `MULTIMEDIA_HTTP_TIMEOUT_SEC` | `30` | HTTP timeout in seconds |
| `MULTIMEDIA_SEARCH_LIMIT_MAX` | `200` | Max results returned by MCP search tools per media type |
| `MULTIMEDIA_LOG_LEVEL` | `INFO` | MCP server log level |

Claude Desktop config example:

```json
{
  "mcpServers": {
    "multimedia-library-search": {
      "command": "python",
      "args": [
        "/absolute/path/to/multimedia-library-search/cli.py",
        "mcp"
      ],
      "env": {
        "MULTIMEDIA_API_BASE_URL": "http://127.0.0.1:8000"
      }
    }
  }
}
```

Available tools: `health_check`, `get_library_stats`, `list_people`, `get_person`, `search_by_name`, `search_by_photo`, `get_media_info`, `compile_highlight_reel`, `check_compile_status`, `create_photo_collage`, and `check_collage_status`. For search tools, `limit` caps each media list separately, so a request with `limit=5` may return up to 5 videos and 5 photos.

For a simple HTTP API smoke test against a running FastAPI instance:

```bash
python examples/api_calls.py Alice
```

This example calls the FastAPI HTTP API directly. Full MCP transport verification should be done through MCP Inspector or an MCP-compatible client by confirming tool discovery and calling `health_check`, `list_people`, and one search tool.

MCP is intended for localhost/offline use. Do not expose the FastAPI or MCP process on a network without adding an authentication layer first.

---

## Web UI

| Page | URL | Description |
|---|---|---|
| Index | `/` | Start indexing, watch live progress, see all indexed videos and photos |
| Label | `/label` | Assign names to person clusters; merge duplicates; trigger re-clustering |
| Search | `/search` | Search by name or reference photo; play results in-browser; compile highlight reels and collages |

### Label page

- **Cluster Unlabeled Faces** — incremental pass; labels preserved, safe to run any time
- **Full Re-cluster** — restructures from scratch; existing labels are erased
- Both controls accept `eps` and `min-samples` with inline tooltips
- Select two cards and click **Merge Selected** to combine duplicate clusters

### Search page

**Search by name** — type a labeled person's name (autocomplete from existing labels). Results are split into a **Videos** tab and a **Photos** tab; the tab showing more results is active by default.

**Search by photo** — upload any image containing a face. Results are ranked by match confidence (closest embedding distance first) across both videos and photos.

**Videos tab** — results grouped by unique video, sorted by most appearances first. Each card shows three evenly-sampled scene frames, a scene count, and scene chips. The tab count reflects unique videos, not individual face detections.

**Photos tab** — photo results shown as a scrollable grid. Click any card to open it in a full-screen lightbox with keyboard navigation (`←` / `→` to browse, `Esc` to close).

**Scene chips** — each chip represents one detected scene. A brief isolated appearance shows as a single timestamp (`1:23`); a continuous stretch shows as a range (`0:00–5:52`). Clicking any chip opens the video at that moment.

**Video player** — custom player with:
- Timestamp markers on the scrubber (yellow lines at every detected appearance)
- Hover a marker to see the time; click to jump to it
- `Space` — play / pause
- `←` / `→` — seek ±5 seconds
- `Esc` — close
- Fullscreen button

### Highlight Reel

Appears when named persons have video results. Compiles all appearances into a single downloadable MP4 using FFmpeg.

| Setting | Default | Description |
|---|---|---|
| Clip length (sec) | 30 | Duration of each snippet, centered on the detected scene midpoint. |
| Scene gap (sec) | 30 | Detections within this many seconds are treated as one scene → one clip. |
| Max clips per video | 5 | Cap on scenes taken from a single video, so one long video doesn't dominate. |
| Clip order | Earliest first | Order clips by recording date ascending, descending, or randomly. |

Recording date comes from the ffprobe container tag, filename date pattern, or file mtime — in that priority order. Run `python cli.py backfill-dates` to populate dates for already-indexed videos.

### Photo Collage

Appears when named persons have photo results. Compiles matching photos into a downloadable JPEG using a masonry grid layout.

| Setting | Default | Description |
|---|---|---|
| Columns | 3 | Number of columns in the grid. |
| Sort order | Earliest first | Sort photos by capture date ascending, descending, or randomly. |
| Date captions | on | Overlay the capture date on each photo cell. |

Capture date is read from EXIF `DateTimeOriginal`, falling back to file mtime.

---

## API

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/index` | Start indexing a directory (`directory_path`, `interval_sec`) |
| `GET` | `/api/index/status` | Indexing progress (videos + photos) |
| `GET` | `/api/health` | App, SQLite, and ChromaDB readiness |
| `GET` | `/api/stats` | Video/photo/face/person counts |
| `POST` | `/api/cluster` | Start a cluster job (`incremental`, `eps`, `min_samples`) |
| `GET` | `/api/cluster/status` | Cluster job progress |
| `GET` | `/api/persons` | List all person clusters (id, name, thumbnail, face_count) |
| `GET` | `/api/persons/{id}` | Get one person cluster, including sample thumbnails |
| `POST` | `/api/persons/{id}/label` | Save a name for a person |
| `POST` | `/api/persons/merge` | Merge two clusters (`source_id`, `target_id`) |
| `GET` | `/api/search?name=Alice` | Find all appearances of a named person (videos + photos) |
| `POST` | `/api/search/photo` | Upload a face photo; returns ranked matches across videos and photos |
| `GET` | `/api/video/{id}` | Stream a video file with HTTP range request support |
| `GET` | `/api/video/{id}/info` | JSON video metadata and local API paths |
| `GET` | `/api/frame/{id}?t=47` | Extract a single JPEG frame at the given timestamp (cached) |
| `GET` | `/api/photo/{id}` | Serve the original photo file |
| `GET` | `/api/photo/{id}/info` | JSON photo metadata and local API paths |
| `GET` | `/api/photo/{id}/preview?size=600` | Serve a downscaled JPEG preview (max 1200px) |
| `POST` | `/api/compile` | Start a highlight reel job (`person_id`, `clip_duration_sec`, `merge_gap_sec`, `max_clips_per_video`, `order`) |
| `GET` | `/api/compile/{job_id}` | Poll reel job status and progress |
| `GET` | `/api/compile/{job_id}/download` | Download the finished MP4 |
| `POST` | `/api/collage` | Start a collage job (`person_id`, `columns`, `sort`, `captions`) |
| `GET` | `/api/collage/{job_id}` | Poll collage job status and progress |
| `GET` | `/api/collage/{job_id}/download` | Download the finished JPEG |

---

## Project structure

```
multimedia-library-search/
├── requirements.txt
├── cli.py                   # CLI entry point
├── app/
│   ├── config.py            # Paths and constants
│   ├── database.py          # SQLite (videos, photos, persons tables)
│   ├── chroma.py            # ChromaDB face embeddings store
│   ├── indexer.py           # Video frame extraction + InsightFace pipeline
│   ├── photo_indexer.py     # Photo loading + InsightFace pipeline
│   ├── clusterer.py         # Full DBSCAN + incremental clusterer
│   ├── compiler.py          # Scene merging + FFmpeg highlight reel
│   ├── collage.py           # Masonry photo collage builder
│   └── api/
│       ├── index.py         # /api/index
│       ├── cluster.py       # /api/cluster
│       ├── persons.py       # /api/persons
│       ├── search.py        # /api/search, /api/search/photo
│       ├── video.py         # /api/video, /api/frame
│       ├── photo.py         # /api/photo
│       ├── compile.py       # /api/compile
│       └── collage.py       # /api/collage
└── templates/
    ├── base.html            # Nav + shared layout
    ├── index.html           # Indexing page
    ├── label.html           # Labeling + clustering page
    └── search.html          # Search page with video player and photo lightbox
```

Generated at runtime (gitignored):

```
data/                # SQLite DB + ChromaDB files
static/thumbnails/   # Face crop PNGs
output/              # Compiled highlight reels and photo collages
```

## Data

All indexed data lives in `data/` and `static/thumbnails/`. Delete those directories to start over.

## Roadmap

- [x] Phase 1 — Indexing pipeline + CLI
- [x] Phase 2 — Face clustering + labeling UI
- [x] Phase 2.5 — Progressive incremental clustering
- [x] Phase 3 — Search by name or reference photo + in-browser playback
- [x] Phase 4 — Highlight reel compilation (FFmpeg clip extraction + concat)
- [x] Phase 5 — Photo library support (indexing, search, collage)
- [x] Phase 6 — LLM tool layer: MCP server + stable read-oriented API endpoints
