# Multimedia Library Search

Search your local video library by face. Point it at a directory of videos, index them, and find every clip a person appears in — entirely offline, no cloud.

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

## Usage

### Index a video directory

```bash
python cli.py index /path/to/videos
```

Options:

| Flag | Default | Description |
|---|---|---|
| `--interval 2.0` | `1.0` | Seconds between sampled keyframes. Higher = faster, lower accuracy. |
| `--gpu` | off | Use CUDA for face inference (requires `onnxruntime-gpu`). |
| `--no-cluster` | off | Skip the automatic incremental cluster pass that runs after indexing. |
| `--eps 0.6` | `0.6` | DBSCAN eps used by the auto-triggered incremental cluster. |

After indexing completes, an incremental cluster pass runs automatically so new faces appear in `/label` without a separate step. Pass `--no-cluster` to skip it (e.g. when batch-indexing several directories before a single cluster run).

### Cluster faces into person identities

```bash
python cli.py cluster              # full DBSCAN — initial setup or restructure
python cli.py cluster --incremental  # assign new faces only, preserve labels
```

**Full cluster** groups all faces from scratch using DBSCAN. Use this on first run or when you want to tune eps and restructure everything. **All existing labels are lost.**

**Incremental cluster** assigns only unlabeled faces (new since the last run) to existing persons using centroid distance, then runs a mini-DBSCAN on the remainder to discover new persons. Existing labels are fully preserved.

Options:

| Flag | Default | Description |
|---|---|---|
| `--incremental` | off | Run incremental mode instead of full DBSCAN. |
| `--eps 0.6` | `0.6` | Grouping radius (euclidean distance on L2-normed embeddings). Lower = tighter clusters. |
| `--min-samples 3` | `3` | Minimum faces required to form a cluster. Lower = fewer faces needed per person. |

### Start the web UI

```bash
python cli.py serve
```

Opens at `http://127.0.0.1:8000`. Options: `--host 0.0.0.0 --port 8000`.

### Check library statistics

```bash
python cli.py stats
```

Output:
```
Videos:  12
Faces:   3,847
Persons: 5 (3 labeled)
```

## Workflow

```
index → (auto-cluster) → label → search
```

1. `python cli.py index /path/to/videos` — extract faces; incremental cluster runs automatically at the end
2. Open `/label` — assign names to each person cluster; trigger re-cluster from the UI if needed
3. Open `/search` — search by name or reference photo *(Phase 3)*

For the first run with no existing clusters, the auto-triggered incremental pass falls back to a full cluster automatically.

## Web UI

| Page | Path | Description |
|---|---|---|
| Index | `/` | Start indexing, watch live progress, see previously indexed videos |
| Label | `/label` | Assign names to person clusters; merge duplicates; trigger clustering |
| Search | `/search` | Search by name or reference photo *(Phase 3)* |

The Label page includes a **Clustering** panel with two actions:

- **Cluster Unlabeled Faces** — incremental pass; safe to run at any time, labels preserved
- **Full Re-cluster** — restructures everything from scratch; existing labels are erased

Both accept eps and min-samples inputs with inline descriptions.

## API

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/index` | Start indexing a directory |
| `GET` | `/api/index/status` | Indexing progress |
| `GET` | `/api/persons` | List all person clusters with face counts |
| `POST` | `/api/persons/{id}/label` | Save a name for a person |
| `POST` | `/api/persons/merge` | Merge two person clusters into one |
| `POST` | `/api/cluster` | Start a cluster job (`incremental`, `eps`, `min_samples`) |
| `GET` | `/api/cluster/status` | Cluster job progress |

## Project structure

```
multimedia-library-search/
├── requirements.txt
├── cli.py                   # CLI entry point
├── app/
│   ├── config.py            # Paths and constants
│   ├── database.py          # SQLite (videos, persons tables)
│   ├── chroma.py            # ChromaDB face embeddings store
│   ├── indexer.py           # Frame extraction + InsightFace pipeline
│   ├── clusterer.py         # Full DBSCAN + incremental clusterer
│   └── api/
│       ├── index.py         # POST /api/index, GET /api/index/status
│       ├── cluster.py       # POST /api/cluster, GET /api/cluster/status
│       └── persons.py       # GET/POST /api/persons — label + merge
└── templates/               # Jinja2 HTML templates
    ├── base.html
    ├── index.html
    └── label.html
```

Generated at runtime (gitignored):

```
data/                # SQLite DB + ChromaDB files
static/thumbnails/   # Face crop PNGs
output/              # Compiled highlight reels (Phase 4)
```

## Data

All indexed data lives in `data/` and `static/thumbnails/`. Delete those directories to reset the library.

## Roadmap

- [x] Phase 1 — Indexing pipeline + CLI
- [x] Phase 2 — Face clustering + labeling UI
- [x] Phase 2.5 — Progressive incremental clustering
- [ ] Phase 3 — Search by name or reference photo
- [ ] Phase 4 — Highlight reel compilation
