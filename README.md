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

### Cluster faces into person identities

After indexing, group detected faces into clusters — each cluster becomes one person identity:

```bash
python cli.py cluster
```

Options:

| Flag | Default | Description |
|---|---|---|
| `--eps 0.6` | `0.6` | DBSCAN eps (euclidean distance on L2-normed embeddings). Lower = tighter clusters. |
| `--min-samples 3` | `3` | Minimum faces required to form a cluster. |

Re-running cluster overwrites previous groupings. **Any names you have assigned will be lost**, so label after clustering, not before.

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
index → cluster → label (web UI) → search (web UI)
```

1. `python cli.py index /path/to/videos` — extract and store face embeddings
2. `python cli.py cluster` — group faces into person identities
3. Open `/label` — assign names to each person cluster
4. Open `/search` — search by name or reference photo *(Phase 3)*

## Web UI

| Page | Path | Description |
|---|---|---|
| Index | `/` | Start indexing, watch live progress |
| Label | `/label` | Assign names to person clusters; merge duplicates |
| Search | `/search` | Search by name or reference photo *(Phase 3)* |

## API

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/index` | Start indexing a directory |
| `GET` | `/api/index/status` | Indexing progress (polls every 2 s from the UI) |
| `GET` | `/api/persons` | List all person clusters with face counts |
| `POST` | `/api/persons/{id}/label` | Save a name for a person |
| `POST` | `/api/persons/merge` | Merge two person clusters into one |

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
│   ├── clusterer.py         # DBSCAN face clustering
│   └── api/
│       ├── index.py         # POST /api/index, GET /api/index/status
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
- [ ] Phase 3 — Search by name or reference photo
- [ ] Phase 4 — Highlight reel compilation
