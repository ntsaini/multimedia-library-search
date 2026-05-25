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

### Start the web UI

```bash
python cli.py serve
```

Opens at `http://127.0.0.1:8000`. Options: `--host 0.0.0.0 --port 8000`.

## Web UI

| Page | Path | Description |
|---|---|---|
| Index | `/` | Start indexing, watch live progress |
| Label | `/label` | Assign names to detected face clusters *(Phase 2)* |
| Search | `/search` | Search by name or reference photo *(Phase 3)* |

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
│   └── api/
│       └── index.py         # POST /api/index, GET /api/index/status
└── templates/               # Jinja2 HTML templates
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
- [ ] Phase 2 — Face clustering + labeling UI
- [ ] Phase 3 — Search by name or reference photo
- [ ] Phase 4 — Highlight reel compilation
