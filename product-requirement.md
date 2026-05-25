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

### Phase 5 — Full Body Search (Future)

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

### Phase 5 — Full Body Re-ID (Future)

| ID | Requirement |
|---|---|
| F-28 | Detect full-person bounding boxes alongside face detection during indexing |
| F-29 | Compute body appearance embeddings stored in ChromaDB |
| F-30 | Search fuses face similarity and body similarity scores |
| F-31 | UI provides a toggle to enable full-body matching |

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
