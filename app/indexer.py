import json
import subprocess
import time
import re
import cv2
import numpy as np
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from insightface.app import FaceAnalysis

from app.config import (
    THUMBNAILS_DIR,
    FACE_DET_SIZE,
    THUMBNAIL_SIZE,
    VIDEO_EXTENSIONS,
    MODEL_NAME_DEFAULT,
    MODEL_NAME_HIGH,
)
from app.database import get_connection
from app.chroma import get_collection

index_progress: dict = {
    "status": "idle",
    "videos_total": 0,
    "videos_done": 0,
    "current_video": "",
    "faces_found": 0,
    "started_at": None,
    "eta_sec": None,
    "error": None,
}


def _init_face_analysis(use_gpu: bool) -> FaceAnalysis:
    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if use_gpu
        else ["CPUExecutionProvider"]
    )
    fa = FaceAnalysis(name=MODEL_NAME_DEFAULT, providers=providers)
    fa.prepare(ctx_id=0, det_size=FACE_DET_SIZE)
    return fa


def _extract_recording_date(path: Path) -> str | None:
    """Return ISO 8601 recording date, trying ffprobe → filename → mtime."""
    # 1. ffprobe container tag
    try:
        r = subprocess.run(
            [
                "ffprobe", "-v", "quiet",
                "-print_format", "json",
                "-show_entries", "format_tags=creation_time",
                str(path),
            ],
            capture_output=True,
            timeout=10,
        )
        if r.returncode == 0:
            ct = (json.loads(r.stdout or b"{}")
                  .get("format", {}).get("tags", {}).get("creation_time"))
            if ct:
                dt = datetime.fromisoformat(ct.replace("Z", "+00:00"))
                return dt.astimezone().replace(tzinfo=None).isoformat(timespec="seconds")
    except Exception:
        pass

    # 2. Filename patterns
    name = path.stem
    # YYYY-MM-DD HH-MM-SS  /  YYYY-MM-DD_HH-MM-SS  /  YYYY-MM-DDTHH:MM:SS
    m = re.search(r'(\d{4})-(\d{2})-(\d{2})[ _T](\d{2})[-:.](\d{2})[-:.](\d{2})', name)
    if m:
        try:
            return datetime(*map(int, m.groups())).isoformat(timespec="seconds")
        except ValueError:
            pass
    # YYYYMMDD_HHMMSS  /  YYYYMMDD-HHMMSS  (Android / GoPro style)
    m = re.search(r'(\d{4})(\d{2})(\d{2})[_-](\d{2})(\d{2})(\d{2})', name)
    if m:
        try:
            return datetime(*map(int, m.groups())).isoformat(timespec="seconds")
        except ValueError:
            pass

    # 3. File mtime
    try:
        return datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds")
    except Exception:
        return None


def _safe_id(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_\-]", "_", text)


def _save_thumbnail(frame: np.ndarray, bbox, face_id: str) -> str:
    THUMBNAILS_DIR.mkdir(parents=True, exist_ok=True)
    x1, y1, x2, y2 = (int(v) for v in bbox)
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return ""
    crop = cv2.resize(frame[y1:y2, x1:x2], THUMBNAIL_SIZE)
    abs_path = THUMBNAILS_DIR / f"{face_id}.png"
    cv2.imwrite(str(abs_path), crop)
    return f"thumbnails/{face_id}.png"


def _extract_frames(cap: cv2.VideoCapture, interval_sec: float):
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_interval = max(1, int(fps * interval_sec))
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_interval == 0:
            yield frame, frame_idx / fps
        frame_idx += 1


def prune_stale_videos() -> int:
    """Remove indexed data for videos no longer on disk. Returns number of videos pruned."""
    from app.chroma import get_collection
    from app.config import THUMBNAILS_DIR

    db = get_connection()
    rows = db.execute("SELECT id, path, filename FROM videos").fetchall()
    stale = [r for r in rows if not Path(r["path"]).exists()]

    if not stale:
        db.close()
        return 0

    collection = get_collection()
    for row in stale:
        result = collection.get(
            where={"video_id": {"$eq": row["id"]}},
            include=["metadatas"],
        )
        face_ids = result.get("ids") or []
        for meta in (result.get("metadatas") or []):
            thumb = meta.get("thumbnail_path")
            if thumb:
                p = THUMBNAILS_DIR / Path(thumb).name
                if p.exists():
                    p.unlink()
        if face_ids:
            collection.delete(ids=face_ids)
        db.execute("DELETE FROM videos WHERE id = ?", (row["id"],))

    db.commit()

    person_rows = db.execute("SELECT id FROM persons").fetchall()
    for p in person_rows:
        if not collection.get(where={"person_id": {"$eq": p["id"]}}, include=[])["ids"]:
            db.execute("DELETE FROM persons WHERE id = ?", (p["id"],))

    db.commit()
    db.close()
    return len(stale)


def run_indexer(
    directory: str,
    interval_sec: float = 1.0,
    use_gpu: bool = False,
    auto_cluster: bool = True,
    eps: float = 0.6,
) -> None:
    global index_progress

    index_progress.update({
        "status": "running",
        "videos_total": 0,
        "videos_done": 0,
        "current_video": "",
        "faces_found": 0,
        "started_at": time.time(),
        "eta_sec": None,
        "error": None,
    })

    try:
        pruned = prune_stale_videos()
        if pruned:
            print(f"Pruned {pruned} stale video(s) no longer on disk.")

        fa = _init_face_analysis(use_gpu=use_gpu)
        collection = get_collection()
        db = get_connection()

        video_paths = sorted(
            p for p in Path(directory).rglob("*")
            if p.suffix.lower() in VIDEO_EXTENSIONS
        )

        index_progress["videos_total"] = len(video_paths)
        total_faces = 0
        skipped = 0

        with tqdm(total=len(video_paths), desc="Indexing", unit="video") as outer:
            for video_path in video_paths:
                index_progress["current_video"] = video_path.name
                outer.set_postfix(file=video_path.name, faces=total_faces)

                if db.execute(
                    "SELECT 1 FROM videos WHERE path = ?", (str(video_path),)
                ).fetchone():
                    skipped += 1
                    index_progress["videos_done"] += 1
                    outer.update(1)
                    continue

                cap = cv2.VideoCapture(str(video_path))
                if not cap.isOpened():
                    index_progress["videos_done"] += 1
                    outer.update(1)
                    continue

                fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
                raw_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                # CAP_PROP_FRAME_COUNT is unreliable; use only for display, not bar total
                duration_sec = (raw_count / fps) if fps > 0 and raw_count > 0 else 0.0

                face_ids: list[str] = []
                embeddings: list[list[float]] = []
                metadatas: list[dict] = []
                video_stem = _safe_id(video_path.stem)

                with tqdm(desc="  Frames", unit="frame", leave=False) as inner:
                    for frame, timestamp in _extract_frames(cap, interval_sec):
                        inner.set_postfix(t=f"{timestamp:.0f}/{duration_sec:.0f}s")

                        for face_idx, face in enumerate(fa.get(frame)):
                            if face.normed_embedding is None:
                                continue

                            face_id = _safe_id(
                                f"{video_stem}_{timestamp:.3f}_{face_idx}"
                            )
                            thumbnail_path = _save_thumbnail(frame, face.bbox, face_id)

                            face_ids.append(face_id)
                            embeddings.append(face.normed_embedding.tolist())
                            metadatas.append({
                                "timestamp_sec": float(timestamp),
                                "person_id": "unlabeled",
                                "thumbnail_path": thumbnail_path,
                                # video_id patched in after SQLite insert below
                                "video_id": -1,
                            })
                            total_faces += 1

                        inner.update(1)
                        index_progress["faces_found"] = total_faces

                cap.release()

                # Insert video row and patch video_id into collected metadatas
                recorded_at = _extract_recording_date(video_path)
                cursor = db.execute(
                    "INSERT INTO videos (path, filename, duration_sec, recorded_at)"
                    " VALUES (?, ?, ?, ?)",
                    (str(video_path), video_path.name, duration_sec, recorded_at),
                )
                db.commit()
                video_id = cursor.lastrowid

                for m in metadatas:
                    m["video_id"] = video_id

                if face_ids:
                    collection.upsert(
                        ids=face_ids,
                        embeddings=embeddings,
                        metadatas=metadatas,
                    )

                index_progress["videos_done"] += 1
                outer.update(1)
                outer.set_postfix(file=video_path.name, faces=total_faces)

                elapsed = time.time() - index_progress["started_at"]
                done = index_progress["videos_done"]
                vt = index_progress["videos_total"]
                rate = done / elapsed if elapsed > 0 else 0
                index_progress["eta_sec"] = ((vt - done) / rate) if rate > 0 else None

        db.close()

        elapsed = time.time() - index_progress["started_at"]
        h = int(elapsed // 3600)
        m = int((elapsed % 3600) // 60)
        s = int(elapsed % 60)
        newly_indexed = len(video_paths) - skipped
        if newly_indexed == 0:
            print(f"\nAll {skipped} video(s) already indexed — nothing to do.")
        else:
            skip_note = f" ({skipped} already indexed)" if skipped else ""
            print(
                f"\nDone. {newly_indexed} video(s) indexed{skip_note} | "
                f"{total_faces:,} faces | {h:02d}:{m:02d}:{s:02d}"
            )

        if auto_cluster and newly_indexed > 0:
            index_progress["current_video"] = "Clustering new faces…"
            from app.clusterer import run_incremental_clusterer
            run_incremental_clusterer(eps=eps)

        index_progress["status"] = "done"

    except Exception as exc:
        index_progress["status"] = "error"
        index_progress["error"] = str(exc)
        raise
