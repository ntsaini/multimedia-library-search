import time
import re
import cv2
import numpy as np
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
                cursor = db.execute(
                    "INSERT INTO videos (path, filename, duration_sec) VALUES (?, ?, ?)",
                    (str(video_path), video_path.name, duration_sec),
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
