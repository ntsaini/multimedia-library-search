import numpy as np
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

from app.config import PHOTO_EXTENSIONS
from app.database import get_connection
from app.chroma import get_collection
from app.indexer import (
    index_progress,
    _init_face_analysis,
    _save_thumbnail,
    _safe_id,
)


def _extract_taken_at(path: Path) -> str | None:
    """Return ISO 8601 capture date from EXIF DateTimeOriginal, fallback to mtime."""
    try:
        from PIL import Image
        img = Image.open(path)
        exif = img.getexif()
        if exif:
            dt_str = exif.get(36867) or exif.get(306)  # DateTimeOriginal or DateTime
            if dt_str:
                dt = datetime.strptime(dt_str, "%Y:%m:%d %H:%M:%S")
                return dt.isoformat(timespec="seconds")
    except Exception:
        pass
    try:
        return datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds")
    except Exception:
        return None


def _load_photo(path: Path) -> np.ndarray | None:
    """Load photo via cv2; fall back to Pillow for HEIC and other formats."""
    import cv2
    img = cv2.imread(str(path))
    if img is not None:
        return img
    try:
        from PIL import Image
        pil = Image.open(path).convert("RGB")
        return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    except Exception:
        return None


def run_photo_indexer(
    directory: str,
    use_gpu: bool = False,
    auto_cluster: bool = True,
    eps: float = 0.6,
    _finalize: bool = True,
    paths_override: list | None = None,
) -> None:
    index_progress["status"] = "running"
    index_progress["error"] = None

    try:
        if paths_override is not None:
            photo_paths = [Path(p) for p in paths_override]
        else:
            photo_paths = sorted(
                p for p in Path(directory).rglob("*")
                if p.suffix.lower() in PHOTO_EXTENSIONS
            )

        index_progress["photos_total"] = len(photo_paths)
        index_progress["photos_done"] = 0

        if not photo_paths:
            print("No photo files found.")
            if _finalize:
                index_progress["status"] = "done"
            return

        fa = _init_face_analysis(use_gpu=use_gpu)
        collection = get_collection()
        db = get_connection()

        total_faces = 0
        skipped = 0

        try:
            with tqdm(total=len(photo_paths), desc="Photos", unit="photo") as bar:
                for photo_path in photo_paths:
                    index_progress["current_video"] = photo_path.name
                    bar.set_postfix(file=photo_path.name, faces=total_faces)

                    if paths_override is None and db.execute(
                        "SELECT 1 FROM photos WHERE path = ?", (str(photo_path),)
                    ).fetchone():
                        skipped += 1
                        index_progress["photos_done"] += 1
                        bar.update(1)
                        continue

                    frame = _load_photo(photo_path)
                    if frame is None:
                        index_progress["photos_done"] += 1
                        bar.update(1)
                        continue

                    taken_at = _extract_taken_at(photo_path)
                    cursor = db.execute(
                        "INSERT INTO photos (path, filename, taken_at) VALUES (?, ?, ?)",
                        (str(photo_path), photo_path.name, taken_at),
                    )
                    db.commit()
                    photo_id = cursor.lastrowid

                    photo_stem = _safe_id(photo_path.stem)
                    face_ids: list[str] = []
                    embeddings: list[list[float]] = []
                    metadatas: list[dict] = []

                    fh, fw = frame.shape[:2]
                    for face_idx, face in enumerate(fa.get(frame)):
                        if face.normed_embedding is None:
                            continue

                        # "photo__" prefix avoids any collision with video face IDs
                        face_id = _safe_id(f"photo__{photo_stem}_{face_idx}")
                        thumbnail_path = _save_thumbnail(frame, face.bbox, face_id)

                        x1, y1, x2, y2 = (float(v) for v in face.bbox)
                        face_ids.append(face_id)
                        embeddings.append(face.normed_embedding.tolist())
                        metadatas.append({
                            "person_id": "unlabeled",
                            "thumbnail_path": thumbnail_path,
                            "media_type": "photo",
                            "photo_id": photo_id,
                            "timestamp_sec": 0.0,
                            "video_id": -1,
                            "det_score": round(float(face.det_score), 4) if face.det_score is not None else 1.0,
                            "face_x1": round(x1 / fw, 4),
                            "face_y1": round(y1 / fh, 4),
                            "face_x2": round(x2 / fw, 4),
                            "face_y2": round(y2 / fh, 4),
                        })
                        total_faces += 1

                    if face_ids:
                        collection.upsert(
                            ids=face_ids,
                            embeddings=embeddings,
                            metadatas=metadatas,
                        )

                    index_progress["photos_done"] += 1
                    index_progress["faces_found"] = (index_progress.get("faces_found") or 0) + len(face_ids)
                    bar.update(1)
                    bar.set_postfix(file=photo_path.name, faces=total_faces)
        finally:
            db.close()

        newly_indexed = len(photo_paths) - skipped
        if newly_indexed == 0:
            print(f"\nAll {skipped} photo(s) already indexed — nothing to do.")
        else:
            skip_note = f" ({skipped} already indexed)" if skipped else ""
            print(f"\nDone. {newly_indexed} photo(s) indexed{skip_note} | {total_faces:,} faces")

        if _finalize:
            if auto_cluster and newly_indexed > 0:
                index_progress["current_video"] = "Clustering new faces…"
                from app.clusterer import run_incremental_clusterer
                run_incremental_clusterer(eps=eps)
            index_progress["status"] = "done"

    except Exception as exc:
        index_progress["status"] = "error"
        index_progress["error"] = str(exc)
        raise
