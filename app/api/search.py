from fastapi import APIRouter, File, HTTPException, UploadFile

import cv2
import numpy as np
from insightface.app import FaceAnalysis

from app.chroma import get_collection
from app.config import FACE_DET_SIZE, MODEL_NAME_DEFAULT
from app.database import get_connection

router = APIRouter()

_fa: FaceAnalysis | None = None


def _get_face_analysis() -> FaceAnalysis:
    global _fa
    if _fa is None:
        _fa = FaceAnalysis(name=MODEL_NAME_DEFAULT, providers=["CPUExecutionProvider"])
        _fa.prepare(ctx_id=0, det_size=FACE_DET_SIZE)
    return _fa


def _fetch_video_map(conn, video_ids: list) -> dict:
    if not video_ids:
        return {}
    placeholders = ",".join("?" * len(video_ids))
    rows = conn.execute(
        f"SELECT id, filename FROM videos WHERE id IN ({placeholders})", video_ids
    ).fetchall()
    return {r["id"]: r["filename"] for r in rows}


def _dedup_by_minute(hits: list) -> list:
    """Keep one hit per (video_id, minute) to avoid flooding results."""
    seen: set = set()
    out = []
    for h in hits:
        key = (h["video_id"], int((h["timestamp_sec"] or 0) // 60))
        if key not in seen:
            seen.add(key)
            out.append(h)
    return out


@router.get("/api/search")
def search_by_name(name: str = ""):
    name = name.strip()
    if not name:
        return []

    conn = get_connection()
    # Exact match first, then partial
    rows = conn.execute(
        "SELECT id, name FROM persons WHERE name IS NOT NULL AND LOWER(name) LIKE LOWER(?)",
        (f"%{name}%",),
    ).fetchall()

    if not rows:
        conn.close()
        return []

    collection = get_collection()
    hits = []

    for person_row in rows:
        person_id = person_row["id"]
        person_name = person_row["name"]

        result = collection.get(
            where={"person_id": {"$eq": person_id}},
            include=["metadatas"],
        )
        if not result["ids"]:
            continue

        video_ids = list({m.get("video_id") for m in result["metadatas"]})
        video_map = _fetch_video_map(conn, video_ids)

        for meta in result["metadatas"]:
            vid_id = meta.get("video_id")
            hits.append(
                {
                    "video_id": vid_id,
                    "filename": video_map.get(vid_id, "unknown"),
                    "timestamp_sec": meta.get("timestamp_sec"),
                    "thumbnail_path": meta.get("thumbnail_path"),
                    "person_name": person_name,
                }
            )

    conn.close()
    hits.sort(key=lambda h: (h["video_id"], h["timestamp_sec"] or 0))
    hits = _dedup_by_minute(hits)
    return hits[:200]


@router.post("/api/search/photo")
async def search_by_photo(file: UploadFile = File(...)):
    data = await file.read()
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "Could not decode image")

    fa = _get_face_analysis()
    faces = fa.get(img)
    if not faces:
        raise HTTPException(422, "No face detected in the uploaded image")

    embedding = faces[0].normed_embedding.tolist()

    collection = get_collection()
    total = collection.count()
    if total == 0:
        return []

    results = collection.query(
        query_embeddings=[embedding],
        n_results=min(50, total),
        include=["metadatas", "distances"],
    )

    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    conn = get_connection()
    video_ids = list({m.get("video_id") for m in metadatas})
    video_map = _fetch_video_map(conn, video_ids)

    person_ids = list({m.get("person_id", "unlabeled") for m in metadatas})
    placeholders = ",".join("?" * len(person_ids))
    person_rows = conn.execute(
        f"SELECT id, name FROM persons WHERE id IN ({placeholders})", person_ids
    ).fetchall()
    person_map = {r["id"]: r["name"] for r in person_rows}
    conn.close()

    hits = []
    for meta, dist in zip(metadatas, distances):
        if dist > 0.5:
            continue
        vid_id = meta.get("video_id")
        person_id = meta.get("person_id", "unlabeled")
        hits.append(
            {
                "video_id": vid_id,
                "filename": video_map.get(vid_id, "unknown"),
                "timestamp_sec": meta.get("timestamp_sec"),
                "thumbnail_path": meta.get("thumbnail_path"),
                "distance": round(float(dist), 3),
                "person_name": person_map.get(person_id),
            }
        )

    return hits
