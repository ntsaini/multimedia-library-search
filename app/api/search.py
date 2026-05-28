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
        f"SELECT id, filename, duration_sec FROM videos WHERE id IN ({placeholders})", video_ids
    ).fetchall()
    return {r["id"]: {"filename": r["filename"], "duration_sec": r["duration_sec"]} for r in rows}


def _fetch_photo_map(conn, photo_ids: list) -> dict:
    if not photo_ids:
        return {}
    placeholders = ",".join("?" * len(photo_ids))
    rows = conn.execute(
        f"SELECT id, filename, taken_at FROM photos WHERE id IN ({placeholders})", photo_ids
    ).fetchall()
    return {r["id"]: {"filename": r["filename"], "taken_at": r["taken_at"]} for r in rows}


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


def _dedup_photos(hits: list) -> list:
    """Keep one hit per photo_id."""
    seen: set = set()
    out = []
    for h in hits:
        pid = h["photo_id"]
        if pid not in seen:
            seen.add(pid)
            out.append(h)
    return out


def _dedup_photos_best(hits: list) -> list:
    """Keep the lowest-distance hit per photo_id (for photo-upload search)."""
    best: dict = {}
    for h in hits:
        pid = h["photo_id"]
        if pid not in best or h["distance"] < best[pid]["distance"]:
            best[pid] = h
    return list(best.values())


def _split_metas(metas: list) -> tuple[list, list]:
    """Split face metadata into (video_metas, photo_metas)."""
    videos, photos = [], []
    for m in metas:
        if m.get("media_type") == "photo":
            photos.append(m)
        else:
            videos.append(m)
    return videos, photos


@router.get("/api/search")
def search_by_name(name: str = ""):
    name = name.strip()
    if not name:
        return {"videos": [], "photos": []}

    conn = get_connection()
    rows = conn.execute(
        "SELECT id, name FROM persons WHERE name IS NOT NULL AND LOWER(name) LIKE LOWER(?)",
        (f"%{name}%",),
    ).fetchall()

    if not rows:
        conn.close()
        return {"videos": [], "photos": []}

    person_name_map = {r["id"]: r["name"] for r in rows}
    person_ids = list(person_name_map)

    collection = get_collection()
    result = collection.get(
        where={"person_id": {"$in": person_ids}},
        include=["metadatas"],
    )

    video_hits: list = []
    photo_hits: list = []

    if result["ids"]:
        video_metas, photo_metas = _split_metas(result["metadatas"])

        video_ids = list({m.get("video_id") for m in video_metas if m.get("video_id", -1) != -1})
        video_map = _fetch_video_map(conn, video_ids)
        for meta in video_metas:
            vid_id = meta.get("video_id")
            vinfo = video_map.get(vid_id) or {}
            video_hits.append({
                "video_id": vid_id,
                "filename": vinfo.get("filename", "unknown"),
                "duration_sec": vinfo.get("duration_sec"),
                "timestamp_sec": meta.get("timestamp_sec"),
                "thumbnail_path": meta.get("thumbnail_path"),
                "person_name": person_name_map.get(meta.get("person_id")),
                "person_id": meta.get("person_id"),
            })

        photo_ids = list({m.get("photo_id") for m in photo_metas if m.get("photo_id") is not None})
        photo_map = _fetch_photo_map(conn, photo_ids)
        for meta in photo_metas:
            ph_id = meta.get("photo_id")
            pinfo = photo_map.get(ph_id) or {}
            photo_hits.append({
                "photo_id": ph_id,
                "filename": pinfo.get("filename", "unknown"),
                "taken_at": pinfo.get("taken_at"),
                "thumbnail_path": meta.get("thumbnail_path"),
                "person_name": person_name_map.get(meta.get("person_id")),
                "person_id": meta.get("person_id"),
            })

    conn.close()

    video_hits.sort(key=lambda h: (h["video_id"], h["timestamp_sec"] or 0))
    video_hits = _dedup_by_minute(video_hits)

    photo_hits = _dedup_photos(photo_hits)
    photo_hits.sort(key=lambda h: (h["taken_at"] or "", h["filename"]))

    return {"videos": video_hits[:200], "photos": photo_hits[:200]}


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
        return {"videos": [], "photos": []}

    results = collection.query(
        query_embeddings=[embedding],
        n_results=min(100, total),
        include=["metadatas", "distances"],
    )

    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    conn = get_connection()

    video_metas_dist = [(m, d) for m, d in zip(metadatas, distances)
                        if d <= 0.5 and m.get("media_type") != "photo"]
    photo_metas_dist = [(m, d) for m, d in zip(metadatas, distances)
                        if d <= 0.5 and m.get("media_type") == "photo"]

    # Videos
    video_ids = list({m.get("video_id") for m, _ in video_metas_dist if m.get("video_id", -1) != -1})
    video_map = _fetch_video_map(conn, video_ids)

    all_ids = list({m.get("person_id", "unlabeled") for m, d in zip(metadatas, distances) if d <= 0.5})
    placeholders = ",".join("?" * len(all_ids))
    person_rows = conn.execute(
        f"SELECT id, name FROM persons WHERE id IN ({placeholders})", all_ids
    ).fetchall() if all_ids else []
    person_map = {r["id"]: r["name"] for r in person_rows}

    # Photos
    photo_ids = list({m.get("photo_id") for m, _ in photo_metas_dist if m.get("photo_id") is not None})
    photo_map = _fetch_photo_map(conn, photo_ids)

    conn.close()

    video_hits = []
    for meta, dist in video_metas_dist:
        vid_id = meta.get("video_id")
        person_id = meta.get("person_id", "unlabeled")
        vinfo = video_map.get(vid_id) or {}
        video_hits.append({
            "video_id": vid_id,
            "filename": vinfo.get("filename", "unknown"),
            "duration_sec": vinfo.get("duration_sec"),
            "timestamp_sec": meta.get("timestamp_sec"),
            "thumbnail_path": meta.get("thumbnail_path"),
            "distance": round(float(dist), 3),
            "person_name": person_map.get(person_id),
            "person_id": person_id if person_id != "unlabeled" else None,
        })

    photo_hits_raw = []
    for meta, dist in photo_metas_dist:
        ph_id = meta.get("photo_id")
        person_id = meta.get("person_id", "unlabeled")
        pinfo = photo_map.get(ph_id) or {}
        photo_hits_raw.append({
            "photo_id": ph_id,
            "filename": pinfo.get("filename", "unknown"),
            "taken_at": pinfo.get("taken_at"),
            "thumbnail_path": meta.get("thumbnail_path"),
            "distance": round(float(dist), 3),
            "person_name": person_map.get(person_id),
            "person_id": person_id if person_id != "unlabeled" else None,
        })

    video_hits = _dedup_by_minute(video_hits)

    photo_hits = _dedup_photos_best(photo_hits_raw)
    photo_hits.sort(key=lambda h: h["distance"])

    return {"videos": video_hits, "photos": photo_hits}
