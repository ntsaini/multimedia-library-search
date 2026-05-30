import json
import uuid

import numpy as np
from fastapi import APIRouter, Form, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from app.chroma import get_collection
from app.database import get_connection
from app.services.person_service import get_person, list_people

router = APIRouter()


class MergeRequest(BaseModel):
    source_id: str
    target_id: str


@router.get("/api/persons")
def list_persons():
    return list_people(include_unnamed=True)


@router.get("/api/persons/{person_id}")
def get_person_by_id(person_id: str):
    person = get_person(person_id)
    if person is None:
        raise HTTPException(status_code=404, detail="Person not found")
    return person


@router.post("/api/persons/merge")
def merge_persons(req: MergeRequest):
    collection = get_collection()

    result = collection.get(
        where={"person_id": {"$eq": req.source_id}},
        include=["metadatas"],
    )
    source_face_ids = result["ids"]
    source_metas = result["metadatas"]

    if source_face_ids:
        for m in source_metas:
            m["person_id"] = req.target_id
        batch_size = 500
        for i in range(0, len(source_face_ids), batch_size):
            collection.update(
                ids=source_face_ids[i : i + batch_size],
                metadatas=source_metas[i : i + batch_size],
            )

    conn = get_connection()
    rows = conn.execute(
        "SELECT id, face_count, centroid FROM persons WHERE id IN (?, ?)",
        (req.source_id, req.target_id),
    ).fetchall()
    by_id = {r["id"]: r for r in rows}

    src = by_id.get(req.source_id)
    tgt = by_id.get(req.target_id)

    if src and tgt:
        n_s, n_t = src["face_count"], tgt["face_count"]
        merged_centroid = None
        if src["centroid"] and tgt["centroid"]:
            c_s = np.array(json.loads(src["centroid"]), dtype=np.float32)
            c_t = np.array(json.loads(tgt["centroid"]), dtype=np.float32)
            merged = (c_t * n_t + c_s * n_s) / (n_t + n_s)
            norm = np.linalg.norm(merged)
            merged_centroid = json.dumps((merged / norm if norm > 0 else merged).tolist())

        conn.execute(
            "UPDATE persons SET face_count = ?, centroid = ? WHERE id = ?",
            (n_t + n_s, merged_centroid, req.target_id),
        )

    conn.execute("DELETE FROM persons WHERE id = ?", (req.source_id,))
    conn.commit()
    conn.close()

    return {"status": "merged"}


@router.post("/api/faces/{face_id}/promote")
def promote_face_to_person(face_id: str, name: str = Form(default="")):
    """Create a new single-face person from an unclustered noise face."""
    collection = get_collection()
    result = collection.get(ids=[face_id], include=["embeddings", "metadatas"])
    if not result["ids"]:
        raise HTTPException(status_code=404, detail="Face not found")

    meta = result["metadatas"][0]
    if meta.get("person_id") != "unlabeled":
        raise HTTPException(status_code=400, detail="Face is already assigned to a person")

    embedding = np.array(result["embeddings"][0], dtype=np.float32)
    norm = np.linalg.norm(embedding)
    centroid = embedding / norm if norm > 0 else embedding

    person_id = str(uuid.uuid4())
    thumbnail_path = meta.get("thumbnail_path", "")
    label = name.strip() or None

    conn = get_connection()
    conn.execute(
        "INSERT INTO persons (id, name, thumbnail_path, face_count, samples, centroid)"
        " VALUES (?, ?, ?, ?, ?, ?)",
        (person_id, label, thumbnail_path, 1, json.dumps([]), json.dumps(centroid.tolist())),
    )
    conn.commit()
    conn.close()

    meta["person_id"] = person_id
    collection.update(ids=[face_id], metadatas=[meta])

    return {"person_id": person_id, "thumbnail_path": thumbnail_path}


@router.post("/api/persons/{person_id}/label")
def label_person(person_id: str, name: str = Form(...)):
    name = name.strip()
    conn = get_connection()
    conn.execute(
        "UPDATE persons SET name = ? WHERE id = ?",
        (name if name else None, person_id),
    )
    conn.commit()
    conn.close()
    display = name if name else "Unknown"
    return HTMLResponse(f'<span class="save-ok">&#10003; Saved as "{display}"</span>')
