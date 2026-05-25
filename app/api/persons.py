from fastapi import APIRouter, Form
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from app.chroma import get_collection
from app.database import get_connection

router = APIRouter()


class MergeRequest(BaseModel):
    source_id: str
    target_id: str


@router.get("/api/persons")
def list_persons():
    conn = get_connection()
    rows = conn.execute(
        "SELECT id, name, thumbnail_path, face_count FROM persons ORDER BY face_count DESC"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


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
    source_row = conn.execute(
        "SELECT face_count FROM persons WHERE id = ?", (req.source_id,)
    ).fetchone()
    if source_row:
        conn.execute(
            "UPDATE persons SET face_count = face_count + ? WHERE id = ?",
            (source_row["face_count"], req.target_id),
        )
    conn.execute("DELETE FROM persons WHERE id = ?", (req.source_id,))
    conn.commit()
    conn.close()

    return {"status": "merged"}


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
