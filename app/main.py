from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import json
from app.config import BASE_DIR, OUTPUT_DIR, THUMBNAILS_DIR
from app.database import init_db, get_connection
from app.chroma import get_collection
from app.api import index as index_api
from app.api import persons as persons_api
from app.api import cluster as cluster_api
from app.api import search as search_api
from app.api import video as video_api
from app.api import compile as compile_api
from app.api import photo as photo_api
from app.api import collage as collage_api
from app.api import system as system_api


@asynccontextmanager
async def lifespan(app: FastAPI):
    THUMBNAILS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    init_db()
    get_collection()
    yield


app = FastAPI(title="Multimedia Library Search", lifespan=lifespan)

app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

app.include_router(index_api.router)
app.include_router(persons_api.router)
app.include_router(cluster_api.router)
app.include_router(search_api.router)
app.include_router(video_api.router)
app.include_router(compile_api.router)
app.include_router(photo_api.router)
app.include_router(collage_api.router)
app.include_router(system_api.router)


@app.get("/")
def home(request: Request):
    conn = get_connection()
    videos = conn.execute(
        "SELECT filename, path, duration_sec, indexed_at FROM videos ORDER BY indexed_at DESC"
    ).fetchall()
    photos = conn.execute(
        "SELECT filename, path, taken_at, indexed_at FROM photos ORDER BY indexed_at DESC"
    ).fetchall()
    conn.close()
    return templates.TemplateResponse(request, "index.html", {
        "videos": [dict(v) for v in videos],
        "photos": [dict(p) for p in photos],
    })


@app.get("/search")
def search_page(request: Request):
    conn = get_connection()
    persons = conn.execute(
        "SELECT id, name FROM persons WHERE name IS NOT NULL ORDER BY name"
    ).fetchall()
    conn.close()
    return templates.TemplateResponse(request, "search.html", {"persons": [dict(p) for p in persons]})


@app.get("/label")
def label_page(request: Request):
    conn = get_connection()
    rows = conn.execute(
        "SELECT id, name, thumbnail_path, face_count, samples FROM persons ORDER BY face_count DESC"
    ).fetchall()
    conn.close()

    persons = []
    for row in rows:
        person = dict(row)
        person["samples"] = json.loads(person["samples"] or "[]")
        persons.append(person)

    collection = get_collection()
    unlabeled_result = collection.get(
        where={"person_id": {"$eq": "unlabeled"}}, include=["metadatas"]
    )
    DET_THRESHOLD = 0.6
    all_unlabeled = [
        {
            "id": fid,
            "thumbnail_path": m.get("thumbnail_path", ""),
            "media_type": m.get("media_type", "video"),
            "det_score": m.get("det_score"),
        }
        for fid, m in zip(unlabeled_result["ids"], unlabeled_result["metadatas"])
        if m.get("thumbnail_path")
    ]
    # Sort: scored faces by det_score desc, unscored faces at end
    all_unlabeled.sort(
        key=lambda f: f["det_score"] if f["det_score"] is not None else -1,
        reverse=True,
    )
    visible_faces = [f for f in all_unlabeled if f["det_score"] is None or f["det_score"] >= DET_THRESHOLD]
    hidden_faces = [f for f in all_unlabeled if f["det_score"] is not None and f["det_score"] < DET_THRESHOLD]

    return templates.TemplateResponse(request, "label.html", {
        "persons": persons,
        "visible_faces": visible_faces,
        "hidden_faces": hidden_faces,
        "unlabeled_count": len(unlabeled_result["ids"]),
    })
