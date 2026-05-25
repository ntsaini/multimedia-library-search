from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import json
from app.config import BASE_DIR, THUMBNAILS_DIR
from app.database import init_db, get_connection
from app.chroma import get_collection
from app.api import index as index_api
from app.api import persons as persons_api


@asynccontextmanager
async def lifespan(app: FastAPI):
    THUMBNAILS_DIR.mkdir(parents=True, exist_ok=True)
    init_db()
    get_collection()
    yield


app = FastAPI(title="Multimedia Library Search", lifespan=lifespan)

app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

app.include_router(index_api.router)
app.include_router(persons_api.router)


@app.get("/")
def home(request: Request):
    conn = get_connection()
    videos = conn.execute(
        "SELECT filename, path, duration_sec, indexed_at FROM videos ORDER BY indexed_at DESC"
    ).fetchall()
    conn.close()
    return templates.TemplateResponse(request, "index.html", {"videos": [dict(v) for v in videos]})


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

    return templates.TemplateResponse(request, "label.html", {"persons": persons})
