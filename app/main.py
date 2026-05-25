from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.config import BASE_DIR, THUMBNAILS_DIR
from app.database import init_db
from app.chroma import get_collection
from app.api import index as index_api


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


@app.get("/")
def home(request: Request):
    return templates.TemplateResponse(request, "index.html")
