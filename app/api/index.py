import threading
from fastapi import APIRouter
from pydantic import BaseModel

from app.indexer import index_progress, run_indexer

router = APIRouter()


class IndexRequest(BaseModel):
    directory_path: str
    interval_sec: float = 1.0
    auto_cluster: bool = True
    eps: float = 0.6


@router.post("/api/index")
def start_index(req: IndexRequest):
    if index_progress["status"] == "running":
        return {"status": "already_running"}
    thread = threading.Thread(
        target=run_indexer,
        kwargs={
            "directory": req.directory_path,
            "interval_sec": req.interval_sec,
            "auto_cluster": req.auto_cluster,
            "eps": req.eps,
        },
        daemon=True,
    )
    thread.start()
    return {"status": "started"}


@router.get("/api/index/status")
def get_status():
    p = index_progress
    total = p["videos_total"]
    done = p["videos_done"]
    return {
        **p,
        "progress": (done / total) if total > 0 else 0.0,
    }
