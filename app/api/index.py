import time
import threading
from pathlib import Path
from fastapi import APIRouter
from pydantic import BaseModel

from app.config import VIDEO_EXTENSIONS, PHOTO_EXTENSIONS
from app.indexer import index_progress, run_indexer, clear_paths_from_index
from app.photo_indexer import run_photo_indexer

router = APIRouter()

_EPS_VIDEO = 0.7
_EPS_PHOTO = 1.0


class IndexRequest(BaseModel):
    directory_path: str
    interval_sec: float = 1.0
    auto_cluster: bool = True


class ForceIndexRequest(BaseModel):
    paths: list[str]
    interval_sec: float = 1.0
    auto_cluster: bool = True


def _run_full_index(directory: str, interval_sec: float, use_gpu: bool, auto_cluster: bool) -> None:
    dir_path = Path(directory)
    v_total = sum(1 for p in dir_path.rglob("*") if p.suffix.lower() in VIDEO_EXTENSIONS)
    ph_total = sum(1 for p in dir_path.rglob("*") if p.suffix.lower() in PHOTO_EXTENSIONS)
    index_progress.update({
        "status": "running",
        "videos_total": v_total,
        "videos_done": 0,
        "photos_total": ph_total,
        "photos_done": 0,
        "current_video": "",
        "faces_found": 0,
        "started_at": time.time(),
        "eta_sec": None,
        "error": None,
    })
    try:
        run_indexer(directory, interval_sec, use_gpu, auto_cluster=False, _finalize=False)
    except Exception:
        return
    try:
        run_photo_indexer(directory, use_gpu=use_gpu, auto_cluster=False, _finalize=False)
    except Exception:
        return
    if auto_cluster:
        index_progress["current_video"] = "Clustering new faces…"
        from app.clusterer import run_incremental_clusterer
        run_incremental_clusterer(eps_video=_EPS_VIDEO, eps_photo=_EPS_PHOTO)
    index_progress["current_video"] = ""
    index_progress["status"] = "done"


def _run_selective_reindex(paths: list[str], interval_sec: float,
                           use_gpu: bool, auto_cluster: bool) -> None:
    video_paths = [p for p in paths if Path(p).suffix.lower() in VIDEO_EXTENSIONS]
    photo_paths = [p for p in paths if Path(p).suffix.lower() in PHOTO_EXTENSIONS]

    index_progress.update({
        "status": "running",
        "videos_total": len(video_paths),
        "videos_done": 0,
        "photos_total": len(photo_paths),
        "photos_done": 0,
        "current_video": f"Clearing existing data for {len(paths)} file(s)…",
        "faces_found": 0,
        "started_at": time.time(),
        "eta_sec": None,
        "error": None,
    })

    clear_paths_from_index(paths)
    index_progress["current_video"] = "Loading face detector…"

    try:
        if video_paths:
            run_indexer("", interval_sec, use_gpu, auto_cluster=False, _finalize=False,
                        paths_override=video_paths)
        if photo_paths:
            run_photo_indexer("", use_gpu=use_gpu, auto_cluster=False, _finalize=False,
                              paths_override=photo_paths)
    except Exception:
        return

    if auto_cluster:
        index_progress["current_video"] = "Clustering new faces…"
        from app.clusterer import run_incremental_clusterer
        run_incremental_clusterer(eps_video=_EPS_VIDEO, eps_photo=_EPS_PHOTO)
    index_progress["current_video"] = ""
    index_progress["status"] = "done"


@router.post("/api/index/force")
def force_index(req: ForceIndexRequest):
    if index_progress["status"] == "running":
        return {"status": "already_running"}
    if not req.paths:
        return {"status": "error", "detail": "No paths provided"}
    threading.Thread(
        target=_run_selective_reindex,
        kwargs={
            "paths": req.paths,
            "interval_sec": req.interval_sec,
            "use_gpu": False,
            "auto_cluster": req.auto_cluster,
        },
        daemon=True,
    ).start()
    return {"status": "started", "count": len(req.paths)}


@router.post("/api/index")
def start_index(req: IndexRequest):
    if index_progress["status"] == "running":
        return {"status": "already_running"}
    threading.Thread(
        target=_run_full_index,
        kwargs={
            "directory": req.directory_path,
            "interval_sec": req.interval_sec,
            "use_gpu": False,
            "auto_cluster": req.auto_cluster,
        },
        daemon=True,
    ).start()
    return {"status": "started"}


@router.get("/api/index/status")
def get_status():
    p = index_progress
    v_total = p["videos_total"]
    v_done = p["videos_done"]
    ph_total = p["photos_total"]
    ph_done = p["photos_done"]
    total = v_total + ph_total
    done = v_done + ph_done
    return {
        **p,
        "progress": (done / total) if total > 0 else 0.0,
    }
