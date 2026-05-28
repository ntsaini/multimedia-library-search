import threading
from fastapi import APIRouter
from pydantic import BaseModel

from app.clusterer import run_clusterer, run_incremental_clusterer

router = APIRouter()

cluster_progress: dict = {
    "status": "idle",   # idle | running | done | error
    "mode": None,       # "incremental" | "full"
    "result": None,
    "error": None,
}


class ClusterRequest(BaseModel):
    incremental: bool = True
    eps_video: float = 0.7
    eps_photo: float = 1.0
    min_samples: int = 3


def _run(incremental: bool, eps_video: float, eps_photo: float, min_samples: int) -> None:
    try:
        if incremental:
            result = run_incremental_clusterer(
                eps_video=eps_video, eps_photo=eps_photo, min_samples=min_samples
            )
        else:
            result = run_clusterer(
                eps_video=eps_video, eps_photo=eps_photo, min_samples=min_samples
            )
        cluster_progress["result"] = result
        cluster_progress["status"] = "done"
    except Exception as exc:
        cluster_progress["status"] = "error"
        cluster_progress["error"] = str(exc)


@router.post("/api/cluster")
def start_cluster(req: ClusterRequest):
    if cluster_progress["status"] == "running":
        return {"status": "already_running"}
    cluster_progress.update({
        "status": "running",
        "mode": "incremental" if req.incremental else "full",
        "result": None,
        "error": None,
    })
    threading.Thread(
        target=_run,
        kwargs={
            "incremental": req.incremental,
            "eps_video": req.eps_video,
            "eps_photo": req.eps_photo,
            "min_samples": req.min_samples,
        },
        daemon=True,
    ).start()
    return {"status": "started"}


@router.get("/api/cluster/status")
def get_cluster_status():
    return cluster_progress
