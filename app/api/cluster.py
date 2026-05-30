import threading
from fastapi import APIRouter
from pydantic import BaseModel

from app.clusterer import run_clusterer, run_incremental_clusterer
from app.config import LABELED_FACES_DIR
from app.labeled_faces import auto_label_persons, count_candidate_refs

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
    auto_label: bool = True
    label_threshold: float = 0.6
    label_margin: float = 0.08


class AutoLabelRequest(BaseModel):
    threshold: float = 0.6
    margin: float = 0.08
    overwrite: bool = False


def _run(
    incremental: bool,
    eps_video: float,
    eps_photo: float,
    min_samples: int,
    auto_label: bool,
    label_threshold: float,
    label_margin: float,
) -> None:
    try:
        kwargs = dict(
            eps_video=eps_video,
            eps_photo=eps_photo,
            min_samples=min_samples,
            auto_label=auto_label,
            label_threshold=label_threshold,
            label_margin=label_margin,
        )
        if incremental:
            result = run_incremental_clusterer(**kwargs)
        else:
            result = run_clusterer(**kwargs)
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
            "auto_label": req.auto_label,
            "label_threshold": req.label_threshold,
            "label_margin": req.label_margin,
        },
        daemon=True,
    ).start()
    return {"status": "started"}


@router.get("/api/cluster/status")
def get_cluster_status():
    return cluster_progress


@router.get("/api/cluster/label-refs")
def get_label_refs():
    exists = LABELED_FACES_DIR.exists() and LABELED_FACES_DIR.is_dir()
    return {
        "exists": exists,
        "candidate_people": count_candidate_refs(LABELED_FACES_DIR) if exists else 0,
        "directory": LABELED_FACES_DIR.name,
    }


@router.post("/api/cluster/auto-label")
def post_auto_label(req: AutoLabelRequest):
    return auto_label_persons(
        label_dir=LABELED_FACES_DIR,
        threshold=req.threshold,
        margin=req.margin,
        overwrite=req.overwrite,
    )
