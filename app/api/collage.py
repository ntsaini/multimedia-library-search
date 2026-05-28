import os
import threading
import uuid

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from app.collage import _jobs, run_collage

router = APIRouter()


class CollageRequest(BaseModel):
    person_id: str
    columns: int = 3
    sort: str = "asc"
    captions: bool = True


@router.post("/api/collage")
def start_collage(req: CollageRequest):
    job_id = str(uuid.uuid4())
    _jobs[job_id] = {
        "status": "running",
        "progress": 0.0,
        "photos_total": 0,
        "photos_done": 0,
        "error": None,
        "output_path": None,
        "filename": None,
    }
    threading.Thread(
        target=run_collage,
        args=(job_id, req.person_id, req.columns, req.sort, req.captions),
        daemon=True,
    ).start()
    return {"job_id": job_id}


@router.get("/api/collage/{job_id}")
def get_collage_status(job_id: str):
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    return job


@router.get("/api/collage/{job_id}/download")
def download_collage(job_id: str):
    job = _jobs.get(job_id)
    if not job or job["status"] != "done":
        raise HTTPException(404, "Collage not ready")
    path = job.get("output_path")
    if not path or not os.path.isfile(path):
        raise HTTPException(404, "Output file not found on disk")
    mt = "image/jpeg" if path.endswith(".jpg") else "image/png"
    return FileResponse(path, media_type=mt, filename=job["filename"])
