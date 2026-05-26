import os
import threading
import uuid

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from app.compiler import _jobs, run_compile

router = APIRouter()


class CompileRequest(BaseModel):
    person_id: str
    clip_duration_sec: int = 30
    merge_gap_sec: float = 30.0
    max_clips_per_video: int = 5


@router.post("/api/compile")
def start_compile(req: CompileRequest):
    job_id = str(uuid.uuid4())
    _jobs[job_id] = {
        "status": "running",
        "progress": 0.0,
        "segments_total": 0,
        "segments_done": 0,
        "error": None,
        "output_path": None,
        "filename": None,
    }
    threading.Thread(
        target=run_compile,
        args=(
            job_id,
            req.person_id,
            float(req.clip_duration_sec),
            req.merge_gap_sec,
            req.max_clips_per_video,
        ),
        daemon=True,
    ).start()
    return {"job_id": job_id}


@router.get("/api/compile/{job_id}")
def get_compile_status(job_id: str):
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    return job


@router.get("/api/compile/{job_id}/download")
def download_reel(job_id: str):
    job = _jobs.get(job_id)
    if not job or job["status"] != "done":
        raise HTTPException(404, "Reel not ready")
    path = job.get("output_path")
    if not path or not os.path.isfile(path):
        raise HTTPException(404, "Output file not found on disk")
    return FileResponse(path, media_type="video/mp4", filename=job["filename"])
