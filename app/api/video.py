import os
from functools import lru_cache

import cv2
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response, StreamingResponse

from app.database import get_connection

router = APIRouter()


@lru_cache(maxsize=500)
def _extract_frame(path: str, t_sec: int) -> bytes:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return b""
    cap.set(cv2.CAP_PROP_POS_MSEC, t_sec * 1000)
    ret, frame = cap.read()
    if not ret and t_sec > 0:
        cap.set(cv2.CAP_PROP_POS_MSEC, max(0, t_sec - 1) * 1000)
        ret, frame = cap.read()
    cap.release()
    if not ret:
        return b""
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
    return buf.tobytes()


@router.get("/api/frame/{video_id}")
def get_frame(video_id: int, t: float = 0.0):
    conn = get_connection()
    row = conn.execute("SELECT path FROM videos WHERE id = ?", (video_id,)).fetchone()
    conn.close()

    if not row:
        raise HTTPException(404, "Video not found")

    path = row["path"]
    if not os.path.isfile(path):
        raise HTTPException(404, "Video file not found on disk")

    data = _extract_frame(path, int(t))
    if not data:
        raise HTTPException(404, "Could not extract frame at that timestamp")

    return Response(content=data, media_type="image/jpeg",
                    headers={"Cache-Control": "public, max-age=3600"})

_MIME = {
    ".mp4": "video/mp4",
    ".avi": "video/x-msvideo",
    ".mov": "video/quicktime",
    ".mkv": "video/x-matroska",
}


def _stream(path: str, start: int, end: int):
    chunk = 256 * 1024
    with open(path, "rb") as f:
        f.seek(start)
        remaining = end - start + 1
        while remaining > 0:
            data = f.read(min(chunk, remaining))
            if not data:
                break
            remaining -= len(data)
            yield data


@router.get("/api/video/{video_id}")
def serve_video(video_id: int, request: Request):
    conn = get_connection()
    row = conn.execute("SELECT path FROM videos WHERE id = ?", (video_id,)).fetchone()
    conn.close()

    if not row:
        raise HTTPException(404, "Video not found")

    path = row["path"]
    if not os.path.isfile(path):
        raise HTTPException(404, "Video file not found on disk")

    file_size = os.path.getsize(path)
    ext = os.path.splitext(path)[1].lower()
    media_type = _MIME.get(ext, "video/mp4")

    range_header = request.headers.get("range")
    if not range_header:
        return StreamingResponse(
            _stream(path, 0, file_size - 1),
            media_type=media_type,
            headers={"Content-Length": str(file_size), "Accept-Ranges": "bytes"},
        )

    try:
        raw = range_header.strip().removeprefix("bytes=")
        start_s, _, end_s = raw.partition("-")
        start = int(start_s) if start_s else 0
        end = int(end_s) if end_s else file_size - 1
    except ValueError:
        raise HTTPException(416, "Invalid Range header")

    if start >= file_size or start > end:
        raise HTTPException(
            416,
            "Requested range not satisfiable",
            headers={"Content-Range": f"bytes */{file_size}"},
        )

    end = min(end, file_size - 1)
    length = end - start + 1

    return StreamingResponse(
        _stream(path, start, end),
        status_code=206,
        media_type=media_type,
        headers={
            "Content-Range": f"bytes {start}-{end}/{file_size}",
            "Content-Length": str(length),
            "Accept-Ranges": "bytes",
        },
    )


@router.get("/api/video/{video_id}/info")
def video_info(video_id: int):
    conn = get_connection()
    row = conn.execute(
        "SELECT id, path, filename, duration_sec, recorded_at, indexed_at FROM videos WHERE id = ?",
        (video_id,),
    ).fetchone()
    conn.close()

    if not row:
        raise HTTPException(404, "Video not found")

    path = row["path"]
    return {
        "id": row["id"],
        "media_type": "video",
        "filename": row["filename"],
        "path": path,
        "duration_sec": row["duration_sec"],
        "recorded_at": row["recorded_at"],
        "indexed_at": row["indexed_at"],
        "exists": os.path.isfile(path),
        "api_path": f"/api/video/{video_id}",
        "frame_api_path": f"/api/frame/{video_id}",
    }
