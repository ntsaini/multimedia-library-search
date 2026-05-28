import io

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, Response

from app.database import get_connection

router = APIRouter()


@router.get("/api/photo/{photo_id}")
def serve_photo(photo_id: int):
    conn = get_connection()
    row = conn.execute("SELECT path FROM photos WHERE id = ?", (photo_id,)).fetchone()
    conn.close()
    if not row:
        raise HTTPException(404, "Photo not found")
    return FileResponse(row["path"])


@router.get("/api/photo/{photo_id}/preview")
def serve_photo_preview(photo_id: int, size: int = 600):
    """Return a JPEG downscaled to `size` px on the longest side (default 600)."""
    conn = get_connection()
    row = conn.execute("SELECT path FROM photos WHERE id = ?", (photo_id,)).fetchone()
    conn.close()
    if not row:
        raise HTTPException(404, "Photo not found")
    size = max(100, min(size, 1200))
    try:
        from PIL import Image
        img = Image.open(row["path"]).convert("RGB")
        img.thumbnail((size, size), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=82)
        return Response(content=buf.getvalue(), media_type="image/jpeg")
    except Exception as exc:
        raise HTTPException(500, f"Could not generate preview: {exc}")
