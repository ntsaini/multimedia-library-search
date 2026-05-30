from fastapi import APIRouter

from app.chroma import get_collection
from app.database import get_connection

router = APIRouter()


@router.get("/api/health")
def health_check():
    sqlite_ok = False
    chromadb_ok = False

    try:
        conn = get_connection()
        conn.execute("SELECT 1").fetchone()
        conn.close()
        sqlite_ok = True
    except Exception:
        sqlite_ok = False

    try:
        get_collection().count()
        chromadb_ok = True
    except Exception:
        chromadb_ok = False

    return {
        "status": "ok" if sqlite_ok and chromadb_ok else "error",
        "sqlite": sqlite_ok,
        "chromadb": chromadb_ok,
    }


@router.get("/api/stats")
def library_stats():
    conn = get_connection()
    videos = conn.execute("SELECT COUNT(*) FROM videos").fetchone()[0]
    photos = conn.execute("SELECT COUNT(*) FROM photos").fetchone()[0]
    persons = conn.execute("SELECT COUNT(*) FROM persons").fetchone()[0]
    labeled = conn.execute(
        "SELECT COUNT(*) FROM persons WHERE name IS NOT NULL"
    ).fetchone()[0]
    conn.close()

    faces = get_collection().count()

    return {
        "videos": videos,
        "photos": photos,
        "faces": faces,
        "persons": persons,
        "labeled_persons": labeled,
    }
