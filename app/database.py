import sqlite3
from app.config import DB_PATH


def get_connection() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    conn = get_connection()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS videos (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            path         TEXT UNIQUE NOT NULL,
            filename     TEXT NOT NULL,
            duration_sec REAL,
            indexed_at   TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS persons (
            id             TEXT PRIMARY KEY,
            name           TEXT,
            thumbnail_path TEXT,
            face_count     INTEGER DEFAULT 0,
            created_at     TEXT DEFAULT (datetime('now'))
        );
    """)
    conn.commit()
    conn.close()
