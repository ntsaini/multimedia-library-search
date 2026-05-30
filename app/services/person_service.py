import json

from app.database import get_connection


def _parse_samples(value: str | None) -> list[str]:
    if not value:
        return []
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return []
    return parsed if isinstance(parsed, list) else []


def list_people(
    include_unnamed: bool = True,
    limit: int | None = None,
    name_query: str | None = None,
) -> list[dict]:
    clauses = []
    params: list = []

    if not include_unnamed:
        clauses.append("name IS NOT NULL")
    if name_query:
        clauses.append("name IS NOT NULL AND LOWER(name) LIKE LOWER(?)")
        params.append(f"%{name_query.strip()}%")

    sql = "SELECT id, name, thumbnail_path, face_count FROM persons"
    if clauses:
        sql += " WHERE " + " AND ".join(f"({c})" for c in clauses)
    sql += " ORDER BY face_count DESC"
    if limit is not None:
        sql += " LIMIT ?"
        params.append(max(0, int(limit)))

    conn = get_connection()
    rows = conn.execute(sql, params).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_person(person_id: str) -> dict | None:
    conn = get_connection()
    row = conn.execute(
        "SELECT id, name, thumbnail_path, face_count, samples FROM persons WHERE id = ?",
        (person_id,),
    ).fetchone()
    conn.close()
    if not row:
        return None
    person = dict(row)
    person["samples"] = _parse_samples(person.get("samples"))
    return person
