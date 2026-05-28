import random

from PIL import Image, ImageDraw, ImageFont

from app.chroma import get_collection
from app.config import OUTPUT_DIR, COLLAGE_CELL_SIZE

_jobs: dict = {}

# Hard cap on photos included — keeps the collage comfortably viewable
_MAX_PHOTOS = 60


def _load_font(size: int) -> ImageFont.ImageFont:
    for path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/TTF/DejaVuSans.ttf",
        "/Library/Fonts/Arial.ttf",
    ]:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            pass
    return ImageFont.load_default()


def _col_width_for(n_photos: int, base: int) -> int:
    """Slightly scale down column width for larger sets so the canvas stays manageable."""
    if n_photos <= 24:
        return base
    if n_photos <= 48:
        return max(220, int(base * 0.80))
    return max(200, int(base * 0.70))


def run_collage(
    job_id: str,
    person_id: str,
    columns: int,
    sort: str,
    captions: bool,
) -> None:
    """Runs in a background thread; mutates _jobs[job_id] throughout."""
    job = _jobs[job_id]

    try:
        # ── 1. Collect photos for this person ───────────────────────────────
        collection = get_collection()
        result = collection.get(
            where={
                "$and": [
                    {"person_id": {"$eq": person_id}},
                    {"media_type": {"$eq": "photo"}},
                ]
            },
            include=["metadatas"],
        )
        metas = result.get("metadatas") or []

        seen: dict = {}
        for m in metas:
            pid = m.get("photo_id")
            if pid is not None and pid not in seen:
                seen[pid] = m

        if not seen:
            job.update({"status": "error", "error": "No photos found for this person"})
            return

        photo_ids = list(seen.keys())
        from app.database import get_connection
        conn = get_connection()
        ph = ",".join("?" * len(photo_ids))
        rows = conn.execute(
            f"SELECT id, path, taken_at FROM photos WHERE id IN ({ph})", photo_ids
        ).fetchall()
        person_row = conn.execute(
            "SELECT name FROM persons WHERE id = ?", (person_id,)
        ).fetchone()
        conn.close()

        photo_info = {
            r["id"]: {"path": r["path"], "taken_at": r["taken_at"] or ""}
            for r in rows
        }
        raw_name = person_row["name"] if person_row and person_row["name"] else "unknown"
        safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in raw_name)

        photos = [
            {"photo_id": pid, "path": photo_info[pid]["path"], "taken_at": photo_info[pid]["taken_at"]}
            for pid in photo_ids if pid in photo_info
        ]

        if sort == "desc":
            photos.sort(key=lambda p: p["taken_at"], reverse=True)
        elif sort == "random":
            random.shuffle(photos)
        else:
            photos.sort(key=lambda p: p["taken_at"])

        # Cap total photos
        if len(photos) > _MAX_PHOTOS:
            photos = photos[:_MAX_PHOTOS]

        job["photos_total"] = len(photos)

        # ── 2. Determine column width ────────────────────────────────────────
        col_w = _col_width_for(len(photos), COLLAGE_CELL_SIZE)
        font  = _load_font(max(9, col_w // 24))

        # ── 3. Load and scale photos; assign to shortest column (masonry) ───
        col_heights = [0] * columns
        placements: list[tuple[int, int, Image.Image, str]] = []

        for i, photo in enumerate(photos):
            try:
                img = Image.open(photo["path"]).convert("RGB")
                w, h = img.size
                new_h = int(h * col_w / w)
                img = img.resize((col_w, new_h), Image.LANCZOS)
            except Exception:
                img = Image.new("RGB", (col_w, col_w * 2 // 3), (180, 176, 170))

            shortest = min(range(columns), key=lambda c: col_heights[c])
            x = shortest * col_w
            y = col_heights[shortest]
            col_heights[shortest] += img.height
            placements.append((x, y, img, photo["taken_at"]))

            job["photos_done"] = i + 1
            job["progress"]    = round((i + 1) / len(photos) * 0.6, 3)

        # ── 4. Compose canvas ────────────────────────────────────────────────
        canvas_w = columns * col_w
        canvas_h = max(col_heights)
        canvas   = Image.new("RGB", (canvas_w, canvas_h), (30, 30, 30))

        for idx, (x, y, img, taken_at) in enumerate(placements):
            canvas.paste(img, (x, y))

            if captions and taken_at:
                date_str = taken_at[:10]
                overlay  = Image.new("RGBA", (img.width, 20), (0, 0, 0, 140))
                canvas_rgba = canvas.crop((x, y + img.height - 20, x + img.width, y + img.height)).convert("RGBA")
                canvas_rgba = Image.alpha_composite(canvas_rgba, overlay)
                canvas.paste(canvas_rgba.convert("RGB"), (x, y + img.height - 20))
                d = ImageDraw.Draw(canvas)
                d.text(
                    (x + img.width // 2, y + img.height - 10),
                    date_str,
                    fill=(220, 216, 208),
                    anchor="mm",
                    font=font,
                )

            job["progress"] = round(0.6 + (idx + 1) / len(placements) * 0.4, 3)

        # ── 5. Save ──────────────────────────────────────────────────────────
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_path = OUTPUT_DIR / f"{safe_name}_{job_id[:8]}.jpg"
        canvas.save(str(output_path), "JPEG", quality=88, optimize=True)

        job.update({
            "status":      "done",
            "progress":    1.0,
            "output_path": str(output_path),
            "filename":    output_path.name,
            "photo_count": len(photos),
        })

    except Exception as exc:
        job.update({"status": "error", "error": str(exc)})
