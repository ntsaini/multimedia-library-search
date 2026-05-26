import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path

from app.chroma import get_collection
from app.config import OUTPUT_DIR
from app.database import get_connection

# In-memory job state — sufficient for a single-user local tool
_jobs: dict = {}


def merge_timestamps(timestamps: list, merge_gap_sec: float = 30.0) -> list:
    """
    Group a sorted list of timestamps into (scene_start, scene_end) tuples.
    Two timestamps belong to the same scene if they are within merge_gap_sec of
    each other.  Returns a list of (start, end) pairs.
    """
    if not timestamps:
        return []
    segments: list = []
    for t in sorted(timestamps):
        if segments and t <= segments[-1][1] + merge_gap_sec:
            segments[-1] = (segments[-1][0], t)
        else:
            segments.append((t, t))
    return segments


def _ffmpeg(*args) -> subprocess.CompletedProcess:
    return subprocess.run(["ffmpeg", "-y", *args], capture_output=True)


def run_compile(
    job_id: str,
    person_id: str,
    clip_duration_sec: float,
    merge_gap_sec: float,
    max_clips_per_video: int,
    order: str = "asc",
) -> None:
    """Runs in a background thread; mutates _jobs[job_id] throughout."""
    job = _jobs[job_id]

    try:
        # ── Fetch all face timestamps for this person ──────────────────────
        collection = get_collection()
        result = collection.get(
            where={"person_id": {"$eq": person_id}},
            include=["metadatas"],
        )
        metas = result.get("metadatas") or []
        if not metas:
            job.update({"status": "error", "error": "No faces found for this person"})
            return

        video_ts: dict = defaultdict(list)
        for m in metas:
            video_ts[m["video_id"]].append(float(m.get("timestamp_sec") or 0))

        # ── Fetch video paths + durations from SQLite ──────────────────────
        conn = get_connection()
        vid_ids = list(video_ts.keys())
        ph = ",".join("?" * len(vid_ids))
        rows = conn.execute(
            f"SELECT id, path, duration_sec, recorded_at, indexed_at"
            f" FROM videos WHERE id IN ({ph})", vid_ids
        ).fetchall()
        person_row = conn.execute(
            "SELECT name FROM persons WHERE id = ?", (person_id,)
        ).fetchone()
        conn.close()

        video_info = {
            r["id"]: {
                "path": r["path"],
                "duration_sec": r["duration_sec"],
                "sort_key": r["recorded_at"] or r["indexed_at"] or "",
            }
            for r in rows
        }
        raw_name = person_row["name"] if (person_row and person_row["name"]) else "unknown"
        safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in raw_name)

        # ── Build clip list ────────────────────────────────────────────────
        # Strategy:
        #   1. Group detections into scenes using merge_gap_sec.
        #   2. If a video has more scenes than max_clips_per_video, evenly sample.
        #   3. For each chosen scene, cut a clip of clip_duration_sec centred on
        #      the scene midpoint — so a 9-minute continuous-presence scene becomes
        #      one short representative clip, not the whole video.
        segments = []
        half = clip_duration_sec / 2.0

        for vid_id, timestamps in video_ts.items():
            if vid_id not in video_info:
                continue
            info = video_info[vid_id]
            path = info["path"]
            cap = float(info["duration_sec"] or 1e9)

            scenes = merge_timestamps(timestamps, merge_gap_sec)

            # Evenly sample if there are more scenes than the per-video cap
            if max_clips_per_video > 0 and len(scenes) > max_clips_per_video:
                n = max_clips_per_video
                step = (len(scenes) - 1) / (n - 1) if n > 1 else 0
                scenes = [scenes[round(i * step)] for i in range(n)]

            for s_start, s_end in scenes:
                mid = (s_start + s_end) / 2.0
                start = max(0.0, mid - half)
                end = min(cap, mid + half)
                if end > start:
                    segments.append({
                        "path": path,
                        "start": start,
                        "end": end,
                        "vid_id": vid_id,
                        "sort_key": info["sort_key"],
                    })

        if order == "desc":
            segments.sort(key=lambda s: (s["sort_key"], s["start"]), reverse=True)
        elif order == "random":
            import random
            random.shuffle(segments)
        else:  # "asc" / default
            segments.sort(key=lambda s: (s["sort_key"], s["start"]))

        if not segments:
            job.update({"status": "error", "error": "No segments to compile"})
            return

        job["segments_total"] = len(segments)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_path = OUTPUT_DIR / f"{safe_name}_{job_id[:8]}.mp4"

        # ── Cut individual clips ───────────────────────────────────────────
        with tempfile.TemporaryDirectory() as tmpdir:
            clip_paths = []

            for i, seg in enumerate(segments):
                clip = Path(tmpdir) / f"clip_{i:04d}.mp4"

                # -ss before -i = fast keyframe seek; -c copy = lossless
                r = _ffmpeg(
                    "-ss", str(seg["start"]),
                    "-to", str(seg["end"]),
                    "-i",  seg["path"],
                    "-c",  "copy",
                    str(clip),
                )
                if r.returncode != 0 or not clip.exists():
                    # Fallback: re-encode (handles codec/container mismatches)
                    _ffmpeg(
                        "-ss", str(seg["start"]),
                        "-to", str(seg["end"]),
                        "-i",  seg["path"],
                        "-c:v", "libx264", "-preset", "fast",
                        "-c:a", "aac",
                        str(clip),
                    )

                if clip.exists():
                    clip_paths.append(clip)

                job["segments_done"] = i + 1
                job["progress"]      = round((i + 1) / len(segments) * 0.9, 3)

            if not clip_paths:
                job.update({"status": "error", "error": "FFmpeg produced no clips"})
                return

            # ── Concatenate ────────────────────────────────────────────────
            concat_list = Path(tmpdir) / "concat.txt"
            concat_list.write_text("\n".join(f"file '{p}'" for p in clip_paths))

            r = _ffmpeg(
                "-f", "concat", "-safe", "0",
                "-i", str(concat_list),
                "-c", "copy",
                "-movflags", "+faststart",
                str(output_path),
            )
            if r.returncode != 0:
                job.update({
                    "status": "error",
                    "error": "Concat failed: " + r.stderr.decode(errors="replace")[:300],
                })
                return

        job.update({
            "status":      "done",
            "progress":    1.0,
            "output_path": str(output_path),
            "filename":    output_path.name,
        })

    except Exception as exc:
        job.update({"status": "error", "error": str(exc)})
