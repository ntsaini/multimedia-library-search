import argparse
import shutil
import sys
from pathlib import Path


def check_dependencies() -> None:
    if shutil.which("ffmpeg") is None:
        sys.exit("Error: ffmpeg not found. Install it and ensure it is on your PATH.")


def _force_clear_directory(directory: str) -> None:
    """Delete all DB rows and ChromaDB face entries for media under directory."""
    from app.database import get_connection
    from app.indexer import clear_paths_from_index

    prefix = str(Path(directory).resolve())
    db = get_connection()
    video_rows = db.execute(
        "SELECT path FROM videos WHERE path LIKE ?", (prefix + "%",)
    ).fetchall()
    photo_rows = db.execute(
        "SELECT path FROM photos WHERE path LIKE ?", (prefix + "%",)
    ).fetchall()
    db.close()

    paths = [r["path"] for r in video_rows] + [r["path"] for r in photo_rows]
    n_v, n_p = len(video_rows), len(photo_rows)

    if not paths:
        print("No existing entries found for this directory — nothing to clear.")
        return

    print(f"Clearing {n_v} video(s) and {n_p} photo(s) from index…")
    clear_paths_from_index(paths)
    print("Cleared. Files will be reprocessed on next index run.")


def cmd_index(args) -> None:
    from app.database import init_db
    from app.chroma import get_collection
    from app.indexer import run_indexer
    from app.photo_indexer import run_photo_indexer

    init_db()
    get_collection()

    if args.force:
        _force_clear_directory(args.directory)

    do_videos = not args.photos_only
    do_photos = not args.videos_only
    do_cluster = not args.no_cluster

    if do_videos and do_photos:
        # Both phases: suppress per-phase clustering, run exactly once at the end
        run_indexer(
            directory=args.directory,
            interval_sec=args.interval,
            use_gpu=args.gpu,
            auto_cluster=False,
            eps=args.eps,
            _finalize=False,
        )
        run_photo_indexer(
            directory=args.directory,
            use_gpu=args.gpu,
            auto_cluster=False,
            eps=args.eps,
            _finalize=False,
        )
        if do_cluster:
            from app.clusterer import run_incremental_clusterer
            run_incremental_clusterer(eps=args.eps)
    elif do_videos:
        run_indexer(
            directory=args.directory,
            interval_sec=args.interval,
            use_gpu=args.gpu,
            auto_cluster=do_cluster,
            eps=args.eps,
            _finalize=True,
        )
    else:  # photos only
        run_photo_indexer(
            directory=args.directory,
            use_gpu=args.gpu,
            auto_cluster=do_cluster,
            eps=args.eps,
            _finalize=True,
        )


def cmd_cluster(args) -> None:
    from app.database import init_db
    from app.chroma import get_collection
    from app.clusterer import run_clusterer, run_incremental_clusterer

    init_db()
    get_collection()
    if args.incremental:
        run_incremental_clusterer(eps=args.eps, min_samples=args.min_samples)
    else:
        run_clusterer(eps=args.eps, min_samples=args.min_samples)


def cmd_stats(args) -> None:
    from app.database import get_connection
    from app.chroma import get_collection

    db = get_connection()
    videos = db.execute("SELECT COUNT(*) FROM videos").fetchone()[0]
    photos = db.execute("SELECT COUNT(*) FROM photos").fetchone()[0]
    persons = db.execute("SELECT COUNT(*) FROM persons").fetchone()[0]
    labeled = db.execute(
        "SELECT COUNT(*) FROM persons WHERE name IS NOT NULL"
    ).fetchone()[0]
    db.close()

    faces = get_collection().count()

    print(f"Videos:  {videos}")
    print(f"Photos:  {photos}")
    print(f"Faces:   {faces:,}")
    print(f"Persons: {persons} ({labeled} labeled)")


def cmd_prune(args) -> None:
    from pathlib import Path
    from app.database import init_db, get_connection
    from app.indexer import prune_stale_media

    init_db()

    if args.dry_run:
        db = get_connection()
        video_rows = db.execute("SELECT path, filename FROM videos").fetchall()
        photo_rows = db.execute("SELECT path, filename FROM photos").fetchall()
        stale_v = [r for r in video_rows if not Path(r["path"]).exists()]
        stale_p = [r for r in photo_rows if not Path(r["path"]).exists()]
        db.close()
        if not stale_v and not stale_p:
            print("No stale media found.")
        else:
            if stale_v:
                print(f"Would remove {len(stale_v)} stale video(s):")
                for r in stale_v:
                    print(f"  {r['filename']}")
            if stale_p:
                print(f"Would remove {len(stale_p)} stale photo(s):")
                for r in stale_p:
                    print(f"  {r['filename']}")
        return

    n_vid, n_photo = prune_stale_media()
    if n_vid or n_photo:
        parts = []
        if n_vid:   parts.append(f"{n_vid} video(s)")
        if n_photo: parts.append(f"{n_photo} photo(s)")
        print(f"Removed {' and '.join(parts)} and their associated data.")
    else:
        print("No stale media found — nothing to do.")


def cmd_trim_thumbnails(args) -> None:
    from app.database import init_db, get_connection
    from app.config import THUMBNAILS_DIR
    from app.clusterer import trim_thumbnails
    import json

    init_db()

    if args.dry_run:
        db = get_connection()
        rows = db.execute("SELECT thumbnail_path, samples FROM persons").fetchall()
        db.close()
        keep = set()
        for row in rows:
            if row["thumbnail_path"]:
                keep.add(Path(row["thumbnail_path"]).name)
            for p in json.loads(row["samples"] or "[]"):
                if p:
                    keep.add(Path(p).name)
        to_delete = [p for p in THUMBNAILS_DIR.glob("*.png") if p.name not in keep]
        freed = sum(p.stat().st_size for p in to_delete)
        print(f"Would delete {len(to_delete):,} thumbnail(s) ({freed / 1024 / 1024:.1f} MB), keeping {len(keep)}.")
        return

    deleted = trim_thumbnails()
    if deleted:
        print(f"Trimmed {deleted:,} redundant thumbnail(s).")
    else:
        print("Nothing to trim — thumbnail directory is already minimal.")


def cmd_backfill_dates(args) -> None:
    from pathlib import Path
    from app.database import init_db, get_connection
    from app.indexer import _extract_recording_date

    init_db()
    db = get_connection()
    rows = db.execute(
        "SELECT id, path, filename FROM videos WHERE recorded_at IS NULL"
    ).fetchall()

    if not rows:
        print("All videos already have a recorded_at date — nothing to do.")
        db.close()
        return

    print(f"Backfilling dates for {len(rows)} video(s)…")
    updated = 0
    for row in rows:
        path = Path(row["path"])
        date = _extract_recording_date(path)
        db.execute(
            "UPDATE videos SET recorded_at = ? WHERE id = ?", (date, row["id"])
        )
        status = date or "NULL"
        print(f"  {row['filename']}: {status}")
        if date:
            updated += 1

    db.commit()
    db.close()
    print(f"\nDone. {updated}/{len(rows)} video(s) updated.")


def cmd_serve(args) -> None:
    import uvicorn
    uvicorn.run("app.main:app", host=args.host, port=args.port, reload=False)


def main() -> None:
    check_dependencies()

    parser = argparse.ArgumentParser(description="Multimedia Library Search CLI")
    subs = parser.add_subparsers(dest="command")

    p_index = subs.add_parser("index", help="Index a directory of videos")
    p_index.add_argument("directory", help="Path to directory containing video files")
    p_index.add_argument(
        "--interval", type=float, default=1.0,
        help="Seconds between sampled keyframes (default: 1.0)",
    )
    p_index.add_argument(
        "--gpu", action="store_true",
        help="Use GPU (CUDAExecutionProvider) for face inference",
    )
    p_index.add_argument(
        "--no-cluster", action="store_true",
        help="Skip automatic incremental clustering after indexing",
    )
    p_index.add_argument(
        "--eps", type=float, default=0.6,
        help="DBSCAN eps passed to the auto-triggered incremental cluster (default: 0.6)",
    )
    p_index.add_argument(
        "--force", action="store_true",
        help="Remove existing index entries for this directory before indexing, forcing full reprocessing",
    )
    media_grp = p_index.add_mutually_exclusive_group()
    media_grp.add_argument(
        "--videos-only", action="store_true",
        help="Index only video files; skip photos",
    )
    media_grp.add_argument(
        "--photos-only", action="store_true",
        help="Index only photo files; skip videos",
    )
    p_index.set_defaults(func=cmd_index)

    p_cluster = subs.add_parser("cluster", help="Cluster faces into person identities")
    p_cluster.add_argument(
        "--incremental", action="store_true",
        help="Assign only new unlabeled faces; preserve existing persons and labels",
    )
    p_cluster.add_argument(
        "--eps", type=float, default=0.6,
        help="DBSCAN eps in euclidean space on L2-normed embeddings (default: 0.6)",
    )
    p_cluster.add_argument(
        "--min-samples", type=int, default=3,
        help="DBSCAN min_samples — minimum faces to form a cluster (default: 3)",
    )
    p_cluster.set_defaults(func=cmd_cluster)

    p_stats = subs.add_parser("stats", help="Show library statistics")
    p_stats.set_defaults(func=cmd_stats)

    p_prune = subs.add_parser(
        "prune",
        help="Remove stale data for videos/photos deleted from disk",
    )
    p_prune.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be removed without making changes",
    )
    p_prune.set_defaults(func=cmd_prune)

    p_trim = subs.add_parser(
        "trim-thumbnails",
        help="Delete redundant face thumbnails, keeping only label-page samples",
    )
    p_trim.add_argument(
        "--dry-run", action="store_true",
        help="Show how many thumbnails would be deleted without deleting them",
    )
    p_trim.set_defaults(func=cmd_trim_thumbnails)

    p_bd = subs.add_parser(
        "backfill-dates",
        help="Populate recorded_at for already-indexed videos (one-time, no re-indexing)",
    )
    p_bd.set_defaults(func=cmd_backfill_dates)

    p_serve = subs.add_parser("serve", help="Start the web UI server")
    p_serve.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    p_serve.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000)")
    p_serve.set_defaults(func=cmd_serve)

    args = parser.parse_args()
    if not hasattr(args, "func"):
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
