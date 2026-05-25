import argparse
import shutil
import sys


def check_dependencies() -> None:
    if shutil.which("ffmpeg") is None:
        sys.exit("Error: ffmpeg not found. Install it and ensure it is on your PATH.")


def cmd_index(args) -> None:
    from app.database import init_db
    from app.chroma import get_collection
    from app.indexer import run_indexer

    init_db()
    get_collection()
    run_indexer(
        directory=args.directory,
        interval_sec=args.interval,
        use_gpu=args.gpu,
        auto_cluster=not args.no_cluster,
        eps=args.eps,
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
    persons = db.execute("SELECT COUNT(*) FROM persons").fetchone()[0]
    labeled = db.execute(
        "SELECT COUNT(*) FROM persons WHERE name IS NOT NULL"
    ).fetchone()[0]
    db.close()

    faces = get_collection().count()

    print(f"Videos:  {videos}")
    print(f"Faces:   {faces:,}")
    print(f"Persons: {persons} ({labeled} labeled)")


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
