import json
import uuid
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.cluster import DBSCAN
from tqdm import tqdm

from app.chroma import get_collection
from app.config import BASE_DIR, LABELED_FACES_DIR
from app.database import get_connection

_STATIC_DIR = BASE_DIR / "static"


def _maybe_auto_label(
    auto_label: bool,
    threshold: float,
    margin: float,
) -> dict | None:
    if not auto_label or not LABELED_FACES_DIR.exists():
        return None
    from app.labeled_faces import auto_label_persons
    label_result = auto_label_persons(
        label_dir=LABELED_FACES_DIR,
        threshold=threshold,
        margin=margin,
        overwrite=False,
    )
    print(f"Auto-label: {label_result}")
    return label_result


def _normed(arr: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return arr / norms


def _thumb_exists(p: str) -> bool:
    if not p:
        return False
    path = Path(p)
    return path.exists() if path.is_absolute() else (_STATIC_DIR / p).exists()


def _build_person_records(labels, X, metadatas):
    records = []
    assignments = {}
    for cluster_label in tqdm(sorted({lbl for lbl in labels if lbl >= 0}), desc="Building clusters"):
        indices = np.where(labels == cluster_label)[0]
        person_id = str(uuid.uuid4())
        cluster_vecs = X[indices]
        mean_vec = cluster_vecs.mean(axis=0)
        centroid = mean_vec / max(np.linalg.norm(mean_vec), 1e-9)
        medoid_local = int(np.argmin(np.linalg.norm(cluster_vecs - mean_vec, axis=1)))
        medoid_global = int(indices[medoid_local])

        thumbnail_path = metadatas[medoid_global].get("thumbnail_path", "")
        if not _thumb_exists(thumbnail_path):
            thumbnail_path = next(
                (metadatas[int(i)].get("thumbnail_path", "") for i in indices
                 if _thumb_exists(metadatas[int(i)].get("thumbnail_path", ""))),
                "",
            )

        all_paths = [metadatas[int(i)].get("thumbnail_path", "") for i in indices]
        samples = [p for p in all_paths if _thumb_exists(p) and p != thumbnail_path][:4]
        for i in indices:
            assignments[int(i)] = person_id
        records.append({
            "id": person_id,
            "thumbnail_path": thumbnail_path,
            "face_count": len(indices),
            "samples": json.dumps(samples),
            "centroid": json.dumps(centroid.tolist()),
        })
    return records, assignments


def _insert_persons(conn, records):
    for r in records:
        conn.execute(
            "INSERT INTO persons (id, name, thumbnail_path, face_count, samples, centroid)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (r["id"], None, r["thumbnail_path"], r["face_count"], r["samples"], r["centroid"]),
        )


def _bulk_update_chroma(collection, ids, metadatas, batch_size=500):
    for i in tqdm(range(0, len(ids), batch_size), desc="Updating ChromaDB"):
        collection.update(
            ids=ids[i : i + batch_size],
            metadatas=metadatas[i : i + batch_size],
        )


def trim_thumbnails() -> int:
    from app.config import THUMBNAILS_DIR
    keep = set()
    conn = get_connection()
    rows = conn.execute("SELECT thumbnail_path, samples FROM persons").fetchall()
    conn.close()
    for row in rows:
        if row["thumbnail_path"]:
            keep.add(Path(row["thumbnail_path"]).name)
        for path in json.loads(row["samples"] or "[]"):
            if path:
                keep.add(Path(path).name)
    result = get_collection().get(
        where={"person_id": {"$eq": "unlabeled"}},
        include=["metadatas"],
    )
    for meta in (result["metadatas"] or []):
        p = meta.get("thumbnail_path", "")
        if p:
            keep.add(Path(p).name)
    deleted = 0
    for png in THUMBNAILS_DIR.glob("*.png"):
        if png.name not in keep:
            png.unlink()
            deleted += 1
    return deleted


def _dbscan_group(ids, X, metas, eps, min_samples):
    """Run DBSCAN on one group of faces; return new person records and updated metas."""
    if not ids:
        return [], [], [dict(m) for m in metas]
    labels = DBSCAN(
        eps=eps, min_samples=min_samples, metric="euclidean",
        algorithm="ball_tree", n_jobs=-1,
    ).fit_predict(X)
    records, assignments = _build_person_records(labels, X, metas)
    updated = [dict(m) for m in metas]
    for m in updated:
        m["person_id"] = "unlabeled"
    for idx, pid in assignments.items():
        updated[idx]["person_id"] = pid
    noise = int(np.sum(labels == -1))
    return records, updated, noise


def _assign_to_persons(person_ids, C, person_counts, ids, X, metas, eps):
    """Assign faces to existing persons by centroid distance < eps.

    Returns (new_centroids, new_counts, assigned_mask, updated_metas).
    """
    dot = C @ X.T
    np.clip(dot, -1.0, 1.0, out=dot)
    dists = np.sqrt(np.maximum(0.0, 2.0 - 2.0 * dot))
    best_idx = np.argmin(dists, axis=0)
    best_dist = dists[best_idx, np.arange(len(ids))]
    assigned_mask = best_dist < eps

    person_new_faces = defaultdict(list)
    for j in np.where(assigned_mask)[0]:
        pid = person_ids[best_idx[j]]
        person_new_faces[pid].append((ids[j], metas[j], X[j]))

    new_centroids = {}
    new_counts = {}
    collection = get_collection()
    for pid, faces in person_new_faces.items():
        centroid = C[person_ids.index(pid)].copy()
        n = person_counts[pid]
        for face_id, meta, emb in faces:
            centroid = (centroid * n + emb) / (n + 1)
            n += 1
            meta["person_id"] = pid
        norm = np.linalg.norm(centroid)
        new_centroids[pid] = centroid / norm if norm > 0 else centroid
        new_counts[pid] = n
        face_ids_batch = [f[0] for f in faces]
        metas_batch = [f[1] for f in faces]
        for i in range(0, len(face_ids_batch), 500):
            collection.update(
                ids=face_ids_batch[i : i + 500],
                metadatas=metas_batch[i : i + 500],
            )
    return new_centroids, new_counts, assigned_mask


def run_clusterer(
    eps_video: float = 0.7,
    eps_photo: float = 1.0,
    min_samples: int = 3,
    auto_label: bool = True,
    label_threshold: float = 0.6,
    label_margin: float = 0.08,
) -> dict:
    collection = get_collection()
    print("Fetching embeddings from ChromaDB...")
    result = collection.get(include=["embeddings", "metadatas"])
    ids = result["ids"]
    embeddings = result["embeddings"]
    metadatas = result["metadatas"]

    if not ids:
        print("No faces indexed yet.")
        return {"clusters": 0, "noise": 0}

    X = _normed(np.array(embeddings, dtype=np.float32))

    # Split by media type and cluster each group with its own eps
    video_idx = [i for i, m in enumerate(metadatas) if m.get("media_type") != "photo"]
    photo_idx = [i for i, m in enumerate(metadatas) if m.get("media_type") == "photo"]

    existing = get_connection()
    has_named = existing.execute(
        "SELECT COUNT(*) FROM persons WHERE name IS NOT NULL"
    ).fetchone()[0]
    existing.close()
    if has_named:
        print(f"WARNING: {has_named} named person(s) exist — re-clustering will erase all labels.")

    conn = get_connection()
    conn.execute("DELETE FROM persons")

    all_records = []
    updated_metas = [dict(m) for m in metadatas]
    total_noise = 0

    for indices, eps, label in [(video_idx, eps_video, "video"), (photo_idx, eps_photo, "photo")]:
        if not indices:
            continue
        sub_X = X[indices]
        sub_metas = [metadatas[i] for i in indices]
        print(f"Clustering {len(indices):,} {label} faces (eps={eps}, min_samples={min_samples})...")
        labels = DBSCAN(
            eps=eps, min_samples=min_samples, metric="euclidean",
            algorithm="ball_tree", n_jobs=-1,
        ).fit_predict(sub_X)
        n_clusters = len({l for l in labels if l >= 0})
        noise = int(np.sum(labels == -1))
        total_noise += noise
        print(f"  {label}: {n_clusters} clusters, {noise} noise")
        records, assignments = _build_person_records(labels, sub_X, sub_metas)
        all_records.extend(records)
        for local_idx, pid in assignments.items():
            updated_metas[indices[local_idx]]["person_id"] = pid
        for gi in indices:
            if updated_metas[gi]["person_id"] not in {r["id"] for r in records}:
                updated_metas[gi]["person_id"] = "unlabeled"

    _insert_persons(conn, all_records)
    conn.commit()
    conn.close()

    _bulk_update_chroma(collection, ids, updated_metas)
    trimmed = trim_thumbnails()
    print(f"\nDone. {len(all_records)} persons | {total_noise} noise | {trimmed:,} thumbnails trimmed.")
    result = {"clusters": len(all_records), "noise": total_noise}
    label_result = _maybe_auto_label(auto_label, label_threshold, label_margin)
    if label_result is not None:
        result["auto_label"] = label_result
    return result


def run_incremental_clusterer(
    eps_video: float = 0.7,
    eps_photo: float = 1.0,
    min_samples: int = 3,
    auto_label: bool = True,
    label_threshold: float = 0.6,
    label_margin: float = 0.08,
) -> dict:
    conn = get_connection()
    rows = conn.execute(
        "SELECT id, face_count, centroid FROM persons WHERE centroid IS NOT NULL"
    ).fetchall()
    conn.close()

    if not rows:
        print("No existing clusters — falling back to full cluster.")
        return run_clusterer(
            eps_video=eps_video,
            eps_photo=eps_photo,
            min_samples=min_samples,
            auto_label=auto_label,
            label_threshold=label_threshold,
            label_margin=label_margin,
        )

    person_ids = [r["id"] for r in rows]
    person_counts = {r["id"]: r["face_count"] for r in rows}
    C = np.array([json.loads(r["centroid"]) for r in rows], dtype=np.float32)

    collection = get_collection()
    print("Fetching unlabeled face embeddings...")
    result = collection.get(
        where={"person_id": {"$eq": "unlabeled"}},
        include=["embeddings", "metadatas"],
    )
    if not result["ids"]:
        print("No new faces to process.")
        out = {"assigned": 0, "new_clusters": 0, "noise": 0}
        label_result = _maybe_auto_label(auto_label, label_threshold, label_margin)
        if label_result is not None:
            out["auto_label"] = label_result
        return out

    ids = result["ids"]
    metadatas = list(result["metadatas"])
    X = _normed(np.array(result["embeddings"], dtype=np.float32))

    # Split by media type
    video_idx = [i for i, m in enumerate(metadatas) if m.get("media_type") != "photo"]
    photo_idx = [i for i, m in enumerate(metadatas) if m.get("media_type") == "photo"]

    total_assigned = 0
    total_new_clusters = 0
    total_noise = 0
    all_new_records: list = []
    all_new_centroids: dict = {}
    all_new_counts: dict = {}

    for indices, eps, label in [(video_idx, eps_video, "video"), (photo_idx, eps_photo, "photo")]:
        if not indices:
            continue
        sub_ids = [ids[i] for i in indices]
        sub_X = X[indices]
        sub_metas = [metadatas[i] for i in indices]

        print(f"Processing {len(sub_ids):,} unlabeled {label} faces (eps={eps})...")
        new_centroids, new_counts, assigned_mask = _assign_to_persons(
            person_ids, C, person_counts, sub_ids, sub_X, sub_metas, eps
        )
        all_new_centroids.update(new_centroids)
        all_new_counts.update(new_counts)
        total_assigned += int(assigned_mask.sum())

        unmatched = np.where(~assigned_mask)[0]
        if len(unmatched) >= min_samples:
            unmatched_ids = [sub_ids[i] for i in unmatched]
            unmatched_X = sub_X[unmatched]
            unmatched_metas = [sub_metas[i] for i in unmatched]
            print(f"  Running DBSCAN on {len(unmatched)} unmatched {label} faces...")
            records, updated_metas_sub, noise = _dbscan_group(
                unmatched_ids, unmatched_X, unmatched_metas, eps, min_samples
            )
            all_new_records.extend(records)
            total_new_clusters += len(records)
            total_noise += noise
            _bulk_update_chroma(collection, unmatched_ids, updated_metas_sub)
        else:
            total_noise += len(unmatched)

    # Persist centroid updates
    conn = get_connection()
    for pid, centroid in all_new_centroids.items():
        conn.execute(
            "UPDATE persons SET centroid = ?, face_count = ? WHERE id = ?",
            (json.dumps(centroid.tolist()), all_new_counts[pid], pid),
        )
    _insert_persons(conn, all_new_records)
    conn.commit()
    conn.close()

    trimmed = trim_thumbnails()
    print(
        f"\nDone. {total_assigned} assigned | {total_new_clusters} new persons"
        f" | {total_noise} noise | {trimmed:,} thumbnails trimmed."
    )
    out = {"assigned": total_assigned, "new_clusters": total_new_clusters, "noise": total_noise}
    label_result = _maybe_auto_label(auto_label, label_threshold, label_margin)
    if label_result is not None:
        out["auto_label"] = label_result
    return out
