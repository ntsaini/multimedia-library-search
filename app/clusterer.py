import json
import uuid
from collections import defaultdict

import numpy as np
from sklearn.cluster import DBSCAN
from tqdm import tqdm

from app.chroma import get_collection
from app.database import get_connection


def _normed(arr: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return arr / norms


def _build_person_records(labels, X, metadatas):
    """Return list of dicts ready for SQLite INSERT, and a parallel updated_metas list."""
    records = []
    assignments = {}  # face_global_index -> person_id
    for cluster_label in tqdm(sorted({lbl for lbl in labels if lbl >= 0}), desc="Building clusters"):
        indices = np.where(labels == cluster_label)[0]
        person_id = str(uuid.uuid4())
        cluster_vecs = X[indices]
        mean_vec = cluster_vecs.mean(axis=0)
        centroid = mean_vec / max(np.linalg.norm(mean_vec), 1e-9)
        medoid_local = int(np.argmin(np.linalg.norm(cluster_vecs - mean_vec, axis=1)))
        medoid_global = int(indices[medoid_local])
        thumbnail_path = metadatas[medoid_global].get("thumbnail_path", "")
        all_paths = [metadatas[int(i)].get("thumbnail_path", "") for i in indices]
        samples = [p for p in all_paths if p and p != thumbnail_path][:4]
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


def run_clusterer(eps: float = 0.6, min_samples: int = 3) -> dict:
    collection = get_collection()

    print("Fetching embeddings from ChromaDB...")
    result = collection.get(include=["embeddings", "metadatas"])
    ids = result["ids"]
    embeddings = result["embeddings"]
    metadatas = result["metadatas"]

    if not ids:
        print("No faces indexed yet. Run 'index' first.")
        return {"clusters": 0, "noise": 0}

    X = _normed(np.array(embeddings, dtype=np.float32))

    print(f"Clustering {len(ids):,} faces (eps={eps}, min_samples={min_samples})...")
    # euclidean on normed embeddings enables ball_tree → O(n log n) vs O(n²) for cosine
    labels = DBSCAN(
        eps=eps, min_samples=min_samples, metric="euclidean",
        algorithm="ball_tree", n_jobs=-1,
    ).fit_predict(X)

    cluster_labels = sorted({lbl for lbl in labels if lbl >= 0})
    noise_count = int(np.sum(labels == -1))
    print(f"Found {len(cluster_labels)} clusters, {noise_count} noise points.")

    records, assignments = _build_person_records(labels, X, metadatas)

    # Prepare ChromaDB metadata updates (reset all, then assign clusters)
    updated_metas = [dict(m) for m in metadatas]
    for m in updated_metas:
        m["person_id"] = "unlabeled"
    for idx, pid in assignments.items():
        updated_metas[idx]["person_id"] = pid

    existing = get_connection()
    has_named = existing.execute(
        "SELECT COUNT(*) FROM persons WHERE name IS NOT NULL"
    ).fetchone()[0]
    existing.close()
    if has_named:
        print(f"WARNING: {has_named} named person(s) exist — re-clustering will erase all labels.")

    conn = get_connection()
    conn.execute("DELETE FROM persons")
    _insert_persons(conn, records)
    conn.commit()
    conn.close()

    _bulk_update_chroma(collection, ids, updated_metas)

    print(f"\nDone. {len(cluster_labels)} persons | {noise_count} faces unlabeled.")
    return {"clusters": len(cluster_labels), "noise": noise_count}


def run_incremental_clusterer(eps: float = 0.6, min_samples: int = 3) -> dict:
    conn = get_connection()
    rows = conn.execute(
        "SELECT id, face_count, centroid FROM persons WHERE centroid IS NOT NULL"
    ).fetchall()
    conn.close()

    if not rows:
        print("No existing clusters — falling back to full cluster.")
        return run_clusterer(eps=eps, min_samples=min_samples)

    person_ids = [r["id"] for r in rows]
    person_counts = {r["id"]: r["face_count"] for r in rows}
    # C: (P, 512) — one centroid row per person
    C = np.array([json.loads(r["centroid"]) for r in rows], dtype=np.float32)

    collection = get_collection()
    print("Fetching unlabeled face embeddings...")
    result = collection.get(
        where={"person_id": {"$eq": "unlabeled"}},
        include=["embeddings", "metadatas"],
    )
    ids = result["ids"]
    embeddings = result["embeddings"]
    metadatas = result["metadatas"]

    if not ids:
        print("No new faces to process.")
        return {"assigned": 0, "new_clusters": 0, "noise": 0}

    X = _normed(np.array(embeddings, dtype=np.float32))
    print(f"Assigning {len(ids):,} new faces to {len(person_ids)} existing persons...")

    # Batch distance: for normed vectors, euclidean² = 2 - 2·dot
    dot = C @ X.T  # (P, F)
    np.clip(dot, -1.0, 1.0, out=dot)
    dists = np.sqrt(np.maximum(0.0, 2.0 - 2.0 * dot))  # (P, F)

    best_idx = np.argmin(dists, axis=0)   # (F,) — index into person_ids
    best_dist = dists[best_idx, np.arange(len(ids))]  # (F,)
    assigned_mask = best_dist < eps

    # Group assigned faces by person
    person_new_faces = defaultdict(list)  # person_id -> [(face_id, meta, embedding)]
    for j in np.where(assigned_mask)[0]:
        pid = person_ids[best_idx[j]]
        person_new_faces[pid].append((ids[j], metadatas[j], X[j]))

    # Update centroids and ChromaDB for assigned faces
    new_centroids = {}
    new_counts = {}
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

    conn = get_connection()
    for pid, centroid in new_centroids.items():
        conn.execute(
            "UPDATE persons SET centroid = ?, face_count = ? WHERE id = ?",
            (json.dumps(centroid.tolist()), new_counts[pid], pid),
        )
    conn.commit()
    conn.close()

    assigned_count = int(assigned_mask.sum())

    # Mini-DBSCAN on unmatched faces to discover new persons
    unmatched = np.where(~assigned_mask)[0]
    new_clusters = 0
    noise_count = 0

    if len(unmatched) > 0:
        unmatched_ids = [ids[i] for i in unmatched]
        unmatched_X = X[unmatched]
        unmatched_metas = [metadatas[i] for i in unmatched]

        if len(unmatched) >= min_samples:
            print(f"Running DBSCAN on {len(unmatched)} unmatched faces...")
            labels = DBSCAN(
                eps=eps, min_samples=min_samples, metric="euclidean",
                algorithm="ball_tree", n_jobs=-1,
            ).fit_predict(unmatched_X)

            noise_count = int(np.sum(labels == -1))
            records, assignments = _build_person_records(labels, unmatched_X, unmatched_metas)
            new_clusters = len(records)

            updated_metas = [dict(m) for m in unmatched_metas]
            for m in updated_metas:
                m["person_id"] = "unlabeled"
            for idx, pid in assignments.items():
                updated_metas[idx]["person_id"] = pid

            conn = get_connection()
            _insert_persons(conn, records)
            conn.commit()
            conn.close()

            _bulk_update_chroma(collection, unmatched_ids, updated_metas)
        else:
            noise_count = len(unmatched)

    print(f"\nDone. {assigned_count} assigned | {new_clusters} new persons | {noise_count} noise.")
    return {"assigned": assigned_count, "new_clusters": new_clusters, "noise": noise_count}
