import json
import uuid
import numpy as np
from sklearn.cluster import DBSCAN
from tqdm import tqdm

from app.chroma import get_collection
from app.database import get_connection


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

    X = np.array(embeddings, dtype=np.float32)
    # InsightFace normed_embedding is already L2-normalized; renorm for safety
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    X /= norms

    print(f"Clustering {len(ids):,} faces (eps={eps}, min_samples={min_samples})...")
    # euclidean on normed embeddings enables ball_tree → O(n log n) vs O(n²) for cosine
    labels = DBSCAN(
        eps=eps,
        min_samples=min_samples,
        metric="euclidean",
        algorithm="ball_tree",
        n_jobs=-1,
    ).fit_predict(X)

    cluster_labels = sorted({lbl for lbl in labels if lbl >= 0})
    noise_count = int(np.sum(labels == -1))
    print(f"Found {len(cluster_labels)} clusters, {noise_count} noise points.")

    # Build final metadata: start unlabeled, then assign person_ids for clustered faces
    updated_metas = [dict(m) for m in metadatas]
    for m in updated_metas:
        m["person_id"] = "unlabeled"

    person_records = []
    for cluster_label in tqdm(cluster_labels, desc="Building clusters"):
        indices = np.where(labels == cluster_label)[0]
        person_id = str(uuid.uuid4())
        cluster_vecs = X[indices]
        mean_vec = cluster_vecs.mean(axis=0)
        medoid_local = int(np.argmin(np.linalg.norm(cluster_vecs - mean_vec, axis=1)))
        medoid_global = int(indices[medoid_local])
        thumbnail_path = metadatas[medoid_global].get("thumbnail_path", "")
        all_paths = [metadatas[int(i)].get("thumbnail_path", "") for i in indices]
        samples = [p for p in all_paths if p and p != thumbnail_path][:4]
        for i in indices:
            updated_metas[i]["person_id"] = person_id
        person_records.append((person_id, thumbnail_path, len(indices), json.dumps(samples)))

    existing = get_connection()
    has_named = existing.execute(
        "SELECT COUNT(*) FROM persons WHERE name IS NOT NULL"
    ).fetchone()[0]
    existing.close()
    if has_named:
        print(
            f"WARNING: {has_named} named person(s) exist — re-clustering will erase all labels."
        )

    conn = get_connection()
    conn.execute("DELETE FROM persons")
    for person_id, thumbnail_path, face_count, samples_json in person_records:
        conn.execute(
            "INSERT INTO persons (id, name, thumbnail_path, face_count, samples) VALUES (?, ?, ?, ?, ?)",
            (person_id, None, thumbnail_path, face_count, samples_json),
        )
    conn.commit()
    conn.close()

    batch_size = 500
    for i in tqdm(range(0, len(ids), batch_size), desc="Updating ChromaDB"):
        collection.update(
            ids=ids[i : i + batch_size],
            metadatas=updated_metas[i : i + batch_size],
        )

    print(f"\nDone. {len(cluster_labels)} persons | {noise_count} faces unlabeled.")
    return {"clusters": len(cluster_labels), "noise": noise_count}
