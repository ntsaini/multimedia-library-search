import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from app.config import LABELED_FACES_DIR, PHOTO_EXTENSIONS
from app.database import get_connection


def _empty_ref_summary() -> dict:
    return {
        "candidate_refs": 0,
        "valid_refs": 0,
        "invalid_refs": 0,
        "duplicate_refs": 0,
        "zero_face_refs": 0,
        "multi_face_refs": 0,
        "unreadable_refs": 0,
    }


def _load_image(path: Path) -> np.ndarray | None:
    """Load image via cv2; fall back to Pillow for HEIC and other formats."""
    import cv2
    img = cv2.imread(str(path))
    if img is not None:
        return img
    try:
        from PIL import Image
        pil = Image.open(path).convert("RGB")
        return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    except Exception:
        return None


def count_candidate_refs(label_dir: Path = LABELED_FACES_DIR) -> int:
    """Cheap count of candidate reference images — no InsightFace, no decoding."""
    if not label_dir.exists() or not label_dir.is_dir():
        return 0
    return sum(
        1 for p in label_dir.iterdir()
        if p.is_file() and p.suffix.lower() in PHOTO_EXTENSIONS
    )


def load_reference_embeddings(
    label_dir: Path = LABELED_FACES_DIR,
    fa=None,
) -> tuple[dict[str, np.ndarray], dict]:
    """Scan label_dir for reference images and return {name: normalized_embedding}.

    Strict: only images with exactly one detected face produce a reference.
    Duplicate filename stems are dropped (first wins; rest counted as duplicates).
    """
    summary = _empty_ref_summary()
    refs: dict[str, np.ndarray] = {}

    if not label_dir.exists() or not label_dir.is_dir():
        return refs, summary

    candidates = sorted(
        p for p in label_dir.iterdir()
        if p.is_file() and p.suffix.lower() in PHOTO_EXTENSIONS
    )
    summary["candidate_refs"] = len(candidates)

    if not candidates:
        return refs, summary

    # Group by name first so duplicates are countable without invoking InsightFace
    by_name: dict[str, list[Path]] = defaultdict(list)
    for path in candidates:
        by_name[path.stem].append(path)

    if fa is None:
        from app.indexer import _init_face_analysis
        fa = _init_face_analysis(use_gpu=False)

    for name, paths in by_name.items():
        # First path is the canonical reference; remaining paths are duplicates
        primary = paths[0]
        summary["duplicate_refs"] += len(paths) - 1

        frame = _load_image(primary)
        if frame is None:
            summary["unreadable_refs"] += 1
            summary["invalid_refs"] += 1
            print(f"  [labeled-faces] skip {primary.name}: unreadable")
            continue

        faces = fa.get(frame)
        if len(faces) == 0:
            summary["zero_face_refs"] += 1
            summary["invalid_refs"] += 1
            print(f"  [labeled-faces] skip {primary.name}: no face detected")
            continue
        if len(faces) > 1:
            summary["multi_face_refs"] += 1
            summary["invalid_refs"] += 1
            print(f"  [labeled-faces] skip {primary.name}: {len(faces)} faces detected (expected 1)")
            continue

        emb = faces[0].normed_embedding
        if emb is None:
            summary["invalid_refs"] += 1
            print(f"  [labeled-faces] skip {primary.name}: no embedding")
            continue

        emb = np.asarray(emb, dtype=np.float32)
        norm = float(np.linalg.norm(emb))
        if norm > 0:
            emb = emb / norm
        refs[name] = emb

    summary["valid_refs"] = len(refs)
    return refs, summary


def auto_label_persons(
    label_dir: Path = LABELED_FACES_DIR,
    threshold: float = 0.6,
    margin: float = 0.08,
    overwrite: bool = False,
    fa=None,
) -> dict:
    """Match person centroids against reference embeddings; apply confident names.

    A cluster is labeled only when best_dist < threshold AND
    (second_best_dist - best_dist) >= margin. With a single reference, the
    margin check is considered satisfied.
    """
    refs, summary = load_reference_embeddings(label_dir, fa=fa)
    result = {
        "labeled": 0,
        "skipped_existing": 0,
        "ambiguous": 0,
        "no_match": 0,
        **summary,
    }

    if not refs:
        return result

    ref_names = list(refs.keys())
    R = np.stack([refs[n] for n in ref_names]).astype(np.float32)

    conn = get_connection()
    rows = conn.execute(
        "SELECT id, name, centroid FROM persons WHERE centroid IS NOT NULL"
    ).fetchall()

    updates: list[tuple[str, str]] = []  # (name, person_id)

    for row in rows:
        if row["name"] is not None and not overwrite:
            result["skipped_existing"] += 1
            continue

        try:
            centroid = np.asarray(json.loads(row["centroid"]), dtype=np.float32)
        except Exception:
            continue
        norm = float(np.linalg.norm(centroid))
        if norm > 0:
            centroid = centroid / norm

        # Euclidean distance to every reference
        dists = np.linalg.norm(R - centroid, axis=1)
        order = np.argsort(dists)
        best_idx = int(order[0])
        best_dist = float(dists[best_idx])
        if len(order) > 1:
            second_dist = float(dists[int(order[1])])
            gap = second_dist - best_dist
        else:
            gap = float("inf")  # single reference — margin trivially satisfied

        if best_dist < threshold and gap >= margin:
            updates.append((ref_names[best_idx], row["id"]))
        elif best_dist < threshold:
            result["ambiguous"] += 1
        else:
            result["no_match"] += 1

    for name, person_id in updates:
        conn.execute(
            "UPDATE persons SET name = ? WHERE id = ?",
            (name, person_id),
        )
    conn.commit()
    conn.close()

    result["labeled"] = len(updates)
    return result
