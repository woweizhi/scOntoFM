"""Run unseen (novel) cell-type classification with ontology-inferred centroids.

Steps:
- Read the unseen CL ids and their cell counts (``CL_id_obs.csv`` with columns
  ``CL_id, obs_count``, next to the per-CL query files) and keep CLs with at
  least ``--min-cells-per-cl`` cells.
- Load precomputed per-cell embeddings of the unseen cells
  (``<embedding-root>/<model-dir>/adata_CL_xxxxxxx/adata_CL_xxxxxxx_embeddings.npy``,
  i.e. the output layout of ``scripts/run_inference.py --run_id <model-dir>``),
  optionally capped to ``--sample-per-cl`` cells per CL.
- Load the seen cell-type centroids (``build_seen_centroids.py``) and infer
  centroids for unseen types from the LCA distance matrix.
- Classify each cell against the inferred unseen centroids (cosine distance)
  and annotate the ontology relation (exact/parent/child/sibling/no_match).

Outputs (``<output-root>/<model-dir>/<output-prefix>/``):
  inferred_unseen_centroids.pkl, unseen_cell_predictions.pkl,
  unseen_celltype_accuracy.pkl, unseen_ontology_accuracy.pkl

Example (run from this folder):
    python novel_cell_prediction.py \
        --model-dir scOntoFM \
        --sample-per-cl 0 \
        --min-cells-per-cl 2000 \
        --output-prefix minCL2000
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import pickle
import sys
from collections import defaultdict
from typing import Dict, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
import torch
from datasets import Dataset
from scipy.stats import spearmanr
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path

from onto_unseenCell_classifier import (
    _create_collate_fn,
    classify_against_unseen,
    infer_unseen_centroids,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from scOntoFM.ontology import get_id_mapper, import_cell_ontology  # noqa: E402
from scOntoFM.paths import CL_OBO_PATH, LCA_DISTANCE_PATH  # noqa: E402


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Default paths (relative to this folder) and knobs; all overridable from the CLI.
HERE = Path(__file__).resolve().parent
EMBEDDING_ROOT = str(HERE / "output" / "unseen_CL_emb")
SEEN_CENTROID_ROOT = str(HERE / "output" / "seen_CL_emb")
OUTPUT_ROOT = str(HERE / "output" / "novelCL_result")
DISTANCE_PATH = str(LCA_DISTANCE_PATH)
BASIC_OBO_FILE = str(CL_OBO_PATH)
DEFAULT_MODEL_DIR = "scOntoFM"
SAMPLE_PER_CL = 100
MIN_CELLS_PER_CL = 0
UNSEEN_COUNT_CSV_PATH = str(HERE / "czi_query" / "data" / "unseen_CL" / "CL_id_obs.csv")
RANDOM_SEED = 0
OUTPUT_PREFIX = "test"


def load_data_id_map(pickle_path: str) -> Mapping[str, Mapping[str, Sequence[int]]]:
    with open(pickle_path, "rb") as f:
        return pickle.load(f)


def _flatten_indices(nested_map: Mapping[str, Mapping[str, Sequence[int]]]) -> Mapping[str, List[int]]:
    """Flatten nested index map into {celltype: [all_indices]} for convenience."""

    flat: Dict[str, List[int]] = {}
    for celltype, dataset_map in nested_map.items():
        idxs: List[int] = []
        for indices in dataset_map.values():
            idxs.extend(int(i) for i in indices)
        flat[celltype] = idxs
    return flat


def _normalize_cl_id(cl_id: str) -> str:
    """Normalize CL identifiers to the canonical ``CL:xxxxxxx`` format."""

    if cl_id.startswith("adata_"):
        cl_id = cl_id.split("adata_", 1)[1]
    if cl_id.startswith("CL_"):
        return cl_id.replace("CL_", "CL:", 1)
    return cl_id


def _cl_id_from_folder(folder_name: str) -> str:
    """Extract a normalized CL id from a folder like ``adata_CL_0000000``."""

    return _normalize_cl_id(folder_name)


def _find_embedding_file(folder_path: str, folder_name: str) -> str:
    """Locate the embedding file inside a per-CL folder."""

    candidates = [
        os.path.join(folder_path, f"{folder_name}_embeddings.npy"),
        os.path.join(folder_path, f"{folder_name}_embedding.npy"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    npy_files = [f for f in os.listdir(folder_path) if f.endswith(".npy")]
    raise FileNotFoundError(
        f"No embedding file found in {folder_path}. Tried: {candidates}. "
        f"Found: {npy_files}"
    )


def load_precomputed_embeddings(
    embedding_root: str,
    model_dir: str,
    target_cls: Optional[Set[str]],
    sample_per_cl: Optional[int],
    random_seed: int = RANDOM_SEED,
) -> Tuple[np.ndarray, List[Tuple[str, int]], Dict[str, int]]:
    """Load precomputed cell embeddings from disk instead of running the model.

    Args:
        embedding_root: Root directory containing per-model subdirectories.
        model_dir: Subdirectory name under ``embedding_root`` to use.
        target_cls: Optional set of normalized CL ids to keep; if ``None`` keeps all.
        sample_per_cl: Optional cap of cells per CL; ``None`` uses all cells.
        random_seed: Seed for deterministic subsampling.

    Returns:
        - Stacked cell embeddings (N, D).
        - Evaluation pairs [(true_cl, synthetic_row_idx)].
        - Per-CL counts after sampling.
    """

    model_path = os.path.join(embedding_root, model_dir)
    if not os.path.isdir(model_path):
        raise FileNotFoundError(f"Model directory not found: {model_path}")

    rng = np.random.default_rng(random_seed)
    embeddings: List[np.ndarray] = []
    eval_pairs: List[Tuple[str, int]] = []
    per_cl_counts: Dict[str, int] = {}
    global_idx = 0

    for folder in sorted(os.listdir(model_path)):
        folder_path = os.path.join(model_path, folder)
        if not os.path.isdir(folder_path):
            continue

        cl_id = _cl_id_from_folder(folder)
        if target_cls and cl_id not in target_cls:
            continue

        embedding_file = _find_embedding_file(folder_path, folder)
        arr = np.load(embedding_file)
        if arr.ndim != 2:
            raise ValueError(
                f"Embedding file {embedding_file} must be 2D, got shape {arr.shape}"
            )

        selected_idx = np.arange(arr.shape[0])
        if sample_per_cl and sample_per_cl > 0 and arr.shape[0] > sample_per_cl:
            selected_idx = rng.choice(arr.shape[0], size=sample_per_cl, replace=False)
        arr = arr[selected_idx]
        per_cl_counts[cl_id] = int(arr.shape[0])

        for emb in arr:
            embeddings.append(emb.astype(np.float32))
            eval_pairs.append((cl_id, global_idx))
            global_idx += 1

    if not embeddings:
        raise RuntimeError(
            "No embeddings were loaded. Check target cell types, model directory, and files."
        )

    stacked = np.vstack(embeddings).astype(np.float32)
    return stacked, eval_pairs, per_cl_counts


def load_distance_and_nodes(pickle_path: str) -> Tuple[np.ndarray, List[str]]:
    with open(pickle_path, "rb") as f:
        data = pickle.load(f)
    return data["Distance"], data["nodes"]


def load_unseen_from_csv(csv_path: str, min_cells: int) -> Tuple[List[str], Dict[str, int]]:
    """Load unseen CL ids and counts from a CSV and filter by size threshold."""

    unseen: List[str] = []
    counts: Dict[str, int] = {}

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        if "CL_id" not in reader.fieldnames or "obs_count" not in reader.fieldnames:
            raise ValueError(f"CSV {csv_path} must have columns CL_id and obs_count")
        for row in reader:
            cl = _normalize_cl_id(row["CL_id"])
            try:
                cnt = int(row["obs_count"])
            except (TypeError, ValueError):
                continue
            counts[cl] = cnt
            if cnt >= min_cells:
                unseen.append(cl)

    return unseen, counts


def compute_cell_embeddings(
    dataset: Dataset,
    indices: Sequence[int],
    model: torch.nn.Module,
    token_dict: Mapping[str, int],
    batch_size: int,
    num_workers: int,
    device: str,
) -> np.ndarray:
    """Run model to obtain CLS embeddings for the selected indices."""

    subset = dataset.select(list(indices))
    collate_fn = _create_collate_fn(token_dict)
    loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )

    embeddings: List[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc="Encoding sampled cells", leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids,
                output_hidden_states=True,
            )
            cls_emb = outputs.hidden_states[-1][:, 0, :].detach()
            embeddings.append(cls_emb.cpu())

    if not embeddings:
        raise RuntimeError("No embeddings were produced from the model.")

    return torch.cat(embeddings, dim=0).numpy().astype(np.float32)


def classify_sampled_cells(
    cell_embeddings: np.ndarray,
    eval_pairs: Sequence[Tuple[str, int]],
    unseen_centroids: Mapping[str, np.ndarray],
) -> List[Dict[str, object]]:
    """Classify each sampled cell against unseen centroids."""

    results: List[Dict[str, object]] = []
    for emb, (true_cl, row_idx) in zip(cell_embeddings, eval_pairs):
        pred, dist, dist_map = classify_against_unseen(emb, unseen_centroids)
        results.append(
            {
                "row_index": int(row_idx),
                "true_cl": true_cl,
                "pred_cl": pred,
                "cosine_distance": dist,
                "all_distances": dist_map,
            }
        )
    return results


def _safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman correlation with NaN/constant safeguards."""

    coef, _ = spearmanr(x, y)
    if np.isnan(coef):
        return 0.0
    return float(coef)


def _compute_query_similarity_vector(
    cell_embedding: np.ndarray,
    seen_centroids: Mapping[str, np.ndarray],
    seen_order: Sequence[str],
) -> np.ndarray:
    """Cosine similarities between query embedding and ordered seen centroids."""

    seen_matrix = np.stack([seen_centroids[name] for name in seen_order], axis=0)
    cell_vec = np.asarray(cell_embedding, dtype=np.float32)

    cell_norm = np.linalg.norm(cell_vec)
    centroid_norms = np.linalg.norm(seen_matrix, axis=1)

    cell_norm = cell_norm if cell_norm != 0 else 1e-12
    centroid_norms[centroid_norms == 0] = 1e-12

    sim = (seen_matrix @ cell_vec) / (centroid_norms * cell_norm)
    return sim.astype(np.float32)


def compute_ontology_profile_matrix(
    distance_matrix: np.ndarray,
    nodes: Sequence[str],
    seen_nodes: Sequence[str],
    unseen_nodes: Sequence[str],
) -> np.ndarray:
    """Build unseen x seen ontology correlation matrix via Spearman ranks."""

    if not seen_nodes or not unseen_nodes:
        return np.zeros((0, 0), dtype=np.float32)

    node_to_idx = {n: i for i, n in enumerate(nodes)}
    seen_idx = [node_to_idx[n] for n in seen_nodes]
    unseen_idx = [node_to_idx[n] for n in unseen_nodes]

    full_similarity = -np.asarray(distance_matrix, dtype=np.float64)
    seen_profiles = full_similarity[seen_idx, :]

    profile_mat = np.zeros((len(unseen_idx), len(seen_idx)), dtype=np.float32)
    for i, u_idx in enumerate(unseen_idx):
        u_profile = full_similarity[u_idx, :]
        for j, s_profile in enumerate(seen_profiles):
            profile_mat[i, j] = _safe_spearman(u_profile, s_profile)

    return profile_mat


def classify_against_unseen_by_profile(
    cell_embedding: np.ndarray,
    seen_centroids: Mapping[str, np.ndarray],
    seen_order: Sequence[str],
    unseen_order: Sequence[str],
    ontology_profile_matrix: np.ndarray,
) -> Tuple[str, float, Dict[str, float]]:
    """Match query similarity profile to ontology-implied unseen profiles."""

    if ontology_profile_matrix.size == 0:
        raise ValueError("Ontology profile matrix is empty; cannot classify.")

    query_vec = _compute_query_similarity_vector(cell_embedding, seen_centroids, seen_order)

    scores: Dict[str, float] = {}
    for idx, unseen_name in enumerate(unseen_order):
        expected_profile = ontology_profile_matrix[idx, :]
        score = _safe_spearman(query_vec, expected_profile)
        scores[unseen_name] = score

    best_name = max(scores, key=scores.get)
    return best_name, scores[best_name], scores


def classify_sampled_cells_by_profile(
    cell_embeddings: np.ndarray,
    eval_pairs: Sequence[Tuple[str, int]],
    seen_centroids: Mapping[str, np.ndarray],
    seen_order: Sequence[str],
    unseen_order: Sequence[str],
    ontology_profile_matrix: np.ndarray,
) -> List[Dict[str, object]]:
    """Classify each sampled cell by ontology profile matching."""

    results: List[Dict[str, object]] = []
    for emb, (true_cl, row_idx) in zip(cell_embeddings, eval_pairs):
        pred, score, score_map = classify_against_unseen_by_profile(
            emb, seen_centroids, seen_order, unseen_order, ontology_profile_matrix
        )
        results.append(
            {
                "row_index": int(row_idx),
                "true_cl": true_cl,
                "profile_pred_cl": pred,
                "profile_score": float(score),
                "profile_scores": score_map,
            }
        )
    return results


def _load_ontology_graph(obo_path: str = BASIC_OBO_FILE) -> Tuple[object, Dict[str, str]]:
    """Load the Cell Ontology graph and a name -> CL id mapper."""

    onto_graph = import_cell_ontology(obo_path)
    id2name = get_id_mapper(onto_graph)
    name2id = {value: key for key, value in id2name.items()}
    return onto_graph, name2id


def _match_type(n1: str, n2: str, graph) -> str:
    """Ontology-aware match category following provided logic."""

    if n1 == n2:
        return "exact"
    preds1 = set(graph.predecessors(n1))
    preds2 = set(graph.predecessors(n2))
    if preds1 and preds2 and not preds1.isdisjoint(preds2):
        return "sibling"
    if n1 in preds2:
        return "parent"
    if n2 in preds1:
        return "child"
    return "no_match"


def annotate_ontology_matches(
    results: Sequence[Mapping[str, object]],
    graph,
    name2id: Mapping[str, str],
) -> Tuple[List[Dict[str, object]], Dict[str, Dict[str, float]]]:
    """Add ontology match category per cell and return summary fractions."""

    def to_node(label: str) -> str:
        if label in graph:
            return label
        return name2id.get(label, "")

    annotated: List[Dict[str, object]] = []
    counts: Dict[str, int] = defaultdict(int)
    categories = ["exact", "parent", "child", "sibling", "no_match"]

    for row in results:
        true_label = row["true_cl"]
        pred_label = row["pred_cl"]
        t_node = to_node(true_label)
        p_node = to_node(pred_label)
        if not t_node or not p_node:
            match = "no_match"
        else:
            match = _match_type(t_node, p_node, graph)
        counts[match] += 1
        new_row = dict(row)
        new_row["ontology_match"] = match
        annotated.append(new_row)

    total = len(results)
    summary: Dict[str, Dict[str, float]] = {}
    for cat in categories:
        c = counts.get(cat, 0)
        summary[cat] = {"count": float(c), "fraction": (c / total) if total else 0.0}

    return annotated, summary


def per_celltype_accuracy(results: Sequence[Mapping[str, object]]) -> Dict[str, Dict[str, float]]:
    """Compute accuracy grouped by true unseen cell type."""

    counts: Dict[str, List[int]] = defaultdict(lambda: [0, 0])
    for row in results:
        true_cl = row["true_cl"]
        pred_cl = row["pred_cl"]
        counts[true_cl][1] += 1
        if true_cl == pred_cl:
            counts[true_cl][0] += 1

    summary: Dict[str, Dict[str, float]] = {}
    for cl, (correct, total) in counts.items():
        acc = correct / total if total else 0.0
        summary[cl] = {"correct": float(correct), "total": float(total), "accuracy": acc}
    return summary


def main(
    model_dir: str = DEFAULT_MODEL_DIR,
    embedding_root: str = EMBEDDING_ROOT,
    sample_per_cl: int = SAMPLE_PER_CL,
    min_cells_per_cl: int = MIN_CELLS_PER_CL,
    unseen_csv_path: str = UNSEEN_COUNT_CSV_PATH,
    output_prefix: str = OUTPUT_PREFIX,
    output_root: str = OUTPUT_ROOT,
    seen_centroid_root: str = SEEN_CENTROID_ROOT,
    seen_centroid_path: Optional[str] = None,
    distance_path: str = DISTANCE_PATH,
    obo_path: str = BASIC_OBO_FILE,
):
    logger.info("Loading unseen CL ids from %s (min cells=%d)", unseen_csv_path, min_cells_per_cl)
    unseen_cls, unseen_counts = load_unseen_from_csv(unseen_csv_path, min_cells_per_cl)
    target_cls = set(unseen_cls)
    if not target_cls:
        raise RuntimeError(
            f"No unseen cell types passed the min-cells threshold {min_cells_per_cl} in {unseen_csv_path}"
        )

    logger.info(
        "Using %d unseen cell types after filtering; sample cap per CL=%s",
        len(target_cls),
        "all" if sample_per_cl <= 0 else sample_per_cl,
    )
    sample_cap = None if sample_per_cl <= 0 else sample_per_cl
    logger.info(
        "Loading precomputed embeddings for %d unseen cell types from %s (model=%s)",
        len(target_cls),
        embedding_root,
        model_dir,
    )
    cell_embeddings, eval_pairs, per_cl_counts = load_precomputed_embeddings(
        embedding_root=embedding_root,
        model_dir=model_dir,
        target_cls=target_cls,
        sample_per_cl=sample_cap,
    )

    loaded_cls = set(per_cl_counts)
    missing_cls = sorted(target_cls - loaded_cls)
    if missing_cls:
        preview = ", ".join(missing_cls[:10])
        if len(missing_cls) > 10:
            preview += " ..."
        logger.warning(
            "Missing embeddings for %d unseen cell types (skipped): %s",
            len(missing_cls),
            preview,
        )

    logger.info(
        "Loaded %d cell embeddings across %d unseen cell types", cell_embeddings.shape[0], len(loaded_cls)
    )
    if cell_embeddings.size == 0:
        raise RuntimeError("No embeddings loaded; aborting.")

    out_dir = Path(output_root) / model_dir / output_prefix
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading seen centroids and ontology distances...")
    if seen_centroid_path is None:
        seen_centroid_path = os.path.join(seen_centroid_root, model_dir, "seen_celltype_centroids.pkl")
    logger.info("Seen centroids: %s", seen_centroid_path)
    with open(seen_centroid_path, "rb") as f:
        seen_centroids = pickle.load(f)
    dis_mat, nodes = load_distance_and_nodes(distance_path)

    inferred, seen_nodes, unseen_nodes, _ = infer_unseen_centroids(
        dis_mat, nodes, seen_centroids
    )

    active_unseen = sorted(loaded_cls & set(inferred))
    unseen_centroids = {cl: inferred[cl] for cl in active_unseen}
    # save the unseen_centroids for downstream analysis
    with open(os.path.join(out_dir, "inferred_unseen_centroids.pkl"), "wb") as f:
        pickle.dump(unseen_centroids, f)
    logger.info("Saved inferred unseen centroids to %s", os.path.join(out_dir, "inferred_unseen_centroids.pkl"))
    if not unseen_centroids:
        raise RuntimeError("No inferred unseen centroids overlap with loaded cell types.")

    logger.info(
        "Inferred %d unseen centroids (seen nodes: %d, unseen nodes: %d, kept: %d)",
        len(inferred),
        len(seen_nodes),
        len(unseen_nodes),
        len(unseen_centroids),
    )

    logger.info("Classifying sampled cells against inferred centroids...")
    results = classify_sampled_cells(cell_embeddings, eval_pairs, unseen_centroids)

    logger.info("Loading ontology graph for hierarchical accuracy...")
    onto_graph, name2id = _load_ontology_graph(obo_path)
    results, ontology_summary = annotate_ontology_matches(results, onto_graph, name2id)

    # Simple summary by accuracy over sampled cells.
    correct = sum(1 for r in results if r["true_cl"] == r["pred_cl"])
    acc = correct / len(results)
    logger.info("Sampled accuracy over %d cells (overall): %.4f", len(results), acc)

    logger.info("Ontology match distribution (fractions over sampled cells):")
    for cat in ["exact", "parent", "child", "sibling", "no_match"]:
        stats = ontology_summary.get(cat, {"count": 0.0, "fraction": 0.0})
        logger.info("  %s: count=%d fraction=%.4f", cat, int(stats["count"]), stats["fraction"])

    per_cl = per_celltype_accuracy(results)
    for cl, stats in sorted(per_cl.items(), key=lambda kv: kv[1]["total"], reverse=True):
        logger.info(
            "Per-CL accuracy %s -> correct=%d total=%d acc=%.4f",
            cl,
            int(stats["correct"]),
            int(stats["total"]),
            stats["accuracy"],
        )

    # Persist per-cell results for downstream analysis.
    with open(os.path.join(out_dir, "unseen_cell_predictions.pkl"), "wb") as f:
        pickle.dump(results, f)
    logger.info("Saved per-cell predictions to %s", os.path.join(out_dir, "unseen_cell_predictions.pkl"))

    with open(os.path.join(out_dir, "unseen_celltype_accuracy.pkl"), "wb") as f:
        pickle.dump(per_cl, f)
    logger.info("Saved per-celltype accuracy to %s", os.path.join(out_dir, "unseen_celltype_accuracy.pkl"))

    with open(os.path.join(out_dir, "unseen_ontology_accuracy.pkl"), "wb") as f:
        pickle.dump(ontology_summary, f)
    logger.info("Saved ontology-based accuracy summary to %s", os.path.join(out_dir, "unseen_ontology_accuracy.pkl"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classify novel cell types using precomputed embeddings")
    parser.add_argument(
        "--model-dir",
        default=DEFAULT_MODEL_DIR,
        help="Subdirectory under embedding_root containing unseen cell embeddings (the inference run_id, e.g. scOntoFM)",
    )
    parser.add_argument(
        "--embedding-root",
        default=EMBEDDING_ROOT,
        help="Root directory that holds per-model unseen embeddings",
    )
    parser.add_argument(
        "--sample-per-cl",
        type=int,
        default=SAMPLE_PER_CL,
        help="Max cells per unseen CL to load; set <=0 to use all",
    )
    parser.add_argument(
        "--min-cells-per-cl",
        type=int,
        default=MIN_CELLS_PER_CL,
        help="filter CLs its total cell number less than the threshold",
    )
    parser.add_argument(
        "--unseen-csv",
        default=UNSEEN_COUNT_CSV_PATH,
        help="CSV with columns CL_id, obs_count to pick unseen cell types",
    )
    parser.add_argument(
        "--output-root",
        default=OUTPUT_ROOT,
        help="Base directory to write prediction outputs; results go under <output-root>/<model-dir>",
    )
    parser.add_argument(
        "--output-prefix",
        default=OUTPUT_PREFIX,
        help="output folder prefix identifier"
    )
    parser.add_argument(
        "--seen-centroid-root",
        default=SEEN_CENTROID_ROOT,
        help="Root holding <model-dir>/seen_celltype_centroids.pkl (written by build_seen_centroids.py)",
    )
    parser.add_argument(
        "--seen-centroid-path",
        default=None,
        help="Explicit seen-centroid pickle; overrides --seen-centroid-root/<model-dir>/seen_celltype_centroids.pkl",
    )
    parser.add_argument(
        "--distance-path",
        default=DISTANCE_PATH,
        help="Pickle with the ontology distance matrix ({'Distance', 'nodes'}); default: LCA distance asset",
    )
    parser.add_argument(
        "--obo-path",
        default=BASIC_OBO_FILE,
        help="Cell Ontology .obo file used for hierarchical (parent/child/sibling) matching",
    )
    args = parser.parse_args()

    main(
        model_dir=args.model_dir,
        embedding_root=args.embedding_root,
        sample_per_cl=args.sample_per_cl,
        min_cells_per_cl=args.min_cells_per_cl,
        output_prefix=args.output_prefix,
        unseen_csv_path=args.unseen_csv,
        output_root=args.output_root,
        seen_centroid_root=args.seen_centroid_root,
        seen_centroid_path=args.seen_centroid_path,
        distance_path=args.distance_path,
        obo_path=args.obo_path,
    )

