"""Utilities for unseen cell-type classification via ontology-guided centroids.

This module provides three main steps:
1) Compute centroids for the seen cell types directly from per-cell embeddings.
2) Infer centroids for unseen types by projecting LCA distances into the
   embedding space (distance -> similarity -> weighted average of seen centroids).
3) Classify a new cell by nearest centroid (cosine distance) among the unseen types.

Run as a script, it computes the seen cell-type centroids by encoding the
pretraining corpus with the model (last-layer <cls> embedding averaged per cell
type, using the CL id map ``{CL_id: {dataset_id: [row_indices]}}``) and writes
them to ``--out``.

Inputs : tokenized pretraining HF dataset (``--dataset_path``), CL id map,
         LCA distance matrix, token dictionary, model weights.
Output : ``seen_celltype_centroids.pkl`` ({CL_id: centroid vector}).

Example (from the repo root, GPU recommended):
    python downstream/4_novel_celltype/onto_unseenCell_classifier.py \
        --dataset_path path/to/scLLM_onto_dataset \
        --out downstream/4_novel_celltype/seen_celltype_centroids.pkl
"""

from __future__ import annotations

import argparse
import logging
import pickle
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from datasets import Dataset
from numpy.typing import NDArray
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertForMaskedLM

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from scOntoFM.paths import CL_ID_MAP_PATH, LCA_DISTANCE_PATH, MODEL_DIR, TOKEN_DICT_PATH  # noqa: E402


logger = logging.getLogger(__name__)


# Candidate column names that may store per-cell embeddings in the HF dataset.
EMBEDDING_CANDIDATES: Tuple[str, ...] = (
    "cell_embedding",
    "embedding",
    "emb",
    "cell_emb",
    "cls_embedding",
    "representation",
    "hidden_state",
    "features",
)


def nested_defaultdict():
    """Helper for nested defaultdict (kept for compatibility)."""
    return defaultdict(list)


def _create_collate_fn(token_dict: Mapping[str, int]):
    """Pad `input_ids` and build attention masks for inference."""

    pad_token = token_dict["<pad>"]

    def collate_fn(batch):
        seqs = [torch.tensor(b["input_ids"], dtype=torch.long) for b in batch]
        input_ids = pad_sequence(seqs, batch_first=True, padding_value=pad_token)
        attention_mask = (input_ids != pad_token).long()
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    return collate_fn


def _infer_embedding_key(dataset: Dataset, preferred: Optional[str] = None) -> str:
    """Select an embedding column from the dataset.

    Args:
        dataset: HF dataset containing per-cell embeddings.
        preferred: Optional explicit column name to use first.

    Raises:
        KeyError: if no candidate column is found.
    """

    candidates: Iterable[str]
    if preferred:
        candidates = (preferred, *EMBEDDING_CANDIDATES)
    else:
        candidates = EMBEDDING_CANDIDATES

    for key in candidates:
        if key in dataset.column_names:
            return key
    raise KeyError(
        "No embedding column found. Available columns: "
        f"{dataset.column_names}. Tried: {[preferred] if preferred else [] + list(EMBEDDING_CANDIDATES)}"
    )


def _flatten_indices(nested_map: Mapping[str, Mapping[str, Sequence[int]]]) -> Mapping[str, List[int]]:
    """Flatten nested index map into {celltype: [all_indices]} for convenience."""

    flat: Dict[str, List[int]] = {}
    for celltype, dataset_map in nested_map.items():
        idxs: List[int] = []
        for indices in dataset_map.values():
            idxs.extend(int(i) for i in indices)
        flat[celltype] = idxs
    return flat


def compute_celltype_centroids(
    cl_id_map: Mapping[str, Mapping[str, Sequence[int]]],
    dataset: Dataset,
    embedding_key: Optional[str] = None,
) -> Dict[str, NDArray[np.float32]]:
    """Average per-cell embeddings to obtain one centroid per cell type.

    Args:
        cl_id_map: Nested dict {celltype: {dataset_id: [row_indices]}}.
        dataset: HF dataset holding per-cell embeddings.
        embedding_key: Optional explicit column name with embeddings.

    Returns:
        Dict mapping celltype -> centroid vector (float32).
    """

    key = _infer_embedding_key(dataset, preferred=embedding_key)
    flat_map = _flatten_indices(cl_id_map)
    centroids: Dict[str, NDArray[np.float32]] = {}

    for celltype, indices in flat_map.items():
        if not indices:
            logger.warning("No indices found for cell type %s; skipping", celltype)
            continue

        subset = dataset.select(indices)
        # Convert to numpy array; HF will materialize only the requested column.
        emb_array = np.asarray(subset[key], dtype=np.float32)
        if emb_array.ndim != 2:
            raise ValueError(
                f"Embedding column '{key}' for cell type {celltype} is not 2D; "
                f"shape={emb_array.shape}"
            )
        centroids[celltype] = emb_array.mean(axis=0)

    return centroids


def compute_celltype_centroids_from_model(
    cl_id_map: Mapping[str, Mapping[str, Sequence[int]]],
    dataset: Dataset,
    model: torch.nn.Module,
    token_dict: Mapping[str, int],
    batch_size: int = 32,
    num_workers: int = 4,
    device: Optional[str] = None,
) -> Dict[str, NDArray[np.float32]]:
    """Compute centroids per cell type by running the model over each subset.

    The centroid uses the last-layer CLS embedding averaged across all cells
    belonging to that cell type (aggregating across datasets via the provided
    index map).

    Args:
        cl_id_map: Nested dict {celltype: {dataset_id: [row_indices]}}.
        dataset: HF dataset holding tokenized sequences.
        model: Pretrained model that returns `hidden_states` when called.
        token_dict: Vocabulary mapping containing at least the `<pad>` token.
        batch_size: Batch size for the per-celltype DataLoader.
        num_workers: Dataloader workers.
        device: Torch device string; defaults to CUDA if available else CPU.

    Returns:
        Dict mapping celltype -> centroid vector (float32).
    """

    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(device)
    model.eval()

    flat_map = _flatten_indices(cl_id_map)
    collate_fn = _create_collate_fn(token_dict)

    centroids: Dict[str, NDArray[np.float32]] = {}

    for celltype, indices in flat_map.items():
        if not indices:
            logger.warning("No indices found for cell type %s; skipping", celltype)
            continue

        subset = dataset.select(indices)
        dl = DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn,
        )

        cell_embs: List[torch.Tensor] = []
        with torch.no_grad():
            for batch in tqdm(dl, desc=f"Embeddings for {celltype}", leave=False):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=input_ids,
                    output_hidden_states=True,
                )
                hidden_states = outputs.hidden_states
                cls_emb = hidden_states[-1][:, 0, :].detach()
                cell_embs.append(cls_emb.cpu())

        if not cell_embs:
            logger.warning("No embeddings collected for cell type %s; skipping", celltype)
            continue

        stacked = torch.cat(cell_embs, dim=0)
        centroids[celltype] = stacked.mean(dim=0).numpy().astype(np.float32)

    return centroids


def _split_seen_unseen(
    nodes: Sequence[str],
    seen_centroids: Mapping[str, NDArray[np.float32]],
) -> Tuple[List[str], List[str]]:
    """Split ontology nodes into seen (have centroids) and unseen (missing)."""

    seen = [n for n in nodes if n in seen_centroids]
    unseen = [n for n in nodes if n not in seen_centroids]
    return seen, unseen


def _similarity_weights(
    distance_matrix: NDArray[np.float64],
    seen_idx: Sequence[int],
    unseen_idx: Sequence[int],
    gamma: float = 1.0,
    kernel: str = "exp",
) -> NDArray[np.float64]:
    """Convert LCA distances (unseen x seen) into normalized similarity weights."""

    submat = distance_matrix[np.ix_(unseen_idx, seen_idx)]
    if kernel == "exp":
        weights = np.exp(-gamma * submat)
    elif kernel == "inverse":
        weights = 1.0 / (1.0 + submat)
    else:
        raise ValueError(f"Unsupported kernel '{kernel}'. Use 'exp' or 'inverse'.")

    row_sums = weights.sum(axis=1, keepdims=True)
    # Avoid divide-by-zero when an unseen type has no reachable neighbors.
    row_sums[row_sums == 0.0] = 1.0
    return weights / row_sums


def infer_unseen_centroids(
    distance_matrix: NDArray[np.float64],
    nodes: Sequence[str],
    seen_centroids: Mapping[str, NDArray[np.float32]],
    gamma: float = 1.0,
    kernel: str = "exp",
) -> Tuple[Dict[str, NDArray[np.float32]], List[str], List[str], NDArray[np.float64]]:
    """Project ontology distances to estimate centroids for unseen cell types.

    Args:
        distance_matrix: Full (n+m) x (n+m) LCA distance matrix ordered by ``nodes``.
        nodes: Cell type IDs corresponding to matrix axes.
        seen_centroids: Dict of already-computed centroids for seen types.
        gamma: Scale for exponential kernel (only used when kernel='exp').
        kernel: 'exp' (Gaussian RBF) or 'inverse' (1 / (1 + D)).

    Returns:
        - Dict of inferred centroids for unseen types.
        - Ordered list of seen node IDs used.
        - Ordered list of unseen node IDs generated.
        - Normalized weight matrix W^ (shape m x n).
    """

    seen_nodes, unseen_nodes = _split_seen_unseen(nodes, seen_centroids)
    if not unseen_nodes:
        logger.info("No unseen cell types found; nothing to infer.")
        return {}, seen_nodes, unseen_nodes, np.zeros((0, 0), dtype=np.float64)

    node_to_idx = {n: i for i, n in enumerate(nodes)}
    seen_idx = [node_to_idx[n] for n in seen_nodes]
    unseen_idx = [node_to_idx[n] for n in unseen_nodes]

    W_hat = _similarity_weights(distance_matrix, seen_idx, unseen_idx, gamma=gamma, kernel=kernel)

    P_seen = np.stack([seen_centroids[n] for n in seen_nodes], axis=0)  # (n, d)
    P_novel = W_hat @ P_seen  # (m, d)

    inferred: Dict[str, NDArray[np.float32]] = {
        ct: P_novel[i].astype(np.float32) for i, ct in enumerate(unseen_nodes)
    }

    return inferred, seen_nodes, unseen_nodes, W_hat


def classify_against_unseen(
    cell_embedding: Sequence[float],
    unseen_centroids: Mapping[str, NDArray[np.float32]],
) -> Tuple[str, float, Dict[str, float]]:
    """Nearest-centroid classification using cosine distance.

    Args:
        cell_embedding: Embedding vector for the query cell.
        unseen_centroids: Dict of unseen cell type -> centroid vector.

    Returns:
        Tuple of (predicted_celltype, min_distance, per_type_distance_dict).
    """

    if not unseen_centroids:
        raise ValueError("No unseen centroids provided for classification.")

    names = list(unseen_centroids.keys())
    centroid_matrix = np.stack([unseen_centroids[n] for n in names], axis=0)

    cell_vec = np.asarray(cell_embedding, dtype=np.float32)
    cell_norm = np.linalg.norm(cell_vec)
    centroid_norms = np.linalg.norm(centroid_matrix, axis=1)

    # Guard against zero vectors.
    cell_norm = cell_norm if cell_norm != 0 else 1e-12
    centroid_norms[centroid_norms == 0] = 1e-12

    cosine_sim = (centroid_matrix @ cell_vec) / (centroid_norms * cell_norm)
    cosine_dist = 1.0 - cosine_sim

    best_idx = int(np.argmin(cosine_dist))
    best_name = names[best_idx]
    distances = {name: float(cosine_dist[i]) for i, name in enumerate(names)}

    return best_name, float(cosine_dist[best_idx]), distances


def load_resources(
    distance_path: str = str(LCA_DISTANCE_PATH),
    dataset_path: Optional[str] = None,
    cl_map_path: str = str(CL_ID_MAP_PATH),
) -> Tuple[NDArray[np.float64], List[str], Dataset, Mapping[str, Mapping[str, Sequence[int]]]]:
    """Convenience loader for distance matrix, dataset, and index map."""

    with open(distance_path, "rb") as f:
        logger.info("Loading precomputed LCA distance matrix from %s", distance_path)
        data = pickle.load(f)

    dis_mat = data["Distance"]
    nodes = data["nodes"]

    dataset = Dataset.load_from_disk(dataset_path)

    with open(cl_map_path, "rb") as f:
        logger.info("Loading CL id map from %s", cl_map_path)
        nested_map = pickle.load(f)

    return dis_mat, nodes, dataset, nested_map


def parse_args():
    parser = argparse.ArgumentParser(description="Compute seen cell-type centroids from the pretraining corpus")
    parser.add_argument("--dataset_path", required=True, help="Tokenized pretraining HF dataset (load_from_disk).")
    parser.add_argument("--distance_path", default=str(LCA_DISTANCE_PATH), help="LCA distance pickle.")
    parser.add_argument("--cl_map_path", default=str(CL_ID_MAP_PATH), help="CL id map {CL: {dataset_id: [rows]}}.")
    parser.add_argument("--token_dict_path", default=str(TOKEN_DICT_PATH))
    parser.add_argument("--model_path", default=str(MODEL_DIR))
    parser.add_argument("--out", default="seen_celltype_centroids.pkl", help="Output centroid pickle.")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = parse_args()

    dis_mat, nodes, dataset, cl_map = load_resources(
        distance_path=args.distance_path, dataset_path=args.dataset_path, cl_map_path=args.cl_map_path
    )
    with open(args.token_dict_path, "rb") as f:
        token_dict = pickle.load(f)
    model = BertForMaskedLM.from_pretrained(args.model_path)

    centroids = compute_celltype_centroids_from_model(
        cl_map, dataset, model, token_dict,
        batch_size=args.batch_size, num_workers=args.num_workers, device=args.device
    )

    # save centroids for seen cell types
    with open(args.out, "wb") as f:
        pickle.dump(centroids, f)
    logger.info("Saved %d seen centroids to %s", len(centroids), args.out)

    unseen_centroids, seen_nodes, unseen_nodes, W = infer_unseen_centroids(
        dis_mat, nodes, centroids
    )
    logger.info("Inferred %d unseen centroids from %d seen types", len(unseen_centroids), len(seen_nodes))

