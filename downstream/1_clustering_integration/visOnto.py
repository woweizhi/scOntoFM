"""scOntoSim: agreement between embedding geometry and the Cell Ontology graph.

For every cell type (``--celltype_key``, CL term IDs) present in both the AnnData and
the reference graph, a trimmed-mean centroid of the cell embeddings is computed
(``--trim_prop`` trimmed from each tail). Pairwise Euclidean centroid distances are then
correlated, row by row, with a reference cell-type distance matrix, and the per-row
Spearman / Kendall tau / Pearson correlations are averaged.

Reference graphs (pickle with keys ``Distance`` (n x n) and ``nodes`` (CL IDs)):
  * LCA (default, shipped): assets/celltype_similarity_LCA_deepest_allnodes.pkl
  * onClass RWR (scGraph-RWR, optional external asset): onClass_rwr_dist.pkl

Inputs
  <base_dir>/<model>/<data_id>/<data_id>_embeddings.npy   (from scripts/run_inference.py)
  <adata_dir>/<data_id>.h5ad                              (row-aligned with the embedding)
Outputs (<output or base_dir>/<run_id>/)
  <data_id>_graph_similarity.csv                 one row per model
  <data_id>_graph_similarity_per_celltype.csv    per-cell-type correlations

Example (from this folder)
  python visOnto.py --base_dir ../../output --adata_dir ../../data/pantissue/filtered \
      --model scOntoFM --data_id Heart --trim_prop 0.2 \
      --output ./output --run_id scOntoGraphSim_SOTA
"""

import argparse
import logging
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from scipy.stats import kendalltau, pearsonr, spearmanr, trim_mean

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from scOntoFM.paths import LCA_DISTANCE_PATH  # noqa: E402

LOG_LEVEL = logging.DEBUG

logging.root.setLevel(LOG_LEVEL)
stream = logging.StreamHandler()
stream.setLevel(LOG_LEVEL)
log = logging.getLogger("graphSimilarity")
log.setLevel(LOG_LEVEL)
log.addHandler(stream)


def load_embedding(embedding_path: Path, model_name: str) -> np.ndarray:
    """Load a saved embedding array from disk."""
    if not embedding_path.exists():
        raise FileNotFoundError(f"embedding file not found at {embedding_path}")
    emb = np.load(embedding_path)
    log.info(f"loaded embedding for model {model_name} from {embedding_path}")
    return emb


def load_reference_graph(reference_path: Path) -> Tuple[np.ndarray, List[str]]:
    """Load the reference ontology graph distance matrix and node order."""
    if not reference_path.exists():
        raise FileNotFoundError(f"reference graph not found at {reference_path}")

    if reference_path.suffix in {".pkl", ".pickle"}:
        with reference_path.open("rb") as f:
            data = pickle.load(f)
        if "Distance" not in data or "nodes" not in data:
            raise ValueError("reference pickle must contain 'Distance' and 'nodes' keys")
        dist_mat = np.asarray(data["Distance"], dtype=float)
        nodes: List[str] = list(data["nodes"])
    elif reference_path.suffix == ".npz":
        with np.load(reference_path) as data:
            if "Distance" not in data or "nodes" not in data:
                raise ValueError("npz reference must store 'Distance' and 'nodes'")
            nodes = list(data["nodes"])
            dist_mat = np.asarray(data["Distance"], dtype=float)
    elif reference_path.suffix == ".npy":
        raise ValueError(".npy references are ambiguous; please provide a pickle or npz with node names")
    else:
        raise ValueError(f"unsupported reference format for {reference_path.suffix}")

    if dist_mat.shape[0] != dist_mat.shape[1]:
        raise ValueError("reference distance matrix must be square")
    if dist_mat.shape[0] != len(nodes):
        raise ValueError("reference matrix size and node list length must match")
    return dist_mat, nodes


def extract_submatrix(similarity: np.ndarray, node_list: Sequence[str], target_nodes: Sequence[str]) -> Tuple[np.ndarray, List[str]]:
    """Extract a submatrix aligned to the provided target node order."""
    idx = {n: i for i, n in enumerate(node_list)}
    filtered_nodes = [n for n in target_nodes if n in idx]
    if not filtered_nodes:
        raise ValueError("none of the requested cell types are present in the reference graph")
    filtered_indices = [idx[n] for n in filtered_nodes]
    submat = similarity[np.ix_(filtered_indices, filtered_indices)]
    return submat, filtered_nodes


def euclidean_distance_matrix_trimmed(
    cell_types: Sequence[str],
    adata: AnnData,
    obsm_key: str,
    celltype_key: str,
    trim_prop: float = 0.2,
) -> np.ndarray:
    """Compute pairwise Euclidean distances between trimmed-mean centroids."""

    centroids: Dict[str, np.ndarray] = {}
    for ct in cell_types:
        X = adata[adata.obs[celltype_key] == ct].obsm[obsm_key]
        if X.shape[0] == 0:
            raise ValueError(f"no cells found for cell type {ct}")
        centroids[ct] = np.asarray(trim_mean(X, proportiontocut=trim_prop, axis=0))

    n = len(cell_types)
    dist_mat = np.zeros((n, n))
    for i, ct1 in enumerate(cell_types):
        v1 = centroids[ct1]
        for j in range(i, n):
            v2 = centroids[cell_types[j]]
            dist = float(np.linalg.norm(v1 - v2))
            dist_mat[i, j] = dist
            dist_mat[j, i] = dist
    return dist_mat


def graph_similarity(embedding_dist: np.ndarray, reference_dist: np.ndarray) -> Dict[str, float]:
    """Row-wise correlation (Spearman, Kendall, Pearson) against the reference graph."""

    if embedding_dist.shape != reference_dist.shape:
        raise ValueError("embedding and reference distance matrices must have the same shape")

    n = embedding_dist.shape[0]
    spear_list: List[float] = []
    kendall_list: List[float] = []
    pearson_list: List[float] = []

    for i in range(n):
        emb_row = embedding_dist[:, i]
        ref_row = reference_dist[:, i]
        mask = np.isfinite(emb_row) & np.isfinite(ref_row)
        if mask.sum() < 2:
            spear_list.append(np.nan)
            kendall_list.append(np.nan)
            pearson_list.append(np.nan)
            continue
        spear_list.append(spearmanr(emb_row[mask], ref_row[mask]).correlation)
        kendall_list.append(kendalltau(emb_row[mask], ref_row[mask]).correlation)
        pearson_list.append(pearsonr(emb_row[mask], ref_row[mask]).statistic)

    return {
        "spearman_list": spear_list,
        "kendall_list": kendall_list,
        "pearson_list": pearson_list,
        "spearman": float(np.nanmean(spear_list)) if np.any(np.isfinite(spear_list)) else np.nan,
        "kendall_tau": float(np.nanmean(kendall_list)) if np.any(np.isfinite(kendall_list)) else np.nan,
        "pearson": float(np.nanmean(pearson_list)) if np.any(np.isfinite(pearson_list)) else np.nan,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Graph similarity between cell embeddings and ontology reference.")
    parser.add_argument("--base_dir", required=True, help="Base directory containing model subfolders.")
    parser.add_argument("--model", required=True, nargs="+", help="One or more model names to evaluate.")
    parser.add_argument("--data_id", required=True, help="Dataset identifier used to locate embeddings and h5ad.")
    parser.add_argument("--adata_dir", default="../../data", help="Directory containing the h5ad file.")
    parser.add_argument("--celltype_key", default="cell_type_ontology_term_id", help="Obs column holding cell type identifiers.")
    parser.add_argument(
        "--reference_path",
        default=str(LCA_DISTANCE_PATH),
        help="Path to the reference distance graph (LCA pickle by default, or onClass RWR pickle).",
    )
    parser.add_argument("--trim_prop", type=float, default=0.2, help="Proportion to trim from each tail for centroid means.")
    parser.add_argument(
        "--output",
        help="Output directory. Defaults to <base_dir> if not set.",
    )
    parser.add_argument(
        "--run_id",
        default="scOntoGraphSim",
        help="Optional run identifier to create a subdirectory under the output directory.",
    )

    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    adata_path = Path(args.adata_dir) / f"{args.data_id}.h5ad"
    if not adata_path.exists():
        raise FileNotFoundError(f"h5ad file not found at {adata_path}")

    adata = sc.read_h5ad(adata_path)
    ref_matrix, ref_nodes = load_reference_graph(Path(args.reference_path))

    adata_cell_types = list(pd.unique(adata.obs[args.celltype_key]))
    aligned_cell_types = [ct for ct in ref_nodes if ct in adata_cell_types]
    if len(aligned_cell_types) < 2:
        raise ValueError("need at least two overlapping cell types between adata and reference graph")
    if len(aligned_cell_types) != len(ref_nodes):
        missing = set(adata_cell_types) - set(aligned_cell_types)
        if missing:
            log.warning("%d cell types in adata not present in reference and will be ignored", len(missing))

    celltype_mask = adata.obs[args.celltype_key].isin(aligned_cell_types)
    adata_filtered = adata[celltype_mask].copy()
    ref_submatrix, aligned_nodes = extract_submatrix(ref_matrix, ref_nodes, aligned_cell_types)

    results = []
    row_results = []
    for model_name in args.model:
        emb_dir = base_dir / model_name / args.data_id
        embedding_path = emb_dir / f"{args.data_id}_embeddings.npy"
        emb_values = load_embedding(embedding_path, model_name)
        if emb_values.shape[0] != adata.n_obs:
            raise ValueError(
                f"embedding for {model_name} has {emb_values.shape[0]} rows but adata has {adata.n_obs} cells (before filtering)"
            )

        emb_values = emb_values[celltype_mask.to_numpy(), :]
        adata_filtered.obsm[model_name] = emb_values
        dist_mat = euclidean_distance_matrix_trimmed(
            aligned_nodes,
            adata_filtered,
            model_name,
            celltype_key=args.celltype_key,
            trim_prop=args.trim_prop,
        )

        similarity = graph_similarity(dist_mat, ref_submatrix)
        log.info(
            "model=%s spearman=%.4f kendall_tau=%.4f pearson=%.4f",
            model_name,
            similarity["spearman"],
            similarity["kendall_tau"],
            similarity["pearson"],
        )

        results.append(
            {
                "model": model_name,
                "spearman": similarity["spearman"],
                "kendall_tau": similarity["kendall_tau"],
                "pearson": similarity["pearson"],
                "cell_types": len(aligned_nodes),
                "trim_prop": args.trim_prop,
                "reference": Path(args.reference_path).name,
            }
        )

        for idx, ct in enumerate(aligned_nodes):
            row_results.append(
                {
                    "model": model_name,
                    "cell_type": ct,
                    "spearman": similarity["spearman_list"][idx],
                    "kendall_tau": similarity["kendall_list"][idx],
                    "pearson": similarity["pearson_list"][idx],
                }
            )

    output_dir = Path(args.output) if args.output else base_dir
    output_dir = output_dir / args.run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / f"{args.data_id}_graph_similarity.csv"
    df = pd.DataFrame(results)
    df.to_csv(summary_path, index=False)
    log.info("saved graph similarity metrics to %s", summary_path)

    row_path = output_dir / f"{args.data_id}_graph_similarity_per_celltype.csv"
    df_rows = pd.DataFrame(row_results)
    df_rows.to_csv(row_path, index=False)
    log.info("saved per-celltype graph similarity metrics to %s", row_path)


if __name__ == "__main__":
    main()