"""Spatial gene imputation using reference single-cell embeddings.

This script loads spatial transcriptomics data and reference single-cell data,
computes cosine-similarity-based weights between their embeddings, and imputes
genes that are missing in spatial data. For every gene fold in ``gene_splits.json``
the held-out test genes of each spot are imputed as the top-k softmax-weighted
average of the most similar reference cells, and Pearson / Spearman / Kendall tau
against the measured (log1p) values are reported per gene.

Embedding layout with ``--result-embedding-root R`` (the output dirs of the
inference scripts):
  spatial folds : R/<spatial-dataset-id>/<model>/fold_<k>_adata/fold_<k>_adata_embeddings.npy
                  (run_inference_imputation.py --output_dir R --run_id <model>)
  SC reference  : R/<model>/<sc-dataset-id>/<sc-dataset-id>_embeddings.npy
                  (scripts/run_inference.py --output_dir R --run_id <model>)

Outputs (``<output-dir>/<spatial-dataset-id>/<model>/``):
  <spatial-dataset-id>_<model>_imputation_metrics.csv          per-gene metrics
  <spatial-dataset-id>_<model>_imputation_metrics_summary.csv  mean / median

Example (from this folder):
    python run_imputation.py --spatial data/SEAAD_donor_H21_032.h5ad \
        --reference data/SEAAD_ref_donor_H21_032.h5ad --model scOntoFM --top-k 50 \
        --gene-splits ./cache/imputation_arrow --output-dir ./imputation_outputs \
        --sc-dataset-id SEAAD_ref_donor_H21_032 --spatial-dataset-id SEAAD_donor_H21_032 \
        --result-embedding-root ../../output
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torch.nn.functional as F
from scipy import stats
from scipy import sparse
from tqdm import tqdm


def cosine_similarity_shared_einsum(
    sc_emb: torch.Tensor,
    st_emb: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Pairwise cosine similarity between spatial and single-cell embeddings."""

    st_norm = F.normalize(st_emb, p=2, dim=1, eps=eps)
    sc_norm = F.normalize(sc_emb, p=2, dim=1, eps=eps)
    return torch.einsum("id,jd->ij", st_norm, sc_norm)


def topk_softmax(similarity_matrix: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Keep top-k neighbors per spatial spot and normalize rows to sum to 1."""

    k = min(k, similarity_matrix.shape[1])
    topk_vals, topk_indices = torch.topk(similarity_matrix, k, dim=1)
    masked = torch.full_like(similarity_matrix, float("-inf"))
    masked.scatter_(1, topk_indices, topk_vals)
    weights = F.softmax(masked, dim=1)
    return weights, topk_indices


def to_dense(matrix) -> np.ndarray:
    """Convert a SciPy sparse or array-like matrix to a dense numpy array."""

    if sparse.issparse(matrix):
        return matrix.toarray()
    return np.asarray(matrix)


def load_sc_embedding(root: Path, model: str, dataset_id: str) -> np.ndarray:
    """Load single-cell reference embedding from result root."""

    candidates = [
        root / model / dataset_id / f"{dataset_id}_embeddings.npy",
        root / model / f"{dataset_id}_embeddings.npy",
    ]

    for candidate in candidates:
        if candidate.exists():
            arr = np.load(candidate, allow_pickle=True)
            if arr.shape == ():
                obj = arr.item()
                if isinstance(obj, dict):
                    for key in ("embeddings", "embedding", "data"):
                        if key in obj:
                            return np.asarray(obj[key])
                return np.asarray(obj)
            return np.asarray(arr)

    raise FileNotFoundError(
        f"No reference embedding found for dataset '{dataset_id}' and model '{model}' under '{root}'."
    )


def load_fold_embedding(root: Path, model: str, fold_name: str) -> np.ndarray:
    """Load a per-fold embedding saved as .npy for a given model."""

    candidates = [
        root / model / f"{fold_name}_adata_embeddings.npy",
        root / model / fold_name / f"{fold_name}_adata_embeddings.npy",
        root / model / f"{fold_name}_adata" / f"{fold_name}_adata_embeddings.npy",
    ]

    for candidate in candidates:
        if candidate.exists():
            arr = np.load(candidate, allow_pickle=True)
            if arr.shape == ():
                obj = arr.item()
                if isinstance(obj, dict):
                    for key in ("embeddings", "embedding", "data"):
                        if key in obj:
                            return np.asarray(obj[key])
                return np.asarray(obj)
            return np.asarray(arr)

    raise FileNotFoundError(
        f"No embedding file found for fold '{fold_name}' and model '{model}' under '{root}'."
    )


def impute_for_embedding(
    spatial_path: Path,
    seq_data: sc.AnnData,
    model: str,
    top_k: int,
    gene_splits: dict,
    output_dir: Path,
    spatial_template: Optional[str] = None,
    spatial_embedding_root: Optional[Path] = None,
    result_embedding_root: Optional[Path] = None,
    sc_dataset_id: Optional[str] = None,
    spatial_dataset_id: Optional[str] = None,
) -> tuple[List[Sequence[str]], List[np.ndarray]]:
    """Run similarity weighting and imputation for one model's embedding."""

    test_gene_list: List[Sequence[str]] = []
    st_imputed_cv: List[np.ndarray] = []
    metrics_frames: List[pd.DataFrame] = []

    data_id = spatial_path.stem
    sc_id = sc_dataset_id or "reference"
    spatial_id = spatial_dataset_id or data_id

    for fold_name, split in tqdm(gene_splits.items(), desc=f"CV imputation ({model})"):
        try:
            fold_idx = int(str(fold_name).split("_")[-1])
        except ValueError:
            fold_idx = None

        if spatial_template:
            spatial_path_fold = Path(spatial_template.format(data_id=data_id, fold_idx=fold_idx))
        else:
            spatial_path_fold = spatial_path

        spatial_data = sc.read_h5ad(spatial_path_fold)
        sc.pp.log1p(spatial_data)
        if result_embedding_root is not None:
            spatial_root = result_embedding_root / spatial_id
            st_emb_np = load_fold_embedding(spatial_root, model=model, fold_name=fold_name)
            if st_emb_np.shape[0] != spatial_data.n_obs:
                raise ValueError(
                    f"Spatial embedding rows ({st_emb_np.shape[0]}) do not match observations ({spatial_data.n_obs})"
                )
            st_emb = torch.as_tensor(st_emb_np, dtype=torch.float32)
            sc_emb_np = load_sc_embedding(result_embedding_root, model, sc_id)
            if sc_emb_np.shape[0] != seq_data.n_obs:
                raise ValueError(
                    f"Reference embedding rows ({sc_emb_np.shape[0]}) do not match observations ({seq_data.n_obs})"
                )
            sc_emb = torch.as_tensor(sc_emb_np, dtype=torch.float32)
        else:
            if spatial_embedding_root is not None:
                st_emb_np = load_fold_embedding(spatial_embedding_root, model, fold_name)
                if st_emb_np.shape[0] != spatial_data.n_obs:
                    raise ValueError(
                        f"Spatial embedding rows ({st_emb_np.shape[0]}) do not match observations ({spatial_data.n_obs})"
                    )
                st_emb = torch.as_tensor(st_emb_np, dtype=torch.float32)
            else:
                if model not in spatial_data.obsm:
                    raise KeyError(f"Embedding '{model}' not found in spatial obsm for fold '{fold_name}'.")
                st_emb = torch.as_tensor(spatial_data.obsm[model], dtype=torch.float32)
            sc_emb = torch.as_tensor(seq_data.obsm[model], dtype=torch.float32)

        with torch.no_grad():
            similarity_matrix = cosine_similarity_shared_einsum(sc_emb, st_emb)
            sim_matrix, topk_indices = topk_softmax(similarity_matrix, top_k)

        test_genes = [g for g in split["test_genes"] if g in seq_data.var_names]
        if test_genes:
            seq_data_unseen = seq_data[:, test_genes]
            seq_unseen_tensor = torch.as_tensor(to_dense(seq_data_unseen.X), dtype=torch.float32)
            with torch.no_grad():
                impute_st_raw = torch.matmul(sim_matrix, seq_unseen_tensor).cpu().numpy()
            test_gene_list.append(test_genes)
            st_imputed_cv.append(impute_st_raw)

            # Compute per-fold metrics on the observed genes being tested.
            raw_X = to_dense(spatial_data[:, test_genes].X)
            pearson = []
            spearman = []
            kendalltau_vals = []

            for v1, v2 in zip(raw_X.T, impute_st_raw.T):
                pearson_r = stats.pearsonr(v1.reshape(-1), v2.reshape(-1))
                spearman_r = stats.spearmanr(v1.reshape(-1), v2.reshape(-1))
                kentou = stats.kendalltau(v1.reshape(-1), v2.reshape(-1))

                if np.isnan(v1).any() or np.isnan(v2).any():
                    print("NaN detected in data!")
                if np.isinf(v1).any() or np.isinf(v2).any():
                    print("Infinity detected in data!")

                pearson.append(pearson_r[0])
                spearman.append(spearman_r[0])
                kendalltau_vals.append(kentou.statistic)

            df_sc = pd.DataFrame(
                {
                    "Pearson": pearson,
                    "Spearman": spearman,
                    "Kendalltou": kendalltau_vals,
                }
            )
            metrics_frames.append(df_sc)

        fold_dir = output_dir / spatial_dataset_id / model
        fold_dir.mkdir(parents=True, exist_ok=True)

    # Optionally report aggregated metrics for visibility.
    if metrics_frames:
        all_metric = pd.concat(metrics_frames, axis=0)

        mean_summary = pd.concat(metrics_frames, axis=0).mean()
        median_summary = pd.concat(metrics_frames, axis=0).median()

        # combine two summaries
        combined_summary = pd.concat([mean_summary, median_summary], axis=1)
        combined_summary.columns = ["mean", "median"]
        print(f"Combined Metrics summary for {model}:\n{combined_summary}")

        # save all metrics to csv, and two summary to csv
        all_metric.to_csv(fold_dir / f"{spatial_dataset_id}_{model}_imputation_metrics.csv", index=False)
        combined_summary.to_csv(fold_dir / f"{spatial_dataset_id}_{model}_imputation_metrics_summary.csv", index=True)
    
    return test_gene_list, st_imputed_cv


def main() -> None:
    parser = argparse.ArgumentParser(description="Spatial imputation via embedding similarity.")
    parser.add_argument(
        "--spatial",
        type=Path,
        default=Path("./SEAAD_donor_H21_032.h5ad"),
        help="Path to spatial h5ad with spatial embeddings in obsm.",
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=Path("./SEAAD_ref_donor_H21_032.h5ad"),
        help="Path to reference single-cell h5ad with embeddings in obsm.",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name identifying which embedding set to use (folder name under result root or obsm key).",
    )
    parser.add_argument("--top-k", type=int, default=50, help="Top-k neighbors to keep per spatial spot.")
    parser.add_argument(
        "--gene-splits",
        type=Path,
        default=Path("./gene_splits.json"),
        help="Path to gene_splits.json containing precomputed fold train/test genes.",
    )
    parser.add_argument(
        "--spatial-template",
        type=str,
        default="../../data/spatial_imputation_splits/{data_id}/fold_{fold_idx}_adata.h5ad",
        help="Template for per-fold spatial h5ad (currently not used: metrics are computed on --spatial).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./imputation_outputs"),
        help="Directory to store similarity and imputed outputs.",
    )
    parser.add_argument(
        "--spatial-embedding-root",
        type=Path,
        default=None,
        help=(
            "Optional root directory containing per-model k-fold spatial embeddings saved as .npy files. "
            "When provided, embeddings are loaded from this directory instead of spatial obsm."
        ),
    )
    parser.add_argument(
        "--result-embedding-root",
        type=Path,
        default=None,
        help=(
            "Root directory containing SOTA inference outputs. "
            "If provided, both spatial (per-fold) and reference embeddings are loaded from this root."
        ),
    )
    parser.add_argument(
        "--sc-dataset-id",
        type=str,
        default=None,
        help="Dataset identifier for the reference embedding subfolder/file name.",
    )
    parser.add_argument(
        "--spatial-dataset-id",
        type=str,
        default=None,
        help="Dataset identifier for the spatial embedding subfolder.",
    )

    args = parser.parse_args()

    spatial_data = sc.read_h5ad(args.spatial)
    seq_data = sc.read_h5ad(args.reference)
    sc.pp.log1p(seq_data)

    sc_dataset_id = args.sc_dataset_id or args.reference.stem
    spatial_dataset_id = args.spatial_dataset_id or args.spatial.stem

    gene_splits_path = Path(args.gene_splits) / spatial_dataset_id / "gene_splits.json" if Path(args.gene_splits).is_dir() else Path(args.gene_splits)

    with open(gene_splits_path, "r", encoding="utf-8") as f:
        gene_splits = json.load(f)


    impute_for_embedding(
        spatial_path=args.spatial,
        seq_data=seq_data,
        model=args.model,
        top_k=args.top_k,
        gene_splits=gene_splits,
        output_dir=args.output_dir,
        spatial_template=None,  # --spatial-template is intentionally not used
        spatial_embedding_root=args.spatial_embedding_root,
        result_embedding_root=args.result_embedding_root,
        sc_dataset_id=sc_dataset_id,
        spatial_dataset_id=spatial_dataset_id,
    )


if __name__ == "__main__":
    main()
