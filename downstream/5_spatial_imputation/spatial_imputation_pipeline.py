"""Step 1 of spatial imputation: gene-wise K-fold splits + per-fold tokenization.

The spatial (ST) dataset is preprocessed (library-size normalization, cached),
restricted to genes with a known median, and its genes are split into
``--n_folds`` folds (``KFold(shuffle=True, random_state=42)``). For each fold the
cells are tokenized using only the *train* genes, so the model never sees the
held-out *test* genes that are imputed later.

Outputs (``<output_dir>/<dataset_id>/``):
  fold_<k>/            tokenized HF dataset (``input_ids``) for fold k
  gene_splits.json     {"fold_<k>": {"train_genes": [...], "test_genes": [...]}}

Example (from the repo root):
    python downstream/5_spatial_imputation/spatial_imputation_pipeline.py \
        --raw_data_path data/SEAAD_donor_H21_032.h5ad \
        --cache_dir downstream/5_spatial_imputation/cache
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
import scanpy as sc
import scipy.sparse as sp
from datasets import Dataset
from sklearn.model_selection import KFold
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from scOntoFM.paths import GENE_MEDIAN_PATH, TOKEN_DICT_PATH  # noqa: E402
from scOntoFM.preprocessing import preprocess_single_h5ad  # noqa: E402
from scOntoFM.tokenization import (  # noqa: E402
    build_gene_lookup_arrays,
    get_special_tokens,
    load_gene_median_dict,
    load_token_dictionary,
    process_chunk_fully_vectorized,
)

HERE = Path(__file__).resolve().parent
CACHE_DIR = HERE / "cache"
TOP_K = 2000
N_FOLDS = 10
KFOLD_SEED = 42


def preprocess_input(raw_path: Path, cache_dir: Path) -> Path:
    """Run filtering/normalization once and return the cached path."""

    out_path = preprocess_single_h5ad(
        str(raw_path),
        0,  # min_genes_per_cell
        0,  # min_cells_per_gene
        1e4,  # target_sum
        str(cache_dir),
    )
    return Path(out_path)


def shared_genes(adata: sc.AnnData, gene_median_dict: dict) -> List[str]:
    """Return genes present in the AnnData that also have medians."""

    return [g for g in adata.var_names.tolist() if g in gene_median_dict]


def generate_gene_folds(genes: List[str], n_splits: int, seed: int) -> Iterable[Tuple[int, List[str], List[str]]]:
    """Yield (fold_idx, train_genes, test_genes) with 80/20 splits."""

    gene_array = np.array(genes)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(gene_array)):
        yield fold_idx, gene_array[train_idx].tolist(), gene_array[test_idx].tolist()


def tokenize_on_genes(
    adata: sc.AnnData,
    genes: List[str],
    gene_median_dict: dict,
    token_dict: dict,
    cls_token: int,
    eos_token: int,
    top_k: int,
) -> List[List[int]]:
    """Tokenize the dataset restricted to the provided gene set."""

    adata_slice = adata[:, genes]
    X = sp.csr_matrix(adata_slice.X) if not sp.issparse(adata_slice.X) else adata_slice.X

    median_array, token_array = build_gene_lookup_arrays(genes, gene_median_dict, token_dict)

    tokens_list = process_chunk_fully_vectorized(
        X_chunk=X,
        median_array=median_array,
        token_array=token_array,
        cls_token=cls_token,
        eos_token=eos_token,
        top_k=top_k,
    )
    return tokens_list


def save_fold_outputs(fold_dir: Path, dataset: Dataset) -> None:
    """Persist tokenized data for one fold."""

    os.makedirs(fold_dir, exist_ok=True)
    dataset.save_to_disk(fold_dir)


def save_gene_splits(out_dir: Path, fold_splits: dict) -> None:
    """Write a single nested JSON with all fold splits."""

    os.makedirs(out_dir, exist_ok=True)
    split_path = out_dir / "gene_splits.json"
    with open(split_path, "w", encoding="utf-8") as f:
        json.dump(fold_splits, f, indent=2)


def run_spatial_imputation_pipeline(
    raw_data_path: Path,
    cache_dir: Path = CACHE_DIR,
    output_dir: Optional[Path] = None,
    dataset_id: Optional[str] = None,
    token_dict_path: Path = TOKEN_DICT_PATH,
    gene_median_path: Path = GENE_MEDIAN_PATH,
    n_folds: int = N_FOLDS,
    kfold_seed: int = KFOLD_SEED,
    top_k: int = TOP_K,
) -> None:
    output_dir = Path(output_dir) if output_dir is not None else Path(cache_dir) / "imputation_arrow"

    preprocessed_path = preprocess_input(Path(raw_data_path), Path(cache_dir))
    adata = sc.read_h5ad(preprocessed_path)

    token_dict = load_token_dictionary(token_dict_path)
    gene_median_dict = load_gene_median_dict(gene_median_path)
    cls_token, eos_token = get_special_tokens(token_dict)

    subset_genes = shared_genes(adata, gene_median_dict)
    if len(subset_genes) < n_folds:
        raise ValueError("Not enough shared genes to perform the requested folds.")

    dataset_id = dataset_id or Path(preprocessed_path).stem
    folds = list(generate_gene_folds(subset_genes, n_folds, kfold_seed))

    fold_splits = {}

    for fold_idx, train_genes, test_genes in tqdm(folds, desc="Tokenizing folds"):
        tokens_list = tokenize_on_genes(
            adata=adata,
            genes=train_genes,
            gene_median_dict=gene_median_dict,
            token_dict=token_dict,
            cls_token=cls_token,
            eos_token=eos_token,
            top_k=top_k,
        )

        dataset = Dataset.from_dict({"input_ids": tokens_list})
        fold_dir = output_dir / dataset_id / f"fold_{fold_idx}"
        save_fold_outputs(fold_dir, dataset)

        fold_splits[f"fold_{fold_idx}"] = {
            "train_genes": train_genes,
            "test_genes": test_genes,
        }

    save_gene_splits(output_dir / dataset_id, fold_splits)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Gene K-fold split + per-fold tokenization of a spatial dataset")
    p.add_argument("--raw_data_path", type=Path, required=True, help="Raw-count spatial .h5ad (Ensembl ids as var_names).")
    p.add_argument("--dataset_id", default=None, help="Output sub-folder name; defaults to the input file stem.")
    p.add_argument("--cache_dir", type=Path, default=CACHE_DIR, help="Where the preprocessed .h5ad is cached.")
    p.add_argument("--output_dir", type=Path, default=None, help="Fold datasets root; default <cache_dir>/imputation_arrow.")
    p.add_argument("--token_dict_path", type=Path, default=TOKEN_DICT_PATH)
    p.add_argument("--gene_median_path", type=Path, default=GENE_MEDIAN_PATH)
    p.add_argument("--n_folds", type=int, default=N_FOLDS)
    p.add_argument("--kfold_seed", type=int, default=KFOLD_SEED)
    p.add_argument("--top_k", type=int, default=TOP_K, help="Max gene tokens per cell.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_spatial_imputation_pipeline(
        raw_data_path=args.raw_data_path,
        cache_dir=args.cache_dir,
        output_dir=args.output_dir,
        dataset_id=args.dataset_id,
        token_dict_path=args.token_dict_path,
        gene_median_path=args.gene_median_path,
        n_folds=args.n_folds,
        kfold_seed=args.kfold_seed,
        top_k=args.top_k,
    )