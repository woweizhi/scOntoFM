"""Rank-value tokenization of single-cell expression profiles.

For every cell:
  1. divide each (library-size normalized) gene value by that gene's non-zero
     median across the pretraining corpus (``gene_medians.pkl``);
  2. rank expressed genes by the median-normalized value (descending);
  3. keep the top ``top_k`` (default 2,000) genes and map them to token IDs;
  4. wrap the sequence as ``[<cls>] + gene tokens + [<eos>]`` (max length 2,002).

Special tokens in ``token_dictionary.pkl``: ``<pad>=0, <mask>=1, <cls>=2, <eos>=3``;
the remaining 19,937 tokens are Ensembl gene IDs.
"""

import os
import pickle
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import anndata as ad
import numpy as np
import pyarrow as pa
import scipy.sparse as sp

from .preprocessing import OBS_META_FIELDS

TOP_K_TOKENS = 2000
CHUNK_CELLS = 10000
ARROW_META_FIELDS = OBS_META_FIELDS + ["dataset_id"]


# --------------------------------------------------------------------------- #
# Vocabulary / gene medians
# --------------------------------------------------------------------------- #
def load_pickle_file(file_path: str):
    with open(file_path, "rb") as f:
        return pickle.load(f)


def load_token_dictionary(dict_path: str) -> Dict[str, int]:
    """Gene (Ensembl ID) / special token -> token ID."""
    return load_pickle_file(dict_path)


def load_gene_median_dict(median_path: str) -> Dict[str, float]:
    """Gene (Ensembl ID) -> non-zero median expression in the pretraining corpus."""
    return load_pickle_file(median_path)


def get_special_tokens(token_dict: Dict[str, int]) -> Tuple[int, int]:
    """Return ``(cls_token_id, eos_token_id)``."""
    cls_token = token_dict.get("<cls>", None)
    eos_token = token_dict.get("<eos>", None)
    if cls_token is None or eos_token is None:
        raise ValueError("CLS or EOS tokens not found in dictionary")
    return cls_token, eos_token


def build_gene_lookup_arrays(
    gene_names: Sequence[str],
    gene_median_dict: Dict[str, float],
    token_dict: Dict[str, int],
) -> Tuple[np.ndarray, np.ndarray]:
    """Per-column lookup arrays of gene medians (default 1.0) and token IDs (-1 = unknown)."""
    n_genes = len(gene_names)
    median_array = np.ones(n_genes, dtype=np.float32)
    token_array = np.full(n_genes, -1, dtype=np.int32)
    for idx, gene_name in enumerate(gene_names):
        if gene_name in gene_median_dict and gene_median_dict[gene_name] != 0:
            median_array[idx] = gene_median_dict[gene_name]
        if gene_name in token_dict:
            token_array[idx] = token_dict[gene_name]
    return median_array, token_array


# --------------------------------------------------------------------------- #
# Tokenization
# --------------------------------------------------------------------------- #
def process_chunk_fully_vectorized(
    X_chunk: sp.csr_matrix,
    median_array: np.ndarray,
    token_array: np.ndarray,
    cls_token: int,
    eos_token: int,
    top_k: int,
) -> List[List[int]]:
    """Tokenize a (cells x genes) sparse chunk into lists of token IDs."""
    X_chunk = X_chunk.tocsr(copy=False).astype(np.float32)
    n_cells = X_chunk.shape[0]

    # median normalization on the CSR data array
    X_normalized = X_chunk.copy()
    X_normalized.data = X_normalized.data / median_array[X_normalized.indices]

    tokens_per_cell = []
    for cell_idx in range(n_cells):
        start_idx, end_idx = X_normalized.indptr[cell_idx], X_normalized.indptr[cell_idx + 1]
        if start_idx == end_idx:  # empty cell
            tokens_per_cell.append([cls_token, eos_token])
            continue

        gene_indices = X_normalized.indices[start_idx:end_idx]
        normalized_values = X_normalized.data[start_idx:end_idx]

        ranks = np.argsort(np.argsort(normalized_values)) + 1
        top_k_limit = min(top_k, len(ranks))
        if top_k_limit < len(ranks):
            top_indices = np.argpartition(-ranks, top_k_limit - 1)[:top_k_limit]
            top_indices = top_indices[np.argsort(-ranks[top_indices])]
        else:
            top_indices = np.argsort(-ranks)

        selected_tokens = token_array[gene_indices[top_indices]]
        valid_tokens = selected_tokens[selected_tokens >= 0]
        tokens_per_cell.append([cls_token] + valid_tokens.tolist() + [eos_token])

    return tokens_per_cell


def shared_genes(adata: ad.AnnData, gene_median_dict: dict) -> List[str]:
    """Genes of ``adata`` (in their original order) that have a median value."""
    return [g for g in adata.var_names.tolist() if g in gene_median_dict]


def tokenize_adata(
    adata: ad.AnnData,
    genes: List[str],
    gene_median_dict: dict,
    token_dict: dict,
    top_k: int = TOP_K_TOKENS,
) -> List[List[int]]:
    """Tokenize every cell of ``adata`` restricted to ``genes``."""
    cls_token, eos_token = get_special_tokens(token_dict)
    adata_slice = adata[:, genes]
    X = sp.csr_matrix(adata_slice.X) if not sp.issparse(adata_slice.X) else adata_slice.X
    median_array, token_array = build_gene_lookup_arrays(genes, gene_median_dict, token_dict)
    return process_chunk_fully_vectorized(
        X_chunk=X,
        median_array=median_array,
        token_array=token_array,
        cls_token=cls_token,
        eos_token=eos_token,
        top_k=top_k,
    )


# --------------------------------------------------------------------------- #
# Arrow export (used to build the pretraining corpus)
# --------------------------------------------------------------------------- #
def extract_metadata(adata: ad.AnnData, dataset_id: str, meta_fields: List[str]) -> Dict[str, List]:
    meta_dict = {}
    n_obs = adata.n_obs
    for field in meta_fields:
        if field in adata.obs.columns:
            meta_dict[field] = adata.obs[field].to_list()
        elif field == "dataset_id":
            meta_dict[field] = [dataset_id] * n_obs
        else:
            meta_dict[field] = [None] * n_obs
    return meta_dict


def create_arrow_batch(
    tokens_list: List[List[int]],
    meta_dict: Dict[str, List],
    start_idx: int,
    end_idx: int,
    meta_fields: List[str],
) -> pa.Table:
    arrays = {"input_ids": pa.array(tokens_list, type=pa.list_(pa.int32()))}
    for field in meta_fields:
        arrays[field] = pa.array(meta_dict[field][start_idx:end_idx])
    return pa.table(arrays)


def make_arrow_for_dataset(
    h5ad_path: Path,
    out_dir: Path,
    token_dict_path: Path,
    gene_median_path: Path,
    top_k: int = TOP_K_TOKENS,
    chunk_cells: int = CHUNK_CELLS,
) -> Path:
    """Tokenize a preprocessed ``.h5ad`` and write ``<out_dir>/<stem>.arrow``.

    The Arrow table holds ``input_ids`` plus the obs metadata in
    ``ARROW_META_FIELDS`` (incl. ``cell_type_ontology_term_id`` and ``dataset_id``).
    """
    adata = ad.read_h5ad(h5ad_path)
    token_dict = load_token_dictionary(token_dict_path)
    gene_median_dict = load_gene_median_dict(gene_median_path)
    cls_token, eos_token = get_special_tokens(token_dict)

    # NOTE: the column order of the subset defines the lookup arrays; the same
    # ordered list is used for slicing and for the lookup tables.
    subset_genes = list(set(adata.var_names.tolist()).intersection(gene_median_dict.keys()))
    adata = adata[:, subset_genes]
    X = sp.csr_matrix(adata.X) if not sp.issparse(adata.X) else adata.X
    median_array, token_array = build_gene_lookup_arrays(subset_genes, gene_median_dict, token_dict)

    data_id = Path(h5ad_path).stem
    meta_dict = extract_metadata(adata, data_id, ARROW_META_FIELDS)

    batches = []
    n_cells = X.shape[0]
    for start_idx in range(0, n_cells, chunk_cells):
        end_idx = min(n_cells, start_idx + chunk_cells)
        tokens_list = process_chunk_fully_vectorized(
            X[start_idx:end_idx], median_array, token_array, cls_token, eos_token, top_k
        )
        batches.append(create_arrow_batch(tokens_list, meta_dict, start_idx, end_idx, ARROW_META_FIELDS))

    final_table = pa.concat_tables(batches) if batches else pa.table(
        {"input_ids": pa.array([], type=pa.list_(pa.int32()))}
    )

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{data_id}.arrow")
    with pa.OSFile(str(out_path), "wb") as sink:
        with pa.RecordBatchFileWriter(sink, final_table.schema) as writer:
            writer.write_table(final_table)
    return Path(out_path)
