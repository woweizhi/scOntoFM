"""AnnData preprocessing for scOntoFM.

Pipeline (applied per dataset):
  1. (optional) keep only ``is_primary_data == True`` cells (CELLxGENE convention);
  2. take raw counts from ``adata.raw.X`` (fallback: ``adata.X``);
  3. drop cells with < ``min_genes_per_cell`` detected genes, then genes detected
     in < ``min_cells_per_gene`` cells;
  4. library-size normalize every cell to ``target_sum`` (no log transform);
  5. write a new AnnData with normalized values in ``.X`` (``.raw`` dropped).

``adata.var_names`` are expected to be Ensembl gene IDs (CELLxGENE convention).
"""

import gc
import logging
import os
from pathlib import Path
from typing import List, Tuple

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp

logger = logging.getLogger(__name__)

# obs-level metadata carried into tokenized Arrow files
OBS_META_FIELDS = [
    "assay_ontology_term_id", "disease_ontology_term_id", "organism_ontology_term_id",
    "tissue_ontology_term_id", "is_primary_data", "self_reported_ethnicity_ontology_term_id",
    "development_stage_ontology_term_id", "sex_ontology_term_id", "cell_type_ontology_term_id",
    "tissue_type", "cell_type", "assay", "disease", "organism", "sex", "tissue",
    "self_reported_ethnicity", "development_stage", "observation_joinid",
]


def read_dataset_ids(csv_path: str) -> List[str]:
    """Read a CSV with a ``dataset_id`` column and return the non-empty IDs."""
    df = pd.read_csv(csv_path)
    if "dataset_id" not in df.columns:
        raise ValueError("CSV must contain a column named 'dataset_id'")
    ids = df["dataset_id"].astype(str).str.strip()
    return ids[ids != ""].tolist()


def ensure_dir(p: str) -> None:
    Path(p).mkdir(parents=True, exist_ok=True)


def _as_csr(m) -> sp.csr_matrix:
    if sp.isspmatrix_csr(m):
        return m
    if sp.issparse(m):
        return m.tocsr()
    return sp.csr_matrix(m)


def get_raw_counts(adata: ad.AnnData) -> Tuple[sp.csr_matrix, np.ndarray, np.ndarray]:
    """Return (raw counts CSR, var_names, obs_names), preferring ``adata.raw``."""
    if (adata.raw is None) or (adata.raw.X is None):
        X = _as_csr(adata.X)
        return X, adata.var_names.to_numpy(), adata.obs_names.to_numpy()
    X = _as_csr(adata.raw.X)
    return X, adata.raw.var_names.to_numpy(), adata.obs_names.to_numpy()


def filter_cells_and_genes_csr(
    X: sp.csr_matrix,
    var_names: np.ndarray,
    obs_names: np.ndarray,
    min_genes_per_cell: int,
    min_cells_per_gene: int,
) -> Tuple[sp.csr_matrix, np.ndarray, np.ndarray]:
    """Filter cells by #detected genes first, then genes by #expressing cells."""
    cell_nnz = np.diff(X.indptr)
    keep_cells = cell_nnz >= min_genes_per_cell
    if not np.any(keep_cells):
        return sp.csr_matrix((0, X.shape[1]), dtype=X.dtype), var_names, obs_names[keep_cells]

    Xc = X[keep_cells, :]
    obs_kept = obs_names[keep_cells]

    gene_nnz = np.diff(Xc.tocsc().indptr)
    keep_genes = gene_nnz >= min_cells_per_gene
    if not np.any(keep_genes):
        return sp.csr_matrix((Xc.shape[0], 0), dtype=X.dtype), var_names[keep_genes], obs_kept

    return Xc[:, keep_genes].tocsr(), var_names[keep_genes], obs_kept


def libsize_normalize_csr(X: sp.csr_matrix, target_sum: float) -> sp.csr_matrix:
    """Scale every row (cell) to sum to ``target_sum`` (cf. ``sc.pp.normalize_total``)."""
    rsums = np.array(X.sum(axis=1)).ravel()
    scale = np.zeros_like(rsums, dtype=np.float64)
    nonzero = rsums > 0
    scale[nonzero] = target_sum / rsums[nonzero]
    X = X.tocsr(copy=True)
    for i in range(X.shape[0]):
        if nonzero[i]:
            start, end = X.indptr[i], X.indptr[i + 1]
            X.data[start:end] *= scale[i]
    return X


def preprocess_single_h5ad(
    h5ad_path: str,
    min_genes_per_cell: int,
    min_cells_per_gene: int,
    target_sum: float,
    out_dir: str,
    primary_data_only: bool = True,
    overwrite: bool = False,
) -> str:
    """Preprocess one ``.h5ad`` file and write the result to ``out_dir/<filename>``.

    If the output file already exists it is reused unless ``overwrite=True``.
    Returns the path of the preprocessed file.
    """
    ensure_dir(out_dir)
    out_path = os.path.join(out_dir, Path(h5ad_path).name)
    if os.path.exists(out_path) and not overwrite:
        logger.info(f"Preprocessed file already exists for {h5ad_path}, skipping.")
        return out_path

    adata = sc.read_h5ad(h5ad_path)
    if primary_data_only and "is_primary_data" in adata.obs.columns:
        if adata.obs["is_primary_data"].dtype != bool:
            adata.obs["is_primary_data"] = adata.obs["is_primary_data"].astype(bool)
        adata = adata[adata.obs["is_primary_data"], :]

    X_raw, var_names, obs_names = get_raw_counts(adata)
    Xf, var_kept, obs_kept = filter_cells_and_genes_csr(
        X_raw, var_names, obs_names, min_genes_per_cell, min_cells_per_gene
    )
    Xn = libsize_normalize_csr(Xf, target_sum)

    try:
        new_obs = adata.obs.loc[obs_kept].copy()
    except Exception:
        new_obs = pd.DataFrame(index=obs_kept)

    try:
        var_df = getattr(adata.raw, "var", None)
        if var_df is None:
            var_df = adata.var
        var_df = var_df.set_index(var_df.index.astype(str))
        new_var = var_df.loc[var_kept].copy()
    except Exception:
        new_var = pd.DataFrame(index=var_kept)

    del adata
    gc.collect()
    new_adata = ad.AnnData(X=Xn, obs=new_obs, var=new_var)
    new_adata.raw = None

    logger.info(f"Preprocessed data in {h5ad_path}")
    new_adata.write_h5ad(out_path)
    return out_path
