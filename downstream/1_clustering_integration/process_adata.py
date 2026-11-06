"""Prepare the pan-tissue CELLxGENE datasets for clustering / integration evaluation.

For each dataset: keep only ``is_primary_data == True`` cells (if the column exists) and
move raw counts from ``adata.raw.X`` (with ``adata.raw.var_names``) into ``adata.X`` as CSR;
if ``adata.raw`` is missing, ``adata.X`` is kept. The resulting files are the inputs to
``scripts/run_inference.py`` and the ``--adata_dir`` of the evaluation scripts here.

Inputs   <adata-base-path>/<dataset_id>.h5ad
Outputs  <output-base-path>/<dataset_id>.h5ad

Example (from this folder)
  python process_adata.py --adata-base-path ../../data/pantissue \
      --output-base-path ../../data/pantissue/filtered \
      --dataset-ids HIHA Heart Lung Liver Kidney Pancreas Breast_TNBC Adipose_tissue Bone_marrow
"""

import argparse
from pathlib import Path

import scanpy as sc
import scipy.sparse as sp


def _as_csr(m):
    if sp.isspmatrix_csr(m):
        return m
    if sp.issparse(m):
        return m.tocsr()
    # dense -> CSR
    return sp.csr_matrix(m)


def main():
    parser = argparse.ArgumentParser(description="Filter adata, favor raw counts when available.")
    parser.add_argument(
        "--adata-base-path",
        default="../../data/pantissue",
        help="Directory containing input .h5ad files.",
    )
    parser.add_argument(
        "--output-base-path",
        default="../../data/pantissue/filtered",
        help="Directory to write filtered .h5ad files.",
    )
    parser.add_argument(
        "--dataset-ids",
        nargs="+",
        required=True,
        help="Dataset IDs (without .h5ad) to process.",
    )
    args = parser.parse_args()

    Path(args.output_base_path).mkdir(parents=True, exist_ok=True)

    for dataset_id in args.dataset_ids:
        print(f"Processing dataset: {dataset_id}")
        adata = sc.read_h5ad(f"{args.adata_base_path}/{dataset_id}.h5ad")

        print(f"Original number of cells: {adata.n_obs}")

        if "is_primary_data" in adata.obs.columns:
            # Ensure boolean
            if adata.obs["is_primary_data"].dtype != bool:
                adata.obs["is_primary_data"] = adata.obs["is_primary_data"].astype(bool)
            adata = adata[adata.obs["is_primary_data"], :].copy()
            print(f"Number of primary cells: {adata.n_obs}")

        if (adata.raw is None) or (adata.raw.X is None):
            print("Warning: adata.raw is missing; using adata.X instead.")
            adata.X = _as_csr(adata.X)
            adata.write_h5ad(f"{args.output_base_path}/{dataset_id}.h5ad")
            continue

        adata.X = _as_csr(adata.raw.X)
        adata.var_names = adata.raw.var_names
        adata.raw = None
        adata.write_h5ad(f"{args.output_base_path}/{dataset_id}.h5ad")

        del adata  # free memory


if __name__ == "__main__":
    main()