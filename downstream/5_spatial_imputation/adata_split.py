"""Step 2 of spatial imputation: write one spatial .h5ad per gene fold.

Each ``fold_<k>_adata.h5ad`` keeps only the *train* genes of fold k (from
``gene_splits.json`` written by ``spatial_imputation_pipeline.py``). These per-fold
files are the inputs for embedding the spatial data with baseline models.
scOntoFM embeds the pre-tokenized fold datasets directly
(``run_inference_imputation.py``).

Input : <base-data-dir>/<dataset_id>.h5ad, <gene-splits-dir>/<dataset_id>/gene_splits.json
Output: <output-dir>/fold_<k>_adata.h5ad
        (default <base-data-dir>/spatial_imputation_splits/<dataset_id>/)

Example (from the repo root):
    python downstream/5_spatial_imputation/adata_split.py --dataset_id SEAAD_donor_H21_032 \
        --base-data-dir data
"""

import argparse
import json
from pathlib import Path

import scanpy as sc

HERE = Path(__file__).resolve().parent


def load_gene_splits(gene_splits_path: Path) -> dict:
    with gene_splits_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def split_adata(adata_path: Path, gene_splits_path: Path, output_dir: Path) -> None:
    gene_splits = load_gene_splits(gene_splits_path)
    adata = sc.read_h5ad(adata_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    gene_set: list[str] = []

    for fold_name, split in gene_splits.items():
        train_genes = split["train_genes"]
        test_genes = split["test_genes"]
        gene_set.extend(test_genes)

        adata_split = adata[:, train_genes].copy()
        adata_split.write_h5ad(output_dir / f"{fold_name}_adata.h5ad")
        print(f"Saved {fold_name} with {adata_split.n_vars} genes.")

    print("Total unique genes in all folds:", len(set(gene_set)))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Split adata by precomputed gene folds.")
    parser.add_argument(
        "--dataset_id",
        required=True,
        help="Dataset identifier used for locating input and output files.",
    )
    parser.add_argument(
        "--base-data-dir",
        default="data",
        help="Root directory containing <dataset_id>.h5ad (default: ./data).",
    )
    parser.add_argument(
        "--gene-splits-dir",
        default=str(HERE / "cache" / "imputation_arrow"),
        help="Directory containing gene_splits.json under <dataset-id>/.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional output directory. Defaults to <base-data-dir>/spatial_imputation_splits/<dataset-id>/.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    dataset_id = args.dataset_id
    base_data_dir = Path(args.base_data_dir)
    gene_splits_path = Path(args.gene_splits_dir) / dataset_id / "gene_splits.json"
    adata_path = base_data_dir / f"{dataset_id}.h5ad"
    output_dir = Path(args.output_dir) if args.output_dir else base_data_dir / "spatial_imputation_splits" / dataset_id

    split_adata(adata_path=adata_path, gene_splits_path=gene_splits_path, output_dir=output_dir)

if __name__ == "__main__":
    main()
