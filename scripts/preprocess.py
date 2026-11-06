#!/usr/bin/env python
"""Preprocess raw-count ``.h5ad`` files (filter + library-size normalization).

Example
-------
python scripts/preprocess.py --input examples/data/demo_pancreas.h5ad \
    --output_dir ./cache/preprocessed_h5ad
python scripts/preprocess.py --input ./raw_h5ad_dir --output_dir ./processed \
    --min_genes_per_cell 200 --min_cells_per_gene 5          # pretraining-corpus setting
"""

import argparse
import logging
from pathlib import Path

from tqdm import tqdm

import _bootstrap  # noqa: F401
from scOntoFM.preprocessing import preprocess_single_h5ad

logging.basicConfig(level=logging.INFO)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True, help="An .h5ad file or a directory of .h5ad files.")
    p.add_argument("--output_dir", required=True, help="Directory for the preprocessed .h5ad files.")
    p.add_argument("--min_genes_per_cell", type=int, default=0)
    p.add_argument("--min_cells_per_gene", type=int, default=0)
    p.add_argument("--target_sum", type=float, default=1e4)
    p.add_argument("--keep_non_primary", action="store_true",
                   help="Keep cells with is_primary_data == False (default: drop them).")
    p.add_argument("--overwrite", action="store_true", help="Recompute even if the output exists.")
    return p.parse_args()


def main():
    args = parse_args()
    inp = Path(args.input)
    files = sorted(inp.glob("*.h5ad")) if inp.is_dir() else [inp]
    for h5ad_path in tqdm(files, desc="Preprocessing H5AD files"):
        out = preprocess_single_h5ad(
            str(h5ad_path), args.min_genes_per_cell, args.min_cells_per_gene, args.target_sum,
            args.output_dir, primary_data_only=not args.keep_non_primary, overwrite=args.overwrite,
        )
        print(f"Preprocessed: {out}")


if __name__ == "__main__":
    main()
