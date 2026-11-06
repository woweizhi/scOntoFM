#!/usr/bin/env python
"""Embed pre-tokenized gene-split folds for spatial imputation.

Input: ``<dataset_path>/<dataset_id>/fold_*/`` HuggingFace datasets produced by
``downstream/5_spatial_imputation/spatial_imputation_pipeline.py``.
Output: ``<output_dir>/<dataset_id>/<run_id>/<fold>_adata/<fold>_adata_embeddings.npy``.

Example
-------
python downstream/5_spatial_imputation/run_inference_imputation.py --run_id scOntoFM \
    --dataset_id SEAAD_donor_H21_032 --output_dir ./output
"""

import argparse
import re
import sys
from pathlib import Path
from typing import List

import numpy as np
import scanpy as sc
from datasets import Dataset

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scOntoFM import plotting  # noqa: E402
from scOntoFM.model import POOLING_CHOICES, embed_cells, load_model  # noqa: E402
from scOntoFM.paths import MODEL_DIR, TOKEN_DICT_PATH  # noqa: E402
from scOntoFM.tokenization import load_token_dictionary  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run_id", required=True)
    p.add_argument("--dataset_id", required=True)
    p.add_argument("--dataset_path", default=str(HERE / "cache" / "imputation_arrow"))
    p.add_argument("--fold_regex", default=r"fold_\d+.*", help="Regex matching fold sub-directories.")
    p.add_argument("--output_dir", default="./output")
    p.add_argument("--model_path", default=str(MODEL_DIR))
    p.add_argument("--token_dict_path", default=str(TOKEN_DICT_PATH))
    p.add_argument("--pooling", choices=POOLING_CHOICES, default="cls")
    p.add_argument("--obsm_name", default="scOntoFM")
    p.add_argument("--color_obs", nargs="+", default=["cell_type"])
    p.add_argument("--save_data", action="store_true", help="Write --adata_path with embeddings + UMAP.")
    p.add_argument("--adata_path", default=None)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=8)
    return p.parse_args()


def list_fold_dirs(base_path: Path, fold_regex: str) -> List[Path]:
    pattern = re.compile(fold_regex)
    return sorted(p for p in base_path.iterdir() if p.is_dir() and pattern.match(p.name))


def main():
    args = parse_args()
    dataset_root = Path(args.dataset_path) / args.dataset_id
    fold_dirs = list_fold_dirs(dataset_root, args.fold_regex)
    if not fold_dirs and dataset_root.is_dir():
        fold_dirs = [dataset_root]
    if not fold_dirs:
        raise FileNotFoundError(f"No fold directories found in {dataset_root}")

    token_dict = load_token_dictionary(args.token_dict_path)
    print(f"Loading model from {args.model_path} ...")
    model, device = load_model(args.model_path)
    output_root = Path(args.output_dir) / args.dataset_id / args.run_id

    for fold_dir in fold_dirs:
        print(f"Loading dataset from {fold_dir} ...")
        dataset = Dataset.load_from_disk(str(fold_dir))
        cell_emb, _ = embed_cells(model, dataset, token_dict, device, pooling=args.pooling,
                                  batch_size=args.batch_size, num_workers=args.num_workers)
        print(f"[{fold_dir.name}] Cell embeddings shape: {tuple(cell_emb.shape)}")

        fold_out = output_root / f"{fold_dir.name}_adata"
        fold_out.mkdir(parents=True, exist_ok=True)
        np.save(fold_out / f"{fold_dir.name}_adata_embeddings.npy", cell_emb.numpy())

        if args.save_data and args.adata_path:
            plotting.config_rc(dpi=300, font_size=8, lw=0.5)
            adata = sc.read_h5ad(args.adata_path)
            adata.obsm.pop("X_umap", None)
            adata.obsm.pop("umap", None)
            adata.obsm[args.obsm_name] = cell_emb.numpy()
            sc.pp.neighbors(adata, use_rep=args.obsm_name, n_neighbors=10)
            sc.tl.umap(adata)
            for obs in args.color_obs:
                plotting.plot_obs(adata, obs, args.obsm_name, fold_out, args.dataset_id)
            adata.write_h5ad(fold_out / f"{args.dataset_id}.h5ad")


if __name__ == "__main__":
    main()
