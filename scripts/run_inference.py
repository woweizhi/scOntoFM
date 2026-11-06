#!/usr/bin/env python
"""End-to-end scOntoFM inference: raw ``.h5ad`` -> preprocess -> tokenize -> embeddings.

Outputs (``<output_dir>/<run_id>/<dataset_id>/``)
  <dataset_id>_embeddings.npy        cell embeddings (n_cells x 256), rows aligned with
                                     the preprocessed AnnData
  <dataset_id>_hf_dataset/           tokenized cells (input_ids)
  <dataset_id>_gene_embeddings.npy   contextual gene embeddings  (--save_gene_embeddings)
  <dataset_id>.h5ad + UMAP PDFs      AnnData with obsm[<obsm_name>]  (--save_data)

Example
-------
python scripts/run_inference.py --input_h5ad examples/data/demo_pancreas.h5ad \
    --run_id demo --color_obs cell_type --save_data
"""

import argparse
from pathlib import Path

import numpy as np
import scanpy as sc

import _bootstrap  # noqa: F401
from scOntoFM import plotting
from scOntoFM.data import build_hf_dataset
from scOntoFM.model import POOLING_CHOICES, embed_cells, load_model
from scOntoFM.paths import GENE_MEDIAN_PATH, MODEL_DIR, TOKEN_DICT_PATH
from scOntoFM.preprocessing import preprocess_single_h5ad
from scOntoFM.tokenization import TOP_K_TOKENS


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_argument_group("inputs")
    g.add_argument("--input_h5ad", required=True, help="Raw-count AnnData (Ensembl IDs as var_names).")
    g.add_argument("--dataset_id", default=None, help="Defaults to the input filename stem.")
    g.add_argument("--run_id", default="inference", help="Sub-directory name for this run.")

    g = p.add_argument_group("preprocessing / tokenization")
    g.add_argument("--preprocess_cache", default="./cache/preprocessed_h5ad")
    g.add_argument("--min_genes_per_cell", type=int, default=0)
    g.add_argument("--min_cells_per_gene", type=int, default=0)
    g.add_argument("--target_sum", type=float, default=1e4)
    g.add_argument("--keep_non_primary", action="store_true",
                   help="Keep cells with is_primary_data == False (default: drop them).")
    g.add_argument("--top_k_tokens", type=int, default=TOP_K_TOKENS)
    g.add_argument("--token_dict_path", default=str(TOKEN_DICT_PATH))
    g.add_argument("--gene_median_path", default=str(GENE_MEDIAN_PATH))

    g = p.add_argument_group("model / outputs")
    g.add_argument("--model_path", default=str(MODEL_DIR))
    g.add_argument("--pooling", choices=POOLING_CHOICES, default="cls",
                   help="Cell embedding: <cls> token (default) or masked mean over tokens.")
    g.add_argument("--save_gene_embeddings", action="store_true",
                   help="Also save contextual gene embeddings (for contextual GO analysis).")
    g.add_argument("--output_dir", default="./output")
    g.add_argument("--obsm_name", default="scOntoFM")
    g.add_argument("--color_obs", nargs="+", default=["cell_type"])
    g.add_argument("--save_data", action="store_true", help="Write AnnData with embeddings + UMAP plots.")

    g = p.add_argument_group("performance")
    g.add_argument("--batch_size", type=int, default=32)
    g.add_argument("--num_workers", type=int, default=8)
    return p.parse_args()


def save_annotated_adata(adata, embeddings, args, out_dir: Path, dataset_id: str):
    plotting.config_rc(dpi=300, font_size=8, lw=0.5)
    adata.obsm.pop("X_umap", None)
    adata.obsm.pop("umap", None)
    adata.obsm[args.obsm_name] = embeddings
    sc.pp.neighbors(adata, use_rep=args.obsm_name, n_neighbors=10)
    sc.tl.umap(adata)

    sc.settings.figdir = out_dir
    sc.set_figure_params(facecolor="white", figsize=(plotting.centimeter * 6, plotting.centimeter * 6),
                         dpi_save=300, fontsize=8, vector_friendly=True)
    color_obs = [c for c in args.color_obs if c in adata.obs.columns]
    for obs in color_obs:
        plotting.plot_obs(adata, obs, args.obsm_name, out_dir, dataset_id)
    if color_obs:
        sc.pl.umap(adata, color=color_obs, edges_color="none", title=f"Cell Embeddings colored by {color_obs}",
                   frameon=False, save=f"{dataset_id}.pdf", show=False)

    path = out_dir / f"{dataset_id}.h5ad"
    adata.write_h5ad(path)
    print(f"Saved AnnData with obsm['{args.obsm_name}'] to {path}")


def main():
    args = parse_args()
    dataset_id = args.dataset_id or Path(args.input_h5ad).stem

    preprocessed_path = Path(preprocess_single_h5ad(
        args.input_h5ad, args.min_genes_per_cell, args.min_cells_per_gene, args.target_sum,
        args.preprocess_cache, primary_data_only=not args.keep_non_primary,
    ))
    print(f"Preprocessed data cached at {preprocessed_path}")

    print("Tokenizing preprocessed data ...")
    dataset, token_dict = build_hf_dataset(preprocessed_path, args.token_dict_path,
                                           args.gene_median_path, args.top_k_tokens)

    print(f"Loading model from {args.model_path} ...")
    model, device = load_model(args.model_path)
    cell_emb, gene_emb = embed_cells(model, dataset, token_dict, device, pooling=args.pooling,
                                     batch_size=args.batch_size, num_workers=args.num_workers,
                                     return_gene_embeddings=args.save_gene_embeddings)
    print(f"Cell embeddings shape: {tuple(cell_emb.shape)}")

    out_dir = Path(args.output_dir) / args.run_id / dataset_id
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(out_dir / f"{dataset_id}_hf_dataset"))
    np.save(out_dir / f"{dataset_id}_embeddings.npy", cell_emb.numpy())
    if gene_emb is not None:
        print(f"Gene embeddings shape: {tuple(gene_emb.shape)}")
        np.save(out_dir / f"{dataset_id}_gene_embeddings.npy", gene_emb.numpy())
    print(f"Saved embeddings to {out_dir}")

    if args.save_data:
        save_annotated_adata(sc.read_h5ad(preprocessed_path), cell_emb.numpy(), args, out_dir, dataset_id)


if __name__ == "__main__":
    main()
