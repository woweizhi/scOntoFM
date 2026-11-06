#!/usr/bin/env python
"""Tokenize preprocessed ``.h5ad`` files.

Two output formats:
  --format hf     HuggingFace ``datasets`` folder with ``input_ids`` (inference input)
  --format arrow  Arrow IPC file with ``input_ids`` + obs metadata (pretraining corpus)

Example
-------
python scripts/tokenize_h5ad.py --input ./cache/preprocessed_h5ad/demo_pancreas.h5ad \
    --output_dir ./cache/tokenized --format hf
"""

import argparse
from pathlib import Path

from tqdm import tqdm

import _bootstrap  # noqa: F401
from scOntoFM.data import build_hf_dataset
from scOntoFM.paths import GENE_MEDIAN_PATH, TOKEN_DICT_PATH
from scOntoFM.tokenization import TOP_K_TOKENS, make_arrow_for_dataset


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True, help="A preprocessed .h5ad file or a directory of them.")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--format", choices=["hf", "arrow"], default="hf")
    p.add_argument("--token_dict_path", default=str(TOKEN_DICT_PATH))
    p.add_argument("--gene_median_path", default=str(GENE_MEDIAN_PATH))
    p.add_argument("--top_k_tokens", type=int, default=TOP_K_TOKENS)
    p.add_argument("--chunk_cells", type=int, default=4096, help="Cells per chunk (arrow format).")
    return p.parse_args()


def main():
    args = parse_args()
    inp = Path(args.input)
    files = sorted(inp.glob("*.h5ad")) if inp.is_dir() else [inp]
    out_dir = Path(args.output_dir)
    for h5ad_path in tqdm(files, desc="Tokenizing"):
        if args.format == "arrow":
            out = make_arrow_for_dataset(h5ad_path, out_dir, args.token_dict_path, args.gene_median_path,
                                         top_k=args.top_k_tokens, chunk_cells=args.chunk_cells)
        else:
            ds, _ = build_hf_dataset(h5ad_path, args.token_dict_path, args.gene_median_path, args.top_k_tokens)
            out = out_dir / f"{h5ad_path.stem}_hf_dataset"
            ds.save_to_disk(str(out))
        print(f"Tokenized: {out}")


if __name__ == "__main__":
    main()
