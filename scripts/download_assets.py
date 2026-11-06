#!/usr/bin/env python
"""Download the large scOntoFM files hosted on the HuggingFace Hub.

Model repo  https://huggingface.co/xiangyguo/scOntoFM  (assets/ below; weights also ship in git)

    assets/celltype_similarity_LCA_deepest_allnodes.pkl   78 MB  LCA distances between all CL terms
    assets/CL_id_map.pkl                                  138 MB  {CL id: {dataset id: [row indices]}}
    assets/triplet_ind_hard_top50.pkl                     468 MB  27.5M (anchor, positive, negative) triplets

Dataset repo  https://huggingface.co/datasets/xiangyguo/scOntoFM-pretrain-corpus  (~167 GB)

    The tokenized pretraining corpus (HuggingFace ``save_to_disk`` folder, 29,160,112 cells, of
    which the 27,525,118 cells with a valid Cell Ontology label are used in training; see the
    dataset card). Only needed to re-run pretraining or rebuild the triplet index.

The assets are only needed for pretraining / triplet mining and some downstream analyses;
inference only needs the files tracked in ``resources/``.

Usage
-----
    python scripts/download_assets.py                                # assets -> assets/
    python scripts/download_assets.py --files CL_id_map.pkl
    python scripts/download_assets.py --corpus --corpus_dir data/scLLM_onto_dataset
"""

import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ASSETS_DIR = REPO_ROOT / "assets"

HF_REPO_ID = "xiangyguo/scOntoFM"
HF_SUBFOLDER = "assets"
CORPUS_REPO_ID = "xiangyguo/scOntoFM-pretrain-corpus"

ASSET_FILES = [
    "celltype_similarity_LCA_deepest_allnodes.pkl",
    "CL_id_map.pkl",
    "triplet_ind_hard_top50.pkl",
]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo_id", default=HF_REPO_ID)
    p.add_argument("--files", nargs="+", default=ASSET_FILES)
    p.add_argument("--local_dir", default=str(ASSETS_DIR))
    p.add_argument("--corpus", action="store_true",
                   help="Download the pretraining corpus (~167 GB) instead of the assets.")
    p.add_argument("--corpus_repo_id", default=CORPUS_REPO_ID)
    p.add_argument("--corpus_dir", default=str(REPO_ROOT / "data" / "scLLM_onto_dataset"),
                   help="Target folder; load it with datasets.Dataset.load_from_disk(corpus_dir).")
    p.add_argument("--max_workers", type=int, default=8)
    return p.parse_args()


def main():
    args = parse_args()
    from huggingface_hub import hf_hub_download, snapshot_download

    if args.corpus:
        path = snapshot_download(repo_id=args.corpus_repo_id, repo_type="dataset",
                                 local_dir=args.corpus_dir, allow_patterns=["*.arrow", "*.json"],
                                 max_workers=args.max_workers)
        print(f"corpus downloaded to {path}")
        return

    local_dir = Path(args.local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)
    for name in args.files:
        path = hf_hub_download(repo_id=args.repo_id, repo_type="model", filename=f"{HF_SUBFOLDER}/{name}",
                               local_dir=str(local_dir.parent))
        print(f"downloaded {path}")


if __name__ == "__main__":
    main()
