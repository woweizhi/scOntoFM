"""Average per-cell embeddings of each seen cell type into one centroid per CL.

Input : ``<embedding_root>/<model>/adata_CL_xxxxxxx/adata_CL_xxxxxxx_embeddings.npy``
        (embeddings of the ``seen_CL`` query cells, i.e. the output layout of
        ``scripts/run_inference.py --output_dir <embedding_root> --run_id <model>``).
Output: ``seen_celltype_centroids.pkl`` = {"CL:xxxxxxx": mean embedding}, written to
        ``<embedding_root>/<model>/`` (the location ``novel_cell_prediction.py`` reads
        by default) unless ``--out`` is given.

Example (from the repo root):
    python downstream/4_novel_celltype/build_seen_centroids.py \\
        --embedding_root downstream/4_novel_celltype/output/seen_CL_emb --model scOntoFM
"""

import argparse
import pickle
from pathlib import Path
from typing import Dict

import numpy as np
from numpy.typing import NDArray

HERE = Path(__file__).resolve().parent


def build_seen_centroids(base_dir: Path) -> Dict[str, NDArray[np.float32]]:
    """Compute {CL_id: mean embedding} over the per-CL folders in ``base_dir``."""

    folder_names = [p.name for p in base_dir.iterdir() if p.is_dir()]

    centroids: Dict[str, NDArray[np.float32]] = {}

    for file in folder_names:
        # load embeddings for CLs
        file_path = base_dir / file / f"{file}_embeddings.npy"
        CL_emb = np.load(file_path)

        # calculate the centroid CL embedding
        CL_emb_centroid = CL_emb.mean(axis=0)
        CL_id = file.split("adata_", 1)[1].replace("_", ":")
        centroids[CL_id] = CL_emb_centroid

    return centroids


def main():
    parser = argparse.ArgumentParser(description="Build seen cell-type centroids from per-CL embeddings")
    parser.add_argument("--embedding_root", default=str(HERE / "output" / "seen_CL_emb"),
                        help="Root holding one sub-folder per model (inference run_id).")
    parser.add_argument("--model", nargs="+", default=["scOntoFM"],
                        help="Model sub-folder(s) under --embedding_root.")
    parser.add_argument("--out", default=None,
                        help="Output pickle (single model only). Default: <embedding_root>/<model>/seen_celltype_centroids.pkl")
    args = parser.parse_args()

    if args.out is not None and len(args.model) > 1:
        parser.error("--out can only be used with a single --model")

    for model in args.model:
        base_dir = Path(args.embedding_root) / model
        centroids = build_seen_centroids(base_dir)

        # save centroids for seen cell types
        final_centroid_emb_path = Path(args.out) if args.out else base_dir / "seen_celltype_centroids.pkl"
        with open(final_centroid_emb_path, "wb") as f:
            pickle.dump(centroids, f)
        print(f"[{model}] saved {len(centroids)} centroids to {final_centroid_emb_path}")


if __name__ == "__main__":
    main()
