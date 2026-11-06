#!/usr/bin/env python
"""Ontology-guided triplet mining for scOntoFM pretraining.

For every cell (anchor) one triplet ``(anchor, positive, negative)`` is drawn:
  * positive: a cell of the same cell type, preferentially from a *different*
    source dataset (same dataset only if the type occurs in a single dataset);
  * negative: a cell from one of the top-``k`` (default 50) hardest negative cell
    types, i.e. the ontologically closest types by LCA distance, excluding the
    anchor type itself and its direct ancestors/descendants (LCA distance <= 1).
    Ties at the k-th distance are randomly down-sampled (tie-aware selection).

Cell types are processed in order of ontology depth; per-depth triplets are saved
to ``<out_dir>/depth_<d>_all_triplets.npy`` and all triplets to ``--out`` as a
list of ``(a, p, n)`` tuples (``triplet_ind_hard_top50.pkl``, 27,525,118 triplets).

Example
-------
python pretraining/triplets/generate_triplets.py \
    --cl_id_map assets/CL_id_map.pkl \
    --lca_path assets/celltype_similarity_LCA_deepest_allnodes.pkl \
    --out assets/triplet_ind_hard_top50.pkl
"""

import argparse
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import networkx as nx
import numpy as np
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from scOntoFM.ontology import import_cell_ontology  # noqa: E402

ROOT_TERM = "CL:0000000"


def build_celltype_hierarchy_depth(G: nx.DiGraph, CL_list) -> Tuple[Dict[int, List[str]], Dict[str, int]]:
    """Shortest-path depth of each term from the root; returns (depth -> terms, term -> depth)."""
    depth2celltype, celltype2depth = {}, {}
    for node in CL_list:
        level = nx.shortest_path_length(G, ROOT_TERM, node)
        depth2celltype.setdefault(level, []).append(node)
        celltype2depth[node] = level
    return depth2celltype, celltype2depth


def build_hard_negative_lookup_tieaware(
    celltype_list: list,
    dist_matrix: np.ndarray,
    depth_dict: dict,
    top_k: int = 3,
    random_state: int = 42,
) -> dict:
    """{cell type: [hard negative cell types]} using the ``top_k`` closest valid types."""
    rng = np.random.default_rng(random_state)
    hard_negatives = {}

    for i, ct in enumerate(tqdm(celltype_list, desc="Computing hard negatives (tie-aware)")):
        dists = dist_matrix[i].copy()
        dists[dist_matrix[i] <= 1] = np.inf          # self + ancestors/descendants

        valid_mask = np.isfinite(dists)              # also drops NaN (no common ancestor)
        valid_dists = dists[valid_mask]
        valid_indices = np.where(valid_mask)[0]
        if len(valid_indices) == 0:
            hard_negatives[ct] = []
            continue

        order = np.argsort(valid_dists)
        sorted_dists = valid_dists[order]
        sorted_indices = valid_indices[order]

        if len(sorted_dists) <= top_k:
            selected_indices = sorted_indices
        else:
            kth_value = sorted_dists[top_k - 1]
            selected_indices = sorted_indices[sorted_dists <= kth_value]
            if len(selected_indices) > top_k:
                selected_indices = rng.choice(selected_indices, size=top_k, replace=False)

        hard_negatives[ct] = [celltype_list[j] for j in selected_indices]
    return hard_negatives


def generate_triplets_celltype_fast_ds(nested_map, celltypes, hierarchy_level, triplets_per_anchor=1,
                                       hard_negative_lookup=None, rng_seed=42, verbose=True):
    """Return an (N, 3) array of (anchor, positive, negative) row indices for ``celltypes``."""
    rng = np.random.default_rng(rng_seed)
    all_triplets = []
    hard_negative_lookup = hard_negative_lookup or {ct: [c for c in celltypes if c != ct] for ct in celltypes}

    for anchor_ct in celltypes:
        ds_dict = nested_map[anchor_ct]
        anchor_pool = [idx for indices in ds_dict.values() for idx in indices]
        if len(anchor_pool) == 0:
            continue

        negative_pool = []
        for neg_ct in hard_negative_lookup.get(anchor_ct, []):
            if neg_ct in nested_map:
                for _, neg_indices in nested_map[neg_ct].items():
                    negative_pool.extend(neg_indices)
        if len(negative_pool) == 0:
            continue
        negative_pool = np.array(negative_pool, dtype=np.int64)

        dataset_ids = list(ds_dict.keys())
        num_datasets = len(dataset_ids)

        for dsid, anchor_indices in ds_dict.items():
            anchor_indices = np.array(anchor_indices, dtype=np.int64)
            if len(anchor_indices) == 0:
                continue
            all_anchors = np.repeat(anchor_indices, triplets_per_anchor)

            if num_datasets == 1:
                # single dataset: positives from the same dataset, no self-pairs
                positive_pool = np.array(anchor_indices, dtype=np.int64)
                all_positives = rng.choice(positive_pool, size=len(all_anchors), replace=True)
                self_mask = all_anchors == all_positives
                while np.any(self_mask):
                    all_positives[self_mask] = rng.choice(positive_pool, size=np.sum(self_mask), replace=True)
                    self_mask = all_anchors == all_positives
            else:
                # positives from the other datasets of the same cell type
                positive_pool = np.concatenate(
                    [np.array(ds_dict[d], dtype=np.int64) for d in dataset_ids if d != dsid]
                )
                all_positives = rng.choice(positive_pool, size=len(all_anchors), replace=True)

            all_negatives = rng.choice(negative_pool, size=len(all_anchors), replace=True)
            all_triplets.append(np.stack([all_anchors, all_positives, all_negatives], axis=1))
            if verbose:
                print(f"Process triplets for celltype '{anchor_ct}', dataset '{dsid}' at level {hierarchy_level}")

    if all_triplets:
        return np.concatenate(all_triplets, axis=0)
    return np.zeros((0, 3), dtype=np.int64)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cl_id_map", default=str(REPO_ROOT / "assets" / "CL_id_map.pkl"))
    p.add_argument("--lca_path", default=str(REPO_ROOT / "assets" / "celltype_similarity_LCA_deepest_allnodes.pkl"))
    p.add_argument("--obo", default=str(REPO_ROOT / "resources" / "ontology" / "cl-basic.obo"))
    p.add_argument("--top_k", type=int, default=50, help="Number of hard-negative cell types per anchor type.")
    p.add_argument("--triplets_per_anchor", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_dir", default="./triplet_index_ontology_curriculum_top50",
                   help="Directory for per-depth .npy files.")
    p.add_argument("--out", default="triplet_ind_hard_top50.pkl")
    p.add_argument("--save_lookup", default=None, help="Optionally pickle the hard-negative lookup here.")
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.cl_id_map, "rb") as f:
        nested_map = pickle.load(f)
    with open(args.lca_path, "rb") as f:
        celltype_distance = pickle.load(f)
    onto_graph = import_cell_ontology(args.obo)

    CL_list = list(nested_map.keys())
    depth_celltype, celltype_depth = build_celltype_hierarchy_depth(onto_graph, CL_list)

    # LCA distances restricted to cell types present in the corpus
    idx_map = {n: i for i, n in enumerate(celltype_distance["nodes"])}
    share_CL = [n for n in CL_list if n in idx_map]
    subset_indices = [idx_map[n] for n in share_CL]
    subset_distance_matrix = celltype_distance["Distance"][np.ix_(subset_indices, subset_indices)]

    hard_negatives = build_hard_negative_lookup_tieaware(share_CL, subset_distance_matrix, celltype_depth,
                                                         top_k=args.top_k)
    if args.save_lookup:
        with open(args.save_lookup, "wb") as f:
            pickle.dump(hard_negatives, f)

    print("\nGenerating triplets...")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_triplets = []
    for depth in sorted(depth_celltype.keys()):
        triplets_ind = generate_triplets_celltype_fast_ds(
            nested_map, depth_celltype[depth], depth, triplets_per_anchor=args.triplets_per_anchor,
            hard_negative_lookup=hard_negatives, rng_seed=args.seed,
        )
        all_triplets.append(triplets_ind)
        np.save(out_dir / f"depth_{depth}_all_triplets", triplets_ind)

    triplets_list = list(map(tuple, np.concatenate(all_triplets, axis=0).tolist()))
    with open(args.out, "wb") as f:
        pickle.dump(triplets_list, f)
    print(f"Saved {len(triplets_list)} triplets to {args.out}")


if __name__ == "__main__":
    main()
