#!/usr/bin/env python
"""Pairwise LCA (lowest common ancestor) distances between all Cell Ontology terms.

For two terms u, v the common ancestors are found by walking up the is_a /
relationship graph; the "deepest" common ancestor (max depth from the root, ties
broken by shorter total path) is selected, and the distance is
``d(u, LCA) + d(v, LCA)``. Pairs without a common ancestor are NaN.

Output: a pickle ``{"Distance": (N, N) array, "nodes": [CL ids]}`` covering every
term of ``cl-basic.obo`` (used by triplet mining and the ontology-alignment loss).

Example
-------
python pretraining/ontology/build_lca_distance.py \
    --obo resources/ontology/cl-basic.obo \
    --out assets/celltype_similarity_LCA_deepest_allnodes.pkl
"""

import argparse
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import networkx as nx
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from scOntoFM.ontology import get_id_mapper, import_cell_ontology  # noqa: E402


def precompute_depths(g: nx.DiGraph, direction: str = "child_to_parent") -> None:
    """Store the longest-path depth from the root(s) as node attribute ``depth``."""
    gr = g.reverse(copy=False) if direction == "child_to_parent" else g
    if not nx.is_directed_acyclic_graph(gr):
        raise ValueError("Graph must be a DAG.")
    depths = {}
    for node in nx.topological_sort(gr):
        preds = list(gr.predecessors(node))
        depths[node] = 0 if not preds else max(depths[p] + 1 for p in preds)
    nx.set_node_attributes(g, depths, "depth")


def lca_distance(
    g: nx.DiGraph,
    u: Any,
    v: Any,
    direction: str = "child_to_parent",
    method: str = "deepest",
) -> Tuple[Optional[Any], Optional[int], Dict[str, Any]]:
    """Return ``(lca, distance, info)`` for terms ``u`` and ``v``."""
    g_up = g if direction == "child_to_parent" else g.reverse(copy=False)

    try:
        dists_u = nx.shortest_path_length(g_up, source=u)
    except nx.NetworkXNoPath:
        dists_u = {u: 0}
    try:
        dists_v = nx.shortest_path_length(g_up, source=v)
    except nx.NetworkXNoPath:
        dists_v = {v: 0}

    commons = set(dists_u.keys()) & set(dists_v.keys())
    if not commons:
        return None, None, {"reason": "no common ancestor"}

    if "depth" in g.nodes[u]:
        get_depth = lambda n: g.nodes[n].get("depth", 0)  # noqa: E731
    else:  # slow fallback; call precompute_depths() first
        gr = g.reverse(copy=True) if direction == "child_to_parent" else g.copy()
        local_depths = {}
        for node in nx.topological_sort(gr):
            preds = list(gr.predecessors(node))
            local_depths[node] = 0 if not preds else max(local_depths[p] + 1 for p in preds)
        get_depth = lambda n: local_depths.get(n, 0)  # noqa: E731

    best_key, best_res = None, None
    for a in commons:
        du, dv = dists_u[a], dists_v[a]
        total = du + dv
        dpth = get_depth(a)
        if method == "deepest":
            key = (dpth, -total)
        elif method == "shortest":
            key = (-total, dpth)
        else:
            raise ValueError("method must be 'deepest' or 'shortest'")
        if (best_key is None) or (key > best_key):
            best_key = key
            best_res = (a, total, du, dv, dpth)

    if best_res is None:
        return None, None, {"reason": "no valid ancestor"}
    best_a, best_total, best_du, best_dv, best_depth = best_res
    return best_a, int(best_total), {
        "du": int(best_du), "dv": int(best_dv), "depth[lca]": int(best_depth),
        "method": method, "direction": direction,
    }


def extract_submatrix(similarity, node_list, target_nodes):
    """Sub-matrix of ``similarity`` for ``target_nodes`` (kept in the given order)."""
    idx = {n: i for i, n in enumerate(node_list)}
    filtered_nodes = [n for n in target_nodes if n in idx]
    filtered_indices = [idx[n] for n in filtered_nodes]
    return similarity[np.ix_(filtered_indices, filtered_indices)], filtered_nodes


def onto_distance_matrix(nodes, G, method):
    precompute_depths(G, direction="parent_to_child")
    lca_dist_matrix = np.zeros((len(nodes), len(nodes)))
    for i, ct1 in enumerate(nodes):
        for j, ct2 in enumerate(nodes):
            if i <= j:
                if ct1 is None or ct2 is None:
                    lca_dist = np.nan
                else:
                    _, lca_dist, _ = lca_distance(G, ct1, ct2, direction="parent_to_child", method=method)
                lca_dist_matrix[i, j] = lca_dist
                lca_dist_matrix[j, i] = lca_dist
    return lca_dist_matrix


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--obo", default=str(REPO_ROOT / "resources" / "ontology" / "cl-basic.obo"))
    p.add_argument("--method", choices=["deepest", "shortest"], default="deepest")
    p.add_argument("--out", default=None,
                   help="Output pickle (default: celltype_similarity_LCA_<method>_allnodes.pkl).")
    return p.parse_args()


def main():
    args = parse_args()
    onto_graph = import_cell_ontology(args.obo)
    id2name = get_id_mapper(onto_graph)
    print(f"Cell ontology graph: {onto_graph.number_of_nodes()} nodes, {onto_graph.number_of_edges()} edges")
    first = list(onto_graph.nodes)[0]
    print(f"Example node: {first}, name: {id2name[first]}")

    nodes = list(onto_graph.nodes)
    lca_dist_mat = onto_distance_matrix(nodes, onto_graph, method=args.method)

    out = args.out or f"celltype_similarity_LCA_{args.method}_allnodes.pkl"
    with open(out, "wb") as f:
        pickle.dump({"Distance": lca_dist_mat, "nodes": nodes}, f)
    print(f"Saved LCA distance matrix {lca_dist_mat.shape} to {out}")


if __name__ == "__main__":
    main()
