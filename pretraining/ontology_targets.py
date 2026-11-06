"""Ontology supervision targets used during pretraining.

* ``dis_mat``: pairwise LCA distance between all Cell Ontology terms
  (``celltype_similarity_LCA_deepest_allnodes.pkl``, built by
  ``ontology/build_lca_distance.py``); NaN where two terms share no ancestor.
* ``depths``: shortest-path depth of every term from the root ``CL:0000000``
  (-1 if unreachable).
"""

import logging
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import networkx as nx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scOntoFM.ontology import import_cell_ontology  # noqa: E402

ROOT_TERM = "CL:0000000"  # "cell"


def load_lca_distance(path: str) -> Tuple[np.ndarray, List[str]]:
    with open(path, "rb") as f:
        logging.info("Loading precomputed LCA distance matrix")
        data = pickle.load(f)
    return data["Distance"], data["nodes"]


def compute_depths(obo_path: str, nodes: List[str], root: str = ROOT_TERM) -> List[int]:
    onto_graph = import_cell_ontology(obo_path)
    celltype2depth: Dict[str, int] = {}
    for node in nodes:
        if nx.has_path(onto_graph, root, node):
            celltype2depth[node] = nx.shortest_path_length(onto_graph, root, node)
        else:
            celltype2depth[node] = -1
    return [celltype2depth.get(n, 0) for n in nodes]


def load_ontology_targets(lca_path: str, obo_path: str):
    """Return ``(dis_mat, nodes, CL2idx, depths_list)``."""
    dis_mat, nodes = load_lca_distance(lca_path)
    cl2idx = {cl: i for i, cl in enumerate(nodes)}
    depths_list = compute_depths(obo_path, nodes)
    return dis_mat, nodes, cl2idx, depths_list
