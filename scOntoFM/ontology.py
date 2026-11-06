"""Cell Ontology (CL) utilities."""

from pathlib import Path
from typing import Union

import networkx as nx
import obonet

from .paths import CL_OBO_PATH


def import_cell_ontology(url: Union[str, Path] = CL_OBO_PATH) -> nx.DiGraph:
    """Read a CL ``.obo`` file (local path or URL) as a parent -> child ``DiGraph``.

    obonet returns child -> parent edges; the graph is reversed so that edges
    point from general to specific terms. The release ships ``cl-basic.obo``
    (CL 2025-07-30); the latest version is at
    http://purl.obolibrary.org/obo/cl/cl-basic.obo
    """
    graph = obonet.read_obo(str(url)).reverse()
    return nx.DiGraph(graph)


def get_id_mapper(graph: nx.DiGraph) -> dict:
    """CL term ID -> term name."""
    return {id_: data.get("name") for id_, data in graph.nodes(data=True)}
