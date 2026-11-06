"""Default locations of the files shipped with the repository."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

RESOURCES_DIR = REPO_ROOT / "resources"
VOCAB_DIR = RESOURCES_DIR / "vocab"
TOKEN_DICT_PATH = VOCAB_DIR / "token_dictionary.pkl"
GENE_MEDIAN_PATH = VOCAB_DIR / "gene_medians.pkl"
GENE_NAME_ID_PATH = VOCAB_DIR / "gene_name_id_dict.pkl"  # gene symbol -> Ensembl ID
CL_OBO_PATH = RESOURCES_DIR / "ontology" / "cl-basic.obo"

MODEL_DIR = RESOURCES_DIR / "weights" / "scOntoFM"
MAIN_FIGURE_PATH = RESOURCES_DIR / "main_figure.png"

ASSETS_DIR = REPO_ROOT / "assets"
LCA_DISTANCE_PATH = ASSETS_DIR / "celltype_similarity_LCA_deepest_allnodes.pkl"
CL_ID_MAP_PATH = ASSETS_DIR / "CL_id_map.pkl"
TRIPLET_INDEX_PATH = ASSETS_DIR / "triplet_ind_hard_top50.pkl"
