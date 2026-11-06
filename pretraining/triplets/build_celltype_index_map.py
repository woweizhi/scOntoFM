#!/usr/bin/env python
"""Index the pretraining corpus by cell type and source dataset.

Output: ``{CL_id: {dataset_id: [global row indices]}}`` (``CL_id_map.pkl``), used
for triplet mining. Malignant cells (CL:0001064) and a few terms absent from /
not usable in the pretraining corpus are removed.

Example
-------
python pretraining/triplets/build_celltype_index_map.py \
    --dataset_path /path/to/scLLM_onto_dataset --out assets/CL_id_map.pkl
"""

import argparse
import pickle
from collections import defaultdict
from typing import Dict, List

from datasets import Dataset
from tqdm import tqdm

# CL:0001064 malignant cell; CL:0000651, CL:0010003, 'unknown' not in the pretraining
# ontology set; CL:0000402 removed for the current pretraining corpus version.
EXCLUDED_CELL_TYPES = ["CL:0001064", "CL:0000651", "CL:0010003", "unknown", "CL:0000402"]


def nested_defaultdict():
    return defaultdict(list)


def build_celltype_dataset_index_map(
    dataset,
    celltype_col: str = "cell_type_ontology_term_id",
    dataset_id_col: str = "dataset_id",
    sample_limit_per_group: int = None,
) -> Dict[str, Dict[str, List[int]]]:
    """Build ``{celltype: {dataset_id: [row indices]}}`` with a pandas groupby."""
    print(f"Building celltype-dataset-index map from {len(dataset)} samples...")
    df = dataset.to_pandas()[[celltype_col, dataset_id_col]]
    df["global_index"] = df.index
    df.dropna(subset=[celltype_col, dataset_id_col], inplace=True)

    nested = defaultdict(nested_defaultdict)
    for (ct, dsid), group in tqdm(df.groupby([celltype_col, dataset_id_col]), desc="Building index map"):
        indices = group["global_index"].tolist()
        if sample_limit_per_group:
            indices = indices[:sample_limit_per_group]
        nested[ct][dsid] = indices
    return {k: dict(v) for k, v in nested.items()}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset_path", required=True, help="Tokenized corpus (HF dataset on disk).")
    p.add_argument("--out", default="CL_id_map.pkl")
    p.add_argument("--exclude", nargs="*", default=EXCLUDED_CELL_TYPES)
    return p.parse_args()


def main():
    args = parse_args()
    dataset = Dataset.load_from_disk(args.dataset_path)
    if "input_ids" in dataset.column_names:  # not needed, saves memory
        dataset = dataset.remove_columns(["input_ids"])

    data_id_map = build_celltype_dataset_index_map(dataset)
    for ct in args.exclude:
        data_id_map.pop(ct, None)

    with open(args.out, "wb") as f:
        pickle.dump(data_id_map, f)
    print(f"Saved index map ({len(data_id_map)} cell types) to {args.out}")


if __name__ == "__main__":
    main()
