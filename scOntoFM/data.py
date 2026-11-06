"""Tokenized HuggingFace datasets and batch collation for inference."""

from pathlib import Path
from typing import Dict, Tuple

import scanpy as sc
import torch
from datasets import Dataset
from torch.nn.utils.rnn import pad_sequence

from .tokenization import (
    TOP_K_TOKENS,
    load_gene_median_dict,
    load_token_dictionary,
    shared_genes,
    tokenize_adata,
)


def build_hf_dataset(
    preprocessed_path: Path,
    token_dict_path: str,
    gene_median_path: str,
    top_k: int = TOP_K_TOKENS,
) -> Tuple[Dataset, Dict[str, int]]:
    """Tokenize a preprocessed ``.h5ad`` into an in-memory ``Dataset({"input_ids"})``.

    Row ``i`` of the dataset corresponds to cell ``i`` of the preprocessed AnnData.
    """
    adata = sc.read_h5ad(preprocessed_path)
    gene_median_dict = load_gene_median_dict(gene_median_path)
    token_dict = load_token_dictionary(token_dict_path)

    genes = shared_genes(adata, gene_median_dict)
    if len(genes) == 0:
        raise ValueError(
            "No overlapping genes between AnnData.var_names and the gene median dictionary. "
            "scOntoFM expects Ensembl gene IDs (e.g. ENSG00000141510) as var_names."
        )

    tokens_list = tokenize_adata(adata, genes, gene_median_dict, token_dict, top_k)
    return Dataset.from_dict({"input_ids": tokens_list}), token_dict


def create_collate_fn(token_dict: Dict[str, int]):
    """Right-pad ``input_ids`` with ``<pad>`` and build the attention mask."""
    pad_token = token_dict["<pad>"]

    def collate_fn(batch):
        seqs = [torch.tensor(b["input_ids"], dtype=torch.long) for b in batch]
        input_ids = pad_sequence(seqs, batch_first=True, padding_value=pad_token)
        attention_mask = (input_ids != pad_token).long()
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    return collate_fn
