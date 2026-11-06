"""Model loading and embedding extraction.

scOntoFM is a 6-layer BERT encoder (hidden 256, 4 heads, 19,941-token vocab,
max length 2,002) stored in HuggingFace ``BertForMaskedLM`` format.

Cell embeddings
  * ``cls``  (default): last-layer hidden state of the ``<cls>`` token; this is the
    representation optimized by the ontology-alignment and triplet objectives.
  * ``mean``: attention-masked mean of all last-layer token states.

Contextual gene embeddings
  ``hidden_states[-1][:, 1:-1, :].mean(dim=2)``: one value per gene position
  (special tokens at both ends removed), consumed by
  ``downstream/3_gene_function/GO_embedding_contextualize.py`` together with the
  tokenized dataset (which gives the gene token at each position).
"""

from typing import Dict, Optional, Tuple

import torch
from datasets import Dataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertForMaskedLM

from .data import create_collate_fn

POOLING_CHOICES = ("cls", "mean")


def load_model(model_path: str, device: Optional[torch.device] = None) -> Tuple[BertForMaskedLM, torch.device]:
    """Load scOntoFM weights in eval mode on ``device`` (default: first GPU, else CPU)."""
    model = BertForMaskedLM.from_pretrained(model_path)
    model.config.output_hidden_states = True
    model.eval()
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model, device


def get_cls_embedding(hidden_states) -> torch.Tensor:
    """Last-layer ``<cls>`` state, (B, D)."""
    return hidden_states[-1][:, 0, :].clone()


def average_hidden_states(hidden_states, attention_mask: torch.Tensor) -> torch.Tensor:
    """Attention-masked mean of the last-layer token states, (B, D)."""
    last_hidden = hidden_states[-1]
    mask = attention_mask.unsqueeze(-1).float()
    sum_hidden = (last_hidden * mask).sum(dim=1)
    token_count = mask.sum(dim=1).clamp(min=1e-5)
    return sum_hidden / token_count


def contextual_gene_embedding(hidden_states) -> torch.Tensor:
    """Per-position contextual gene values with ``<cls>``/last token removed, (B, L-2)."""
    return hidden_states[-1][:, 1:-1, :].mean(dim=2)


@torch.no_grad()
def embed_cells(
    model: BertForMaskedLM,
    dataset: Dataset,
    token_dict: Dict[str, int],
    device: torch.device,
    pooling: str = "cls",
    batch_size: int = 32,
    num_workers: int = 8,
    return_gene_embeddings: bool = False,
):
    """Embed every cell of a tokenized dataset.

    Returns ``cell_embeddings`` (N, D) and, if requested, the zero-padded
    contextual gene embeddings (N, L_max-2); otherwise ``None``.
    """
    if pooling not in POOLING_CHOICES:
        raise ValueError(f"pooling must be one of {POOLING_CHOICES}, got {pooling!r}")

    dl = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=create_collate_fn(token_dict),
    )

    cell_embeddings, gene_embeddings = [], []
    for batch in tqdm(dl, desc="Generating cell embeddings", unit="batch"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        hidden_states = model(input_ids=input_ids, attention_mask=attention_mask).hidden_states

        if pooling == "cls":
            cell_emb = get_cls_embedding(hidden_states)
        else:
            cell_emb = average_hidden_states(hidden_states, attention_mask)
        cell_embeddings.append(cell_emb.cpu())

        if return_gene_embeddings:
            gene_embeddings.append(contextual_gene_embedding(hidden_states).cpu())

    cell_embeddings = torch.cat(cell_embeddings, dim=0)
    if not return_gene_embeddings:
        return cell_embeddings, None
    flat = [seq for b in gene_embeddings for seq in b]
    return cell_embeddings, pad_sequence(flat, batch_first=True, padding_value=0)
