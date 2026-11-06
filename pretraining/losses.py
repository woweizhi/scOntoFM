"""Ontology-aware objectives of scOntoFM.

Total loss = MLM + triplet_alpha * dynamic-margin triplet + ontoAlign_beta * ontology alignment.
"""

import torch
import torch.nn.functional as F
from torchsort import soft_rank


def get_cls_embedding(hidden_states) -> torch.Tensor:
    """Cell embedding = last-layer ``<cls>`` state, (B, D)."""
    return hidden_states[-1][:, 0, :]


def compute_pairwise_cosine_sim(embeddings: torch.Tensor) -> torch.Tensor:
    normalized = F.normalize(embeddings, p=2, dim=1)
    return torch.mm(normalized, normalized.t())


def compute_ontology_alignment_loss(embeddings, cell_type_ids, LCA_dist_mat):
    """Differentiable row-wise Spearman alignment between embedding and ontology distances.

    For each cell, the ranking of its cosine distances to the other B-1 cells in the
    batch is matched to the ranking of the corresponding LCA ontology distances
    (soft ranks via ``torchsort``, L2 regularization 0.1). Loss = 1 - mean Spearman.
    """
    embeddings = embeddings.float()
    LCA_dist_mat = LCA_dist_mat.float()
    B = embeddings.size(0)
    device = embeddings.device

    d_emb = 1 - compute_pairwise_cosine_sim(embeddings)                     # [0, 2]
    d_onto = LCA_dist_mat.index_select(0, cell_type_ids).index_select(1, cell_type_ids)

    mask = ~torch.eye(B, dtype=torch.bool, device=device)                   # drop diagonal
    d_emb_rows = d_emb.masked_select(mask).view(B, B - 1)
    d_onto_rows = d_onto.masked_select(mask).view(B, B - 1)

    pred_ranks = soft_rank(d_emb_rows, regularization_strength=0.1, regularization="l2")
    with torch.no_grad():
        target_ranks = soft_rank(d_onto_rows, regularization_strength=0.1, regularization="l2")

    vx = pred_ranks - pred_ranks.mean(dim=1, keepdim=True)
    vy = target_ranks - target_ranks.mean(dim=1, keepdim=True)
    numerator = (vx * vy).sum(dim=1)
    denominator = torch.sqrt((vx ** 2).sum(dim=1)) * torch.sqrt((vy ** 2).sum(dim=1))
    correlations = numerator / (denominator + 1e-6)
    return 1 - correlations.mean()


def dynamic_triplet_loss(anchor_emb, positive_emb, negative_emb,
                         cell_type_ids_a, cell_type_ids_n,
                         onto_distances, cell_type_depths,
                         margin_alpha=1.0):
    """Euclidean triplet loss with an ontology-dependent margin.

    margin = margin_alpha * ((1 - d_onto(a, n)) + depth(a) + 0.1)

    Ontologically closer (harder) negatives and deeper (more specific) anchor
    cell types receive a larger margin. ``onto_distances`` and ``cell_type_depths``
    are normalized to [0, 1].
    """
    d_ap = F.pairwise_distance(anchor_emb, positive_emb, p=2, eps=1e-6)
    d_an = F.pairwise_distance(anchor_emb, negative_emb, p=2, eps=1e-6)

    d_onto = onto_distances[cell_type_ids_a, cell_type_ids_n]
    depth_a = cell_type_depths[cell_type_ids_a]
    dynamic_margin = margin_alpha * ((1.0 - d_onto) + depth_a + 0.1)

    return F.relu(d_ap - d_an + dynamic_margin).mean()
