"""Triplet dataset and MLM-masking collator for pretraining."""

from typing import Dict, Sequence, Tuple

import torch
from torch.nn.utils.rnn import pad_sequence

MLM_PROBABILITY = 0.15


class TripletDataset(torch.utils.data.Dataset):
    """Maps a list of ``(anchor, positive, negative)`` row indices to tokenized cells."""

    def __init__(self, hf_dataset, triplet_indices: Sequence[Tuple[int, int, int]]):
        self.dataset = hf_dataset
        self.triplet_indices = triplet_indices

    def __len__(self):
        return len(self.triplet_indices)

    def __getitem__(self, idx):
        a_idx, p_idx, n_idx = self.triplet_indices[idx]
        return self.dataset[a_idx], self.dataset[p_idx], self.dataset[n_idx]


class TripletCollator:
    """Pad anchor / positive / negative cells and apply BERT-style MLM masking.

    15% of non-special tokens are selected; of those 80% -> ``<mask>``,
    10% -> random token, 10% unchanged. ``<pad>``, ``<cls>`` and ``<eos>`` are never
    masked or predicted. Cell types are mapped to rows of the ontology matrices.
    """

    def __init__(self, token_dict: Dict[str, int], cl2idx: Dict[str, int]):
        self.token_dict = token_dict
        self.cl2idx = cl2idx

    def encode(self, batch):
        token_dict = self.token_dict
        seqs = [torch.tensor(b["input_ids"], dtype=torch.long) for b in batch]
        input_ids = pad_sequence(seqs, batch_first=True, padding_value=token_dict["<pad>"])
        attention_mask = (input_ids != token_dict["<pad>"]).long()

        labels = input_ids.clone()
        probability_matrix = torch.full(labels.shape, MLM_PROBABILITY)
        special_tokens_mask = (
            (input_ids == token_dict["<pad>"])
            | (input_ids == token_dict.get("<cls>", token_dict["<pad>"]))
            | (input_ids == token_dict.get("<eos>", token_dict["<pad>"]))
        )
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()

        labels[~masked_indices] = -100
        labels[input_ids == token_dict["<cls>"]] = -100
        labels[input_ids == token_dict["<eos>"]] = -100

        rand = torch.rand(labels.shape)
        input_ids[masked_indices & (rand < 0.8)] = token_dict["<mask>"]
        random_tokens = torch.randint(low=0, high=len(token_dict), size=input_ids.shape)
        replace_random = masked_indices & (rand >= 0.8) & (rand < 0.9)
        input_ids[replace_random] = random_tokens[replace_random]
        return input_ids, attention_mask, labels

    def _cell_type_ids(self, batch):
        return torch.tensor([self.cl2idx[b["cell_type_ontology_term_id"]] for b in batch], dtype=torch.long)

    def __call__(self, batches):
        anchors, positives, negatives = zip(*batches)
        ct_a = self._cell_type_ids(anchors)
        ct_p = self._cell_type_ids(positives)
        ct_n = self._cell_type_ids(negatives)

        out = {}
        for name, part, ct in (("anchor", anchors, ct_a), ("positive", positives, ct_p), ("negative", negatives, ct_n)):
            input_ids, attention_mask, labels = self.encode(part)
            out[name] = {"input_ids": input_ids, "attention_mask": attention_mask,
                         "labels": labels, "cell_type_ids": ct}
        return out
