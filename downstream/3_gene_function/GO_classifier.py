"""Train an MLP to predict GO term membership from gene token embeddings.

Inputs
- Gene embeddings extracted directly from a pretrained single-cell foundation model
  (input word-embedding table of a HuggingFace ``BertForMaskedLM`` checkpoint, e.g.
  scOntoFM in resources/weights/scOntoFM), mapped to gene symbols via ``<dict_dir>/token_dictionary.pkl``
  (Ensembl ID -> token id) and ``<dict_dir>/gene_name_id_dict.pkl`` (symbol -> Ensembl ID);
  or, with ``--embedding_source scgpt``, a precomputed (n_genes x d) ``.npy`` matrix plus a
  row-aligned gene-name ``.npy`` (produced outside this repository).
- GO memberships read from a GMT file (MSigDB/decoupler format). Download
  ``c5.go.v2025.1.Hs.symbols.gmt`` from https://www.gsea-msigdb.org/gsea/msigdb/human/collections.jsp
  (C5 ontology gene sets, GO, gene symbols) into this folder.

Outputs (<output_dir>/)
- go_terms_<model_id>.txt                    GO terms used (columns of the label matrix)
- go_term_mlp_<model_id>_splitXX.pt          trained MLP per CV split
- term_metrics_<model_id>_splitXX.csv        per-GO-term F1 / AP per split
- cv_metrics_<model_id>.csv, cv_metrics_summary_<model_id>.txt   global micro/macro F1 and AP
- term_metrics_summary_<model_id>.csv        per-GO-term mean/std across splits

Example (from this folder)
  python GO_classifier.py --model_path ../../resources/weights/scOntoFM \
      --gmt_path c5.go.v2025.1.Hs.symbols.gmt --output_dir ./go_classifier_out \
      --batch_size 64 --epochs 15 --hidden_dim 128 --lr 1e-4 --model_id scOntoFM
"""

import argparse
import os
import sys
import pickle
import random
import csv
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import decoupler as dc
import numpy as np
import torch
from sklearn.metrics import average_precision_score, f1_score
from sklearn.model_selection import StratifiedShuffleSplit
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertForMaskedLM

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT))
from scOntoFM.paths import MODEL_DIR, VOCAB_DIR  # noqa: E402


GO_NAMESPACES = {"GOBP": "BP", "GOCC": "CC", "GOMF": "MF"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gene function classifier (GO terms)")
    parser.add_argument(
        "--model_path",
        type=str,
        required=False,
        default=str(MODEL_DIR),
        help="Path to a pretrained BertForMaskedLM checkpoint (default: resources/weights/scOntoFM).",
    )
    parser.add_argument(
        "--dict_dir",
        type=str,
        required=False,
        default=str(VOCAB_DIR),
        help="Directory containing token_dictionary.pkl and gene_name_id_dict.pkl (default: resources/vocab).",
    )
    parser.add_argument(
        "--embedding_source",
        type=str,
        choices=["model", "scgpt"],
        default="model",
        help="Use embeddings from a pretrained model checkpoint (model) or local scGPT npy files (scgpt).",
    )
    parser.add_argument(
        "--scgpt_emb_path",
        type=str,
        default="scGPT_gene_embeddings.npy",
        help="Path to scGPT gene embedding matrix (.npy) when embedding_source=scgpt.",
    )
    parser.add_argument(
        "--scgpt_gene_path",
        type=str,
        default="scGPT_gene_names.npy",
        help="Path to scGPT gene names (.npy) aligned to scgpt_emb_path rows.",
    )
    parser.add_argument(
        "--gmt_path",
        type=str,
        default=str(THIS_DIR / "c5.go.v2025.1.Hs.symbols.gmt"),
        help="GO GMT file readable by decoupler (MSigDB C5 GO set; download separately).",
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="scOntoFM",
        help="Identifier tag to distinguish model runs in saved outputs.",
    )
    parser.add_argument("--output_dir", type=str, default="./go_classifier_out", help="Where to store outputs.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training the MLP.")
    parser.add_argument("--hidden_dim", type=int, default=128, help="Hidden size of the MLP.")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate for the MLP.")
    parser.add_argument("--epochs", type=int, default=15, help="Training epochs.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--n_splits", type=int, default=20, help="Number of cross-validation splits.")
    parser.add_argument("--val_split", type=float, default=0.2, help="Validation split fraction.")
    parser.add_argument(
        "--min_genes_per_term",
        type=int,
        default=10,
        help="Keep GO terms with at least this many genes in GMT.",
    )
    parser.add_argument(
        "--max_terms",
        type=int,
        default=0,
        help="If >0, keep only top-K GO terms by gene count.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on (cuda or cpu).",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _namespace_from_term(term: str) -> str:
    for prefix, label in GO_NAMESPACES.items():
        if term.upper().startswith(prefix):
            return label
    return "UNKNOWN"


def load_go_memberships(
    gmt_path: str,
    min_genes_per_term: int = 10,
    max_terms: Optional[int] = None,
) -> Tuple[List[str], np.ndarray, List[str]]:
    """Return genes and a (n_genes, n_terms) multi-hot label matrix for GO terms."""
    gmt = dc.pp.read_gmt(gmt_path)
    gmt["source"] = gmt["source"].astype(str)
    gmt["target"] = gmt["target"].astype(str)
    gmt["namespace"] = gmt["source"].apply(_namespace_from_term)
    gmt = gmt[gmt["namespace"].isin(["BP", "CC", "MF"])]

    if gmt.empty:
        raise ValueError("No BP/CC/MF GO terms found in the provided GMT.")

    term_counts = gmt.groupby("source")["target"].nunique().sort_values(ascending=False)
    kept_terms = term_counts[term_counts >= min_genes_per_term]

    if max_terms is not None and max_terms > 0:
        kept_terms = kept_terms.head(max_terms)

    if kept_terms.empty:
        raise ValueError(
            "No GO terms satisfy min_genes_per_term. Try lowering --min_genes_per_term or increasing --max_terms."
        )

    kept_term_set = set(kept_terms.index.tolist())
    gmt = gmt[gmt["source"].isin(kept_term_set)]

    go_terms = kept_terms.index.tolist()
    term_to_col = {term: i for i, term in enumerate(go_terms)}

    gene_to_terms: Dict[str, set] = defaultdict(set)
    for _, row in gmt.iterrows():
        gene_to_terms[row["target"]].add(row["source"])

    genes: List[str] = []
    labels: List[np.ndarray] = []
    for gene, terms in gene_to_terms.items():
        label = np.zeros(len(go_terms), dtype=np.float32)
        for term in terms:
            col = term_to_col[term]
            label[col] = 1.0
        genes.append(gene)
        labels.append(label)

    return genes, np.asarray(labels, dtype=np.float32), go_terms


def load_embeddings_and_index(
    args: argparse.Namespace, device: torch.device
) -> Tuple[torch.Tensor, Dict[str, int], Optional[int]]:
    """Return embedding table, gene->row index map, and pad token id (if any)."""

    if args.embedding_source == "scgpt":
        emb_np = np.load(args.scgpt_emb_path)
        gene_names = np.load(args.scgpt_gene_path)
        if emb_np.shape[0] != gene_names.shape[0]:
            raise ValueError("scGPT embeddings and gene names have mismatched lengths.")
        gene_to_idx = {str(gene): int(i) for i, gene in enumerate(gene_names)}
        emb_tensor = torch.from_numpy(emb_np)
        return emb_tensor, gene_to_idx, None

    if not args.model_path or not args.dict_dir:
        raise ValueError("model_path and dict_dir are required when embedding_source='model'.")

    print("load {} model and dict".format(args.model_id))
    model = BertForMaskedLM.from_pretrained(
        args.model_path, output_attentions=False, output_hidden_states=True
    ).to(device)
    token_emb = model.get_input_embeddings().weight.detach().cpu()

    with open(os.path.join(args.dict_dir, "token_dictionary.pkl"), "rb") as f:
        vocab = pickle.load(f)
    with open(os.path.join(args.dict_dir, "gene_name_id_dict.pkl"), "rb") as f:
        gene_name_id = pickle.load(f)

    gene_to_idx = {k: vocab[v] for k, v in gene_name_id.items() if v in vocab}
    pad_token_id = vocab.get("<pad>") if isinstance(vocab, dict) else None
    return token_emb, gene_to_idx, pad_token_id


def map_genes_to_embeddings(
    genes: Iterable[str],
    labels: np.ndarray,
    embedding_table: torch.Tensor,
    gene_to_idx: Dict[str, int],
    pad_token_id: Optional[int] = None,
) -> Tuple[torch.Tensor, np.ndarray, List[str]]:
    """Map gene symbols to embedding rows; skip genes not in dicts."""
    kept_embeddings: List[torch.Tensor] = []
    kept_labels: List[np.ndarray] = []
    kept_genes: List[str] = []

    for gene, label in zip(genes, labels):
        token_id = gene_to_idx.get(gene)
        if token_id is None:
            continue
        if pad_token_id is not None and token_id == pad_token_id:
            continue
        if token_id >= embedding_table.shape[0]:
            continue
        kept_embeddings.append(embedding_table[token_id])
        kept_labels.append(label)
        kept_genes.append(gene)

    if not kept_embeddings:
        raise ValueError("No genes from GMT found in embedding dictionary.")

    return torch.stack(kept_embeddings, dim=0), np.asarray(kept_labels, dtype=np.float32), kept_genes


def labels_to_strata(labels: np.ndarray) -> np.ndarray:
    """Use positive-label count bins as a robust proxy for stratification."""
    positive_counts = labels.sum(axis=1).astype(int)
    # Cap large counts to keep a small number of strata buckets.
    positive_counts = np.minimum(positive_counts, 10)
    return np.asarray([str(x) for x in positive_counts])


def generate_cv_splits(
    num_samples: int,
    strata: np.ndarray,
    n_splits: int,
    val_split: float,
    seed: int,
):
    """Yield train/val indices; stratify when feasible, else random split fallback."""
    unique_bins, counts = np.unique(strata, return_counts=True)
    can_stratify = unique_bins.size > 1 and np.all(counts >= 2)

    if can_stratify:
        splitter = StratifiedShuffleSplit(n_splits=n_splits, test_size=val_split, random_state=seed)
        for train_idx, val_idx in splitter.split(np.arange(num_samples), strata):
            yield train_idx, val_idx
        return

    print("Stratified split not feasible for current labels; using random splits instead.")
    n_val = max(1, int(round(num_samples * val_split)))
    if n_val >= num_samples:
        raise ValueError("Validation split leaves no training samples. Decrease --val_split.")

    for split_id in range(1, n_splits + 1):
        rng = np.random.RandomState(seed + split_id)
        perm = rng.permutation(num_samples)
        val_idx = perm[:n_val]
        train_idx = perm[n_val:]
        yield train_idx, val_idx


class GeneDataset(Dataset):
    def __init__(self, embeddings: torch.Tensor, labels: np.ndarray):
        self.embeddings = embeddings
        self.labels = torch.from_numpy(labels)

    def __len__(self) -> int:
        return self.labels.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.embeddings[idx], self.labels[idx]


class GeneFunctionMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 512, dropout: float = 0.2, num_classes: int = 3):
        super().__init__()
        mid_dim = max(hidden_dim // 2, num_classes)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, mid_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mid_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def compute_multilabel_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[float, float, float, float]:
    y_pred = (y_prob >= 0.5).astype(np.float32)
    f1_micro = f1_score(y_true, y_pred, average="micro", zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            ap_micro = average_precision_score(y_true, y_prob, average="micro")
        except ValueError:
            ap_micro = float("nan")
        try:
            ap_macro = average_precision_score(y_true, y_prob, average="macro")
        except ValueError:
            ap_macro = float("nan")

    return float(f1_micro), float(f1_macro), float(ap_micro), float(ap_macro)


def compute_per_term_metrics(y_true: np.ndarray, y_prob: np.ndarray, go_terms: List[str]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    y_pred = (y_prob >= 0.5).astype(np.float32)

    for term_idx, term in enumerate(go_terms):
        y_t = y_true[:, term_idx]
        y_p = y_prob[:, term_idx]
        y_h = y_pred[:, term_idx]

        positives = int(y_t.sum())
        negatives = int(y_t.shape[0] - positives)
        f1 = float(f1_score(y_t, y_h, zero_division=0))

        if positives == 0 or negatives == 0:
            ap = float("nan")
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ap = float(average_precision_score(y_t, y_p))

        rows.append(
            {
                "go_term": term,
                "val_pos": positives,
                "val_neg": negatives,
                "val_f1": f1,
                "val_ap": ap,
            }
        )

    return rows


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    train: bool = True,
) -> Tuple[float, np.ndarray, np.ndarray]:
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    all_targets: List[np.ndarray] = []
    all_probs: List[np.ndarray] = []

    for batch_emb, batch_lab in loader:
        batch_emb = batch_emb.to(device)
        batch_lab = batch_lab.to(device)

        with torch.set_grad_enabled(train):
            logits = model(batch_emb)
            loss = criterion(logits, batch_lab)
            probs = torch.sigmoid(logits)

        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * batch_emb.size(0)
        all_targets.append(batch_lab.detach().cpu().numpy())
        all_probs.append(probs.detach().cpu().numpy())

    y_true = np.vstack(all_targets)
    y_prob = np.vstack(all_probs)

    avg_loss = total_loss / len(loader.dataset)
    return avg_loss, y_true, y_prob


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(args.device)

    embedding_table, gene_to_idx, pad_token_id = load_embeddings_and_index(args, device)

    max_terms = args.max_terms if args.max_terms > 0 else None
    genes, labels, go_terms = load_go_memberships(
        args.gmt_path,
        min_genes_per_term=args.min_genes_per_term,
        max_terms=max_terms,
    )
    gene_embeddings, labels, kept_genes = map_genes_to_embeddings(
        genes,
        labels,
        embedding_table,
        gene_to_idx,
        pad_token_id,
    )
    strata = labels_to_strata(labels)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    go_term_list_path = output_dir / f"go_terms_{args.model_id}.txt"
    with go_term_list_path.open("w") as f:
        for term in go_terms:
            f.write(f"{term}\n")

    print(f"Loaded {len(go_terms)} GO terms and {labels.shape[0]} genes with embeddings.")
    print(f"Saved GO term list to {go_term_list_path}")

    metric_keys = ["val_f1_micro", "val_f1_macro", "val_ap_micro", "val_ap_macro"]
    records: List[Dict[str, object]] = []
    per_term_records: List[Dict[str, object]] = []

    for split_id, (train_idx, val_idx) in enumerate(
        generate_cv_splits(
            num_samples=labels.shape[0],
            strata=strata,
            n_splits=args.n_splits,
            val_split=args.val_split,
            seed=args.seed,
        ),
        start=1,
    ):
        # Split-specific seed to vary shuffling but keep reproducibility.
        set_seed(args.seed + split_id)

        x_train = gene_embeddings[train_idx]
        x_val = gene_embeddings[val_idx]
        y_train = labels[train_idx]
        y_val = labels[val_idx]

        train_ds = GeneDataset(x_train, y_train)
        val_ds = GeneDataset(x_val, y_val)

        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)

        mlp = GeneFunctionMLP(
            input_dim=gene_embeddings.shape[1],
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
            num_classes=labels.shape[1],
        )
        mlp.to(device)

        optimizer = torch.optim.Adam(mlp.parameters(), lr=args.lr)
        criterion = nn.BCEWithLogitsLoss()

        for epoch in range(1, args.epochs + 1):
            train_loss, train_true, train_prob = run_epoch(
                mlp, train_loader, optimizer, criterion, device, train=True
            )
            val_loss, val_true, val_prob = run_epoch(
                mlp, val_loader, optimizer, criterion, device, train=False
            )
            train_f1_micro, train_f1_macro, train_ap_micro, train_ap_macro = compute_multilabel_metrics(
                train_true, train_prob
            )
            val_f1_micro, val_f1_macro, val_ap_micro, val_ap_macro = compute_multilabel_metrics(
                val_true, val_prob
            )
            print(
                f"Split {split_id:02d} Epoch {epoch:02d} | Train loss {train_loss:.4f} F1_micro {train_f1_micro:.3f} F1_macro {train_f1_macro:.3f} AP_micro {train_ap_micro:.3f} AP_macro {train_ap_macro:.3f} | "
                f"Val loss {val_loss:.4f} F1_micro {val_f1_micro:.3f} F1_macro {val_f1_macro:.3f} AP_micro {val_ap_micro:.3f} AP_macro {val_ap_macro:.3f}"
            )

        split_record = {
            "model_id": args.model_id,
            "split": split_id,
            "val_f1_micro": val_f1_micro,
            "val_f1_macro": val_f1_macro,
            "val_ap_micro": val_ap_micro,
            "val_ap_macro": val_ap_macro,
        }
        records.append(split_record)

        split_term_rows = compute_per_term_metrics(val_true, val_prob, go_terms)
        for row in split_term_rows:
            row["model_id"] = args.model_id
            row["split"] = split_id
        per_term_records.extend(split_term_rows)

        split_term_path = output_dir / f"term_metrics_{args.model_id}_split{split_id:02d}.csv"
        with split_term_path.open("w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["model_id", "split", "go_term", "val_pos", "val_neg", "val_f1", "val_ap"],
            )
            writer.writeheader()
            writer.writerows(split_term_rows)

        ckpt_path = output_dir / f"go_term_mlp_{args.model_id}_split{split_id:02d}.pt"
        torch.save(
            {
                "model_state": mlp.state_dict(),
                "args": vars(args),
                "split": split_id,
                "go_terms": go_terms,
            },
            ckpt_path,
        )
        print(f"Saved classifier for split {split_id} to {ckpt_path}")
        print(f"Saved per-term metrics for split {split_id} to {split_term_path}")

    # Aggregate statistics across splits
    summary = {}
    for key in metric_keys:
        values = np.asarray([rec[key] for rec in records], dtype=float)
        summary[f"mean_{key}"] = float(values.mean())
        summary[f"std_{key}"] = float(values.std(ddof=0))

    metrics_path = output_dir / f"cv_metrics_{args.model_id}.csv"
    fieldnames = ["model_id", "split"] + metric_keys
    with metrics_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    summary_path = output_dir / f"cv_metrics_summary_{args.model_id}.txt"
    with summary_path.open("w") as f:
        for key in metric_keys:
            mean_k = summary[f"mean_{key}"]
            std_k = summary[f"std_{key}"]
            f.write(f"{key}: mean={mean_k:.4f}, std={std_k:.4f}\n")

    print("Cross-validation summary:")
    for key in metric_keys:
        print(f"{key}: mean {summary[f'mean_{key}']:.4f} | std {summary[f'std_{key}']:.4f}")
    print(f"Model id: {args.model_id}")
    print(f"Saved per-split metrics to {metrics_path}")
    print(f"Saved summary to {summary_path}")

    # Aggregate GO-term metrics across splits
    per_term_summary: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: {"ap": [], "f1": [], "pos": []})
    for rec in per_term_records:
        term = str(rec["go_term"])
        ap = float(rec["val_ap"])
        f1 = float(rec["val_f1"])
        pos = float(rec["val_pos"])
        if not np.isnan(ap):
            per_term_summary[term]["ap"].append(ap)
        per_term_summary[term]["f1"].append(f1)
        per_term_summary[term]["pos"].append(pos)

    per_term_summary_path = output_dir / f"term_metrics_summary_{args.model_id}.csv"
    with per_term_summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model_id",
                "go_term",
                "mean_val_pos",
                "mean_val_f1",
                "std_val_f1",
                "mean_val_ap",
                "std_val_ap",
            ],
        )
        writer.writeheader()
        for term in go_terms:
            stats = per_term_summary.get(term, {"ap": [], "f1": [], "pos": []})
            f1_arr = np.asarray(stats["f1"], dtype=float)
            ap_arr = np.asarray(stats["ap"], dtype=float)
            pos_arr = np.asarray(stats["pos"], dtype=float)
            writer.writerow(
                {
                    "model_id": args.model_id,
                    "go_term": term,
                    "mean_val_pos": float(pos_arr.mean()) if pos_arr.size else float("nan"),
                    "mean_val_f1": float(f1_arr.mean()) if f1_arr.size else float("nan"),
                    "std_val_f1": float(f1_arr.std(ddof=0)) if f1_arr.size else float("nan"),
                    "mean_val_ap": float(ap_arr.mean()) if ap_arr.size else float("nan"),
                    "std_val_ap": float(ap_arr.std(ddof=0)) if ap_arr.size else float("nan"),
                }
            )

    print(f"Saved GO-term summary metrics to {per_term_summary_path}")


if __name__ == "__main__":
    main()
