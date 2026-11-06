"""Cell-type annotation probe: MLP fine-tuned on frozen cell embeddings with donor splits.

Cells are split by donor (``--donor_col``): one random held-out donor set
(``--num_test_donors`` / ``--test_donors``) or leave-one-donor-out
(``--leave_one_donor_out``). For each split, ``--runs_per_split`` MLPs (different seeds)
are trained on the train-donor cells (early stopping on validation macro-F1) and
evaluated on the test donors:

  * seen-label metrics   accuracy / macro-F1 / weighted-F1 on test cells whose label
                         occurs in the training donors;
  * all-label metrics    the same over all test cells (unseen labels count as errors);
  * ontology-aware eval  each prediction is classified against the Cell Ontology
                         (cl-basic.obo) as exact / parent / child / sibling / no_match,
                         over all test cells and over cells with unseen labels.

Inputs
  --h5ad_path       AnnData with ``obs[donor_col]`` and ``obs[label_col]``, row-aligned
                    with the embedding (e.g. the h5ad given to scripts/run_inference.py)
  embedding         --embedding_path, or
                    <embedding_root>/<model_name>/<dataset_id>/<dataset_id>_embeddings.npy
                    (scripts/run_inference.py output; <model_name> = its --run_id)
  --basic_obo_file  Cell Ontology .obo (default: resources/ontology/cl-basic.obo)

Outputs (<output_dir>, default ./output/<dataset_id>_<model_name>/)
  run_summary.csv, split_seed_mean_std_summary.csv, run_config.json, and per run
  <split>_runXX_seedN/: mlp_model.pt, metrics.json, test_predictions*.csv,
  test_{all,unseen}_ontology_summary.json, train_history.csv, label_mapping.csv

Example (from the repository root)
  python downstream/2_celltype_annotation/celltype_annotation_finetune.py \
      --h5ad_path data/pantissue/filtered/HIHA.h5ad \
      --embedding_root ./output --model_name scOntoFM \
      --donor_col donor_id --label_col cell_type --num_test_donors 2
"""

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import anndata as ad
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
	accuracy_score,
	classification_report,
	f1_score,
)
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from scOntoFM.paths import CL_OBO_PATH  # noqa: E402

# Default output directory of scripts/run_inference.py (``--output_dir ./output`` run from the repo root).
DEFAULT_EMBED_ROOT = REPO_ROOT / "output"
DEFAULT_BASIC_OBO_FILE = str(CL_OBO_PATH)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Donor-based MLP finetuning for cell type annotation from saved embeddings."
	)

	parser.add_argument(
		"--h5ad_path",
		type=str,
		required=True,
		help="Path to the h5ad file containing donor_id and cell type labels in adata.obs.",
	)
	parser.add_argument(
		"--dataset_id",
		type=str,
		default=None,
		help="Dataset id used in embedding file naming; defaults to h5ad stem.",
	)
	parser.add_argument(
		"--embedding_path",
		type=str,
		default=None,
		help="Optional explicit path to embedding .npy file.",
	)
	parser.add_argument(
		"--embedding_root",
		type=str,
		default=str(DEFAULT_EMBED_ROOT),
		help="Root folder of inference outputs (<root>/<model_name>/<dataset_id>/<dataset_id>_embeddings.npy).",
	)
	parser.add_argument(
		"--model_name",
		type=str,
		default="scOntoFM",
		help="Embedding subfolder under embedding_root (the --run_id used for scripts/run_inference.py).",
	)

	parser.add_argument(
		"--donor_col",
		type=str,
		default="donor_id",
		help="Column in adata.obs used for donor split.",
	)
	parser.add_argument(
		"--shared_label_scope",
		type=str,
		choices=["all", "train"],
		default="all",
		help=(
			"How to define train/eval labels shared across donors: "
			"'all' = intersection across train+test donors; "
			"'train' = intersection across train donors only."
		),
	)
	parser.add_argument(
		"--label_col",
		type=str,
		default="cell_type",
		help="Column in adata.obs used as cell type class label.",
	)
	parser.add_argument(
		"--test_donors",
		type=str,
		default=None,
		help="Comma-separated donor ids for test split. If unset, random donor holdout is used.",
	)
	parser.add_argument(
		"--num_test_donors",
		type=int,
		default=1,
		help="How many donors to hold out when test_donors is not provided.",
	)
	parser.add_argument(
		"--leave_one_donor_out",
		action="store_true",
		help="If set, run one split per donor as test donor.",
	)

	parser.add_argument(
		"--hidden_dims",
		type=str,
		default="1024,512",
		help="Comma-separated hidden layer widths for MLP backbone.",
	)
	parser.add_argument(
		"--dropout",
		type=float,
		default=0.2,
		help="Dropout used between MLP hidden layers.",
	)
	parser.add_argument(
		"--batch_norm",
		action="store_true",
		help="Enable BatchNorm1d after each hidden linear layer.",
	)
	parser.add_argument(
		"--standardize",
		action="store_true",
		help="Fit StandardScaler on train embeddings and apply to test.",
	)

	parser.add_argument("--epochs", type=int, default=30)
	parser.add_argument("--batch_size", type=int, default=1024)
	parser.add_argument("--learning_rate", type=float, default=1e-3)
	parser.add_argument("--weight_decay", type=float, default=1e-4)
	parser.add_argument("--val_ratio", type=float, default=0.1)
	parser.add_argument(
		"--patience",
		type=int,
		default=8,
		help="Early stopping patience on validation macro-F1. Disable with <= 0.",
	)
	parser.add_argument("--num_workers", type=int, default=0)
	parser.add_argument("--seed", type=int, default=42)
	parser.add_argument(
		"--runs_per_split",
		type=int,
		default=5,
		help="Number of different-seed runs for each donor split.",
	)
	parser.add_argument("--device", type=str, default="auto")
	parser.add_argument(
		"--skip_ontology_eval",
		action="store_true",
		help="Disable ontology parent/child/sibling checks for donor-specific labels.",
	)
	parser.add_argument(
		"--basic_obo_file",
		type=str,
		default=DEFAULT_BASIC_OBO_FILE,
		help="Path to the Cell Ontology cl-basic.obo (default: resources/ontology/cl-basic.obo).",
	)

	parser.add_argument(
		"--output_dir",
		type=str,
		default=None,
		help=(
			"Directory for model checkpoints, predictions, and metrics. "
			"If omitted, uses ./output/{dataset_id}_{model_name}."
		),
	)

	return parser.parse_args()


def set_seed(seed: int) -> None:
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)


def parse_csv_values(raw: str | None) -> List[str]:
	if raw is None:
		return []
	return [x.strip() for x in raw.split(",") if x.strip()]


def parse_hidden_dims(raw: str) -> List[int]:
	values = [x.strip() for x in raw.split(",") if x.strip()]
	if not values:
		return [512]
	dims = [int(v) for v in values]
	if any(d <= 0 for d in dims):
		raise ValueError("All hidden dims must be positive integers.")
	return dims


def safe_name(value: str) -> str:
	out = []
	for ch in str(value):
		if ch.isalnum() or ch in {"-", "_", "."}:
			out.append(ch)
		else:
			out.append("_")
	return "".join(out).strip("_") or "unknown"


def normalize_cl_id(label: str) -> str:
	if label.startswith("adata_"):
		label = label.split("adata_", 1)[1]
	if label.startswith("CL_"):
		return label.replace("CL_", "CL:", 1)
	return label


def load_ontology_graph(basic_obo_file: str):
	# Imported lazily so that a missing ontology dependency only disables the ontology checks.
	from scOntoFM.ontology import get_id_mapper, import_cell_ontology

	obo_file_path = Path(basic_obo_file)

	onto_graph = import_cell_ontology(str(obo_file_path))
	id2name = get_id_mapper(onto_graph)
	name2id = {value: key for key, value in id2name.items()}
	return onto_graph, name2id


def _match_type(n1: str, n2: str, graph) -> str:
	if n1 == n2:
		return "exact"
	preds1 = set(graph.predecessors(n1))
	preds2 = set(graph.predecessors(n2))
	if preds1 and preds2 and not preds1.isdisjoint(preds2):
		return "sibling"
	if n1 in preds2:
		return "parent"
	if n2 in preds1:
		return "child"
	return "no_match"


def _to_ontology_node(label: str, graph, name2id: Dict[str, str]) -> str:
	label_str = str(label)
	norm_label = normalize_cl_id(label_str)
	if norm_label in graph:
		return norm_label
	if label_str in graph:
		return label_str
	if label_str in name2id:
		return name2id[label_str]
	if norm_label in name2id:
		return name2id[norm_label]
	return ""


def summarize_ontology_matches(
	true_labels: Sequence[str],
	pred_labels: Sequence[str],
	graph,
	name2id: Dict[str, str],
) -> Tuple[List[str], Dict[str, Dict[str, float]]]:
	if len(true_labels) != len(pred_labels):
		raise ValueError("true_labels and pred_labels must have the same length")

	categories = ["exact", "parent", "child", "sibling", "no_match"]
	counts = {cat: 0 for cat in categories}
	matches: List[str] = []

	for true_label, pred_label in zip(true_labels, pred_labels):
		true_node = _to_ontology_node(true_label, graph, name2id)
		pred_node = _to_ontology_node(pred_label, graph, name2id)
		if not true_node or not pred_node:
			match = "no_match"
		else:
			match = _match_type(true_node, pred_node, graph)
		counts[match] += 1
		matches.append(match)

	total = len(matches)
	summary = {
		cat: {
			"count": float(counts[cat]),
			"fraction": (counts[cat] / total) if total else 0.0,
		}
		for cat in categories
	}
	return matches, summary


def shared_labels_across_donors(
	obs_df: pd.DataFrame,
	donors: Sequence[str],
	donor_col: str,
	label_col: str,
) -> List[str]:
	donor_labels = []
	for donor in donors:
		labels = set(obs_df.loc[obs_df[donor_col] == donor, label_col].tolist())
		if labels:
			donor_labels.append(labels)

	if not donor_labels:
		return []

	shared = set.intersection(*donor_labels)
	return sorted(shared)


def resolve_embedding_path(
	embedding_path: str | None,
	embedding_root: str,
	model_name: str,
	dataset_id: str,
) -> Path:
	if embedding_path:
		path = Path(embedding_path)
		if not path.exists():
			raise FileNotFoundError(f"Embedding file not found: {path}")
		return path

	root = Path(embedding_root)
	candidates = [
		root / model_name / dataset_id / f"{dataset_id}_embeddings.npy",
		root / model_name / f"{dataset_id}_embeddings.npy",
	]

	for candidate in candidates:
		if candidate.exists():
			return candidate

	candidate_text = "\n".join(str(x) for x in candidates)
	raise FileNotFoundError(
		"Could not resolve embedding file. Checked:\n" + candidate_text
	)


def load_obs_and_embeddings(
	h5ad_path: Path,
	embedding_path: Path,
	donor_col: str,
	label_col: str,
) -> Tuple[pd.DataFrame, np.ndarray]:
	adata = ad.read_h5ad(h5ad_path)

	if donor_col not in adata.obs.columns:
		raise KeyError(f"{donor_col} not found in adata.obs")
	if label_col not in adata.obs.columns:
		raise KeyError(f"{label_col} not found in adata.obs")

	embeddings = np.load(embedding_path)
	if embeddings.ndim != 2:
		raise ValueError(f"Expected a 2D embedding array, got shape {embeddings.shape}")
	if embeddings.shape[0] != adata.n_obs:
		raise ValueError(
			"Embedding rows do not match number of cells in h5ad: "
			f"{embeddings.shape[0]} vs {adata.n_obs}"
		)

	obs_df = adata.obs[[donor_col, label_col]].copy()
	donor_raw = obs_df[donor_col]
	label_raw = obs_df[label_col]

	valid_mask = (
		donor_raw.notna()
		& label_raw.notna()
		& (donor_raw.astype(str).str.len() > 0)
		& (label_raw.astype(str).str.len() > 0)
	)

	if not np.all(valid_mask.values):
		dropped = int((~valid_mask.values).sum())
		print(f"Dropping {dropped} cells with missing donor or label fields.")

	obs_df = obs_df.loc[valid_mask].copy()
	obs_df[donor_col] = obs_df[donor_col].astype(str)
	obs_df[label_col] = obs_df[label_col].astype(str)
	filtered_embeddings = embeddings[valid_mask.values]

	return obs_df.reset_index(drop=True), filtered_embeddings


def pick_splits(
	donors: Sequence[str],
	leave_one_donor_out: bool,
	explicit_test_donors: Sequence[str],
	num_test_donors: int,
	seed: int,
) -> List[Tuple[str, List[str], List[str]]]:
	donor_list = sorted(set(donors))
	if len(donor_list) < 2:
		raise ValueError("Need at least two unique donors for train/test split.")

	splits: List[Tuple[str, List[str], List[str]]] = []
	if leave_one_donor_out:
		for test_donor in donor_list:
			train_donors = [d for d in donor_list if d != test_donor]
			split_name = f"test_{safe_name(test_donor)}"
			splits.append((split_name, train_donors, [test_donor]))
		return splits

	if explicit_test_donors:
		missing = [d for d in explicit_test_donors if d not in donor_list]
		if missing:
			raise ValueError(f"Provided test donors not found in data: {missing}")
		test_donors = sorted(set(explicit_test_donors))
	else:
		if num_test_donors <= 0:
			raise ValueError("num_test_donors must be > 0")
		if num_test_donors >= len(donor_list):
			raise ValueError("num_test_donors must be smaller than #unique donors")
		rng = np.random.default_rng(seed)
		test_donors = sorted(
			rng.choice(np.array(donor_list), size=num_test_donors, replace=False).tolist()
		)

	train_donors = [d for d in donor_list if d not in test_donors]
	split_name = f"test_{'_'.join(safe_name(d) for d in test_donors)}"
	splits.append((split_name, train_donors, test_donors))
	return splits


class MLPClassifier(nn.Module):
	def __init__(
		self,
		input_dim: int,
		hidden_dims: Sequence[int],
		num_classes: int,
		dropout: float,
		batch_norm: bool,
	) -> None:
		super().__init__()
		layers: List[nn.Module] = []
		current = input_dim
		for hdim in hidden_dims:
			layers.append(nn.Linear(current, hdim))
			if batch_norm:
				layers.append(nn.BatchNorm1d(hdim))
			layers.append(nn.ReLU())
			if dropout > 0:
				layers.append(nn.Dropout(dropout))
			current = hdim

		layers.append(nn.Linear(current, num_classes))
		self.net = nn.Sequential(*layers)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return self.net(x)


def make_loader(
	X: np.ndarray,
	y: np.ndarray,
	batch_size: int,
	shuffle: bool,
	num_workers: int,
) -> DataLoader:
	dataset = TensorDataset(
		torch.from_numpy(X.astype(np.float32)),
		torch.from_numpy(y.astype(np.int64)),
	)
	return DataLoader(
		dataset,
		batch_size=batch_size,
		shuffle=shuffle,
		num_workers=num_workers,
		pin_memory=torch.cuda.is_available(),
		drop_last=False,
	)


def evaluate_model(
	model: nn.Module,
	loader: DataLoader,
	device: torch.device,
	criterion: nn.Module,
) -> Tuple[float, np.ndarray, np.ndarray, Dict[str, float]]:
	model.eval()
	total_loss = 0.0
	total_count = 0
	all_true: List[np.ndarray] = []
	all_pred: List[np.ndarray] = []

	with torch.no_grad():
		for x_batch, y_batch in loader:
			x_batch = x_batch.to(device, non_blocking=True)
			y_batch = y_batch.to(device, non_blocking=True)
			logits = model(x_batch)
			loss = criterion(logits, y_batch)

			preds = torch.argmax(logits, dim=1)
			bs = y_batch.size(0)
			total_loss += float(loss.item()) * bs
			total_count += bs

			all_true.append(y_batch.cpu().numpy())
			all_pred.append(preds.cpu().numpy())

	y_true = np.concatenate(all_true) if all_true else np.array([], dtype=np.int64)
	y_pred = np.concatenate(all_pred) if all_pred else np.array([], dtype=np.int64)
	mean_loss = total_loss / max(total_count, 1)

	metrics = {
		"accuracy": float(accuracy_score(y_true, y_pred)) if total_count else 0.0,
		"macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0))
		if total_count
		else 0.0,
		"weighted_f1": float(
			f1_score(y_true, y_pred, average="weighted", zero_division=0)
		)
		if total_count
		else 0.0,
	}
	return mean_loss, y_true, y_pred, metrics


def predict_labels(
	model: nn.Module,
	X: np.ndarray,
	device: torch.device,
	batch_size: int,
) -> np.ndarray:
	if len(X) == 0:
		return np.array([], dtype=np.int64)

	model.eval()
	preds: List[np.ndarray] = []
	with torch.no_grad():
		for start in range(0, len(X), batch_size):
			end = min(start + batch_size, len(X))
			x = torch.from_numpy(X[start:end].astype(np.float32)).to(
				device, non_blocking=True
			)
			logits = model(x)
			batch_pred = torch.argmax(logits, dim=1).cpu().numpy()
			preds.append(batch_pred)

	return np.concatenate(preds)


def split_train_val(
	X: np.ndarray,
	y: np.ndarray,
	val_ratio: float,
	seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
	if val_ratio <= 0 or len(X) < 20:
		return X, y, X, y

	n = len(X)
	n_val = int(round(n * val_ratio))
	n_val = min(max(n_val, 1), n - 1)

	rng = np.random.default_rng(seed)
	perm = rng.permutation(n)
	val_idx = perm[:n_val]
	train_idx = perm[n_val:]

	return X[train_idx], y[train_idx], X[val_idx], y[val_idx]


def choose_device(raw_device: str) -> torch.device:
	if raw_device == "auto":
		return torch.device("cuda" if torch.cuda.is_available() else "cpu")
	return torch.device(raw_device)


def save_scaler(path: Path, scaler: StandardScaler) -> None:
	np.savez(
		path,
		mean=scaler.mean_.astype(np.float32),
		scale=scaler.scale_.astype(np.float32),
		var=scaler.var_.astype(np.float32),
	)


def evaluate_string_predictions(
	y_true: Sequence[str],
	y_pred: Sequence[str],
) -> Dict[str, float]:
	if len(y_true) != len(y_pred):
		raise ValueError("y_true and y_pred must have same length")
	if len(y_true) == 0:
		return {"accuracy": 0.0, "macro_f1": 0.0, "weighted_f1": 0.0}

	labels = sorted(set(y_true).union(set(y_pred)))
	return {
		"accuracy": float(accuracy_score(y_true, y_pred)),
		"macro_f1": float(
			f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
		),
		"weighted_f1": float(
			f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
		),
	}


def aggregate_split_seed_stats(summary_df: pd.DataFrame) -> pd.DataFrame:
	if summary_df.empty:
		return pd.DataFrame()

	metric_columns = [
		"test_seen_loss",
		"test_seen_accuracy",
		"test_seen_macro_f1",
		"test_seen_weighted_f1",
		"test_all_accuracy",
		"test_all_macro_f1",
		"test_all_weighted_f1",
		"all_test_exact_frac",
		"all_test_parent_frac",
		"all_test_child_frac",
		"all_test_sibling_frac",
		"unseen_test_exact_frac",
		"unseen_test_parent_frac",
		"unseen_test_child_frac",
		"unseen_test_sibling_frac",
	]

	agg_dict: Dict[str, str | List[str]] = {
		"run_name": "count",
		"run_seed": "nunique",
	}
	for col in metric_columns:
		if col in summary_df.columns:
			agg_dict[col] = ["mean", "std"]

	agg_df = summary_df.groupby("split_base_name", as_index=False).agg(agg_dict)
	agg_df.columns = [
		"split_base_name"
		if c[0] == "split_base_name"
		else "_".join([x for x in c if x])
		for c in agg_df.columns
	]

	agg_df = agg_df.rename(
		columns={
			"run_name_count": "n_runs",
			"run_seed_nunique": "n_unique_seeds",
		}
	)

	for col in agg_df.columns:
		if col.endswith("_std"):
			agg_df[col] = agg_df[col].fillna(0.0)

	return agg_df.sort_values("split_base_name").reset_index(drop=True)


def train_single_split(
	args: argparse.Namespace,
	split_base_name: str,
	run_index: int,
	run_seed: int,
	train_donors: Sequence[str],
	test_donors: Sequence[str],
	obs_df: pd.DataFrame,
	embeddings: np.ndarray,
	ontology_graph,
	name2id: Dict[str, str],
) -> Dict[str, float | int | str]:
	split_name = f"{split_base_name}_run{run_index:02d}_seed{run_seed}"
	split_dir = Path(args.output_dir) / split_name
	split_dir.mkdir(parents=True, exist_ok=True)

	donor_col = args.donor_col
	label_col = args.label_col

	train_mask = obs_df[donor_col].isin(train_donors).to_numpy()
	test_mask = obs_df[donor_col].isin(test_donors).to_numpy()

	if train_mask.sum() == 0 or test_mask.sum() == 0:
		raise ValueError(
			f"Invalid split {split_name}: train={train_mask.sum()}, test={test_mask.sum()}"
		)

	train_labels = sorted(obs_df.loc[train_mask, label_col].astype(str).unique().tolist())
	if not train_labels:
		raise ValueError(f"No training labels found in split {split_name}.")

	label_to_idx = {label: idx for idx, label in enumerate(train_labels)}
	idx_to_label = {idx: label for label, idx in label_to_idx.items()}

	X_train = embeddings[train_mask]
	y_train = (
		obs_df.loc[train_mask, label_col].map(label_to_idx).to_numpy(dtype=np.int64)
	)

	test_labels_all = obs_df.loc[test_mask, label_col].astype(str).to_numpy()
	test_seen_mask = np.array([x in label_to_idx for x in test_labels_all], dtype=bool)
	test_unseen_mask = ~test_seen_mask

	global_test_idx = np.flatnonzero(test_mask)
	global_test_seen_idx = global_test_idx[test_seen_mask]
	global_test_unseen_idx = global_test_idx[test_unseen_mask]

	X_test_seen = embeddings[global_test_seen_idx]
	y_test_seen = np.array(
		[label_to_idx[x] for x in test_labels_all[test_seen_mask]], dtype=np.int64
	)
	X_test_all = embeddings[test_mask]
	true_test_labels_all = test_labels_all.tolist()

	scaler = None
	if args.standardize:
		scaler = StandardScaler(with_mean=True, with_std=True)
		X_train = scaler.fit_transform(X_train)
		if len(X_test_seen) > 0:
			X_test_seen = scaler.transform(X_test_seen)
		X_test_all = scaler.transform(X_test_all)
		save_scaler(split_dir / "scaler_stats.npz", scaler)

	X_subtrain, y_subtrain, X_val, y_val = split_train_val(
		X_train, y_train, args.val_ratio, run_seed
	)

	train_loader = make_loader(
		X_subtrain,
		y_subtrain,
		args.batch_size,
		shuffle=True,
		num_workers=args.num_workers,
	)
	val_loader = make_loader(
		X_val,
		y_val,
		args.batch_size,
		shuffle=False,
		num_workers=args.num_workers,
	)
	test_loader_seen = (
		make_loader(
			X_test_seen,
			y_test_seen,
			args.batch_size,
			shuffle=False,
			num_workers=args.num_workers,
		)
		if len(X_test_seen) > 0
		else None
	)

	device = choose_device(args.device)
	model = MLPClassifier(
		input_dim=embeddings.shape[1],
		hidden_dims=parse_hidden_dims(args.hidden_dims),
		num_classes=len(label_to_idx),
		dropout=args.dropout,
		batch_norm=args.batch_norm,
	).to(device)

	criterion = nn.CrossEntropyLoss()
	optimizer = torch.optim.AdamW(
		model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
	)

	best_state = None
	best_score = -np.inf
	epochs_no_improve = 0
	history = []

	for epoch in range(1, args.epochs + 1):
		model.train()
		running_loss = 0.0
		running_count = 0

		for x_batch, y_batch in train_loader:
			x_batch = x_batch.to(device, non_blocking=True)
			y_batch = y_batch.to(device, non_blocking=True)

			optimizer.zero_grad(set_to_none=True)
			logits = model(x_batch)
			loss = criterion(logits, y_batch)
			loss.backward()
			optimizer.step()

			bs = y_batch.size(0)
			running_loss += float(loss.item()) * bs
			running_count += bs

		train_loss = running_loss / max(running_count, 1)
		val_loss, _, _, val_metrics = evaluate_model(model, val_loader, device, criterion)

		record = {
			"epoch": epoch,
			"train_loss": train_loss,
			"val_loss": val_loss,
			"val_accuracy": val_metrics["accuracy"],
			"val_macro_f1": val_metrics["macro_f1"],
			"val_weighted_f1": val_metrics["weighted_f1"],
		}
		history.append(record)
		print(
			f"[{split_name}] epoch {epoch:03d} | "
			f"train_loss={train_loss:.4f} "
			f"val_loss={val_loss:.4f} "
			f"val_macro_f1={val_metrics['macro_f1']:.4f}"
		)

		current_score = val_metrics["macro_f1"]
		if current_score > best_score:
			best_score = current_score
			best_state = {
				"model_state_dict": model.state_dict(),
				"epoch": epoch,
				"val_macro_f1": current_score,
			}
			epochs_no_improve = 0
		else:
			epochs_no_improve += 1

		if args.patience > 0 and epochs_no_improve >= args.patience:
			print(
				f"[{split_name}] Early stopping at epoch {epoch} "
				f"(best val_macro_f1={best_score:.4f})."
			)
			break

	if best_state is not None:
		model.load_state_dict(best_state["model_state_dict"])

	if test_loader_seen is not None:
		test_seen_loss, y_true_seen, y_pred_seen, test_metrics_seen = evaluate_model(
			model, test_loader_seen, device, criterion
		)
	else:
		test_seen_loss = 0.0
		y_true_seen = np.array([], dtype=np.int64)
		y_pred_seen = np.array([], dtype=np.int64)
		test_metrics_seen = {"accuracy": 0.0, "macro_f1": 0.0, "weighted_f1": 0.0}

	test_pred_all_idx = predict_labels(
		model,
		X_test_all,
		device=device,
		batch_size=args.batch_size,
	)
	test_pred_all_labels = [idx_to_label[i] for i in test_pred_all_idx]
	test_metrics_all = evaluate_string_predictions(true_test_labels_all, test_pred_all_labels)

	labels_for_report = sorted(set(true_test_labels_all).union(set(test_pred_all_labels)))
	report = classification_report(
		true_test_labels_all,
		test_pred_all_labels,
		labels=labels_for_report,
		output_dict=True,
		zero_division=0,
	)

	test_rows = obs_df.loc[test_mask].copy().reset_index(drop=True)
	test_rows["true_label"] = true_test_labels_all
	test_rows["pred_label"] = test_pred_all_labels
	test_rows["correct"] = (
		test_rows["true_label"].astype(str) == test_rows["pred_label"].astype(str)
	).astype(int)
	test_rows["label_group"] = np.where(test_seen_mask, "seen_in_train", "unseen_in_train")
	ontology_summary_shared = {
		"exact": {"count": 0.0, "fraction": 0.0},
		"parent": {"count": 0.0, "fraction": 0.0},
		"child": {"count": 0.0, "fraction": 0.0},
		"sibling": {"count": 0.0, "fraction": 0.0},
		"no_match": {"count": 0.0, "fraction": 0.0},
	}

	shared_true_labels = test_rows["true_label"].astype(str).to_list()
	shared_pred_labels = test_rows["pred_label"].astype(str).to_list()
	if ontology_graph is not None:
		shared_matches, ontology_summary_shared = summarize_ontology_matches(
			shared_true_labels,
			shared_pred_labels,
			ontology_graph,
			name2id,
		)
		test_rows["ontology_match"] = shared_matches
	else:
		test_rows["ontology_match"] = "ontology_unavailable"

	specific_rows = test_rows.loc[test_rows["label_group"] == "unseen_in_train"].copy()
	specific_rows = specific_rows.reset_index(drop=True)
	ontology_summary_specific = {
		"exact": {"count": 0.0, "fraction": 0.0},
		"parent": {"count": 0.0, "fraction": 0.0},
		"child": {"count": 0.0, "fraction": 0.0},
		"sibling": {"count": 0.0, "fraction": 0.0},
		"no_match": {"count": 0.0, "fraction": 0.0},
	}

	if len(specific_rows) > 0:
		specific_true_labels = specific_rows["true_label"].astype(str).to_list()
		specific_pred_labels = specific_rows["pred_label"].astype(str).to_list()
		if ontology_graph is not None:
			ontology_matches, ontology_summary_specific = summarize_ontology_matches(
				specific_true_labels,
				specific_pred_labels,
				ontology_graph,
				name2id,
			)
			specific_rows["ontology_match"] = ontology_matches
		else:
			specific_rows["ontology_match"] = "ontology_unavailable"
	else:
		specific_rows["true_label"] = []
		specific_rows["pred_label"] = []
		specific_rows["correct"] = []
		specific_rows["label_group"] = []
		specific_rows["ontology_match"] = []

	torch.save(
		{
			"model_state_dict": model.state_dict(),
			"input_dim": embeddings.shape[1],
			"hidden_dims": parse_hidden_dims(args.hidden_dims),
			"num_classes": len(label_to_idx),
			"label_to_idx": label_to_idx,
			"idx_to_label": idx_to_label,
			"args": vars(args),
			"train_donors": list(train_donors),
			"test_donors": list(test_donors),
		},
		split_dir / "mlp_model.pt",
	)

	pd.DataFrame(
		{
			"label": [idx_to_label[i] for i in range(len(idx_to_label))],
			"label_idx": list(range(len(idx_to_label))),
		}
	).to_csv(split_dir / "label_mapping.csv", index=False)
	pd.DataFrame({"train_label": train_labels}).to_csv(
		split_dir / "train_labels.csv", index=False
	)

	pd.DataFrame(history).to_csv(split_dir / "train_history.csv", index=False)
	test_rows.to_csv(split_dir / "test_predictions.csv", index=False)
	specific_rows.to_csv(split_dir / "test_predictions_unseen_in_train.csv", index=False)

	with open(split_dir / "test_all_ontology_summary.json", "w", encoding="utf-8") as f:
		json.dump(ontology_summary_shared, f, indent=2)

	with open(split_dir / "test_unseen_ontology_summary.json", "w", encoding="utf-8") as f:
		json.dump(ontology_summary_specific, f, indent=2)

	metrics_payload = {
		"run_name": split_name,
		"split_base_name": split_base_name,
		"run_index": int(run_index),
		"run_seed": int(run_seed),
		"split_name": split_name,
		"train_donors": list(train_donors),
		"test_donors": list(test_donors),
		"shared_label_scope": args.shared_label_scope,
		"n_train_labels": int(len(train_labels)),
		"train_cells_total": int(train_mask.sum()),
		"test_cells_total": int(test_mask.sum()),
		"train_cells_used": int(train_mask.sum()),
		"test_cells_seen_in_train": int(test_seen_mask.sum()),
		"test_cells_unseen_in_train": int(test_unseen_mask.sum()),
		"n_classes_total": int(len(label_to_idx)),
		"test_seen_loss": float(test_seen_loss),
		"test_seen_accuracy": float(test_metrics_seen["accuracy"]),
		"test_seen_macro_f1": float(test_metrics_seen["macro_f1"]),
		"test_seen_weighted_f1": float(test_metrics_seen["weighted_f1"]),
		"test_all_accuracy": float(test_metrics_all["accuracy"]),
		"test_all_macro_f1": float(test_metrics_all["macro_f1"]),
		"test_all_weighted_f1": float(test_metrics_all["weighted_f1"]),
		"test_all_ontology_summary": ontology_summary_shared,
		"test_unseen_ontology_summary": ontology_summary_specific,
		"classification_report": report,
	}

	with open(split_dir / "metrics.json", "w", encoding="utf-8") as f:
		json.dump(metrics_payload, f, indent=2)

	print(
		f"[{split_name}] seen_test_loss={test_seen_loss:.4f} "
		f"seen_test_acc={test_metrics_seen['accuracy']:.4f} "
		f"seen_test_macro_f1={test_metrics_seen['macro_f1']:.4f} "
		f"all_test_acc={test_metrics_all['accuracy']:.4f} "
		f"all_test_macro_f1={test_metrics_all['macro_f1']:.4f} "
		f"train_labels={len(train_labels)} "
		f"unseen_test_cells={int(test_unseen_mask.sum())}"
	)
	shared_exact_frac = ontology_summary_shared["exact"]["fraction"]
	shared_parent_frac = ontology_summary_shared["parent"]["fraction"]
	shared_child_frac = ontology_summary_shared["child"]["fraction"]
	shared_sibling_frac = ontology_summary_shared["sibling"]["fraction"]
	print(
		f"[{split_name}] ontology on all test labels: "
		f"exact={shared_exact_frac:.4f} parent={shared_parent_frac:.4f} "
		f"child={shared_child_frac:.4f} sibling={shared_sibling_frac:.4f}"
	)
	if int(test_unseen_mask.sum()) > 0:
		exact_frac = ontology_summary_specific["exact"]["fraction"]
		parent_frac = ontology_summary_specific["parent"]["fraction"]
		child_frac = ontology_summary_specific["child"]["fraction"]
		sibling_frac = ontology_summary_specific["sibling"]["fraction"]
		print(
			f"[{split_name}] ontology on unseen-in-train labels: "
			f"exact={exact_frac:.4f} parent={parent_frac:.4f} "
			f"child={child_frac:.4f} sibling={sibling_frac:.4f}"
		)

	return {
		"run_name": split_name,
		"split_base_name": split_base_name,
		"run_index": int(run_index),
		"run_seed": int(run_seed),
		"split_name": split_name,
		"n_train_donors": len(train_donors),
		"n_test_donors": len(test_donors),
		"n_train_labels": int(len(train_labels)),
		"train_cells_total": int(train_mask.sum()),
		"test_cells_total": int(test_mask.sum()),
		"train_cells_used": int(train_mask.sum()),
		"test_cells_seen_in_train": int(test_seen_mask.sum()),
		"test_cells_unseen_in_train": int(test_unseen_mask.sum()),
		"test_seen_loss": float(test_seen_loss),
		"test_seen_accuracy": float(test_metrics_seen["accuracy"]),
		"test_seen_macro_f1": float(test_metrics_seen["macro_f1"]),
		"test_seen_weighted_f1": float(test_metrics_seen["weighted_f1"]),
		"test_all_accuracy": float(test_metrics_all["accuracy"]),
		"test_all_macro_f1": float(test_metrics_all["macro_f1"]),
		"test_all_weighted_f1": float(test_metrics_all["weighted_f1"]),
		"all_test_exact_frac": float(ontology_summary_shared["exact"]["fraction"]),
		"all_test_parent_frac": float(ontology_summary_shared["parent"]["fraction"]),
		"all_test_child_frac": float(ontology_summary_shared["child"]["fraction"]),
		"all_test_sibling_frac": float(ontology_summary_shared["sibling"]["fraction"]),
		"unseen_test_exact_frac": float(ontology_summary_specific["exact"]["fraction"]),
		"unseen_test_parent_frac": float(ontology_summary_specific["parent"]["fraction"]),
		"unseen_test_child_frac": float(ontology_summary_specific["child"]["fraction"]),
		"unseen_test_sibling_frac": float(
			ontology_summary_specific["sibling"]["fraction"]
		),
	}


def main() -> None:
	args = parse_args()
	set_seed(args.seed)
	if args.runs_per_split <= 0:
		raise ValueError("runs_per_split must be a positive integer.")

	h5ad_path = Path(args.h5ad_path)
	if not h5ad_path.exists():
		raise FileNotFoundError(f"h5ad not found: {h5ad_path}")

	dataset_id = args.dataset_id or h5ad_path.stem
	emb_path = resolve_embedding_path(
		args.embedding_path, args.embedding_root, args.model_name, dataset_id
	)

	auto_output_dir = Path("./output") / f"{safe_name(dataset_id)}_{safe_name(args.model_name)}"
	resolved_output_dir = Path(args.output_dir) if args.output_dir else auto_output_dir
	args.output_dir = str(resolved_output_dir)

	print(f"h5ad: {h5ad_path}")
	print(f"embedding: {emb_path}")
	print(f"dataset_id: {dataset_id}")
	print(f"output_dir: {args.output_dir}")

	obs_df, embeddings = load_obs_and_embeddings(
		h5ad_path,
		emb_path,
		donor_col=args.donor_col,
		label_col=args.label_col,
	)

	donor_values = obs_df[args.donor_col].tolist()
	splits = pick_splits(
		donors=donor_values,
		leave_one_donor_out=args.leave_one_donor_out,
		explicit_test_donors=parse_csv_values(args.test_donors),
		num_test_donors=args.num_test_donors,
		seed=args.seed,
	)

	print(f"Total cells after filtering: {len(obs_df)}")
	print(f"Embedding dim: {embeddings.shape[1]}")
	print(f"Total classes (all labels): {obs_df[args.label_col].nunique()}")
	print(f"Total unique donors: {len(set(donor_values))}")
	print(f"Running {len(splits)} split(s) x {args.runs_per_split} seed run(s) each.")

	ontology_graph = None
	name2id: Dict[str, str] = {}
	if args.skip_ontology_eval:
		print("Ontology evaluation disabled by --skip_ontology_eval.")
	else:
		try:
			ontology_graph, name2id = load_ontology_graph(args.basic_obo_file)
			print("Ontology graph loaded for parent/child/sibling checks.")
		except Exception as exc:
			print(f"Warning: failed to load ontology graph; skipping ontology checks. {exc}")
			ontology_graph = None
			name2id = {}

	out_root = Path(args.output_dir)
	out_root.mkdir(parents=True, exist_ok=True)

	run_summary = []
	for split_idx, (split_name, train_donors, test_donors) in enumerate(splits):
		for run_idx in range(args.runs_per_split):
			run_seed = args.seed * 100000 + split_idx * 1000 + run_idx
			set_seed(run_seed)
			summary = train_single_split(
				args,
				split_base_name=split_name,
				run_index=run_idx + 1,
				run_seed=run_seed,
				train_donors=train_donors,
				test_donors=test_donors,
				obs_df=obs_df,
				embeddings=embeddings,
				ontology_graph=ontology_graph,
				name2id=name2id,
			)
			run_summary.append(summary)

	summary_df = pd.DataFrame(run_summary)
	summary_df.to_csv(out_root / "run_summary.csv", index=False)

	agg_df = aggregate_split_seed_stats(summary_df)
	agg_df.to_csv(out_root / "split_seed_mean_std_summary.csv", index=False)

	with open(out_root / "run_config.json", "w", encoding="utf-8") as f:
		json.dump(vars(args), f, indent=2)

	print("Finished.")
	print(f"Summary saved to: {out_root / 'run_summary.csv'}")
	print(f"Mean/std summary saved to: {out_root / 'split_seed_mean_std_summary.csv'}")
	if len(agg_df) > 0:
		display_cols = [
			"split_base_name",
			"n_runs",
			"test_all_accuracy_mean",
			"test_all_accuracy_std",
			"test_all_macro_f1_mean",
			"test_all_macro_f1_std",
		]
		display_cols = [c for c in display_cols if c in agg_df.columns]
		print("Per-split performance mean/std (all_test):")
		print(agg_df[display_cols].to_string(index=False))


if __name__ == "__main__":
	main()
