"""Visualize GO terms using dataset-specific contextualized gene embeddings.

Workflow
1. Load per-cell tokenized dataset and contextualized gene signals exported by inference.
2. Reconstruct contextualized gene representations for a specific dataset
   (following the same token mapping logic used in gene_func.ipynb).
3. Build one embedding per GO term by averaging embeddings of genes in that term.
4. Run PCA (default 50 dims) followed by UMAP for pattern exploration.

Outputs
- go_term_pca_for_umap_<tag>.csv
- go_term_umap_from_pca_<tag>.csv
- go_term_umap_from_pca_<tag>.png
- go_term_umap_summary_<tag>.txt

Inputs (``<sota_output_root>/<run_id>/<dataset_id>/``, written by scripts/run_inference.py
run with ``--save_gene_embeddings --save_data``)
- <dataset_id>_gene_embeddings.npy   contextual gene values, (n_cells x L-2)
- <dataset_id>_hf_dataset/           tokenized cells (input_ids)
- <dataset_id>.h5ad                  defines the gene universe (var_names = Ensembl IDs,
                                     var[--gene_name_column] = symbols); without --save_data
                                     pass --adata_path to the preprocessed h5ad instead
- GO GMT ``c5.go.v2025.1.Hs.symbols.gmt`` (download from
  https://www.gsea-msigdb.org/gsea/msigdb/human/collections.jsp into this folder)

Example (from the repository root)
  python scripts/run_inference.py --input_h5ad data/Kim2020_Lung_convert.h5ad \
      --run_id scOntoFM --save_gene_embeddings --save_data
  python downstream/3_gene_function/GO_embedding_contextualize.py \
      --dataset_id Kim2020_Lung_convert --run_id scOntoFM --sota_output_root ./output \
      --output_dir downstream/3_gene_function/go_embedding_context_out
"""

import argparse
import csv
import os
import pickle
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import decoupler as dc
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
import torch
from datasets import load_from_disk
from sklearn.decomposition import PCA

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT))
from scOntoFM.paths import TOKEN_DICT_PATH  # noqa: E402


GO_NAMESPACES = {"GOBP": "BP", "GOCC": "CC", "GOMF": "MF"}
VALID_NAMESPACES = {"BP", "CC", "MF"}


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="GO term embedding + PCA->UMAP using contextualized gene embeddings from one dataset"
	)
	parser.add_argument(
		"--dataset_id",
		type=str,
		required=True,
		help="Dataset identifier used in scripts/run_inference.py outputs.",
	)
	parser.add_argument(
		"--run_id",
		type=str,
		default="scOntoFM",
		help="Run identifier under sota_output_root (the --run_id of scripts/run_inference.py).",
	)
	parser.add_argument(
		"--sota_output_root",
		type=str,
		default=str(REPO_ROOT / "output"),
		help="Root folder containing inference outputs organized as <run_id>/<dataset_id>/... "
		"(default: <repo>/output, i.e. scripts/run_inference.py --output_dir ./output).",
	)
	parser.add_argument(
		"--context_emb_path",
		type=str,
		default=None,
		help="Optional explicit path to <dataset_id>_gene_embeddings.npy.",
	)
	parser.add_argument(
		"--hf_dataset_path",
		type=str,
		default=None,
		help="Optional explicit path to <dataset_id>_hf_dataset (HuggingFace dataset directory).",
	)
	parser.add_argument(
		"--adata_path",
		type=str,
		default=None,
		help="Optional explicit path to <dataset_id>.h5ad used to define dataset gene universe/order.",
	)
	parser.add_argument(
		"--token_dict_path",
		type=str,
		default=str(TOKEN_DICT_PATH),
		help="Path to token_dictionary.pkl used for token->gene conversion (default: resources/vocab).",
	)
	parser.add_argument(
		"--gene_name_column",
		type=str,
		default="gene_symbols",
		help="When adata_path is provided, prefer this var column as gene names if present.",
	)
	parser.add_argument(
		"--use_adata_gene_universe",
		action="store_true",
		default=True,
		help="Use genes from adata var table (default behavior matching notebook).",
	)
	parser.add_argument(
		"--no_use_adata_gene_universe",
		action="store_false",
		dest="use_adata_gene_universe",
		help="Disable adata-based gene universe and fallback to token dictionary genes.",
	)
	parser.add_argument(
		"--strip_special_tokens",
		action="store_true",
		default=True,
		help="Remove first/last token from each input_ids sequence (default behavior matching notebook).",
	)
	parser.add_argument(
		"--no_strip_special_tokens",
		action="store_false",
		dest="strip_special_tokens",
		help="Disable trimming of first/last token in input_ids.",
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
		help="Model/run identifier used in output filenames.",
	)
	parser.add_argument(
		"--output_tag",
		type=str,
		default=None,
		help="Optional suffix for output filenames. Default: <model_id>_<dataset_id>",
	)
	parser.add_argument("--output_dir", type=str, default="./go_embedding_context_out", help="Directory for outputs.")
	parser.add_argument(
		"--min_genes_per_term",
		type=int,
		default=5,
		help="Minimum mapped genes required to keep a GO term.",
	)
	parser.add_argument("--umap_n_neighbors", type=int, default=15, help="UMAP n_neighbors parameter.")
	parser.add_argument("--umap_min_dist", type=float, default=0.1, help="UMAP min_dist parameter.")
	parser.add_argument(
		"--umap_metric",
		type=str,
		default="euclidean",
		help="UMAP distance metric.",
	)
	parser.add_argument(
		"--umap_pca_components",
		type=int,
		default=50,
		help="Number of PCA components before UMAP (default 50).",
	)
	parser.add_argument("--plot_point_size", type=float, default=10.0, help="Scatter point size for plots.")
	parser.add_argument("--plot_alpha", type=float, default=0.85, help="Scatter alpha for plots.")
	parser.add_argument("--seed", type=int, default=42, help="Random seed.")
	return parser.parse_args()


def set_seed(seed: int) -> None:
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)


def _namespace_from_term(term: str) -> str:
	term_upper = str(term).upper()
	for prefix, label in GO_NAMESPACES.items():
		if term_upper.startswith(prefix):
			return label
	return "UNKNOWN"


def _resolve_input_paths(args: argparse.Namespace) -> Tuple[Path, Path, Optional[Path]]:
	base = Path(args.sota_output_root) / args.run_id / args.dataset_id

	context_emb_path = Path(args.context_emb_path) if args.context_emb_path else base / f"{args.dataset_id}_gene_embeddings.npy"
	hf_dataset_path = Path(args.hf_dataset_path) if args.hf_dataset_path else base / f"{args.dataset_id}_hf_dataset"
	adata_path = Path(args.adata_path) if args.adata_path else base / f"{args.dataset_id}.h5ad"

	if not context_emb_path.exists():
		raise FileNotFoundError(f"Contextual embedding file not found: {context_emb_path}")
	if not hf_dataset_path.exists():
		raise FileNotFoundError(f"HF dataset path not found: {hf_dataset_path}")

	if args.use_adata_gene_universe and not adata_path.exists():
		raise FileNotFoundError(
			f"use_adata_gene_universe is enabled, but adata file is missing: {adata_path}"
		)

	return context_emb_path, hf_dataset_path, adata_path if adata_path.exists() else None


def _load_token_dict(path: str) -> Dict[str, int]:
	if not os.path.exists(path):
		fallback_path = str(TOKEN_DICT_PATH)
		if os.path.exists(fallback_path):
			path = fallback_path
		else:
			raise FileNotFoundError(f"token_dictionary.pkl not found at {path} and fallback {fallback_path}")

	with open(path, "rb") as f:
		token_dict = pickle.load(f)
	return token_dict


def _load_gene_universe(
	adata_path: Optional[Path],
	token_dict: Dict[str, int],
	use_adata_gene_universe: bool,
	gene_name_column: str,
) -> Tuple[List[str], Optional[List[str]], Optional[List[str]]]:
	if use_adata_gene_universe:
		if adata_path is None:
			raise ValueError("use_adata_gene_universe is enabled but adata_path is unavailable.")
		import scanpy as sc

		adata = sc.read_h5ad(adata_path)
		ensembl_ids = adata.var_names.astype(str).tolist()
		if gene_name_column in adata.var.columns:
			symbols = adata.var[gene_name_column].astype(str).tolist()
		else:
			symbols = ensembl_ids

		# Keep a stable unique symbol universe for downstream indexing.
		seen = set()
		gene_names: List[str] = []
		for sym in symbols:
			sym_clean = str(sym).strip()
			if not sym_clean or sym_clean.lower() == "nan" or sym_clean in seen:
				continue
			seen.add(sym_clean)
			gene_names.append(sym_clean)

		return gene_names, ensembl_ids, symbols

	# Fallback: use all non-special genes in token dictionary.
	gene_names = [gene for gene in token_dict.keys() if not str(gene).startswith("<")]
	return gene_names, None, None


def _lookup_gene_idx(gene: str, gene_to_idx: Dict[str, int]) -> Optional[int]:
	if gene in gene_to_idx:
		return gene_to_idx[gene]
	gene_up = gene.upper()
	if gene_up in gene_to_idx:
		return gene_to_idx[gene_up]
	gene_cap = gene.capitalize()
	if gene_cap in gene_to_idx:
		return gene_to_idx[gene_cap]
	return None


def _build_token_to_gene_idx(
	token_dict: Dict[str, int],
	gene_to_idx: Dict[str, int],
	adata_var_names: Optional[List[str]],
	adata_symbols: Optional[List[str]],
) -> Dict[int, int]:
	# Exact token -> var index conversion via the AnnData Ensembl IDs and symbols.
	if adata_var_names is not None and adata_symbols is not None:
		token_to_gene = {v: k for k, v in token_dict.items()}
		gene_to_varidx = {gene: i for i, gene in enumerate(adata_var_names)}

		token_to_varidx = {
			tok: gene_to_varidx[token_to_gene[tok]]
			for tok in token_to_gene
			if token_to_gene[tok] in gene_to_varidx
		}

		token_to_gene_idx: Dict[int, int] = {}
		for tok, varidx in token_to_varidx.items():
			gene_symbol = str(adata_symbols[varidx]).strip()
			if not gene_symbol or gene_symbol.lower() == "nan":
				continue
			gene_idx = _lookup_gene_idx(gene_symbol, gene_to_idx)
			if gene_idx is not None:
				token_to_gene_idx[int(tok)] = gene_idx
		return token_to_gene_idx

	# Fallback when adata-based Ensembl->symbol mapping is unavailable.
	token_to_gene = {int(v): str(k) for k, v in token_dict.items()}
	token_to_gene_idx: Dict[int, int] = {}
	for token_id, gene in token_to_gene.items():
		if gene.startswith("<"):
			continue
		gene_idx = _lookup_gene_idx(gene, gene_to_idx)
		if gene_idx is not None:
			token_to_gene_idx[token_id] = gene_idx
	return token_to_gene_idx


def _slice_input_ids(input_ids: Sequence[int], strip_special_tokens: bool) -> np.ndarray:
	token_ids = np.asarray(input_ids, dtype=np.int64)
	if strip_special_tokens and token_ids.shape[0] >= 2:
		token_ids = token_ids[1:-1]
	return token_ids


def aggregate_contextual_gene_embeddings(
	dataset,
	context_emb: np.ndarray,
	token_to_gene_idx: Dict[int, int],
	n_genes: int,
	strip_special_tokens: bool,
) -> Tuple[np.ndarray, np.ndarray, str]:
	"""Return gene-level contextualized embeddings, gene occurrence counts, and mode.

	Modes
	- context_emb.ndim == 2: per-token scalar signal -> gene vectors become cell-wise signals (genes x cells).
	- context_emb.ndim == 3: per-token vector signal -> gene vectors are averaged embedding vectors (genes x dim).
	"""
	n_cells = len(dataset)
	if context_emb.shape[0] != n_cells:
		raise ValueError(
			f"Mismatched cell count between context_emb ({context_emb.shape[0]}) and dataset ({n_cells})."
		)

	gene_counts = np.zeros(n_genes, dtype=np.int64)

	if context_emb.ndim == 2:
		rows: List[int] = []
		cols: List[int] = []
		data: List[float] = []

		for cell_idx, example in enumerate(dataset):
			token_ids = _slice_input_ids(example["input_ids"], strip_special_tokens)
			seq_len = token_ids.shape[0]
			if seq_len == 0:
				continue

			cell_values = np.asarray(context_emb[cell_idx, :seq_len], dtype=np.float32)
			mapped_gene_idx = np.asarray([token_to_gene_idx.get(int(t), -1) for t in token_ids], dtype=np.int64)
			valid_mask = mapped_gene_idx >= 0
			if not valid_mask.any():
				continue

			valid_gene_idx = mapped_gene_idx[valid_mask]
			valid_values = cell_values[valid_mask]

			rows.extend([cell_idx] * valid_gene_idx.shape[0])
			cols.extend(valid_gene_idx.tolist())
			data.extend(valid_values.tolist())
			np.add.at(gene_counts, valid_gene_idx, 1)

		cell_gene_mat = sp.csr_matrix((data, (rows, cols)), shape=(n_cells, n_genes), dtype=np.float32)
		# Genes x cells; each row becomes a dataset-specific contextual signal for one gene.
		gene_context = cell_gene_mat.transpose().tocsr().astype(np.float32)
		return gene_context, gene_counts, "scalar-per-token"

	if context_emb.ndim == 3:
		emb_dim = int(context_emb.shape[2])
		gene_sum = np.zeros((n_genes, emb_dim), dtype=np.float64)

		for cell_idx, example in enumerate(dataset):
			token_ids = _slice_input_ids(example["input_ids"], strip_special_tokens)
			seq_len = token_ids.shape[0]
			if seq_len == 0:
				continue

			token_vectors = np.asarray(context_emb[cell_idx, :seq_len, :], dtype=np.float32)
			mapped_gene_idx = np.asarray([token_to_gene_idx.get(int(t), -1) for t in token_ids], dtype=np.int64)
			valid_mask = mapped_gene_idx >= 0
			if not valid_mask.any():
				continue

			valid_gene_idx = mapped_gene_idx[valid_mask]
			valid_vectors = token_vectors[valid_mask]

			np.add.at(gene_sum, valid_gene_idx, valid_vectors)
			np.add.at(gene_counts, valid_gene_idx, 1)

		gene_context = np.zeros((n_genes, emb_dim), dtype=np.float32)
		nonzero = gene_counts > 0
		if np.any(nonzero):
			gene_context[nonzero] = (gene_sum[nonzero] / gene_counts[nonzero, None]).astype(np.float32)
		return gene_context, gene_counts, "vector-per-token"

	raise ValueError(f"Unsupported context_emb ndim={context_emb.ndim}. Expected 2 or 3.")


def load_go_terms(gmt_path: str) -> Dict[str, List[str]]:
	"""Load GO term -> unique genes for BP/CC/MF namespaces only."""
	gmt = dc.pp.read_gmt(gmt_path)
	gmt["namespace"] = gmt["source"].apply(_namespace_from_term)
	gmt = gmt[gmt["namespace"].isin(["BP", "CC", "MF"])]

	term_to_genes: Dict[str, set] = defaultdict(set)
	for _, row in gmt.iterrows():
		term = str(row["source"])
		gene = str(row["target"])
		term_to_genes[term].add(gene)

	return {term: sorted(genes) for term, genes in term_to_genes.items()}


def build_term_embeddings_from_context(
	term_to_genes: Dict[str, List[str]],
	gene_context,
	gene_to_idx: Dict[str, int],
	gene_counts: np.ndarray,
	min_genes_per_term: int,
) -> Tuple[np.ndarray, List[str], List[str], List[int], List[int]]:
	"""Average contextualized gene representations within each GO term."""
	kept_terms: List[str] = []
	kept_namespaces: List[str] = []
	used_gene_counts: List[int] = []
	total_gene_counts: List[int] = []
	term_vectors: List[np.ndarray] = []

	is_sparse_context = sp.issparse(gene_context)

	for term, genes in term_to_genes.items():
		idx_list: List[int] = []
		for gene in genes:
			idx = _lookup_gene_idx(gene, gene_to_idx)
			if idx is None:
				continue
			if idx < 0 or idx >= gene_counts.shape[0]:
				continue
			if gene_counts[idx] <= 0:
				continue
			idx_list.append(idx)

		if len(idx_list) < min_genes_per_term:
			continue

		if is_sparse_context:
			avg_vector = np.asarray(gene_context[idx_list].mean(axis=0)).ravel().astype(np.float32)
		else:
			avg_vector = np.asarray(gene_context[idx_list].mean(axis=0), dtype=np.float32)

		namespace = _namespace_from_term(term)
		if namespace not in VALID_NAMESPACES:
			continue

		kept_terms.append(term)
		kept_namespaces.append(namespace)
		used_gene_counts.append(len(idx_list))
		total_gene_counts.append(len(genes))
		term_vectors.append(avg_vector)

	if not term_vectors:
		raise ValueError("No GO terms left after mapping contextualized genes and min_genes_per_term filtering.")

	return (
		np.vstack(term_vectors).astype(np.float32),
		kept_terms,
		kept_namespaces,
		used_gene_counts,
		total_gene_counts,
	)


def _effective_neighbors(n_samples: int, requested_neighbors: int) -> int:
	if n_samples <= 2:
		return 2
	return max(2, min(requested_neighbors, n_samples - 1))


def _effective_pca_components(n_samples: int, n_features: int, requested_components: int) -> int:
	max_components = min(n_samples - 1, n_features)
	if max_components < 2:
		raise ValueError("Not enough GO terms/features to run PCA->UMAP (need at least 3 terms).")
	return max(2, min(requested_components, max_components))


def compute_umap_coordinates(
	term_matrix: np.ndarray,
	seed: int,
	n_neighbors: int,
	min_dist: float,
	metric: str,
	pca_components: int,
) -> Tuple[np.ndarray, np.ndarray, int]:
	"""Return (pca_for_umap, umap_from_pca, used_pca_components)."""
	try:
		from umap import UMAP
	except ImportError as exc:
		raise ImportError("UMAP is required. Install it with: pip install umap-learn") from exc

	n_samples, n_features = term_matrix.shape
	eff_neighbors = _effective_neighbors(n_samples, n_neighbors)
	eff_pca_components = _effective_pca_components(n_samples, n_features, pca_components)

	pca_for_umap = PCA(n_components=eff_pca_components, random_state=seed).fit_transform(term_matrix)
	umap_from_pca = UMAP(
		n_components=2,
		n_neighbors=eff_neighbors,
		min_dist=min_dist,
		metric=metric,
		random_state=seed,
	).fit_transform(pca_for_umap)

	return pca_for_umap.astype(np.float32), umap_from_pca.astype(np.float32), eff_pca_components


def plot_namespace_scatter(
	coords: np.ndarray,
	namespaces: Sequence[str],
	title: str,
	output_path: Path,
	point_size: float,
	alpha: float,
	xlabel: str = "dim1",
	ylabel: str = "dim2",
) -> None:
	palette = {"BP": "#1f77b4", "CC": "#ff7f0e", "MF": "#2ca02c"}
	fig, ax = plt.subplots(figsize=(8, 6), dpi=160)

	for ns in ["BP", "CC", "MF"]:
		idx = np.where(np.asarray(namespaces) == ns)[0]
		if idx.size == 0:
			continue
		ax.scatter(
			coords[idx, 0],
			coords[idx, 1],
			s=point_size,
			alpha=alpha,
			label=ns,
			c=palette[ns],
			edgecolors="none",
		)

	ax.set_title(title)
	ax.set_xlabel(xlabel)
	ax.set_ylabel(ylabel)
	ax.legend(title="GO namespace", frameon=False)
	fig.tight_layout()
	fig.savefig(output_path)
	plt.close(fig)


def main() -> None:
	args = parse_args()
	set_seed(args.seed)

	context_emb_path, hf_dataset_path, adata_path = _resolve_input_paths(args)
	output_dir = Path(args.output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)

	output_tag = args.output_tag if args.output_tag else f"{args.model_id}_{args.dataset_id}"

	print("Loading token dictionary and gene universe ...")
	token_dict = _load_token_dict(args.token_dict_path)
	gene_names, adata_var_names, adata_symbols = _load_gene_universe(
		adata_path=adata_path,
		token_dict=token_dict,
		use_adata_gene_universe=args.use_adata_gene_universe,
		gene_name_column=args.gene_name_column,
	)
	gene_to_idx = {gene: i for i, gene in enumerate(gene_names)}
	token_to_gene_idx = _build_token_to_gene_idx(
		token_dict=token_dict,
		gene_to_idx=gene_to_idx,
		adata_var_names=adata_var_names,
		adata_symbols=adata_symbols,
	)

	print(f"Loading contextual embedding array from: {context_emb_path}")
	context_emb = np.load(context_emb_path)
	print(f"context_emb shape: {context_emb.shape}")

	print(f"Loading HF dataset from: {hf_dataset_path}")
	dataset = load_from_disk(str(hf_dataset_path))

	print("Aggregating contextualized gene embeddings ...")
	gene_context, gene_counts, context_mode = aggregate_contextual_gene_embeddings(
		dataset=dataset,
		context_emb=context_emb,
		token_to_gene_idx=token_to_gene_idx,
		n_genes=len(gene_names),
		strip_special_tokens=args.strip_special_tokens,
	)

	n_genes_observed = int(np.sum(gene_counts > 0))
	print(f"Context mode: {context_mode}")
	print(f"Genes in universe: {len(gene_names)} | Genes observed in dataset: {n_genes_observed}")

	print("Reading GO terms from GMT ...")
	term_to_genes = load_go_terms(args.gmt_path)
	print(f"Loaded {len(term_to_genes)} BP/CC/MF GO terms before filtering.")

	print("Averaging contextualized gene embeddings per GO term ...")
	term_matrix, terms, namespaces, used_counts, total_counts = build_term_embeddings_from_context(
		term_to_genes=term_to_genes,
		gene_context=gene_context,
		gene_to_idx=gene_to_idx,
		gene_counts=gene_counts,
		min_genes_per_term=args.min_genes_per_term,
	)

	print(
		f"Kept {term_matrix.shape[0]} GO terms with at least {args.min_genes_per_term} mapped genes; "
		f"embedding dim={term_matrix.shape[1]}."
	)

	print(f"Running PCA->UMAP with PCA components={args.umap_pca_components} ...")
	pca_for_umap, umap_from_pca, used_pca_dims = compute_umap_coordinates(
		term_matrix=term_matrix,
		seed=args.seed,
		n_neighbors=args.umap_n_neighbors,
		min_dist=args.umap_min_dist,
		metric=args.umap_metric,
		pca_components=args.umap_pca_components,
	)

	pca_path = output_dir / f"go_term_pca_for_umap_{output_tag}.csv"
	with pca_path.open("w", newline="") as f:
		writer = csv.writer(f)
		header = ["go_term", "namespace", "mapped_genes", "total_genes"] + [f"pc{i+1}" for i in range(used_pca_dims)]
		writer.writerow(header)
		for term, ns, used_n, total_n, coord in zip(terms, namespaces, used_counts, total_counts, pca_for_umap):
			writer.writerow([term, ns, int(used_n), int(total_n)] + [float(v) for v in coord.tolist()])

	umap_from_pca_path = output_dir / f"go_term_umap_from_pca_{output_tag}.csv"
	with umap_from_pca_path.open("w", newline="") as f:
		writer = csv.writer(f)
		writer.writerow(["go_term", "namespace", "umap1", "umap2"])
		for term, ns, coord in zip(terms, namespaces, umap_from_pca):
			writer.writerow([term, ns, float(coord[0]), float(coord[1])])

	umap_from_pca_png = output_dir / f"go_term_umap_from_pca_{output_tag}.png"
	plot_namespace_scatter(
		coords=umap_from_pca,
		namespaces=namespaces,
		title=f"GO term UMAP (from PCA, n_components={used_pca_dims})",
		output_path=umap_from_pca_png,
		point_size=args.plot_point_size,
		alpha=args.plot_alpha,
		xlabel="UMAP1",
		ylabel="UMAP2",
	)

	summary_path = output_dir / f"go_term_umap_summary_{output_tag}.txt"
	with summary_path.open("w") as f:
		f.write(f"dataset_id: {args.dataset_id}\n")
		f.write(f"run_id: {args.run_id}\n")
		f.write(f"model_id: {args.model_id}\n")
		f.write(f"output_tag: {output_tag}\n")
		f.write(f"context_emb_path: {context_emb_path}\n")
		f.write(f"hf_dataset_path: {hf_dataset_path}\n")
		f.write(f"adata_path: {adata_path}\n")
		f.write(f"context_mode: {context_mode}\n")
		f.write(f"use_adata_gene_universe: {args.use_adata_gene_universe}\n")
		f.write(f"gene_name_column: {args.gene_name_column}\n")
		f.write(f"strip_special_tokens: {args.strip_special_tokens}\n")
		f.write(f"n_genes_universe: {len(gene_names)}\n")
		f.write(f"n_genes_observed: {n_genes_observed}\n")
		f.write(f"n_terms_kept: {term_matrix.shape[0]}\n")
		f.write(f"embedding_dim: {term_matrix.shape[1]}\n")
		f.write(f"umap_n_neighbors: {args.umap_n_neighbors}\n")
		f.write(f"umap_min_dist: {args.umap_min_dist}\n")
		f.write(f"umap_metric: {args.umap_metric}\n")
		f.write(f"umap_pca_components_used: {used_pca_dims}\n")

	print("Namespace counts in GO term set:")
	ns_counts = defaultdict(int)
	for ns in namespaces:
		ns_counts[str(ns)] += 1
	for ns in ["BP", "CC", "MF"]:
		print(f"{ns}\t{ns_counts.get(ns, 0)}")

	print(f"Saved PCA-for-UMAP coordinates to: {pca_path}")
	print(f"Saved PCA->UMAP coordinates to: {umap_from_pca_path}")
	print(f"Saved PCA->UMAP plot to: {umap_from_pca_png}")
	print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
	main()
