"""Cluster GO terms using averaged static gene embeddings.

Workflow
1. Load static gene embeddings from either a model checkpoint (input token embedding
   table of a HuggingFace ``BertForMaskedLM`` checkpoint, default resources/weights/scOntoFM) or
   precomputed scGPT .npy files (``--embedding_source scgpt``, produced outside this repo).
2. Build one embedding per GO term by averaging embeddings of genes in that term.
3. Run a simple KMeans clustering and compare clusters with GO namespaces (BP/CC/MF).

Outputs
- go_term_cluster_assignments_<model_id>.csv
- go_term_cluster_contingency_<model_id>.csv
- go_term_cluster_summary_<model_id>.txt
- go_term_pca2d_<model_id>.csv
- go_term_umap_direct_<model_id>.csv
- go_term_umap_from_pca_<model_id>.csv
- go_term_umap_direct_<model_id>.png
- go_term_umap_from_pca_<model_id>.png
- go_term_umap_comparison_<model_id>.png

The GO GMT (``c5.go.v2025.1.Hs.symbols.gmt``) is not redistributed; download it from
https://www.gsea-msigdb.org/gsea/msigdb/human/collections.jsp into this folder.

Example (from this folder)
  python GO_embedding.py --model_path ../../resources/weights/scOntoFM \
      --embedding_source model --gmt_path c5.go.v2025.1.Hs.symbols.gmt \
      --output_dir ./go_embedding_out --model_id scOntoFM
"""

import argparse
import csv
import os
import pickle
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import decoupler as dc
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from transformers import BertForMaskedLM

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT))
from scOntoFM.paths import MODEL_DIR, VOCAB_DIR  # noqa: E402


GO_NAMESPACES = {"GOBP": "BP", "GOCC": "CC", "GOMF": "MF"}
NAMESPACE_TO_ID = {"BP": 0, "CC": 1, "MF": 2}


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="GO term embedding + clustering (BP/CC/MF)")
	parser.add_argument(
		"--model_path",
		type=str,
		required=False,
		default=str(MODEL_DIR),
		help="Path to pretrained BertForMaskedLM checkpoint (for embedding_source=model; default: resources/weights/scOntoFM).",
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
		help="Use embedding table from model checkpoint or local scGPT npy files.",
	)
	parser.add_argument(
		"--scgpt_emb_path",
		type=str,
		default="scGPT_gene_embeddings.npy",
		help="Path to scGPT embedding matrix (.npy), row-aligned with --scgpt_gene_path.",
	)
	parser.add_argument(
		"--scgpt_gene_path",
		type=str,
		default="scGPT_gene_names.npy",
		help="Path to scGPT gene names (.npy).",
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
		help="Run identifier used in saved outputs.",
	)
	parser.add_argument("--output_dir", type=str, default="./go_embedding_out", help="Directory for outputs.")
	parser.add_argument(
		"--min_genes_per_term",
		type=int,
		default=5,
		help="Minimum mapped genes required to keep a GO term.",
	)
	parser.add_argument(
		"--n_clusters",
		type=int,
		default=3,
		help="Number of clusters for KMeans (3 is expected for BP/CC/MF).",
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
		default=30,
		help="Number of PCA components before UMAP for the PCA->UMAP branch.",
	)
	parser.add_argument("--plot_point_size", type=float, default=10.0, help="Scatter point size for plots.")
	parser.add_argument("--plot_alpha", type=float, default=0.85, help="Scatter alpha for plots.")
	parser.add_argument("--seed", type=int, default=42, help="Random seed.")
	parser.add_argument(
		"--device",
		type=str,
		default="cuda" if torch.cuda.is_available() else "cpu",
		help="Device to load model weights (cuda or cpu).",
	)
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


def load_embeddings_and_index(
	args: argparse.Namespace, device: torch.device
) -> Tuple[torch.Tensor, Dict[str, int], Optional[int]]:
	"""Return embedding table, gene->row index map, and optional pad token id."""

	if args.embedding_source == "scgpt":
		emb_np = np.load(args.scgpt_emb_path)
		gene_names = np.load(args.scgpt_gene_path)
		if emb_np.shape[0] != gene_names.shape[0]:
			raise ValueError("scGPT embeddings and gene names have mismatched lengths.")
		gene_to_idx = {str(gene): int(i) for i, gene in enumerate(gene_names)}
		return torch.from_numpy(emb_np), gene_to_idx, None

	if not args.model_path or not args.dict_dir:
		raise ValueError("model_path and dict_dir are required when embedding_source='model'.")

	model = BertForMaskedLM.from_pretrained(
		args.model_path,
		output_attentions=False,
		output_hidden_states=True,
	).to(device)
	token_emb = model.get_input_embeddings().weight.detach().cpu()

	vocab_path = os.path.join(args.dict_dir, "token_dictionary.pkl")
	gene_map_path = os.path.join(args.dict_dir, "gene_name_id_dict.pkl")
	if not os.path.exists(gene_map_path):
		fallback_path = os.path.join(str(VOCAB_DIR), "gene_name_id_dict.pkl")
		if os.path.exists(fallback_path):
			gene_map_path = fallback_path

	with open(vocab_path, "rb") as f:
		vocab = pickle.load(f)
	with open(gene_map_path, "rb") as f:
		gene_name_id = pickle.load(f)

	gene_to_idx = {gene: vocab[token] for gene, token in gene_name_id.items() if token in vocab}
	pad_token_id = vocab.get("<pad>") if isinstance(vocab, dict) else None
	return token_emb, gene_to_idx, pad_token_id


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


def _lookup_token_id(gene: str, gene_to_idx: Dict[str, int]) -> Optional[int]:
	"""Try common symbol variants for robust dictionary matching."""
	if gene in gene_to_idx:
		return gene_to_idx[gene]
	gene_up = gene.upper()
	if gene_up in gene_to_idx:
		return gene_to_idx[gene_up]
	gene_cap = gene.capitalize()
	if gene_cap in gene_to_idx:
		return gene_to_idx[gene_cap]
	return None


def build_term_embeddings(
	term_to_genes: Dict[str, List[str]],
	embedding_table: torch.Tensor,
	gene_to_idx: Dict[str, int],
	min_genes_per_term: int,
	pad_token_id: Optional[int] = None,
) -> Tuple[np.ndarray, List[str], List[str], List[int], List[int]]:
	"""Average gene embeddings within each GO term and return term-level matrix."""
	kept_terms: List[str] = []
	kept_namespaces: List[str] = []
	used_gene_counts: List[int] = []
	total_gene_counts: List[int] = []
	term_vectors: List[np.ndarray] = []

	for term, genes in term_to_genes.items():
		vectors: List[torch.Tensor] = []
		for gene in genes:
			token_id = _lookup_token_id(gene, gene_to_idx)
			if token_id is None:
				continue
			if pad_token_id is not None and token_id == pad_token_id:
				continue
			if token_id >= embedding_table.shape[0]:
				continue
			vectors.append(embedding_table[token_id])

		if len(vectors) < min_genes_per_term:
			continue

		avg_vector = torch.stack(vectors, dim=0).mean(dim=0).cpu().numpy()
		namespace = _namespace_from_term(term)
		if namespace not in NAMESPACE_TO_ID:
			continue

		kept_terms.append(term)
		kept_namespaces.append(namespace)
		used_gene_counts.append(len(vectors))
		total_gene_counts.append(len(genes))
		term_vectors.append(avg_vector)

	if not term_vectors:
		raise ValueError("No GO terms left after mapping genes to embeddings and min_genes_per_term filtering.")

	return (
		np.vstack(term_vectors).astype(np.float32),
		kept_terms,
		kept_namespaces,
		used_gene_counts,
		total_gene_counts,
	)


def compute_contingency(namespaces: Sequence[str], clusters: Sequence[int]) -> Tuple[List[str], List[int], np.ndarray]:
	ns_order = ["BP", "CC", "MF"]
	cluster_ids = sorted(set(int(c) for c in clusters))
	matrix = np.zeros((len(ns_order), len(cluster_ids)), dtype=int)
	cluster_to_col = {cluster_id: i for i, cluster_id in enumerate(cluster_ids)}
	ns_to_row = {ns: i for i, ns in enumerate(ns_order)}

	for ns, cluster_id in zip(namespaces, clusters):
		if ns in ns_to_row and int(cluster_id) in cluster_to_col:
			matrix[ns_to_row[ns], cluster_to_col[int(cluster_id)]] += 1

	return ns_order, cluster_ids, matrix


def cluster_purity(contingency: np.ndarray) -> float:
	if contingency.size == 0:
		return float("nan")
	total = contingency.sum()
	if total == 0:
		return float("nan")
	return float(np.max(contingency, axis=0).sum() / total)


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
	"""Return (direct_umap, pca_umap, used_pca_components)."""
	try:
		from umap import UMAP
	except ImportError as exc:
		raise ImportError("UMAP is required. Install it with: pip install umap-learn") from exc

	n_samples, n_features = term_matrix.shape
	eff_neighbors = _effective_neighbors(n_samples, n_neighbors)
	eff_pca_components = _effective_pca_components(n_samples, n_features, pca_components)

	umap_direct = UMAP(
		n_components=2,
		n_neighbors=eff_neighbors,
		min_dist=min_dist,
		metric=metric,
		random_state=seed,
	).fit_transform(term_matrix)

	pca_for_umap = PCA(n_components=eff_pca_components, random_state=seed).fit_transform(term_matrix)
	umap_from_pca = UMAP(
		n_components=2,
		n_neighbors=eff_neighbors,
		min_dist=min_dist,
		metric=metric,
		random_state=seed,
	).fit_transform(pca_for_umap)

	return umap_direct.astype(np.float32), umap_from_pca.astype(np.float32), eff_pca_components


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


def plot_umap_comparison(
	umap_direct: np.ndarray,
	umap_from_pca: np.ndarray,
	namespaces: Sequence[str],
	output_path: Path,
	point_size: float,
	alpha: float,
) -> None:
	palette = {"BP": "#1f77b4", "CC": "#ff7f0e", "MF": "#2ca02c"}
	fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=160)

	for ax, coords, title in [
		(axes[0], umap_direct, "UMAP (direct on GO-term embeddings)"),
		(axes[1], umap_from_pca, "UMAP (after PCA on GO-term embeddings)"),
	]:
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
		ax.set_xlabel("UMAP1")
		ax.set_ylabel("UMAP2")

	handles, labels = axes[0].get_legend_handles_labels()
	if handles:
		fig.legend(handles, labels, title="GO namespace", loc="upper center", ncol=3, frameon=False)
	fig.tight_layout(rect=[0, 0, 1, 0.95])
	fig.savefig(output_path)
	plt.close(fig)


def main() -> None:
	args = parse_args()
	set_seed(args.seed)
	device = torch.device(args.device)

	output_dir = Path(args.output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)

	print(f"Loading embeddings from source={args.embedding_source} ...")
	embedding_table, gene_to_idx, pad_token_id = load_embeddings_and_index(args, device)

	print("Reading GO terms from GMT ...")
	term_to_genes = load_go_terms(args.gmt_path)
	print(f"Loaded {len(term_to_genes)} BP/CC/MF GO terms before filtering.")

	print("Averaging gene embeddings per GO term ...")
	term_matrix, terms, namespaces, used_counts, total_counts = build_term_embeddings(
		term_to_genes=term_to_genes,
		embedding_table=embedding_table,
		gene_to_idx=gene_to_idx,
		min_genes_per_term=args.min_genes_per_term,
		pad_token_id=pad_token_id,
	)

	print(
		f"Kept {term_matrix.shape[0]} GO terms with at least {args.min_genes_per_term} mapped genes; "
		f"embedding dim={term_matrix.shape[1]}."
	)

	y_true = np.asarray([NAMESPACE_TO_ID[ns] for ns in namespaces], dtype=int)

	print(f"Running KMeans clustering with n_clusters={args.n_clusters} ...")
	kmeans = KMeans(n_clusters=args.n_clusters, random_state=args.seed, n_init=20)
	cluster_ids = kmeans.fit_predict(term_matrix)

	ari = adjusted_rand_score(y_true, cluster_ids)
	nmi = normalized_mutual_info_score(y_true, cluster_ids)
	sil = float("nan")
	if term_matrix.shape[0] > args.n_clusters and len(set(cluster_ids)) > 1:
		sil = silhouette_score(term_matrix, cluster_ids, metric="euclidean")

	ns_order, cluster_order, contingency = compute_contingency(namespaces, cluster_ids)
	purity = cluster_purity(contingency)

	pca2d = PCA(n_components=2, random_state=args.seed).fit_transform(term_matrix)
	umap_direct, umap_from_pca, used_pca_dims = compute_umap_coordinates(
		term_matrix=term_matrix,
		seed=args.seed,
		n_neighbors=args.umap_n_neighbors,
		min_dist=args.umap_min_dist,
		metric=args.umap_metric,
		pca_components=args.umap_pca_components,
	)

	assign_path = output_dir / f"go_term_cluster_assignments_{args.model_id}.csv"
	with assign_path.open("w", newline="") as f:
		writer = csv.writer(f)
		writer.writerow(["go_term", "namespace", "cluster", "mapped_genes", "total_genes", "pca1", "pca2"])
		for term, ns, cluster_id, used_n, total_n, coord in zip(
			terms, namespaces, cluster_ids, used_counts, total_counts, pca2d
		):
			writer.writerow([term, ns, int(cluster_id), int(used_n), int(total_n), float(coord[0]), float(coord[1])])

	contingency_path = output_dir / f"go_term_cluster_contingency_{args.model_id}.csv"
	with contingency_path.open("w", newline="") as f:
		writer = csv.writer(f)
		writer.writerow(["namespace"] + [f"cluster_{cid}" for cid in cluster_order])
		for i, ns in enumerate(ns_order):
			writer.writerow([ns] + contingency[i].astype(int).tolist())

	pca_path = output_dir / f"go_term_pca2d_{args.model_id}.csv"
	with pca_path.open("w", newline="") as f:
		writer = csv.writer(f)
		writer.writerow(["go_term", "namespace", "pca1", "pca2"])
		for term, ns, coord in zip(terms, namespaces, pca2d):
			writer.writerow([term, ns, float(coord[0]), float(coord[1])])

	umap_direct_path = output_dir / f"go_term_umap_direct_{args.model_id}.csv"
	with umap_direct_path.open("w", newline="") as f:
		writer = csv.writer(f)
		writer.writerow(["go_term", "namespace", "umap1", "umap2"])
		for term, ns, coord in zip(terms, namespaces, umap_direct):
			writer.writerow([term, ns, float(coord[0]), float(coord[1])])

	umap_from_pca_path = output_dir / f"go_term_umap_from_pca_{args.model_id}.csv"
	with umap_from_pca_path.open("w", newline="") as f:
		writer = csv.writer(f)
		writer.writerow(["go_term", "namespace", "umap1", "umap2"])
		for term, ns, coord in zip(terms, namespaces, umap_from_pca):
			writer.writerow([term, ns, float(coord[0]), float(coord[1])])

	umap_direct_png = output_dir / f"go_term_umap_direct_{args.model_id}.png"
	umap_from_pca_png = output_dir / f"go_term_umap_from_pca_{args.model_id}.png"
	umap_compare_png = output_dir / f"go_term_umap_comparison_{args.model_id}.png"

	plot_namespace_scatter(
		coords=umap_direct,
		namespaces=namespaces,
		title="GO term UMAP (direct)",
		output_path=umap_direct_png,
		point_size=args.plot_point_size,
		alpha=args.plot_alpha,
		xlabel="UMAP1",
		ylabel="UMAP2",
	)
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
	plot_umap_comparison(
		umap_direct=umap_direct,
		umap_from_pca=umap_from_pca,
		namespaces=namespaces,
		output_path=umap_compare_png,
		point_size=args.plot_point_size,
		alpha=args.plot_alpha,
	)

	summary_path = output_dir / f"go_term_cluster_summary_{args.model_id}.txt"
	with summary_path.open("w") as f:
		f.write(f"model_id: {args.model_id}\n")
		f.write(f"embedding_source: {args.embedding_source}\n")
		f.write(f"n_terms_kept: {term_matrix.shape[0]}\n")
		f.write(f"embedding_dim: {term_matrix.shape[1]}\n")
		f.write(f"kmeans_n_clusters: {args.n_clusters}\n")
		f.write(f"umap_n_neighbors: {args.umap_n_neighbors}\n")
		f.write(f"umap_min_dist: {args.umap_min_dist}\n")
		f.write(f"umap_metric: {args.umap_metric}\n")
		f.write(f"umap_pca_components_used: {used_pca_dims}\n")
		f.write(f"adjusted_rand_index: {ari:.4f}\n")
		f.write(f"normalized_mutual_info: {nmi:.4f}\n")
		f.write(f"silhouette_score: {sil:.4f}\n")
		f.write(f"cluster_purity: {purity:.4f}\n")

	print("Cluster-vs-namespace contingency table:")
	header = "\t".join(["namespace"] + [f"cluster_{cid}" for cid in cluster_order])
	print(header)
	for i, ns in enumerate(ns_order):
		row_vals = "\t".join([str(v) for v in contingency[i].astype(int).tolist()])
		print(f"{ns}\t{row_vals}")

	print("Summary metrics:")
	print(f"ARI={ari:.4f} | NMI={nmi:.4f} | Silhouette={sil:.4f} | Purity={purity:.4f}")
	print(f"Saved assignments to: {assign_path}")
	print(f"Saved contingency to: {contingency_path}")
	print(f"Saved PCA coordinates to: {pca_path}")
	print(f"Saved direct UMAP coordinates to: {umap_direct_path}")
	print(f"Saved PCA->UMAP coordinates to: {umap_from_pca_path}")
	print(f"Saved direct UMAP plot to: {umap_direct_png}")
	print(f"Saved PCA->UMAP plot to: {umap_from_pca_png}")
	print(f"Saved UMAP comparison plot to: {umap_compare_png}")
	print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
	main()
