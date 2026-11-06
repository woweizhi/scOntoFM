"""scIB-style clustering / batch-integration metrics on GPU (rapids_singlecell).

For each model, the cell embedding is attached to the AnnData, a kNN graph is built
on GPU, Louvain clustering is optimized over resolutions (by NMI), and the following
metrics are reported (scIB / scGPT definitions):

  bio conservation : NMI_cluster/label, ARI_cluster/label, ASW_label
  batch correction : graph_conn, ASW_label/batch
  aggregates       : avg_bio, avg_batch, overall = 0.6 * avg_bio + 0.4 * avg_batch

Inputs
  <base_dir>/<model>/<data_id>/<data_id>_embeddings.npy   (from scripts/run_inference.py,
                                                          where <model> is its --run_id)
  <adata_dir>/<data_id>.h5ad                              (cells row-aligned with the embedding;
                                                          pass --adata_dir "" to read
                                                          <base_dir>/<model>/<data_id>/<data_id>.h5ad)
Outputs
  <output>/<run_id>/<data_id>_scib_metrics_results.csv    (one row per model)

Example (from this folder)
  python clustering_eval_rapids.py --base_dir ../../output \
      --adata_dir ../../data/pantissue/filtered \
      --model scOntoFM --data_id Heart \
      --batch_key donor_id --label_key cell_type --exclude_label Other \
      --output ./output --run_id scOntoFM
"""

import argparse
import logging
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import scanpy as sc
import rapids_singlecell as rsc
from anndata import AnnData
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_samples,
    silhouette_score,
)

LOG_LEVEL = logging.DEBUG

logging.root.setLevel(LOG_LEVEL)
stream = logging.StreamHandler()
stream.setLevel(LOG_LEVEL)
log = logging.getLogger("pythonConfig")
log.setLevel(LOG_LEVEL)
log.addHandler(stream)


def _to_numpy(x):
    """Convert numpy or CuPy arrays to a host numpy array."""
    try:
        import cupy  # type: ignore

        if isinstance(x, cupy.ndarray):
            return cupy.asnumpy(x)
    except ImportError:
        pass
    return np.asarray(x)


def nmi_score(adata: AnnData, cluster_key: str, label_key: str, average_method: str = "arithmetic") -> float:
    labels_true = _to_numpy(adata.obs[label_key].to_numpy())
    labels_pred = _to_numpy(adata.obs[cluster_key].to_numpy())
    return float(normalized_mutual_info_score(labels_true, labels_pred, average_method=average_method))


def ari_score(adata: AnnData, cluster_key: str, label_key: str) -> float:
    labels_true = _to_numpy(adata.obs[label_key].to_numpy())
    labels_pred = _to_numpy(adata.obs[cluster_key].to_numpy())
    return float(adjusted_rand_score(labels_true, labels_pred))


def silhouette_label(
    adata: AnnData,
    label_key: str,
    embedding_key: str,
    metric: str = "euclidean",
    scale: bool = True,
) -> float:
    emb = _to_numpy(adata.obsm[embedding_key])
    labels = _to_numpy(adata.obs[label_key].to_numpy())
    if np.unique(labels).size < 2:
        return np.nan
    asw = silhouette_score(emb, labels, metric=metric)
    if scale:
        asw = (asw + 1.0) / 2.0
    return float(asw)


def silhouette_batch(
    adata: AnnData,
    batch_key: str,
    label_key: str,
    embed: str,
    metric: str = "euclidean",
    return_all: bool = False,
    scale: bool = True,
    verbose: bool = False,
) -> float:
    """Batch ASW matching scIB: abs per-cell silhouette on batches, averaged per label."""

    if embed not in adata.obsm:
        raise KeyError(f"{embed} not in obsm")

    sil_per_label = []
    for group in adata.obs[label_key].unique():
        adata_group = adata[adata.obs[label_key] == group]
        n_batches = adata_group.obs[batch_key].nunique()
        if (n_batches == 1) or (n_batches == adata_group.shape[0]):
            continue

        emb_subset = _to_numpy(adata_group.obsm[embed])
        batch_subset = _to_numpy(adata_group.obs[batch_key].to_numpy())
        sil = silhouette_samples(emb_subset, batch_subset, metric=metric)
        sil = [abs(i) for i in sil]
        if scale:
            sil = [1.0 - i for i in sil]
        sil_per_label.extend([(group, score) for score in sil])

    if len(sil_per_label) == 0:
        return np.nan if not return_all else (np.nan, None, None)

    sil_df = pd.DataFrame.from_records(sil_per_label, columns=["group", "silhouette_score"])
    sil_means = sil_df.groupby("group").mean()

    if verbose:
        log.debug(sil_means)

    mean_score = float(sil_means["silhouette_score"].mean())
    if return_all:
        return mean_score, sil_means, sil_df
    return mean_score


def graph_connectivity(adata: AnnData, label_key: str, connectivities_key: str = "connectivities") -> float:
    """Average fraction of cells in the largest component per label."""

    if connectivities_key not in adata.obsp:
        raise ValueError(f"{connectivities_key} not found in adata.obsp; compute neighbors first.")

    conn = adata.obsp[connectivities_key]
    conn = _ensure_cpu_sparse(conn)
    if not isinstance(conn, csr_matrix):
        conn = conn.tocsr()

    per_label_scores = []
    labels = adata.obs[label_key]
    for label in labels.unique():
        mask = labels == label
        idx = np.where(mask.to_numpy())[0]
        if idx.size == 0:
            continue
        subgraph = conn[idx[:, None], idx]
        n_components, comp_labels = connected_components(subgraph, directed=False, return_labels=True)
        if n_components == 0:
            continue
        largest = np.max(np.bincount(comp_labels))
        per_label_scores.append(largest / idx.size)

    if len(per_label_scores) == 0:
        return np.nan
    return float(np.mean(per_label_scores))


# code adapted from https://github.com/microsoft/zero-shot-scfoundation/blob/main/sc_foundation_evals/utils.py
# MODIFIED wrapper for all scib metrics from
# https://github.com/bowang-lab/scGPT/blob/5a69912232e214cda1998f78e5b4a7b5ef09fe06/scgpt/utils/util.py#L267
def eval_scib_metrics(
    adata: AnnData,
    batch_key: str = "str_batch",
    label_key: str = "cell_type",
    embedding_key: str = "X_scGPT",
) -> Dict:
    if "neighbors" in adata.uns:
        log.warning(
            "neighbors in adata.uns found\n"
            f"{adata.uns['neighbors']}\n"
            "to make sure the optimal clustering is calculated for the correct embedding, removing neighbors from adata.uns.\n"
            "Overwriting calculation of neighbors with "
            f"rsc.pp.neighbors(adata, use_rep={embedding_key})."
        )
        adata.uns.pop("neighbors", None)
        rsc.pp.neighbors(adata, use_rep=embedding_key)
        log.info(
            "neighbors in adata.uns removed, new neighbors calculated: "
            f"{adata.uns['neighbors']}"
        )

    results_dict = dict()
    opt_louvain_rapids(
        adata,
        label_key=label_key,
        cluster_key="cluster",
        use_rep=embedding_key,
        function=lambda a, c, l: nmi_score(a, c, l, average_method="arithmetic"),
        verbose=False,
        inplace=True,
        force=True,
    )

    results_dict["NMI_cluster/label"] = nmi_score(
        adata,
        "cluster",
        label_key,
        average_method="arithmetic",
    )

    results_dict["ARI_cluster/label"] = ari_score(
        adata,
        "cluster",
        label_key,
    )

    results_dict["ASW_label"] = silhouette_label(
        adata,
        label_key,
        embedding_key,
        metric="euclidean",
    )

    results_dict["graph_conn"] = graph_connectivity(
        adata,
        label_key=label_key,
    )

    batch_scores = [results_dict["graph_conn"]]
    if len(adata.obs[batch_key].unique()) > 1:
        results_dict["ASW_label/batch"] = silhouette_batch(
            adata,
            batch_key,
            label_key,
            embed=embedding_key,
            metric="euclidean",
        )
        batch_scores.append(results_dict["ASW_label/batch"])
    else:
        results_dict["ASW_label/batch"] = np.nan

    results_dict["avg_bio"] = np.mean(
        [
            results_dict["NMI_cluster/label"],
            results_dict["ARI_cluster/label"],
            results_dict["ASW_label"],
        ]
    )

    results_dict["avg_batch"] = np.mean(batch_scores)

    results_dict["overall"] = 0.6 * results_dict["avg_bio"] + 0.4 * results_dict["avg_batch"]

    log.debug("\n".join([f"{k}: {v:.4f}" for k, v in results_dict.items()]))

    results_dict = {k: v for k, v in results_dict.items() if not np.isnan(v)}

    return results_dict


def _ensure_cpu_sparse(mat):
    """Move a sparse matrix from GPU to CPU if it is a CuPy-backed matrix."""
    try:
        import cupyx  # type: ignore

        if isinstance(mat, cupyx.scipy.sparse.spmatrix):
            return mat.get()
    except ImportError:
        return mat
    return mat


def opt_louvain_rapids(
    adata: AnnData,
    label_key: str,
    cluster_key: str = "opt_louvain",
    use_rep: str | None = None,
    function=lambda a, c, l: ari_score(a, c, l),
    min_resolution: float = 0.1,
    max_resolution: float = 2.0,
    n_iterations: int = 10,
    random_state: int | None = 0,
    plot: bool = False,
    inplace: bool = True,
    force: bool = False,
    verbose: bool = False,
):
    """
    GPU-accelerated version of scib.metrics.clustering.opt_louvain using rapids-singlecell.

    The function searches over Louvain resolutions on GPU and writes the best clustering
    to ``adata.obs[cluster_key]``. Only minimal plotting/logging is kept to mirror the
    scIB API surface that downstream code expects.
    """

    if cluster_key in adata.obs and not force:
        raise ValueError(
            f"{cluster_key} already exists in adata.obs. Set force=True to overwrite."
        )

    if use_rep is None and "neighbors" not in adata.uns:
        raise ValueError("No neighbors present; provide use_rep to compute neighbors on GPU.")

    if use_rep is not None:
        # Compute neighbors on GPU for the requested embedding.
        rsc.pp.neighbors(adata, use_rep=use_rep, random_state=random_state)

    # Convert GPU graphs back to CPU after clustering so scIB metrics work on scipy matrices.
    def _convert_graphs_to_cpu(a: AnnData) -> None:
        if "connectivities" in a.obsp:
            a.obsp["connectivities"] = _ensure_cpu_sparse(a.obsp["connectivities"])
        if "distances" in a.obsp:
            a.obsp["distances"] = _ensure_cpu_sparse(a.obsp["distances"])

    # Pre-generate a reproducible list of resolutions to test.
    rng = np.random.default_rng(random_state)
    resolutions = np.linspace(min_resolution, max_resolution, n_iterations)
    rng.shuffle(resolutions)

    best_score = -np.inf
    best_res = None
    best_labels = None

    for res in resolutions:
        tmp_key = f"{cluster_key}_tmp"
        rsc.tl.louvain(
            adata,
            resolution=float(res),
            key_added=tmp_key,
        )

        score = function(adata, tmp_key, label_key)
        if verbose:
            log.debug(f"resolution={res:.3f}, score={score:.4f}")

        if score > best_score:
            best_score = score
            best_res = res
            best_labels = adata.obs[tmp_key].copy()

        adata.obs.pop(tmp_key)

    if best_labels is None:
        raise RuntimeError("Louvain optimization failed to produce any clustering.")

    adata.obs[cluster_key] = best_labels
    adata.uns[f"{cluster_key}_params"] = {
        "resolution": float(best_res),
        "score": float(best_score),
        "method": "rapids_singlecell.louvain",
    }

    _convert_graphs_to_cpu(adata)

    if not inplace:
        return adata
    return None


def load_embedding(embedding_path: Path, model_name: str) -> np.ndarray:
    if not embedding_path.exists():
        raise FileNotFoundError(f"embedding file not found at {embedding_path}")
    emb = np.load(embedding_path)
    log.info(f"loaded embedding for model {model_name} from {embedding_path}")
    return emb


def main() -> None:
    parser = argparse.ArgumentParser(description="Run scIB metrics for model embeddings.")
    parser.add_argument("--base_dir", required=True, help="Base directory containing model subfolders.")
    parser.add_argument("--model", required=True, nargs="+", help="One or more model names to evaluate.")
    parser.add_argument("--data_id", default="pbmc10k_convert", help="Dataset identifier (used to locate the h5ad file).")
    parser.add_argument(
        "--adata_dir", default="../../data",
        help="Directory containing <data_id>.h5ad; set to '' to use the embedding directory.",
    )
    parser.add_argument("--batch_key", default="batch", help="Batch column in adata.obs.")
    parser.add_argument("--label_key", default="str_labels", help="Label column in adata.obs.")
    parser.add_argument(
        "--exclude_label",
        default="Other",
        help="Obs label to exclude from evaluation (set to empty to disable).",
    )
    parser.add_argument(
        "--output",
        help="Optional output directory. Defaults to <base_dir>; the csv is written to "
        "<output>/<run_id>/<data_id>_scib_metrics_results.csv",
    )
    parser.add_argument(
        "--run_id",
        default="rapids",
        help="Optional run identifier to create a subdirectory under the output directory.",
    )

    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    adata_dir_override = Path(args.adata_dir) if args.adata_dir else None

    results = {}
    for model_name in args.model:
        emb_dir = base_dir / model_name / args.data_id
        adata_dir = adata_dir_override if adata_dir_override else emb_dir
        adata_path = adata_dir / f"{args.data_id}.h5ad"
        if not adata_path.exists():
            raise FileNotFoundError(f"h5ad file not found at {adata_path}")

        adata = sc.read_h5ad(adata_path)

        embedding_path = emb_dir / f"{args.data_id}_embeddings.npy"
        emb_values = load_embedding(embedding_path, model_name)

        if emb_values.shape[0] != adata.n_obs:
            raise ValueError(
                f"embedding for {model_name} has {emb_values.shape[0]} rows but adata has {adata.n_obs} cells"
            )

        obsm_key = model_name
        adata.obsm[obsm_key] = emb_values
        if "cluster" in adata.obs:
            adata.obs.pop("cluster")
        if args.exclude_label:
            adata = adata[~(adata.obs[args.label_key] == args.exclude_label), :]
        log.info(f"running scIB metrics for embedding {obsm_key}")
        results[obsm_key] = eval_scib_metrics(
            adata,
            batch_key=args.batch_key,
            label_key=args.label_key,
            embedding_key=obsm_key,
        )

    df = pd.DataFrame(results).T

    output_dir = Path(args.output) if args.output else base_dir
    if args.run_id:
        output_dir = output_dir / args.run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{args.data_id}_scib_metrics_results.csv"
    df.to_csv(output_path)
    log.info(f"saved scIB metrics to {output_path}")


if __name__ == "__main__":
    main()