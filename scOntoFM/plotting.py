"""Plot style and UMAP helpers."""

import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import scanpy as sc
import seaborn as sns

centimeter = 1 / 2.54  # inches per cm

# qualitative palettes
react_cols_10 = ['#147EB3', '#29A634', '#D1980B', '#D33D17', '#9D3F9D',
                 '#00A396', '#DB2C6F', '#8EB125', '#946638', '#7961DB']
norm_7 = ['#4053d3', '#ddb310', '#b51d14', '#00beff', '#fb49b0', '#00b25d', '#cacaca']
norm_12 = ['#ebac23', '#b80058', '#008cf9', '#006e00', '#00bbad', '#d163e6', '#b24502',
           '#ff9287', '#5954d6', '#00c6f8', '#878500', '#00a76c', '#bdbdbd']


def config_rc(dpi=300, font_size=5, lw=1.0):
    """Publication-style matplotlib defaults (embedded TrueType fonts)."""
    rc = {
        'font.size': font_size,
        'axes.labelsize': font_size,
        'axes.titlesize': font_size,
        'xtick.labelsize': font_size,
        'ytick.labelsize': font_size,
        'figure.dpi': dpi,
        'axes.linewidth': lw,
    }
    sns.set(style='ticks', rc=rc)
    sns.set_context("paper")
    mpl.rcParams.update(rc)
    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['ps.fonttype'] = 42
    mpl.rcParams['font.family'] = "DejaVu Serif"
    mpl.rcParams['axes.unicode_minus'] = False


def vis_sim_matrix(sim_mat, node_list, figsize=8, mark="", out_dir="./output"):
    """Heatmap of a cell-type similarity matrix."""
    plt.subplots(figsize=(figsize * centimeter, figsize * centimeter), dpi=300)
    sns.heatmap(sim_mat, xticklabels=node_list, yticklabels=node_list, cmap='viridis', annot=True, fmt=".1f")
    plt.title(f"{mark} Similarity Matrix")
    plt.xlabel('Cell Types')
    plt.ylabel('Cell Types')
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, f"{mark}_similarity_matrix.pdf"), bbox_inches='tight', dpi=300)


def plot_obs(adata, color_obs, obsm_name, out_file_path, dataset_id):
    """Save a UMAP (already computed in ``adata``) colored by ``color_obs``."""
    fig, ax = plt.subplots(figsize=(centimeter * 5, centimeter * 5))
    sc.pl.umap(adata, color=color_obs, edges_color='none', frameon=False, show=False, ax=ax)
    ax.legend(loc='center left', bbox_to_anchor=[0.95, 0.5], ncol=1, frameon=False,
              markerscale=0.6, labelspacing=0.2, fontsize=8, handleheight=0.8, handlelength=1.0)
    ax.set_title(f"Cell Embeddings colored by {color_obs}", pad=0, y=0.965)
    plt.savefig(os.path.join(out_file_path, f"UMAP_{dataset_id}_{color_obs}.pdf"),
                bbox_inches='tight', format='pdf', dpi=300)
    plt.close(fig)
