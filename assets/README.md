# Large assets (not tracked by git)

| File | Size | Produced by | Used by |
|---|---|---|---|
| `celltype_similarity_LCA_deepest_allnodes.pkl` | 78 MB | `pretraining/ontology/build_lca_distance.py` | pretraining (ontology alignment, dynamic margin), triplet mining, scOntoSim, novel cell types |
| `CL_id_map.pkl` | 138 MB | `pretraining/triplets/build_celltype_index_map.py` | triplet mining, novel cell types |
| `triplet_ind_hard_top50.pkl` | 468 MB | `pretraining/triplets/generate_triplets.py` | pretraining |

Download them with `python scripts/download_assets.py`.
