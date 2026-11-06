"""scOntoFM: an ontology-aligned single-cell foundation model.

Core modules
------------
preprocessing : raw-count extraction, cell/gene filtering, library-size normalization
tokenization  : median-normalized rank-value tokenization (Geneformer-style)
data          : tokenized HuggingFace datasets and batch collation
model         : model loading and cell / contextual-gene embedding extraction
ontology      : Cell Ontology (CL) graph utilities
plotting      : matplotlib style + UMAP helpers
"""

__version__ = "1.0.0"
