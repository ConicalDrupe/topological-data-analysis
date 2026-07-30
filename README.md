# topological-data-analysis
Explorations, Algorithms, and Inferences using topological data analysis

# Dataset Sample
https://www.kaggle.com/datasets/ashery/chexpert

# Creating Embeddings
Use the Vision Backbone: The MedGemma family (such as the 4B multimodal version) relies on a SigLIP-based vision encoder. You can extract pooled image embeddings directly using this module, circumventing the text-generation language model (LLM).

See [`GenerateEmbeddings/README.md`](GenerateEmbeddings/README.md) for details. Quick
start for this repo's pneumothorax cohort (from the repo root, after `uv sync`):
```bash
uv run python GenerateEmbeddings/scripts/run_pneumothorax_embeddings.py --backend siglip --split test
```

# Viewing Pointclouds
https://topology-tool-kit.github.io/installation.html

Two-parameter persistent homology https://rivet.readthedocs.io/en/latest/about.html

