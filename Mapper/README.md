# Inital Code Setup
- Script to load embeddings. Will also include utility to reference appropriate patient image (with path depending on raw or postprocessed image)
- Basic Keppler Mapper script for generating interactive (can use basic UMAP or t-SNE for lense, and a basic out of the box covering and clustering fit for embedings)
- Altered Keppler Mapper that allows previewing images, disease distribution, age distribution within clusters.
- Script that uses G-mapper for optimizing cover in Mapper

## Data to use
Embeddings: /data/embeddings/processed/
Processed Images for tooltip, or in viewer:  /kaggle/processed

## Future Enhancements
- Visual Comparison Techniques. Suppose we generate two mapper graphs that have shared nodes. We want to layer them, so we can visually see where the two graphs differ. 
- Global Network metrics. Compare the overall density, diameter, average path length, and degree distribution. 
- Adjacency Matrix Subtraction: Subtract one matrix from the other to isolate the exact edges that were added, removed, or changed in weight. 

# G-Mapper (For optimizing Cover)
https://github.com/MRC-Mapper/G-Mapper/tree/main
