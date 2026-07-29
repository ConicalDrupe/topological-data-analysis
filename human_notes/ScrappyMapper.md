# An example slideshow with good sources
https://www.slideshare.net/slideshow/tutorial-of-topological-data-analysis-part-3mapper-algorithm/67009432

## Disease Trajectories
Percutaneous lung biopsies -> could lead to pneumothroax
https://link.springer.com/article/10.1007/s00330-017-5058-7
What are other trajectories we can identify? What is the proper intervention?

To process a high-dimensional embedding using the Mapper algorithm logic or advanced TDA in TTK, you will typically use a Python pipeline alongside scikit-learn and ParaView for 3D visualization. Since TTK is optimized for meshes and grids, point clouds require a geometric step (like a neighborhood graph or triangulation) or direct linkage to standard Mapper libraries. [1] 
Here is the high-utility workflow to extract the underlying shape of your embedding.
## 1. The Standard Processing Pipeline

* Dimension Reduction (Lens): Map your high-dimensional embedding down to 1D, 2D, or 3D using t-SNE, UMAP, or PCA.
* Graph/Mesh Construction: Connect neighbors using an Alpha Complex or k-Nearest Neighbors (k-NN) graph to give your embedding geometry.
* Scalar Field Definition: Use the lower-dimensional projection coordinates as the scalar fields (the filter functions) over your newly created mesh.
* Topological Skeletonization: Apply TTK's Reeb graph or Morse-Smale complex filters to extract the simplified skeleton of the data. [2, 3] 

## 2. Implementation Options
## Option A: Native Python (Using giotto-tda or KeplerMapper) [4] 
If you specifically want a literal Mapper graph output (nodes as clusters, edges as overlaps) from your embedding, standalone TDA libraries are often more direct than TTK.

* KeplerMapper: Specialized for high-dimensional data clustering and custom lens projections.
* Giotto-TDA: Offers a scikit-learn compatible MapperPipeline object.

## Option B: TTK with Python & ParaView
If you want to use TTK's high-performance C++ backend for complex structural extraction:

* Load your embedding using numpy and run PCA/t-SNE via scikit-learn.
* Save the coordinates and projections into a .vtp (VTK PolyData) file format.
* Apply the TTK Neighborhood Graph filter followed by the TTK Reeb Graph filter inside ParaView to see the data skeleton. [5, 6] 

------------------------------
To help write the exact code or pipeline, let me know:

* What is the shape of your embedding matrix (e.g., 10,000 rows × 512 dimensions)?
* Do you prefer a pure Python script or a visual ParaView workflow?


[1] [https://arxiv.org](https://arxiv.org/html/2505.07747v1)
[2] [https://github.com](https://github.com/run-llama/llama_index/discussions/12168)
[3] [https://michielh.medium.com](https://michielh.medium.com/embedding-for-dummies-beginners-guide-to-ai-s-hidden-language-6b6650bd408e)
[4] [https://direct.mit.edu](https://direct.mit.edu/netn/article/3/3/763/2177/Generating-dynamical-neuroimaging-spatiotemporal)
[5] [https://uw-madison-datascience.github.io](https://uw-madison-datascience.github.io/2022-10-26-machine-learning-novice-sklearn/05-dimensionality-reduction/index.html)
[6] [https://www.analyticsvidhya.com](https://www.analyticsvidhya.com/blog/2021/07/image-denoising-using-autoencoders-a-beginners-guide-to-deep-learning-project/)
