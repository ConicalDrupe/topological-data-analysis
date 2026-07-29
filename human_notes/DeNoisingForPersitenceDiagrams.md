Chest X-ray grayscale images contain several noise types that corrupt persistence diagrams in TDA:

## Noise sources

Quantum (Poisson) noise is the dominant one — X-ray photon counts follow a Poisson distribution, so pixel intensities have signal-dependent variance. This creates many low-persistence off-diagonal points (spurious small loops and components). Electronic/readout noise (Gaussian, signal-independent) adds a fixed-variance background. Fixed-pattern noise from detector non-uniformity and quantization noise from bit-depth discretization add structured artifacts. Scatter radiation adds low-frequency intensity drift, which shifts birth/death times of genuine features.

In TDA on grayscale (sublevel/superlevel filtration on pixel intensity), every local intensity fluctuation spawns a topological feature. Poisson noise therefore populates the diagram with a dense cloud of points near the diagonal that swamps the handful of high-persistence points corresponding to real anatomy.

Denoising while preserving fine detail

The tension is that aggressive smoothing (Gaussian, median) kills the small vessels, fissures, and micronodules you often care about. Options that preserve edges and fine structure:

Variance-stabilizing transform (Anscombe) + denoise + inverse. Because the noise is Poisson, apply the Anscombe transform to convert it to approximately Gaussian unit-variance, denoise with a Gaussian-noise method, then invert. This respects the actual noise statistics rather than assuming additive noise.
BM3D / non-local means. Patch-based methods exploit self-similarity and preserve texture and edges far better than local filters. BM3D is close to the practical ceiling for classical denoising.
Edge-preserving filters: bilateral, anisotropic diffusion (Perona–Malik), total variation (ROF) — TV in particular suppresses small oscillations while keeping sharp boundaries.
Wavelet/curvelet shrinkage with a Poisson-aware threshold — sparse in the transform domain, keeps localized detail.
Learned denoisers (DnCNN, Noise2Noise/Noise2Void) if you have data; self-supervised variants avoid needing clean targets, which is realistic for medical images.

## TDA-side alternatives (often better than pre-denoising)

Rather than cleaning the image, clean the diagram or make the pipeline robust:

Persistence thresholding / topological simplification. Keep only points with persistence above a noise floor; the Poisson cloud sits near the diagonal and is discarded without touching the image.
Persistence images or persistence landscapes with a weighting function that down-weights near-diagonal points — this builds noise-robustness into the vectorization.
Confidence sets / bottleneck bootstrap (Fasy et al.) to estimate a statistically justified cutoff for which features are signal.
Extended/robust filtrations such as distance-to-measure (DTM) filtrations, which are provably stable to noise and outliers, instead of the raw sublevel-set filtration.
Presmoothing only the filtration function, not the displayed image — you can compute topology on a lightly regularized intensity field while keeping the original for viewing.

Practical recommendation: use DTM or a persistence-weighted vectorization so the small-scale noise is handled inside the TDA rather than by blurring the image, and if you must denoise, do Anscombe + BM3D so you respect the Poisson statistics and lose the least fine detail.
