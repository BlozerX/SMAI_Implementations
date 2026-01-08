# Assignment 2 — K-Means, GMM, Segmentation, PCA, PCA→KNN

This folder contains my notebooks for Assignment 2 and two segmentation videos.

---

## Overview

Assignment 2 is an end-to-end unsupervised learning pipeline. I implemented clustering methods, justified model selection choices, applied clustering to image segmentation, implemented PCA, and then used PCA for downstream classification.

---

## What I did

### 1) K-Means from scratch + choosing K
I implemented K-Means including initialization, assignment/update steps, and convergence checks. I selected K using empirical evidence (objective curves and/or clustering quality criteria depending on the experiment).

### 2) GMM with EM + choosing number of components
I implemented a Gaussian Mixture Model using EM (E-step responsibilities, M-step parameter updates). I monitored log-likelihood trends and selected the number of components using evidence (LL trends and/or information criteria where relevant).

### 3) Image segmentation using clustering
I treated segmentation as clustering in pixel-feature space (e.g., RGB and optionally spatial coordinates). I reconstructed segmentation outputs and checked qualitative coherence across settings.

### 4) Segmentation videos
I saved videos to capture segmentation behavior across iterations / K / feature settings:
- `segmentation_video_1.mp4`
- `segmentation_video_2.mp4`

### 5) PCA from scratch + MNIST-style experiments
I implemented PCA (centering, eigendecomposition/SVD, projection, explained variance). I used PCA for visualization and dimensionality reduction experiments on digit-like datasets.

### 6) PCA → KNN pipeline
I evaluated KNN on PCA-reduced features and compared it against non-reduced baselines to understand trade-offs between accuracy and efficiency.

---

## Notebook map

- `1.ipynb` — setup + initial experiments  
- `2.ipynb` — K-Means + K selection  
- `3.ipynb` — GMM + component selection  
- `4.ipynb` — segmentation pipeline  
- `5.ipynb` — PCA implementation  
- `6.ipynb` — PCA on MNIST + PCA→KNN  
- `7.ipynb` — packaging / final reporting  

---

## How to run

```bash
pip install numpy pandas matplotlib scikit-learn tqdm opencv-python
```

Run in order: 1 → 7.
