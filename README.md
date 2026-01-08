# SMAI Assignments (IIIT-H)

This repository contains my solutions for the SMAI (Statistical Methods in AI) assignments. Each assignment lives in its own folder with the corresponding notebooks and a small set of result artifacts (GIFs/videos) that are useful for quick qualitative verification.

I keep this repo lightweight and easy to review, so I do not include datasets, virtual environments, large checkpoints, or experiment logs.

---

## What each assignment covers

### Assignment 1 — Sampling, KNN, Linear Regression
In this assignment I:
- generated a synthetic student-style dataset and validated the distributions with sanity checks and visualizations
- compared sampling strategies (random vs stratified) and built constrained cohorts (balanced groups / controlled distributions)
- implemented and evaluated KNN across different K values
- ran linear regression experiments and compared regularization (Ridge/Lasso-style behavior) using error metrics and coefficient analysis

### Assignment 2 — K-Means, GMM, Segmentation, PCA, PCA→KNN
In this assignment I:
- implemented K-Means and selected K using empirical evidence
- implemented GMM with EM and selected the number of mixture components
- applied clustering to image segmentation and saved segmentation videos
- implemented PCA and used it for dimensionality reduction and visualization (including MNIST-style experiments)
- evaluated a PCA→KNN classification pipeline and discussed trade-offs

### Assignment 3 — Sequence Modeling, Recurrence Analysis, Latent Experiments
In this assignment I:
- set up a sequence prediction problem with careful splitting to avoid leakage
- analyzed/identified recurrence structure and studied stability/parsimony
- defined evaluation criteria suited to sequence tasks and reported results clearly
- ran representation/latent experiments and included targeted ablations (including a “freeze latent parameters” style test)

### Assignment 4 — Multi-task CNN, Colourization, Trees/Forests
In this assignment I:
- built a multi-task CNN setup on Fashion-MNIST-style data (shared backbone + multiple heads)
- implemented an image colourization pipeline with an encoder–decoder CNN
- implemented/analyzed decision trees and random forests with bias–variance style comparisons

### Assignment 5 — Advanced Assignment Set
This folder contains notebooks 1–4 for Assignment 5. The per-assignment README documents how the notebooks map to the assignment sections and what each notebook contains.

---

## Running the notebooks

Typical dependencies across the assignments:
```bash
pip install numpy pandas matplotlib scikit-learn jupyter
```

If a notebook uses deep learning:
```bash
pip install torch torchvision tqdm
```

If a notebook uses computer vision utilities:
```bash
pip install opencv-python
```

Some notebooks expect datasets to exist locally; since datasets are not committed here, paths may need to be updated at the top of the notebook.

---

## Repository structure

```
assignments/
  A1/
    README.md
    1.ipynb
    2.ipynb
    3.ipynb
  A2/
    README.md
    1.ipynb ... 7.ipynb
    segmentation_video_1.mp4
    segmentation_video_2.mp4
  A3/
    README.md
    1.ipynb
    2.ipynb
    3.ipynb
    cat_comparison.gif
    smiley_comparison.gif
  A4/
    README.md
    1.ipynb
    2.ipynb
    3.ipynb
    Q3/
      ...
  A5/
    README.md
    1.ipynb
    2.ipynb
    3.ipynb
    4.ipynb
```
