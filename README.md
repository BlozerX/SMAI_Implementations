# SMAI (Statistical Methods in AI) — Assignments Repository (IIIT-H)

This repository is an organized, reviewer-friendly archive of my **SMAI assignments**.  
Each assignment folder contains:
- **Only the required notebooks** (named `1.ipynb`, `2.ipynb`, …) that implement the solutions end-to-end
- **A small set of output artifacts** (videos/GIFs) when they are important for quick visual verification

I intentionally keep this repo **clean and lightweight**, so large datasets, virtual environments, checkpoints, and experiment logs are not tracked here.

---

## What the assignments are about (actual content, not just file structure)

### Assignment 1 — Sampling + KNN + Linear Regression
This assignment is about fundamentals:
- **Synthetic dataset generation** (controlled distributions + sanity checks)
- **Sampling strategies** (random vs stratified; cohort construction under constraints)
- **KNN** (distance-based classification + K selection + evaluation)
- **Linear regression** (baseline + regularization; error metrics + interpretation)

You’re demonstrating that you can:
1) generate and validate data correctly,  
2) avoid silent bias introduced by sampling,  
3) implement simple models correctly and evaluate them properly.

---

### Assignment 2 — K-Means, GMM, Image Segmentation, PCA, PCA→KNN
This assignment is a full unsupervised-learning pipeline:
- Implement **K-Means** + justify best K
- Implement **GMM (EM)** + justify component count
- Use clustering for **image segmentation**
- Create **segmentation videos** to visually validate results
- Implement **PCA** (math + projection + variance)
- Apply PCA on MNIST and build **PCA → KNN** classifier

You’re demonstrating that you can build algorithms from scratch and justify choices using evidence.

---

### Assignment 3 — Sequence Modeling + Recurrence Analysis + Latent Experiments
This assignment is “modeling + reasoning”, not just implementation:
- sequence setup and prediction
- identifying/deriving **recurrence structure**
- evaluation criteria that actually match sequence problems
- experiments that are controlled + reproducible
- latent representation experiments (including a **freeze-latent** style ablation)

You’re demonstrating scientific hygiene: setup, evaluation, controlled comparisons, interpretation.

---

### Assignment 4 — Multi-task CNN + Image Colourization + Trees/Forests
This assignment mixes deep learning + classical ML:
- Multi-task CNN on Fashion-MNIST (shared backbone + multiple heads)
- CNN-based image colourization (encoder–decoder style pipeline)
- Decision Trees + Random Forests (implementation + bias/variance analysis)

You’re demonstrating both neural pipelines and classical ML fundamentals.

---

### Assignment 5 — Advanced SMAI Module Set (see folder README)
Assignment 5 contains later-course tasks (the exact breakdown is documented in `assignments/A5/README.md`).  
This repo includes only notebooks `1.ipynb`–`4.ipynb` for A5.

---

## How to run (practical)

Typical deps used across SMAI:
```bash
pip install numpy pandas matplotlib scikit-learn jupyter
```

If any assignment uses deep learning:
```bash
pip install torch torchvision tqdm
```

If any vision processing is used:
```bash
pip install opencv-python
```

> Datasets are not committed here. If a notebook expects local datasets, update the first “Paths/Config” cell.

---

## Assignment folders
- `assignments/A1/`
- `assignments/A2/`
- `assignments/A3/`
- `assignments/A4/`
- `assignments/A5/`

---

## Repo structure (kept at end as requested)

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
