# SMAI Assignments (IIIT-H) — Jupyter Notebook Submissions

This repository contains my solutions for SMAI (Statistical Methods in AI) assignments, organized as one folder per assignment.
The repo is intentionally **not a full training/codebase**: it contains **only** the notebooks and a small set of output artifacts
(videos/GIFs) that are useful to view results.

## Repository Layout



Each assignment folder contains:
- `README.md` describing what’s included and what the notebooks cover
- the selected `*.ipynb` notebooks
- selected outputs (only where explicitly included)

## What is Included vs Excluded

### Included
- Selected notebooks (`1.ipynb`, `2.ipynb`, …) per assignment
- Selected outputs:
  - A2: `segmentation_video_1.mp4`, `segmentation_video_2.mp4`
  - A3: `cat_comparison.gif`, `smiley_comparison.gif`
  - A4: `Q3/` folder only

### Excluded (on purpose)
- Large datasets (`data/`, `Data/`, `Dataset/`)
- Training logs / experiment trackers (`wandb/`, `runs/`)
- Virtual environments (`.venv/`, `venv/`)
- Model checkpoints (`*.pth`) unless explicitly required by the assignment submission format

The goal is a clean, reviewable repo with the core work.

## Running the Notebooks (Local)

### Recommended setup
- Python 3.10+ (or whatever your SMAI environment used)
- Create a fresh venv and install dependencies as needed.

Example:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install numpy pandas matplotlib scikit-learn torch torchvision opencv-python

Assignment Summary (High-Level)
A1 — Sampling + KNN + Regression/Regularization

Focus: classical ML workflow, evaluation, and regularization; notebooks 1–3.

A2 — Clustering/PCA + Segmentation Outputs

Focus: unsupervised learning and/or segmentation tasks; notebooks 1–7 + videos showing segmentation output.

A3 — Representation / Reconstruction + Visual Comparisons

Focus: reconstruction / mapping / learned representation (as per assignment); notebooks 1–3 + GIFs comparing qualitative results.

A4 — Deep Learning Work + Q3 Subtask Folder

Focus: CNN-based tasks and additional Q3 portion; notebooks 1–3 + Q3/ folder only.

A5 — Advanced Topics (Forecasting / Generative Modeling, etc.)

Focus: later-course topics; notebooks 1–4.

Notes for Reviewers / TAs

Notebook numbering matches how the assignment was structured (Q1/Q2/... style).

Outputs included are only the ones that materially help review results.

If anything fails to run due to missing datasets, that is expected unless datasets are provided separately.

