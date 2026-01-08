# Assignment 3 — Sequence Modeling, Recurrence Analysis, Latent Experiments

This folder contains my notebooks for Assignment 3 and two qualitative comparison GIFs.

---

## Overview

Assignment 3 is focused on modeling and evaluation discipline. The goal is to set up sequence experiments correctly, define evaluation criteria clearly, and run controlled experiments (including ablations) with clean reporting.

---

## What I did

### 1) Sequence setup + prediction
I built a sequence prediction setup with careful splitting to avoid leakage. I defined input windows/horizons, trained baseline predictors, and compared them against the main modeling approach.

### 2) Recurrence analysis and stability
I explored recurrence structure and tested stability/parsimony: I preferred simpler recurrences when performance was comparable and I checked behavior beyond short-horizon fits.

### 3) Evaluation criteria and reporting
I used metrics suited to the sequence task, reported results consistently, and included diagnostic plots/tables to understand failure modes.

### 4) Latent/representation experiments + ablations
I ran latent representation experiments and included targeted ablations. One of the key checks was freezing latent parameters to see whether the learned representation is actually carrying useful structure.

---

## Included artifacts

- `cat_comparison.gif`
- `smiley_comparison.gif`

---

## Notebook map

- `1.ipynb` — sequence setup + prediction experiments  
- `2.ipynb` — recurrence analysis + evaluation  
- `3.ipynb` — latent experiments + ablations  

---

## How to run

```bash
pip install numpy pandas matplotlib scikit-learn tqdm
pip install torch torchvision  # if deep learning is used
```

Run in order: 1 → 3.
