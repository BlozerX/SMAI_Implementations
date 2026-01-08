# Assignment 1 — Sampling, KNN, Linear Regression

This folder contains my notebooks for Assignment 1.

---

## Overview

Assignment 1 focuses on core ML fundamentals: data generation/validation, sampling, a distance-based classifier (KNN), and linear regression with regularization.

---

## What I did

### 1) Synthetic dataset generation + validation
I generated a structured student-style dataset with controlled distributions (categories + numeric attributes). I validated that the generated dataset matches the intended distributional constraints using:
- frequency tables for categorical columns
- summary statistics for numeric columns (mean/variance/range)
- plots (histograms / bar charts) for quick sanity checks
- explicit checks/guards to catch invalid values early

### 2) Sampling strategies + cohort construction
I compared sampling approaches and built cohorts under constraints.
- Random sampling: baseline approach
- Stratified sampling: used when I needed representation across strata

For cohort construction, I enforced constraints such as balanced groups or controlled distributions and then verified the result with post-sampling validation (counts, proportions, and plots).

### 3) KNN experiments
I implemented KNN and evaluated it across multiple K values. My focus here was:
- correct distance computation
- handling feature scaling where needed
- reporting metrics across K (and interpreting the bias/variance behavior)

### 4) Linear regression + regularization comparisons
I ran linear regression experiments and compared regularized variants. I reported:
- error metrics (e.g., MAE/RMSE-style)
- coefficient behavior as regularization strength changes
- qualitative interpretation of overfitting vs generalization

---

## Notebook map

- `1.ipynb` — dataset generation + validation  
- `2.ipynb` — sampling + cohort construction  
- `3.ipynb` — KNN + regression experiments  

---

## How to run

```bash
pip install numpy pandas matplotlib scikit-learn
```

Run in order: 1 → 2 → 3.
