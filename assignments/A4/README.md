# Assignment 4 — Multi-task CNN, Colourization, Trees/Forests

This folder contains my notebooks for Assignment 4 and the `Q3/` folder needed for the trees/forests part.

---

## Overview

Assignment 4 mixes deep learning and classical ML:
- a multi-task CNN setup
- an image colourization pipeline
- decision trees and random forests with analysis

---

## What I did

### 1) Multi-task CNN (Fashion-MNIST-style)
I implemented a shared CNN backbone with multiple task-specific heads and trained using a combined loss. I reported per-task metrics and compared how multi-task learning affects performance.

### 2) Image colourization (encoder–decoder CNN)
I implemented a colourization pipeline: grayscale input → predicted color channels. I focused heavily on pipeline correctness (color space conversions, normalization, tensor shapes) and validated results qualitatively.

### 3) Decision trees + random forests
I implemented/analyzed trees and forests, including split criteria and tuning. I compared how forests stabilize single trees and discussed bias–variance style behavior.

---

## Notebook map

- `1.ipynb` — multi-task CNN  
- `2.ipynb` — colourization  
- `3.ipynb` — trees/forests + analysis  
- `Q3/` — supporting code/assets for Q3  

---

## How to run

```bash
pip install numpy pandas matplotlib scikit-learn tqdm
pip install torch torchvision
pip install opencv-python  # if used in colourization
```

Run in order: 1 → 3.
