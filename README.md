# CS771: Machine Learning Assignments & Projects

This repository contains the assignments and course projects completed for **CS771: Introduction to Machine Learning** under Prof. Purushottam Kar.

> **Note:** **Assignment 2** will be uploaded to this repository upon its completion.

## Project: Cryptographic PUF Modeling & Statistical Inference

### Objective
To model Physical Unclonable Functions (PUFs) to solve inference tasks and recover hidden variables from distorted data, demonstrating robustness against adversarial meddling.

### Approach & Algorithms Used
* **Feature Engineering:** Mapped non-linear cryptographic challenges into a **288-D feature space** via polynomial expansion to train a **LinearSVC** classifier.
* **Mathematical Modeling:** Formulated a **latent-variable Maximum Likelihood Estimation (MLE)** framework to model 16-bit and 17-bit Arbiter PUFs under bit-insertion attacks.
* **Optimization:** Implemented an **alternating optimization** algorithm to jointly estimate complex model parameters and recover censored bits.
* **Tuning:** Executed extensive hyperparameter tuning (initializations, convergence thresholds) for the non-convex objectives.

### Results
* Achieved **>99% accuracy** on PUF response predictions.
* Successfully reconstructed the censored dataset by leveraging dual-PUF likelihood models and latent prior distributions.

---
*Author: Anany Bhardwaj*
