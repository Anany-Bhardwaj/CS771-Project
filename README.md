# CS771: Machine Learning Assignments & Projects

This repository contains the assignments and course projects completed for **CS771: Introduction to Machine Learning** under Prof. Purushottam Kar.

## Assignment 1: Cryptographic PUF Modeling & Statistical Inference

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

## Assignment 2: Breaking the Latent Arbiter PUF

### Objective
To break a corrupted Arbiter PUF where intermediate challenge bits have been maliciously modified by modeling the corrupted bits as hidden variables[cite: 1]. The framework was extended to jointly learn a secondary hidden 16-bit PUF responsible for generating the corrupted bits alongside the underlying 17-bit physical PUF model[cite: 1].

### Approach & Algorithms Used
* **Mathematical Modeling:** Formulated a joint likelihood objective to effectively evaluate the generative distributions of both the 17-bit predictive PUF and the 16-bit hidden PUF[cite: 1].
* **Optimization:** Applied a winner-take-all Alternating Optimization (Hard-EM) algorithm to iteratively recover the latent variables (E-Step) and optimize the model weights (M-Step)[cite: 1].
* **Implementation:** Solved the independent convex sub-problems using Logistic Regression with a `liblinear` solver[cite: 2].
* **Robustness:** Handled the highly non-convex joint objective by executing multiple random initializations to avoid bad local minima during convergence[cite: 2].

### Results
* Perfectly recovered the missing latent variables by achieving 100% training accuracy with the dual-PUF model[cite: 1].
* Successfully generalized the learned model to unseen challenges, predicting responses with 99.35% test accuracy on the public dataset[cite: 1].

---
*Author: Anany Bhardwaj*
