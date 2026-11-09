# 🧠 Bayesian Neural Network from Scratch

**Author:** Pushpendra Singh  
**Affiliation:** Indian Institute of Technology Guwahati (IIT Guwahati)  
**License:** MIT

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-Scientific_Computing-013243.svg)](https://numpy.org/)
[![SciPy](https://img.shields.io/badge/SciPy-Optimization-8CAAE6.svg)](https://scipy.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📌 Overview

This repository provides a ground-up mathematical implementation of **Bayesian Neural Networks (BNNs)** in pure **Python, NumPy, and SciPy**, without high-level autograd libraries like PyTorch or TensorFlow.

Unlike conventional (frequentist) neural networks that assign deterministic point estimates to weight parameters ($W$), a Bayesian Neural Network treats all weights as probability distributions:

$$P(W | \mathcal{D}) = \frac{P(\mathcal{D} | W) P(W)}{P(\mathcal{D})}$$

This formulation allows BNNs to explicitly measure **epistemic uncertainty** (model ignorance), making them crucial for safety-critical AI applications such as medical diagnosis, autonomous vehicles, and financial risk forecasting.

---

## 🎯 Repository Contents

| Notebook / Module | Task | Description | Detailed Doc |
| :--- | :--- | :--- | :--- |
| [`BNN_Regression.ipynb`](./BNN_Regression.ipynb) | **Regression** | Non-linear continuous function approximation with 95% predictive uncertainty intervals. | [Read Documentation](./README_REGRESSION.md) |
| [`BNN_Classification.ipynb`](./BNN_Classification.ipynb) | **Classification** | Non-linear decision boundaries with calibrated class confidence estimates. | [Read Documentation](./README_Classification.md) |
| [`NSP.pdf`](./NSP.pdf) | **Theoretical Notes** | Mathematical derivation of variational inference and Monte Carlo approximation. | — |

---

## 🔬 Key Theoretical Highlights

1. **Prior Distribution ($P(W)$)**: Typically initialized as standard Gaussian priors $\mathcal{N}(0, \sigma^2)$.
2. **Likelihood Function ($P(\mathcal{D}|W)$)**: Gaussian likelihood for continuous regression; Softmax/Categorical likelihood for classification.
3. **Posterior Approximation**: Intractable analytical integrals solved via Variational Inference / Bayes by Backprop and Monte Carlo sampling.
4. **Uncertainty Quantification**:
   - High data density $\rightarrow$ Low predictive variance (high confidence).
   - Out-of-Distribution (OOD) regions $\rightarrow$ High predictive variance (capturing epistemic uncertainty).

---

## 🚀 Quickstart

### Prerequisites
```bash
pip install numpy scipy matplotlib jupyter
```

### Running the Notebooks
```bash
# Launch Jupyter Notebook
jupyter notebook
```
- Open `BNN_Regression.ipynb` to explore regression uncertainty curves.
- Open `BNN_Classification.ipynb` to visualize probabilistic classification boundaries.

---

## 📖 Citation & Contact

If you find this implementation helpful, please star the repository!  
Created by **Pushpendra Singh** ([@pushpendras0026](https://github.com/pushpendras0026)) — B.Tech Data Science & AI @ IIT Guwahati.
