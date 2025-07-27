# Bayesian Neural Network for Regression from Scratch

**Project Title:** Bayesian Neural Network for Regression from Scratch

**Author:** Pushpendra Singh

**Date:** July 27, 2025

---

## 1. Introduction: The Bayesian Neural Network (BNN)

This project implements a Bayesian Neural Network (BNN) from scratch using NumPy, specifically for **regression tasks**. It highlights the differences from traditional (frequentist) neural networks and emphasizes BNNs' advantage in quantifying uncertainty for continuous predictions.

### Traditional Neural Networks vs. Bayesian Neural Networks:

| Feature                | Traditional Neural Network (Frequentist)                               | Bayesian Neural Network (Bayesian)                                    |
| :--------------------- | :--------------------------------------------------------------------- | :-------------------------------------------------------------------- |
| **Parameters** | Point estimates (single, fixed values) for weights and biases.         | Probability distributions over weights and biases.                    |
| **Training Goal** | Find the optimal set of parameters ($\theta^*$) that minimizes a loss function (e.g., MSE). | Infer the posterior distribution of parameters, $P(\theta | \text{Data})$. |
| **Output** | A single prediction (e.g., a regression value).                        | A distribution of predictions (e.g., a distribution of regression values). |
| **Uncertainty** | Lacks direct uncertainty quantification. Cannot distinguish between model uncertainty (epistemic) and data noise (aleatoric). | Provides intrinsic uncertainty quantification. Can model both epistemic (due to limited data) and aleatoric (inherent data noise) uncertainty. |
| **Overfitting** | Prone to overfitting, relies heavily on regularization techniques (e.g., dropout, L1/L2). | Less prone to overfitting due to Bayesian regularization (priors on weights).                                |
| **Interpretability** | "Black box" model.                                                     | Provides insights into parameter uncertainty, potentially aiding interpretability. |
| **Computational Cost** | Generally faster to train (optimization).                             | Generally slower to train (sampling or complex optimization for VI).   |

### Why BNNs are Important (Across Industry):

BNNs are gaining traction in industries where **uncertainty quantification is critical** for continuous predictions. Examples include:

* **Financial Forecasting:** Predicting stock prices, market volatility, or bond yields with explicit confidence intervals.
* **Energy Consumption Prediction:** Forecasting electricity demand or building energy usage, where understanding prediction bounds is crucial for resource allocation.
* **Environmental Modeling:** Predicting pollution levels, climate change impacts, or resource availability with quantifiable uncertainty.
* **Engineering & Manufacturing:** Predicting material fatigue, product lifetime, or system performance, where safety margins depend on prediction reliability.
* **Medical Prognosis:** Estimating patient recovery times or disease progression, providing doctors with a range of possible outcomes.

---

## 2. Project Background: MCMC for BNNs

This project utilizes **Markov Chain Monte Carlo (MCMC)**, specifically a **Langevin-informed Metropolis-Hastings (M-H) algorithm**, to sample from the posterior distribution of the neural network's parameters.

**Key Concepts:**

* **Posterior Distribution ($P(\theta | \text{Data})$):** The probability distribution of the model parameters ($\theta$) given the observed training data. It's proportional to Likelihood * Prior: $P(\theta | \text{Data}) \propto P(\text{Data} | \theta) P(\theta)$.
* **Likelihood ($P(\text{Data} | \theta)$):** How probable the observed data is given a specific set of parameters $\theta$. For regression, this is typically a **Gaussian (Normal) log-likelihood**, assuming observations are normally distributed around the model's prediction.
* **Prior ($P(\theta)$):** Our initial belief about the distribution of parameters *before* seeing any data. Here, a Gaussian (Normal) prior is used for weights and biases.
* **MCMC:** A class of algorithms used to sample from complex probability distributions (like the posterior) that are difficult to sample from directly.
* **Metropolis-Hastings (M-H):** A fundamental MCMC algorithm. It proposes new states and accepts/rejects them based on a calculated probability.
* **Langevin MCMC:** An extension of M-H that uses gradient information from the log-posterior to propose new states, making proposals more efficient.
* **Burn-in:** Initial MCMC samples are discarded as the chain may not have converged.
* **Posterior Predictive Distribution:** The prediction for new data is made by averaging predictions from each sampled model: $P(Y_{new} | X_{new}, \text{Data}) = \frac{1}{M} \sum_{i=1}^M P(Y_{new} | X_{new}, \theta_i)$.

---

## 3. Code Structure and Function Explanations

The model is structured into two main classes: `NeuralNetwork` and `MCMC`.

### 3.1. `NeuralNetwork` Class

Defines the architecture and core operations of the neural network (L-layer flexible for regression).

* **`__init__(self, layer_sizes, learning_rate=0.0001)`:**
    * Initializes network structure, weights (Xavier/Glorot init), and biases. Sets `learning_rate`.

* **`initialize_network(self)`:**
    * Internal method to set up initial weights and biases.

* **`sigmoid(self, X)`:**
    * Sigmoid activation: $\sigma(x) = \frac{1}{1 + e^{-x}}$. Includes `np.clip` for numerical stability.

* **`forward_pass(self, X)`:**
    * Computes network output for `X`. Stores intermediate `activations` ($A^{(l)}$) and `zs` ($Z^{(l)}$).
    * Flow: $A^{(0)} = X \rightarrow Z^{(l)} = A^{(l)} W^{(l)} + B^{(l)} \rightarrow A^{(l+1)} = \text{activation}(Z^{(l)})$. The output layer uses a linear activation ($A^{(L-1)} = Z^{(L-1)}$).

* **`backward_pass(self, X, Y)`:**
    * Implements backpropagation for **Mean Squared Error (MSE) Loss** with a linear output layer.
    * Flow:
        1.  Output Delta: $\delta^{(L-1)} = A^{(L-1)} - Y$.
        2.  Hidden Deltas: $\delta^{(l)} = (\delta^{(l+1)} \cdot (W^{(l+1)})^T) \odot \sigma'(Z^{(l)})$.
              <img width="1134" height="330" alt="image" src="https://github.com/user-attachments/assets/68c5a8e9-d8b1-49b1-910a-e4b7e0f03637" />

               <img width="1070" height="298" alt="image" src="https://github.com/user-attachments/assets/f78750c8-54ba-4a88-9ff1-3eed72af9f02" />

    * **Robustness:** Checks for `NaN`/`inf` in gradients and sets them to zero.

* **`encode(self)`:**
    * Flattens `self.weights` and `self.biases` into a single 1D vector `theta`.
    * Flow: $\theta = [W^{(0)}.\text{ravel()}, B^{(0)}.\text{ravel()}, \dots, W^{(L-1)}.\text{ravel()}, B^{(L-1)}.\text{ravel()}]$

* **`decode(self, theta)`:**
    * Reconstructs `self.weights` and `self.biases` from a 1D `theta` vector.
    * Flow: Inverse of `encode`, slicing `theta` based on `layer_sizes` and reshaping.

* **`evaluate_proposal(self, theta, X_data)`:**
    * Performs a forward pass using a *given* `theta` (without modifying `self.weights`/`self.biases`).
    * Flow: Temporarily decodes `theta` into local variables, then executes `forward_pass` logic (with linear output).

* **`langevin_gradient(self, x_data, y_data, theta, depth, batch_size)`:**
    * Computes a gradient-informed proposal by temporarily updating `theta` via mini-batch SGD.
    * Flow:
        1.  Saves current `self.weights`, `self.biases`.
        2.  `self.decode(theta)` (sets model to input `theta`).
        3.  Performs `depth` mini-batch SGD steps (`forward_pass`, `backward_pass`).
        4.  `theta_updated = self.encode()`.
        5.  Restores original `self.weights`, `self.biases`.
        6.  Returns `theta_updated` (mean for MCMC proposal).

### 3.2. `MCMC` Class

 The MCMC sampling process.

* **`__init__(self, n_samples, n_burnin, x_data, y_data, x_test, y_test, layer_sizes, noise_variance=0.1, batch_size=32)`:**
    * Initializes MCMC parameters (`n_samples`, `n_burnin`, `step_theta`, `sigma_squared_prior` (prior variance), `noise_variance` (observation noise variance), `batch_size`).
    * Initializes `NeuralNetwork` model.

* **`evaluate_metrics(predictions, targets)` (staticmethod):**
    * Calculates **RMSE** and **R2 score** for regression. Includes robustness checks for finite inputs.

* **`likelihood_function(self, theta, test=False)`:**
    * Calculates **Gaussian log-likelihood** for regression.
    * Flow: Calls `model.evaluate_proposal(theta, X_data)` for `model_prediction`, then computes Gaussian log-likelihood based on sum of squared errors.
    * **Robustness:** Checks for `NaN`/`inf` in `model_prediction` and `noise_variance` (returns `-inf` likelihood if found).

* **`prior(self, sigma_squared_prior, theta)`:**
    * Calculates log-prior probability: $\log P(\theta) = -\frac{N}{2} \log(2\pi\sigma^2) - \frac{1}{2\sigma^2} \sum \theta_i^2$.

* **`MCMC_sampler(self)`:**
    * Main MCMC loop for generating posterior samples.
    * **Flow (Langevin-informed Metropolis-Hastings):**
        1.  **Initialization:** Sets up storage, initial `theta`, calculates initial `prior_val` and `likelihood`. Handles non-finite initial states.
        2.  **Loop (`ii` from 1 to `n_samples - 1`):**
            * **Propose `theta_proposal`:** Either Langevin-informed (calling `langevin_gradient`) or simple random walk.
            * **Calculate Proposal Likelihoods/Priors:** For `theta_proposal`.
            * **Metropolis-Hastings Acceptance:**
                * Calculates `diff_likelihood`, `diff_prior`.
                * Calculates `diff_prop` (log-ratio of reverse/forward proposal densities) for Langevin; `0` for random walk.
                * `mh_prob = min(1, exp(diff_likelihood + diff_prior + diff_prop))`.
                * **Robustness:** Checks for `NaN`/`inf` in `likelihood_proposal`, `prior_proposal`, `diff_prop` and automatically rejects (`mh_prob = 0`).
            * **Accept/Reject:** Draws `u ~ U(0,1)`. If `u < mh_prob`, accepts `theta_proposal`; otherwise, rejects.
            * **Store Results:** Stores `theta`, predictions, and metrics.
            * **Progress:** Prints status periodically.
        3.  **Burn-in:** Discards the first `n_burnin` samples.
        4.  **Return:** Pandas DataFrame of posterior samples (`pos_theta`).

---

## 4. Comparison of Results

This section presents the performance evaluation of the implemented Bayesian Neural Network (BNN) against a standard (frequentist) Neural Network, both trained on the California Housing regression dataset.

**Performance Evaluation Metrics:**

* **Final Train/Test RMSE (from Posterior Predictive Mean):** For the BNN, this is the Root Mean Squared Error of the average prediction across all accepted posterior samples.
* **Final Train/Test R2 Score (from Posterior Predictive Mean):** For the BNN, this is the R-squared score of the average prediction across all accepted posterior samples.
* **Average Train/Test RMSE/R2 (across accepted samples):** For the BNN, these are diagnostic metrics, showing the typical performance of a single network drawn from the posterior.

---

### **Model Results on California Housing Dataset:**

#### **Bayesian Neural Network (BNN) Results:**
Performance on Regression Dataset<br>

Final Train RMSE (from posterior predictive mean): 0.5098<br>
Final Train R2 (from posterior predictive mean): 0.6019 <br>
Final Test RMSE (from posterior predictive mean): 0.5173 <br>
Final Test R2 (from posterior predictive mean): 0.5908 <br>

Average Train RMSE (across accepted finite samples): 0.5097 <br>
Average Train R2 (across accepted finite samples): 0.6020 <br>
Average Test RMSE (across accepted finite samples): 0.5172 <br>
Average Test R2 (across accepted finite samples): 0.5910 <br>
#### **Frequentist Neural Network Results:**
 Frequentist Neural Network Performance --- <br>
Train RMSE: 0.7208 <br>
Train R2 Score: 0.3200 <br>
Test RMSE: 0.7299 <br>
Test R2 Score: 0.3117 <br>
 ---

### **Comparative Analysis:**

Upon comparing the results on the California Housing dataset:

1.  **RMSE:**
    * The **Bayesian NN** achieves a significantly lower RMSE on both train (~0.510) and test (~0.517) sets.
    * The **Frequentist NN** (simple Linear Regression example) showed higher RMSE values (~0.721 train, ~0.730 test).
    * Lower RMSE indicates better predictive accuracy (smaller average prediction error).

2.  **R2 Score:**
    * The **Bayesian NN** achieves a substantially higher R2 score on both train (~0.602) and test (~0.591).
    * The **Frequentist NN** (Linear Regression) had much lower R2 scores (~0.320 train, ~0.312 test).
    * Higher R2 indicates that a larger proportion of the variance in the dependent variable is predictable from the independent variables, signifying a better fit.

**Discussion:**

In this regression task on the California Housing dataset, the custom-built **Bayesian Neural Network clearly outperforms the simple Frequentist Linear Regression model** in terms of both RMSE and R2 score. This demonstrates the BNN's ability to capture more complex non-linear relationships in the data, which a basic linear model cannot.

Beyond the improved point predictions, the BNN offers the critical advantage of **uncertainty quantification**. While not directly shown in the RMSE/R2 numbers, the collected posterior samples allow for:

* **Prediction Intervals:** Estimating a range (e.g., 95% credible interval) around each prediction, indicating the model's confidence.
* **Model Uncertainty:** Understanding which predictions the model is less certain about, which is invaluable in real-world applications (e.g., in high-stakes financial or engineering predictions).

This project successfully demonstrates the implementation of a robust BNN for regression, highlighting its potential for improved performance and, more importantly, its capacity to provide crucial uncertainty estimates.

---
