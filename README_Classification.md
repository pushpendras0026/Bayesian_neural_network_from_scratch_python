# Bayesian Neural Network from Scratch

**Author:** Pushpendra Singh

**Date:** July 27, 2025

---

## 1. Introduction: The Bayesian Neural Network (BNN) Concept

This project implements a Bayesian Neural Network (BNN) from scratch using NumPy, contrasting it with traditional (frequentist) neural networks and highlighting its advantages, particularly in quantifying uncertainty.

### Traditional Neural Networks vs. Bayesian Neural Networks:

| Feature                | Traditional Neural Network (Frequentist)                               | Bayesian Neural Network (Bayesian)                                    |
| :--------------------- | :--------------------------------------------------------------------- | :-------------------------------------------------------------------- |
| **Parameters** | Point estimates (single, fixed values) for weights and biases.         | Probability distributions over weights and biases.                    |
| **Training Goal** | Find the optimal set of parameters ($\theta^*$) that minimizes a loss function (e.g., MSE, Cross-Entropy). | Infer the posterior distribution of parameters, P(theta/Data). |
| **Output** | A single prediction (e.g., a class probability, a regression value).   | A distribution of predictions (e.g., a distribution of class probabilities, a distribution of regression values). |
| **Uncertainty** | Lacks direct uncertainty quantification. Cannot distinguish between model uncertainty (epistemic) and data noise (aleatoric). | Provides intrinsic uncertainty quantification. Can model both epistemic (due to limited data) and aleatoric (inherent data noise) uncertainty. |
| **Overfitting** | Prone to overfitting, relies heavily on regularization techniques (e.g., dropout, L1/L2). | Less prone to overfitting due to Bayesian regularization (priors on weights).                                |
| **Interpretability** | "Black box" model.                                                     | Provides insights into parameter uncertainty, potentially aiding interpretability. |
| **Computational Cost** | Generally faster to train (optimization).                             | Generally slower to train (sampling or complex optimization for VI).   |

### Why BNNs are Important (Across Industry):

BNNs are gaining traction in industries where **uncertainty quantification is critical**, and merely providing a point estimate isn't sufficient. Examples include:

* **Autonomous Driving:** Predicting a path is not enough; the system needs to know *how confident* it is in that prediction to make safe decisions.
* **Medical Diagnosis:** Knowing "I'm 85% confident it's cancer, but if it's not, the next most likely is condition X with 10% confidence" is more valuable than just a "cancer" prediction.
* **Financial Forecasting:** Predicting stock prices with confidence intervals, or assessing risk with clear uncertainty bounds.
* **Drug Discovery & Materials Science:** Understanding the uncertainty in predicted molecular properties to guide costly experimental validation.
* **Robotics:** For decision-making in uncertain environments, robots need to assess the reliability of their sensory inputs and action outcomes.

---

## 2. Project Background: MCMC for BNNs

This project utilizes **Markov Chain Monte Carlo (MCMC)**, specifically a **Langevin-informed Metropolis-Hastings (M-H) algorithm**, to sample from the posterior distribution of the neural network's parameters.

**Key Concepts:**

* **Posterior Distribution (P( $\theta$|Data)):** The probability distribution of the model parameters ($\theta$) given the observed training data. It's proportional to Likelihood * Prior: $P(\theta | \text{Data}) \propto P(\text{Data} | \theta) P(\theta)$.
* **Likelihood (P(Data|($\theta$))):** How probable the observed data is given a specific set of parameters $\theta$. For classification, this is typically the multinomial (categorical) log-likelihood (Cross-Entropy).
* **Prior (P($\theta$)):** Our initial belief about the distribution of parameters *before* seeing any data. Here, a Gaussian (Normal) prior is used for weights and biases.
* **MCMC:** A class of algorithms used to sample from complex probability distributions (like the posterior) that are difficult to sample from directly.
* **Metropolis-Hastings (M-H):** A fundamental MCMC algorithm. It proposes new states and accepts/rejects them based on a calculated probability.
* **Langevin MCMC:** An extension of M-H that uses gradient information from the log-posterior to propose new states, making proposals more efficient.
* **Burn-in:** Initial MCMC samples are discarded as the chain may not have converged.
* **Posterior Predictive Distribution:** The prediction for new data is made by averaging predictions from each sampled model: $P(Y_{new} | X_{new}, \text{Data}) = \frac{1}{M} \sum_{i=1}^M P(Y_{new} | X_{new}, \theta_i)$.

---

## 3. Code Structure and Function Explanations

The project is structured into two main classes: `NeuralNetwork` and `MCMC`.

### 3.1. `NeuralNetwork` Class

Defines the architecture and core operations of the neural network (L-layer flexible).

* **`__init__(self, layer_sizes, learning_rate=0.01)`:**
    * Initializes network structure, weights (Xavier/Glorot init), and biases. Sets `learning_rate`.

* **`initialize_network(self)`:**
    * Internal method to set up initial weights and biases.

* **`sigmoid(self, X)`:**
    * Sigmoid activation: $\sigma(x) = \frac{1}{1 + e^{-x}}$. Includes `np.clip` for numerical stability.

* **`softmax(self, X)`:**
    * Softmax activation for output: $\text{softmax}(x_j) = \frac{e^{x_j}}{\sum_k e^{x_k}}$.

* **`forward_pass(self, X)`:**
    * Computes network output for `X`. Stores intermediate `activations` ($A^{(l)}$) and `zs` ($Z^{(l)}$).
    * Flow: $A^{(0)} = X \rightarrow Z^{(l)} = A^{(l)} W^{(l)} + B^{(l)} \rightarrow A^{(l+1)} = \text{activation}(Z^{(l)})$.

* **`backward_pass(self, X, Y)`:**
    * Implements backpropagation for **Cross-Entropy Loss** with Softmax output.
    * Flow:
        1.  Output Delta: $\delta^{(L-1)} = A^{(L-1)} - Y$.
        2.  Hidden Deltas: $\delta^{(l)} = (\delta^{(l+1)} \cdot (W^{(l+1)})^T) \odot \sigma'(Z^{(l)})$.
         <img width="1187" height="335" alt="image" src="https://github.com/user-attachments/assets/8489968c-3d8a-4e9b-97a9-3b9e69843590" />

        <img width="1288" height="300" alt="image" src="https://github.com/user-attachments/assets/c0ef45d9-0fc4-4040-921f-8972fb1596b1" />

    * **Robustness:** Checks for `NaN`/`inf` in gradients and sets them to zero.

* **`encode(self)`:**
    * Flattens `self.weights` and `self.biases` into a single 1D vector `theta`.
    * Flow: `theta = [W_0.ravel(), B_0.ravel(), ..., W_L-1.ravel(), B_L-1.ravel()]`.

* **`decode(self, theta)`:**
    * Reconstructs `self.weights` and `self.biases` from a 1D `theta` vector.
    * Flow: Inverse of `encode`, slicing `theta` based on `layer_sizes` and reshaping.

* **`evaluate_proposal(self, theta, X_data)`:**
    * Performs a forward pass using a *given* `theta` (without modifying `self.weights`/`self.biases`).
    * Flow: Temporarily decodes `theta` into local variables, then executes `forward_pass` logic.

* **`langevin_gradient(self, x_data, y_data, theta, depth, batch_size)`:**
    * Computes a gradient-informed proposal.
    * Flow:
        1.  Saves current `self.weights`, `self.biases`.
        2.  `self.decode(theta)` (sets model to input `theta`).
        3.  Performs `depth` mini-batch SGD steps (`forward_pass`, `backward_pass`).
        4.  `theta_updated = self.encode()`.
        5.  Restores original `self.weights`, `self.biases`.
        6.  Returns `theta_updated` (mean for MCMC proposal).

### 3.2. `MCMC` Class

Orchestrates the MCMC sampling process.

* **`__init__(self, n_samples, n_burnin, ..., layer_sizes, noise_variance=0.1, batch_size=32)`:**
    * Initializes MCMC parameters.

* **`accuracy(predictions, targets)` (staticmethod):**
    * Calculates classification accuracy (percentage).

* **`likelihood_function(self, theta, test=False)`:**
    * Calculates **multinomial log-likelihood (Cross-Entropy)**.
    * **Robustness:** Clips `model_prediction` and checks for `NaN`/`inf`.

* **`prior(self, sigma_squared, theta)`:**
    * Calculates log-prior probability: $\log P(\theta) = -\frac{N}{2} \log(2\pi\sigma^2) - \frac{1}{2\sigma^2} \sum \theta_i^2$.

* **`MCMC_sampler(self)`:**
    * Main MCMC loop for generating posterior samples.
    * **Flow (Langevin-informed Metropolis-Hastings):**
        1.  **Initialization:** Sets up storage, initial `theta`, calculates initial `prior_val` and `likelihood`. Handles non-finite initial states.
        2.  **Loop (`ii` from 1 to `n_samples - 1`):**
            * **Propose `theta_proposal`:** Either Langevin-informed or simple random walk.
            * **Calculate Proposal Likelihoods/Priors.**
            * **Metropolis-Hastings Acceptance:** Calculates `mh_prob` using `diff_likelihood`, `diff_prior`, `diff_prop`.
            * **Accept/Reject:** Draws random `u`. If `u < mh_prob`, accepts; otherwise, rejects.
            * **Store Results:** Stores `theta`, predictions, and metrics.
            * **Progress:** Prints status periodically.
        3.  **Burn-in:** Discards the first `n_burnin` samples.
        4.  **Return:** Pandas DataFrame of posterior samples (`pos_theta`).

---

## 4. Comparison of Results

This section presents the performance evaluation of the implemented Bayesian Neural Network (BNN) against a standard (frequentist) Neural Network, both trained on the Iris dataset.

**Performance Evaluation Metrics:**

* **Final Train/Test Accuracy (from Posterior Predictive Mean):** For the BNN, this is obtained by averaging the probabilities across all accepted posterior samples, then classifying based on the highest average probability.
* **Average Train/Test Accuracy (across accepted samples):** For the BNN, this is a diagnostic metric, showing the typical performance of a single network drawn from the posterior.
* **Precision, Recall, F1-score:** Standard classification metrics.

---

### **Model Results on Iris Dataset:**

#### **Bayesian Neural Network (BNN) Results:**
Final Performance Evaluation

Final Train Accuracy (from posterior predictive mean): 99.17%
Final Test Accuracy (from posterior predictive mean): 96.67%

Test Classification Report (from posterior predictive mean):

|           | precision | recall | f1-score | support |
| :-------- | :-------- | :----- | :------- | :------ |
| **setosa** | 1.00      | 1.00   | 1.00     | 10      |
| **versicolor**| 0.90      | 1.00   | 0.95     | 9       |
| **virginica** | 1.00      | 0.91   | 0.95     | 11      |
| **accuracy** |           |        | 0.97     | 30      |
| **macro avg** | 0.97      | 0.97   | 0.97     | 30      |
| **weighted avg**| 0.97    | 0.97   | 0.97     | 30      |

Average Train Accuracy (across accepted samples): 99.10%
Average Test Accuracy (across accepted samples): 96.63%

#### **Frequentist Neural Network Results:**
--- Frequentist Neural Network Performance ---
Train Accuracy: 98.33%
Test Accuracy: 100.00%

Test Classification Report:

|           | precision | recall | f1-score | support |
| :-------- | :-------- | :----- | :------- | :------ |
| **setosa** | 1.00      | 1.00   | 1.00     | 10      |
| **versicolor**| 1.00      | 1.00   | 1.00     | 9       |
| **virginica** | 1.00      | 1.00   | 1.00     | 11      |
| **accuracy** |           |        | 1.00     | 30      |
| **macro avg** | 1.00      | 1.00   | 1.00     | 30      |
| **weighted avg**| 1.00    | 1.00   | 1.00     | 30      |

---

### **Comparative Analysis:**

Upon comparing the results on the Iris dataset:

1.  **Training Accuracy:** Both models show very high training accuracy (BNN at ~99.17%, Frequentist at ~98.33%). This indicates both models are capable of fitting the training data well.
2.  **Test Accuracy:**
    * The **Frequentist NN** achieves a perfect **100.00% Test Accuracy**.
    * The **Bayesian NN** achieves **96.67% Test Accuracy**.

3.  **Classification Report Insights:**
    * The Frequentist NN's report shows perfect precision, recall, and F1-score for all classes on the test set.
    * The BNN's report shows excellent performance, but a slight dip in precision for 'versicolor' (0.90) and recall for 'virginica' (0.91) suggests it made a few more errors on the test set compared to the Frequentist model's perfect score.

**Discussion:**

While the Frequentist Neural Network achieved a perfect 100% accuracy on the test set for this particular run, it's important to consider the context:

* **Small Dataset:** The Iris dataset is small. Perfect scores on small datasets can sometimes be misleading or indicate slight overfitting, especially if the train/test split happens to be "easy."
* **Point Estimate vs. Distribution:** The Frequentist NN provides only a single, deterministic prediction. Its 100% accuracy is a single point estimate of its performance.
* **BNN's Core Advantage:** The BNN, even if its point accuracy (mean predictive) is slightly lower on this specific split, offers a crucial advantage: **uncertainty quantification**. The BNN's `pred_y` and `test_pred_y` arrays (after burn-in) contain the *distribution of predictions* from each individual posterior sample. This allows for:
    * **Confidence Intervals:** You can quantify how confident the model is in its predictions (e.g., "I predict this is class A with 95% confidence").
    * **Robustness to Overfitting:** Bayesian models, through their priors, inherently regularize the parameters, making them generally less prone to severe overfitting, even if a single deterministic run of a frequentist model happens to hit a perfect score on a small dataset.
    * **"Knowing What It Doesn't Know":** A BNN can output high uncertainty for predictions on data points far from the training distribution, whereas a frequentist NN would still output a confident (but potentially wrong) prediction.

In conclusion, for this specific Iris dataset test, the Frequentist model achieved perfect accuracy. However, for real-world applications where data might be noisy, limited, or where decision-making demands an understanding of model confidence, the BNN, despite its slightly lower point accuracy in this example, provides a richer and more robust probabilistic understanding of predictions. Its value lies beyond just the single 'accuracy' number.
