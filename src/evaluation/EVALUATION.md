# Evaluation

Technical reference for the metrics, calibration, and threshold selection modules.

---

## Design principles

Three modules are kept strictly separate so they can be used independently of any model:

- **`metrics.py`** — stateless functions; accept numpy arrays, return dicts.
- **`calibration.py`** — stateful fit/predict objects that map raw scores to calibrated probabilities.
- **`thresholds.py`** — stateless functions that search for an optimal decision boundary.

Accuracy is excluded throughout. Under class imbalance a model that predicts the majority class for every sample achieves high accuracy while being completely useless. All metrics here are sensitive to minority-class performance.

---

## Metrics (`evaluation/metrics.py`)

### Confusion matrix

All threshold-dependent metrics derive from the four entries of the confusion matrix at a given decision threshold $\tau$:

$$\hat{y}_i = \mathbf{1}[\hat{p}_i \geq \tau]$$

| | Predicted positive | Predicted negative |
|---|---|---|
| **Actually positive** | TP | FN |
| **Actually negative** | FP | TN |

### `evaluate(y_true, y_prob, threshold=0.5)`

Returns a dict with the following entries:

#### PR-AUC (primary metric)

Area under the Precision-Recall curve. Precision and recall are computed at every unique predicted probability value, tracing a curve from $(R=0, P=1)$ to $(R=1, P=\text{prevalence})$. The area is estimated by the trapezoidal rule:

$$\text{PR-AUC} = \sum_{k} (R_k - R_{k-1}) \cdot \frac{P_k + P_{k-1}}{2}$$

PR-AUC is the recommended primary metric for imbalanced problems because the PR curve is unaffected by the large number of true negatives. A random classifier achieves PR-AUC $\approx$ prevalence; a perfect classifier achieves 1.0.

#### ROC-AUC

Area under the Receiver Operating Characteristic curve, which plots TPR (recall) against FPR at every threshold. Equivalent to the probability that the model ranks a random positive example above a random negative example:

$$\text{ROC-AUC} = P(\hat{p}_{\text{pos}} > \hat{p}_{\text{neg}})$$

Less informative than PR-AUC under heavy imbalance because it includes TN in both axes, but useful for comparison with external benchmarks.

#### Precision

$$\text{Precision} = \frac{TP}{TP + FP}$$

Fraction of predicted positives that are truly positive. High precision means few false alarms.

#### Recall (Sensitivity, TPR)

$$\text{Recall} = \frac{TP}{TP + FN}$$

Fraction of actual positives that are detected. High recall means few missed detections. For IBD signal detection, recall is the critical operational metric.

#### Specificity (TNR)

$$\text{Specificity} = \frac{TN}{TN + FP}$$

Fraction of actual negatives correctly rejected. Complements recall in threshold analysis.

#### F1 Score

Harmonic mean of precision and recall:

$$F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}} = \frac{2\,TP}{2\,TP + FP + FN}$$

Balances the precision-recall trade-off at a single threshold. Does not account for true negatives, making it appropriate for imbalanced problems.

#### Matthews Correlation Coefficient (MCC)

$$\text{MCC} = \frac{TP \cdot TN - FP \cdot FN}{\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}}$$

The only threshold-dependent metric that uses all four confusion matrix cells symmetrically. Returns values in $[-1, +1]$: +1 is perfect prediction, 0 is no better than random, −1 is perfect inverse prediction. MCC is particularly robust when classes are severely imbalanced because it penalises models that collapse to predicting only one class.

#### G-mean (Geometric Mean)

$$G = \sqrt{\text{Sensitivity} \times \text{Specificity}} = \sqrt{\frac{TP}{TP+FN} \cdot \frac{TN}{TN+FP}}$$

Geometric mean of the true positive rate and true negative rate. Equals zero if either class is predicted at chance, regardless of performance on the other. Useful when both classes must be detected with reasonable accuracy simultaneously.

### `evaluate_at_thresholds(y_true, y_prob, thresholds=None)`

Calls `evaluate` at each of `n=100` linearly spaced thresholds in $[0, 1]$ and returns a list of metric dicts. Used to build threshold sweep curves and to visually identify the operating region before selecting a single threshold.

---

## Calibration (`evaluation/calibration.py`)

Raw neural network outputs are not calibrated probabilities — they are arbitrary scores whose magnitude depends on the model's training dynamics. A model might output 0.9 for all positive examples regardless of true confidence. Calibration maps these scores to values that match empirical frequencies: if the calibrated score is 0.7, approximately 70% of samples with that score should actually be positive.

Calibration is fitted on a held-out **calibration split** — a separate partition that is never seen during training or HPO — and evaluated on the test split at the real-world class prevalence.

### `IsotonicCalibrator`

Fits a piecewise constant monotone function $f : [0,1] \to [0,1]$ via isotonic regression:

$$\min_{f \text{ non-decreasing}} \sum_{i=1}^{N} (y_i - f(\hat{p}_i))^2$$

The solution is the pool-adjacent-violators (PAV) algorithm, which groups consecutive predictions that violate monotonicity and replaces them with their weighted mean. The result is a step function that is non-parametric: it makes no assumption about the shape of the miscalibration.

**When to use:** calibration set $\geq 1\,000$ samples. With fewer samples the step function overfits the calibration data.

**Boundary handling:** predictions outside $[0, 1]$ after transformation are clipped (`out_of_bounds="clip"`).

### `PlattCalibrator`

Fits a logistic sigmoid on top of the raw scores:

$$f(\hat{p}) = \frac{1}{1 + e^{-(A\hat{p} + B)}}$$

where $A$ and $B$ are estimated by maximum likelihood on the calibration set. This is equivalent to fitting a one-dimensional logistic regression with the raw scores as the single feature.

**When to use:** calibration set $< 1\,000$ samples. With only two parameters it is far less prone to overfitting than isotonic regression on small sets.

**Regularisation:** controlled by `C` (inverse strength, default 1.0), passed to `sklearn.linear_model.LogisticRegression`.

### Auto-selection in the trainer

The trainer selects between the two methods based on calibration set size:

```
if len(y_cal) >= min_isotonic_samples:   # default 1000
    use IsotonicCalibrator
else:
    use PlattCalibrator
```

This can be overridden by setting `calibration.method` explicitly in `config/training.yaml`.

### `get_calibrator(method, **kwargs)`

Factory function returning an unfitted calibrator by name: `"isotonic"` or `"platt"`. Keyword arguments are forwarded to the constructor.

---

## Threshold Selection (`evaluation/thresholds.py`)

The default decision threshold of $\tau = 0.5$ is almost never optimal when classes are imbalanced or when false negatives and false positives carry different costs. All three functions sweep $N$ candidate thresholds in $[0, 1]$ (default 500) and return the one that optimises a given objective on the calibration split.

**Important:** threshold selection is always performed on the calibration split, never on the test set, to avoid optimistic bias in the final evaluation.

### `optimal_cost_threshold` (default strategy)

Minimises the total asymmetric business cost:

$$C(\tau) = c_{\text{FP}} \cdot FP(\tau) + c_{\text{FN}} \cdot FN(\tau)$$

$$\tau^* = \arg\min_{\tau \in [0,1]} C(\tau)$$

Default costs: $c_{\text{FP}} = 1.0$, $c_{\text{FN}} = 10.0$. The asymmetry reflects the physical reality that missing a genuine IBD detection (false negative) is ten times worse than triggering an unnecessary follow-up (false positive). This ratio should be set based on the operational cost model of the experiment.

The cost-optimal threshold satisfies:

$$\tau^* \approx \frac{c_{\text{FP}}}{c_{\text{FP}} + c_{\text{FN}}} \cdot \frac{1-\pi}{\pi}$$

where $\pi$ is the class prevalence, showing that higher $c_{\text{FN}}/c_{\text{FP}}$ ratios push the threshold lower (more sensitive), and lower prevalence also pushes it lower.

### `optimal_f1_threshold`

Maximises the F1 score over the calibration split:

$$\tau^* = \arg\max_{\tau \in [0,1]} F_1(\tau) = \arg\max_{\tau} \frac{2\,TP(\tau)}{2\,TP(\tau) + FP(\tau) + FN(\tau)}$$

Use this strategy when precision and recall should be treated symmetrically and no asymmetric cost model is available.

### `optimal_gmean_threshold`

Maximises the geometric mean of sensitivity and specificity:

$$\tau^* = \arg\max_{\tau \in [0,1]} G(\tau) = \arg\max_{\tau} \sqrt{\text{TPR}(\tau) \cdot \text{TNR}(\tau)}$$

This is equivalent to maximising the point on the ROC curve that is furthest from the diagonal in the direction of the upper-left corner. Unlike F1, G-mean penalises equally for failing on either class, making it the most conservative choice under severe imbalance where even a small degradation on the minority class should be detected.

### Choosing a strategy

| Strategy | Use when |
|---|---|
| `cost` | You can assign asymmetric costs to FP and FN (recommended for IBD detection) |
| `f1` | No cost model; want to balance precision and recall |
| `gmean` | Severe imbalance; both classes must perform well simultaneously |

All three strategies are configured in `config/training.yaml` under `threshold.strategy`.