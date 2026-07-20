# Models

Technical reference for the model abstraction layer and the MLP implementation.

---

## BaseModel (`models/base_model.py`)

`BaseModel` is an abstract base class that defines the contract every classifier in this pipeline must satisfy. It uses Python's `abc.ABC` machinery: any subclass that does not implement all three abstract methods raises `TypeError` on instantiation.

### Contract

```
fit(X_train, y_train, X_val, y_val) → self
predict_proba(X)                    → np.ndarray, shape (n,)
save(path)                          → None
```

The trainer calls only these three methods. It never inspects model internals, never imports Keras, and never assumes a specific framework. This makes it straightforward to add an XGBoost, GraphNN, or any other classifier by subclassing `BaseModel` without modifying the training pipeline.

#### `fit`

Receives two pairs of numpy arrays: training data `(X_train, y_train)` and validation data `(X_val, y_val)`. The validation pair is passed explicitly so models can use it for early stopping or internal validation — the trainer does not control how or whether it is used. The method must return `self` to allow method chaining.

```
X_train : (n_train, n_features)  float64
y_train : (n_train,)             int {0, 1}
X_val   : (n_val,   n_features)  float64
y_val   : (n_val,)               int {0, 1}
```

#### `predict_proba`

Returns a one-dimensional array of positive-class probabilities in `[0, 1]`. These are raw model scores — they are calibrated separately by the trainer before threshold selection and final evaluation.

```
X      : (n_samples, n_features)  float64
returns: (n_samples,)             float64 ∈ [0, 1]
```

#### `save`

Persists the fitted model to a directory `path`. The trainer creates the run directory; the model is responsible for writing into it in whatever format it requires (Keras `.keras`, `joblib`, ONNX, etc.).

### Optional override: `suggest_params`

```python
@classmethod
def suggest_params(cls, trial: optuna.Trial) -> dict[str, Any]:
    return {}
```

Called once per Optuna trial during HPO. The returned dict is passed verbatim to `__init__` as the `params` argument. The default returns an empty dict, which causes the trainer to instantiate the model with no params (relying on constructor defaults). Override in each subclass to expose model-specific hyperparameters.

### Convenience method: `predict`

```python
def predict(self, X, threshold=0.5) -> np.ndarray  # int {0, 1}
```

A thin wrapper around `predict_proba` that applies a threshold to produce hard binary labels. Not used by the trainer — intended for interactive evaluation and debugging.

### Class imbalance

`BaseModel` does not inject any class weighting. Each subclass manages its own imbalance strategy via the `params` dict (e.g. `class_weight={0: 1.0, 1: w}` for Keras models). This keeps the trainer agnostic and lets each model choose the most appropriate strategy for its framework.

---

## MLPModel (`models/mlp_model.py`)

A fully-connected multi-layer perceptron for binary IBD / radio classification on 9 tabular delta features.

### Input features

| Column | Physical meaning |
|---|---|
| `delta_t` | Time difference between anchor and partner event |
| `delta_energy` | Energy difference |
| `delta_x` | Displacement along x |
| `delta_y` | Displacement along y |
| `delta_z` | Displacement along z |
| `delta_r` | Difference in transverse radius $r = \sqrt{x^2 + y^2}$ |
| `delta_r2` | Difference in $r^2$ |
| `delta_phi` | Difference in azimuthal angle $\phi = \arctan2(y, x)$ |
| `delta_distance` | Difference in 3D distance $D = \sqrt{x^2 + y^2 + z^2}$ |

All features are differences between the anchor event (subscript 1) and the partner event (subscript 2), computed during preprocessing. The input to the model is a flat vector $\mathbf{x} \in \mathbb{R}^9$.

### Architecture

The network is a feed-forward stack of identical blocks followed by a sigmoid output:

```
Input(9)
  └── [Dense(hᵢ) → BatchNorm → Activation → Dropout(p)] × L
        └── Dense(1, sigmoid)
```

where $L$ is the number of hidden layers, $h_i$ is the number of units in layer $i$, and $p$ is the dropout rate.

**Default configuration** (no HPO):

| Layer | Units | After |
|---|---|---|
| Dense 0 | 256 | BN → ReLU → Dropout(0.3) |
| Dense 1 | 128 | BN → ReLU → Dropout(0.3) |
| Dense 2 | 64 | BN → ReLU → Dropout(0.3) |
| Output | 1 | sigmoid |

Total parameters (default): **45,569**

#### Forward pass

For each hidden layer $i$:

$$\mathbf{z}_i = W_i \mathbf{a}_{i-1} + \mathbf{b}_i$$

$$\hat{\mathbf{z}}_i = \text{BatchNorm}(\mathbf{z}_i) = \gamma_i \cdot \frac{\mathbf{z}_i - \mu_i}{\sqrt{\sigma_i^2 + \epsilon}} + \beta_i$$

$$\mathbf{a}_i = \sigma\!\left(\hat{\mathbf{z}}_i\right) \odot \mathbf{m}_i, \quad \mathbf{m}_i \sim \text{Bernoulli}(1-p)$$

where $\sigma$ denotes the activation function (ReLU, ELU, or tanh), $\odot$ is element-wise multiplication, and $\mathbf{m}_i$ is the dropout mask (applied only during training).

The output layer computes:

$$\hat{y} = \text{sigmoid}(W_{\text{out}} \mathbf{a}_L + b_{\text{out}}) = \frac{1}{1 + e^{-(W_{\text{out}} \mathbf{a}_L + b_{\text{out}})}}$$

#### Loss function

Binary cross-entropy:

$$\mathcal{L} = -\frac{1}{N} \sum_{i=1}^{N} \left[ y_i \log \hat{y}_i + (1 - y_i) \log(1 - \hat{y}_i) \right]$$

With optional L2 regularisation on all Dense kernels $W$:

$$\mathcal{L}_{\text{reg}} = \mathcal{L} + \lambda \sum_l \|W_l\|_F^2$$

#### Optimiser

Adam with default $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-7}$:

$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$$
$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$
$$\theta_t = \theta_{t-1} - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$$

where $\eta$ is the learning rate and $\hat{m}_t$, $\hat{v}_t$ are bias-corrected moment estimates.

#### Callbacks

Two Keras callbacks are attached during `fit`, both monitoring `val_pr_auc`:

**EarlyStopping** — halts training and restores the best-weight checkpoint when `val_pr_auc` has not improved for `patience` epochs.

**ReduceLROnPlateau** — halves the learning rate when `val_pr_auc` has not improved for $\lfloor\text{patience}/3\rfloor$ epochs, subject to a floor of $10^{-6}$.

PR-AUC is used instead of loss or accuracy because it is the primary metric for imbalanced detection problems: it measures the area under the precision-recall curve, which is sensitive to performance on the minority class.

### Hyperparameter search space (Optuna)

| Parameter | Type | Range / choices |
|---|---|---|
| `n_layers` | int | 1, 2, 3 |
| `units_lᵢ` | int (log-uniform) | [32, 512] per layer |
| `activation` | categorical | relu, elu, tanh |
| `dropout_rate` | float (uniform) | [0.0, 0.5] |
| `use_bn` | categorical | True, False |
| `learning_rate` | float (log-uniform) | [1e-5, 1e-2] |
| `batch_size` | categorical | 64, 128, 256, 512 |
| `l2` | float (log-uniform) or 0 | [1e-6, 1e-2] if enabled |
| `patience` | fixed | 10 |
| `epochs` | fixed | 100 |

`n_layers` and `units_lᵢ` are coupled: Optuna first samples `n_layers`, then samples one `units_lᵢ` per layer. This means the search space is variable-dimensional, which TPE handles correctly.

`l2` is gated by a categorical `use_l2` flag: if `False`, `l2` is set to `0.0` and no regulariser is attached, avoiding the overhead of near-zero penalty terms.

### Adding a new model

Subclass `BaseModel`, implement `fit`, `predict_proba`, and `save`, override `suggest_params` to expose your search space, then register the class in `models/__init__.py`:

```python
from .my_model import MyModel
__all__ = ["BaseModel", "MLPModel", "MyModel"]
```

The trainer requires no other changes.